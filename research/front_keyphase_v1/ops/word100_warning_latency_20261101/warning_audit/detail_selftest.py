"""Meaningful synthetic copy/short/missing/native-axis checks, no DP/model."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import hashlib
import numpy as np

U = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('details', U / 'detail_enrichment.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def preserved(a, b):
    if isinstance(a, dict):
        return all(k in b and preserved(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(preserved(v, z) for v, z in zip(a, b))
    return a == b


def main():
    tests = []
    def ck(value, name):
        if not value:
            raise AssertionError(name)
        tests.append(name)
    names = json.loads((D.OLD / 'PUBLIC_INPUTS.json').read_text(encoding='utf-8-sig'))['feature_names']
    def sample():
        return dict(x=np.zeros((32, 69)), mask=np.ones((32, 69), bool),
                    probabilities=np.tile([0., 1., 1., 0., 0., 0., 0.], (32, 1)),
                    frame_id=np.arange(32), timestamp_us=np.arange(32) * 33333, feature_names=names)
    def setup(q, r):
        qc, rc = D._engine().channels(q), D._engine().channels(r)
        groups = {}
        for f in D._engine().FAMILIES:
            valid = qc[f][1][10:16] & rc[f][1][10:16]
            delta = abs(qc[f][0][10:16] - rc[f][0][10:16])
            peak = float(delta[valid].max()) if valid.any() else None
            groups[f] = dict(peak_distance=peak, peak_at={'preserve': 'exact original'}, coverage=.73, distance=.42)
        stage = dict(canonical_stage_id='syn', families=groups, status='DIFFERENCE_CANDIDATE')
        refs = [dict(id=rid, path=[[i, i] for i in range(32)], stages=[deepcopy(stage)]) for rid in ['a', 'b']]
        top = dict(canonical_stage_id='syn', status='DIFFERENCE_CANDIDATE', message='original',
                   features=[dict(name=D._engine().LABELS[f], status='DIFFERENCE_CANDIDATE', distances=[.42, .42]) for f in groups])
        result = dict(references=refs, stages=[top], selected_whole_references=['a', 'b'])
        rows = [dict(clip_id=rid, stages=[dict(canonical_stage_id='syn', frames=[10, 15])]) for rid in ['a', 'b']]
        ranges = {f: {'p90': .2, 'peak': .5} for f in groups}
        return rows, result, ranges
    for duration in [1, 2, 5]:
        q, r = sample(), sample()
        q['x'][10:10+duration, 25] = 1.0
        q['x'][10:10+duration, 39] = .75
        q['probabilities'][10:10+duration, 1] = 0.
        q['probabilities'][10:10+duration, 3] = 1.
        rows, base, ranges = setup(q, r)
        before = deepcopy(base)
        xbefore = q['x'].copy()
        out = D.enrich(q, rows, {'a': r, 'b': r}, base, ranges)
        ck(preserved(base, out) and base == before, f'all existing fields/statuses preserved {duration}')
        ck(np.array_equal(xbefore, q['x']), f'input unchanged {duration}')
        details = out['references'][0]['stages'][0]['detail_enrichment']['families']
        for f in ['hand_axis', 'R_angles', 'R_shape']:
            ep = details[f]['exceedances']['peak_number_exceedance']['episodes']
            ck(len(ep) == 1 and ep[0]['sample_count'] == duration, f'exact {duration}-sample {f} event retained')
            ck(ep[0]['source_pts_span_us'] == (duration - 1) * 33333, f'native elapsed span {duration}/{f}')
            ck(details[f]['original_peak_preserved_in_trace'], f'raw peak preserved {duration}/{f}')
        ck(details['hand_axis']['interpretation']['actual_rotation_interpretation'] == 'UNRESOLVED', '2D difference not3D truth')
    q, r = sample(), sample()
    q['mask'][10:16] = False
    q['probabilities'][10:16] = np.nan
    rows, base, ranges = setup(q, r)
    out = D.enrich(q, rows, {'a': r, 'b': r}, base, ranges)
    for f, d in out['references'][0]['stages'][0]['detail_enrichment']['families'].items():
        ck(d['fully_unshared_query_samples'] == 6, 'missing opportunities retained ' + f)
        ck(all(z['max_abs_residual'] is None for z in d['trace']), 'unknown notzero ' + f)
        ck(not d['exceedances']['peak_number_exceedance']['episodes'], 'no fabricated missing difference ' + f)
    q, r = sample(), sample()
    q['x'][10:15, 25] = 1.
    q['frame_id'][13:] += 2
    rows, base, ranges = setup(q, r)
    out = D.enrich(q, rows, {'a': r, 'b': r}, base, ranges)
    ep = out['references'][0]['stages'][0]['detail_enrichment']['families']['hand_axis']['exceedances']['peak_number_exceedance']['episodes']
    ck([e['sample_count'] for e in ep] == [3, 2], 'source frame gap splits runs, no bridge')
    empty = dict(references=[], stages=[dict(status='INSUFFICIENT_EVIDENCE')])
    ck(preserved(empty, D.enrich({}, [], {}, empty, {})), 'unsupported result preserved')
    report = dict(status='PASS', checks=len(tests), tests=tests, synthetic_only=True, new_DTW=0, observer_calls=0,
                  code_sha256=hashlib.sha256((U / 'detail_enrichment.py').read_bytes()).hexdigest(),
                  selftest_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    p = U / 'warning_audit' / 'DETAIL_SELFTEST.json'
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks': len(tests), 'code_sha256': report['code_sha256']}))


if __name__ == '__main__':
    main()
