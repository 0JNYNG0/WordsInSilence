"""Codex independent reference preparation; no candidate import or model inference."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from verify_reference import reference, assert_contract, filehash, load_json

HERE = Path(__file__).resolve().parent


def run():
    started = time.perf_counter()
    cfg = load_json(HERE / 'CONFIG_LOCAL_ONLY.json')
    for path, expected in cfg['pins'].items():
        assert filehash(path) == expected, path
    for key in ['lab_root', 'pilot_src', 'loader_dir', 'helper_dir', 'irregular_dir']:
        sys.path.insert(0, cfg[key])
    import run_audit as R
    import irregular_audit as I
    import corruptions as C
    import timegrid as TG
    import framerate_impl as F
    from pilot.data import build_channels
    from pilot.web_input import to_native_like
    api = dict(F=F, build_channels=build_channels, to_native_like=to_native_like,
               native_roots=cfg['native_roots'], mp_roots=cfg['mp_roots'])
    for mod in [R, I, C, TG, F, sys.modules['pilot.data'], sys.modules['pilot.web_input']]:
        path = Path(mod.__file__).resolve()
        assert filehash(path) == cfg['pins'][str(path)], str(path)
    rows = dict(native=R.load_native_rows(cfg['native_metadata']),
                mediapipe=R.load_mp_rows(cfg['mp_manifest']))
    old = load_json(cfg['prior_result'])
    regular = load_json(cfg['regular_result'])
    cases = ['clean', 'regular_half', *C.CASES]
    results = {}
    for g, rs in rows.items():
        source_before = R.hash_sources(rs, api['native_roots' if g == 'native' else 'mp_roots'])
        R.check_listed(rs, source_before, g)
        result_rows = []
        for i, row in enumerate(rs):
            base = I.prepare_base(R, api, g, row, cfg['tolerance'])
            assert base['meta']['source_raw_sha256'] == old['groups'][g]['cases'][C.CASES[0]]['rows'][i]['base']['source_raw_sha256']
            old_regular = regular['groups'][g]['rows'][i]
            assert old_regular['id'] == row['id']
            out_cases = {}
            for case in cases:
                if case == 'clean':
                    xy, pr, ts = base['xy'], base['pr'], base['ts']
                elif case == 'regular_half':
                    xy, pr, ts = base['xy'][::2], base['pr'][::2], base['ts'][::2]
                else:
                    cor = C.corrupt(base['xy'], base['pr'], base['ts'], row['id'], case)
                    xy, pr, ts = cor['xy'], cor['present'], cor['ts']
                original_hash = I.raw_hash(R, xy, pr, ts)
                output = reference(xy, pr, ts)
                assert_contract(output, xy, pr, ts)
                assert I.raw_hash(R, xy, pr, ts) == original_hash
                x = build_channels(output['xy'], output['present'], output['ts'])['x']
                source_x = build_channels(xy, pr, ts)['x']
                previous = TG.restore(xy, pr, ts, 'linear')
                global_x = build_channels(previous['xy'], previous['present'], previous['ts'])['x']
                hashes = {'original': R.sha256_array(base['x']), 'unrestored': R.sha256_array(source_x),
                          'global_linear': R.sha256_array(global_x), 'local_insert': R.sha256_array(x)}
                if case == 'clean':
                    assert len(set(hashes.values())) == 1 and output['identity']
                elif case == 'regular_half':
                    for a, key in [('original', 'original'), ('unrestored', 'sparse'), ('global_linear', 'linear')]:
                        assert hashes[a] == old_regular['input_sha256'][key], (g, i, case, a)
                else:
                    prev = old['groups'][g]['cases'][case]['rows'][i]
                    assert prev['id'] == row['id']
                    for a, key in [('original', 'original'), ('unrestored', 'corrupted'), ('global_linear', 'linear')]:
                        assert hashes[a] == prev['case']['input_sha256'][key], (g, i, case, a)
                assert np.isfinite(x).all() and x.dtype == np.float32 and x.shape[1] == 280
                reuse = next((a for a in ['unrestored', 'global_linear', 'original'] if hashes[a] == hashes['local_insert']), None)
                out_cases[case] = dict(input_sha256=hashes, candidate_raw_sha256=I.raw_hash(R, output['xy'], output['present'], output['ts']),
                                       inserted=int((output['exact_source_index'] < 0).sum()),
                                       estimated_joints=int(output['estimated'].sum()),
                                       input_frames=len(ts), output_frames=len(output['ts']),
                                       source_index_sha256=R.sha256_array(output['exact_source_index']),
                                       source_frame_count=int((output['exact_source_index'] >= 0).sum()),
                                       cache_match=reuse)
            result_rows.append(dict(index=i, id=row['id'], cases=out_cases))
        source_after = R.hash_sources(rs, api['native_roots' if g == 'native' else 'mp_roots'])
        assert source_before == source_after
        results[g] = dict(rows=result_rows, sources=source_before,
                          summary={case: dict(cells=len(rs), inserted=sum(r['cases'][case]['inserted'] for r in result_rows),
                                              newly_scored=sum(r['cases'][case]['cache_match'] is None for r in result_rows)) for case in cases})
    assert 'cv2' not in sys.modules and 'mediapipe' not in sys.modules
    for path, expected in cfg['pins'].items():
        assert filehash(path) == expected, path
    report = dict(status='PASS', reviewer='Codex independent Fraction reference; candidate not imported',
                  cells=sum(len(rs) for rs in rows.values()) * len(cases), model_forwards=0, groups=results,
                  protected_hashes=len(cfg['pins']), seconds=time.perf_counter() - started)
    with (HERE / 'INDEPENDENT_PREFLIGHT.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({g: v['summary'] for g, v in results.items()}, ensure_ascii=False, indent=2))
    print('PASS independent input contract', report['cells'], 'cells; no model inference.')


if __name__ == '__main__':
    run()
