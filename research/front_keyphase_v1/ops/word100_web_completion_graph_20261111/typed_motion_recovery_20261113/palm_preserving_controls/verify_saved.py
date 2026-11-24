"""Read-only source/saved-output audit; never regenerate a fixture or run a model."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (U / 'SAVED_AUDIT.json').exists()
    frozen, result, visuals = [read(U / name) for name in ('FIXTURE_FROZEN.json', 'FIXTURE_RESULTS.json', 'VISUALS.json')]
    sources = {r['clip_id']: r for r in frozen['sources']}
    checks = []
    def check(name, yes):
        checks.append(dict(name=name, passed=bool(yes)))
    for path, expected in frozen['pins'].items():
        check('source/code pin:' + path, sha(path) == expected)
    check('contract hash', sha(U / 'FIXTURE_FROZEN.json') == result['frozen_sha256'])
    for row in result['rows']:
        source = sources[row['clip_id']]
        prefix = row['clip_id'] + ':' + row['side'] + ':'
        check(prefix + 'saved hash', sha(row['path']) == row['sha256'])
        with np.load(source['raw_path'], allow_pickle=False) as raw, np.load(row['path'], allow_pickle=False) as z:
            for key in ('score', 'timestamp_us', 'frame_id', 'source_width', 'source_height'):
                check(prefix + key, raw[key].tobytes() == z[key].tobytes())
            check(prefix + 'saved baseline equals immutable source', z['original_xy_px'].tobytes() == raw['xy_px'].tobytes())
            allowed = np.zeros(z['synthetic_point'].shape, bool)
            base = 91 if row['side'] == 'L' else 112
            distal = base + np.array([6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])
            applied = [r['index'] for r in row['frame_report'] if r['applied']]
            allowed[np.ix_(applied, distal)] = True
            check(prefix + 'synthetic mask exact by report', np.array_equal(allowed, z['synthetic_point']))
            check(prefix + 'protected source bytes', z['xy_px'][~allowed].tobytes() == raw['xy_px'][~allowed].tobytes())
            check(prefix + 'requested full denominator', np.array_equal(z['requested_indices'], source['frozen_indices']))
            check(prefix + 'applied/rejected denominator', row['requested_frames'] == row['applied_frames'] + row['rejected_frames'] == len(source['frozen_indices']))
            check(prefix + 'applied count', len(applied) == row['applied_frames'])
            bad = [r['index'] for r in row['frame_report'] if not r['applied']]
            check(prefix + 'rejected unchanged', z['xy_px'][bad].tobytes() == raw['xy_px'][bad].tobytes())
            check(prefix + 'rejection reasons', dict(Counter(r['reason'] for r in row['frame_report'] if not r['applied'])) == row['rejections'])
            # Independently derive exactly which 280 columns can change.
            tcn_allowed = np.zeros(z['original_tcn280'].shape, bool)
            common = 9 if row['side'] == 'L' else 30
            local = 102 if row['side'] == 'L' else 144
            columns = [2*(common+j)+k for j in distal-base for k in (0, 1)] + [local+2*j+k for j in distal-base for k in (0, 1)]
            tcn_allowed[np.ix_(applied, columns)] = True
            check(prefix + 'allowed channels independently derived', np.array_equal(tcn_allowed, z['allowed_tcn_change']))
            check(prefix + 'protected saved channels', z['original_tcn280'][~tcn_allowed].tobytes() == z['control_tcn280'][~tcn_allowed].tobytes())
            check(prefix + 'changed actual distal values', np.any(z['xy_px'][allowed] != raw['xy_px'][allowed]))
    check('source/control cardinality', len(sources) == 12 and len(result['rows']) == 24)
    check('requested summary', result['requested_frames'] == sum(r['requested_frames'] for r in result['rows']))
    check('applied summary', result['applied_frames'] == sum(r['applied_frames'] for r in result['rows']))
    check('rejected summary', result['rejected_frames'] == sum(r['rejected_frames'] for r in result['rows']))
    expected_geometry = sum(len(s['events']) for s in sources.values()) * 2 * 2 * 2 * 3
    check('geometry full denominator', len(result['geometry_checks']) == result['checks'] == expected_geometry)
    check('geometry invariance flags', all(r['exact'] and r['eligible_before'] == r['eligible_after'] for r in result['geometry_checks']))
    for visual in visuals['visuals']:
        check('visual hash:' + visual['path'], sha(visual['path']) == visual['sha256'])
    audit = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL', checks=checks,
        check_count=len(checks), failed=sum(not c['passed'] for c in checks),
        reviewer='Same implementing Codex, separate saved-byte arithmetic; not independent root/human review',
        result_sha256=sha(U / 'FIXTURE_RESULTS.json'), verifier_sha256=sha(Path(__file__)),
        fixture_regenerated=False, model_inference=False, source_modified=False)
    (U / 'SAVED_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(audit['status'], audit['check_count'], 'saved checks; failed', audit['failed'], flush=True)
    assert audit['status'] == 'PASS'


if __name__ == '__main__':
    main()
