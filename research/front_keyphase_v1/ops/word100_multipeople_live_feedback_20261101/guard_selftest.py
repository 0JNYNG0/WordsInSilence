"""Synthetic guard contracts only; no public path or observer calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import evidence_guard as G

U = Path(__file__).resolve().parent


def main():
    checks = []
    def ck(value, label):
        if not value:
            raise AssertionError(label)
        checks.append(label)
    names = json.loads((U / 'REFERENCE_BANK.json').read_text(encoding='utf-8-sig'))['feature_names']
    def array():
        n = 32
        a = dict(x=np.zeros((n, 69)), mask=np.ones((n, 69), bool),
                 probabilities=np.tile([0., 1., 1., 0., 0., 0., 0.], (n, 1)),
                 frame_id=np.arange(n), timestamp_us=np.arange(n) * 33333,
                 feature_names=names)
        for col in G._arm_indices(a):
            a['x'][:, col] = np.arange(n) * .04
        return a
    clean = array()
    q = deepcopy(clean)
    q['mask'][10:14, 39:53] = False
    q['probabilities'][10:14, [0, 1, 3]] = np.nan
    q['mask'][10:14, 28:30] = False
    q['mask'][10:14, 53:55] = False
    q['mask'][10:14, 56:59] = False
    path = [[i, i] for i in range(6)]
    path += [[5, j] for j in range(6, 15)]
    path += [[i, 14] for i in range(6, 15)]
    path += [[i, i] for i in range(15, 32)]
    rows = [{'clip_id': rid, 'stages': [dict(id='local', canonical_stage_id='synthetic', frames=[10, 13])]} for rid in ['one', 'two']]
    refs = {rid: deepcopy(clean) for rid in ['one', 'two']}
    families = G._base().FAMILIES
    features = [dict(name=G._base().LABELS[f], status=G.SIMILAR, coverage=[1., 1.], distances=[.1, .2], peak_distances=[.2, .3]) for f in families]
    stage = dict(id='1', canonical_stage_id='synthetic', status=G.SIMILAR, message='original', features=features)
    base = dict(stages=[stage], selected_whole_references=['one', 'two'],
                references=[dict(id=rid, path=deepcopy(path), stages=[deepcopy(stage)], distance=.2) for rid in refs])
    before = json.dumps(base, sort_keys=True)
    qbefore = {k: v.copy() for k, v in q.items() if isinstance(v, np.ndarray)}
    cache = G.ArmPathCache()
    guarded = G.guard_result(q, rows, refs, base, cache=cache)
    ck(guarded['stages'][0]['status'] == G.INSUFFICIENT, 'missing stage cannot remain similar through primary-path scope escape')
    audit = guarded['guard_audit']['pairs']['one']['stages'][0]
    ck(audit['families']['R_angles']['original']['coverage'] == 1., 'original changed-scope support preserved')
    ck(audit['families']['R_angles']['union']['coverage'] < .65, 'union includes missing query times')
    ck(audit['families']['R_angles']['union']['query_indices'] == [5, 10, 11, 12, 13], 'exact query union, not hull filling')
    ck(audit['families']['R_angles']['union']['reference_expected'] == 56, 'complete original reference denominator')
    ck(audit['families']['R_angles']['union']['query_expected'] == 70, 'each query and attribute opportunity counted once')
    ck(audit['families']['R_angles']['union']['query_observed'] == 14, 'unknown does not count as observed zero difference')
    ck(guarded['references'][0]['path'] == path, 'original full path unchanged')
    ck(guarded['selected_whole_references'] == base['selected_whole_references'], 'same whole references retained')
    ck(json.dumps(base, sort_keys=True) == before, 'original result not mutated')
    for k, v in qbefore.items():
        ck(np.array_equal(q[k], v, equal_nan=True), 'original input immutable: ' + k)
    for f, old in zip(guarded['stages'][0]['features'], base['stages'][0]['features']):
        for key in ['coverage', 'distances', 'peak_distances']:
            ck(f[key] == old[key], 'raw metric preserved ' + old['name'] + '/' + key)
    ck(len(cache.records) == 1 and cache.requests[1]['cache_hit'], 'identical pose arrays reuse same auxiliary path')
    ck(G.arm_key(clean, refs['one']) == G.arm_key(q, refs['one']), 'hand removal does not affect arm path key')
    shifted = deepcopy(clean)
    shifted['x'][0, G._arm_indices(shifted)[0]] += 1e-9
    ck(G.arm_key(clean, refs['one']) != G.arm_key(shifted, refs['one']), 'exact raw pose change cannot reuse key')
    nominal = G.guard_result(clean, rows, refs, base, cache=cache)
    ck(nominal['stages'][0]['status'] == G.SIMILAR, 'fully observed unchanged synthetic remains similar')
    for status in ['DIFFERENCE_CANDIDATE', G.INSUFFICIENT]:
        b = deepcopy(base)
        b['stages'][0]['status'] = status
        b['stages'][0]['features'][0]['status'] = status
        z = G.guard_result(q, rows, refs, b, cache=cache)
        ck(z['stages'][0]['status'] == status, 'existing stage never promoted or relaxed: ' + status)
        ck(z['stages'][0]['features'][0]['status'] == status, 'existing feature never promoted or relaxed: ' + status)
    rest_rows = deepcopy(rows)
    for r in rest_rows:
        r['stages'][0]['confirmed_rest_sides'] = ['R']
    rb = deepcopy(base)
    for f in rb['stages'][0]['features']:
        if f['name'] in [G._base().LABELS[x] for x in ['R_shape', 'R_angles', 'hand_relation']]:
            f['status'] = 'NOT_REQUIRED_REFERENCE_REST'
    rest = G.guard_result(q, rest_rows, refs, rb, cache=cache)
    ck(rest['stages'][0]['features'][0]['status'] == 'NOT_REQUIRED_REFERENCE_REST', 'reference rest remains separate, not passed or missing')
    ck(rest['guard_audit']['pairs']['one']['stages'][0]['families']['R_angles']['union']['reference_expected'] == 0, 'identical reference-only rest coordinate scope')
    missing_pose = deepcopy(clean)
    missing_pose['mask'][:, G._arm_indices(missing_pose)] = False
    low = G.guard_result(missing_pose, rows, refs, base)
    ck(low['stages'][0]['status'] == G.INSUFFICIENT, 'no auxiliary pose evidence cannot establish support')
    ck('AUXILIARY_POSE_CORRESPONDENCE_SUPPORT_BELOW_0_65' in low['stages'][0]['features'][0]['guard_support']['reasons'][0], 'pose failure reason retained')
    empty = dict(stages=[dict(status=G.INSUFFICIENT)], references=[])
    out = G.guard_result({}, [], {}, empty)
    ck(out['stages'] == empty['stages'] and out['guard_audit']['status'] == 'NO_REFERENCE_COMPARISON', 'unsupported adapter result safe without fabricated arrays')
    failed = False
    try:
        bad = deepcopy(base)
        bad['selected_whole_references'] = ['one', 'one']
        G.guard_result(q, rows, refs, bad)
    except ValueError:
        failed = True
    ck(failed, 'duplicate selected whole identity rejected')
    report = dict(status='PASS', checks=len(checks), assertions=checks, synthetic_only=True,
                  public_path_runs=0, observer_calls=0, input_immutable=True,
                  code_sha256=G.sha(U / 'evidence_guard.py'), selftest_sha256=G.sha(__file__),
                  contract_sha256=G.sha(U / 'GUARD_FROZEN.json'), engine_sha256=G.EXPECTED_ENGINE_SHA)
    p = U / 'GUARD_SELFTEST_FINAL.json'
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=len(checks), code_sha256=report['code_sha256'])))


if __name__ == '__main__':
    main()
