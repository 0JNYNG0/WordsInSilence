"""Independent saved guard audit. No producer imports/model/DP/re-alignment."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import traceback
import numpy as np

U = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_guard_independent_helpers', Path(__file__).with_name('verify_guard_saved.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
F = A.FAMILIES
SIM = 'SIMILAR_REFERENCE_RANGE'
UNK = 'INSUFFICIENT_EVIDENCE'
checks = 0
failures = []
pins = {}


def check(value, name):
    global checks
    checks += 1
    if not bool(value):
        failures.append(name)
        if len(failures) >= 30:
            raise AssertionError('First30 independent discrepancies: ' + repr(failures))


def same(a, b):
    if isinstance(b, dict):
        return isinstance(a, dict) and all(k in a and same(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        return isinstance(a, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(b, float):
        return a is not None and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    return a == b


def pin(p, h=None):
    p = str(Path(p).resolve())
    value = A.sha(p)
    check(h is None or h == value, 'pin:' + p)
    check(p not in pins or pins[p] == value, 'stable_pin:' + p)
    pins[p] = value
    return value


def read(p, h=None):
    pin(p, h)
    return A.read(p)


def load(p, h=None):
    pin(p, h)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in ['x', 'mask', 'frame_id', 'timestamp_us']}


def arrays(row):
    out = load(row['input'], row['input_sha256'])
    pin(row['probability_file'], row['probability_sha256'])
    with np.load(row['probability_file'], allow_pickle=False) as z:
        out['probabilities'] = z['probabilities']
        check(np.array_equal(z['frame_id'], out['frame_id']) and
              np.array_equal(z['timestamp_us'], out['timestamp_us']), 'posterior_axes:' + row['clip_id'])
    return out


def support(qm, rm, links, bounds, shape=False):
    """Enumerate cell sets from independently grouped links, no producer logic."""
    qq = defaultdict(set)
    rr = defaultdict(set)
    for q, r in links:
        qq[int(q)].add(int(r))
        rr[int(r)].add(int(q))
    frames = range(bounds[0], bounds[1] + 1)
    dim = rm.shape[1]
    re = {(r, d) for r in frames for d in range(dim) if rm[r, d]}
    ro = {(r, d) for r, d in re if any(qm[q, d] for q in rr[r])}
    qe = {(q, d) for q, refs in qq.items() for d in range(dim) if any(rm[r, d] for r in refs)}
    qo = {(q, d) for q, d in qe if qm[q, d]}
    de = [sum(d == j for _, d in re) for j in range(dim)]
    do = [sum(d == j for _, d in ro) for j in range(dim)]
    dqe = [sum(d == j for _, d in qe) for j in range(dim)]
    dqo = [sum(d == j for _, d in qo) for j in range(dim)]
    rc, qc = len(ro) / len(re) if re else None, len(qo) / len(qe) if qe else None
    dc = [min(do[i] / max(de[i], 1), dqo[i] / max(dqe[i], 1)) for i in range(dim)]
    cov = min(rc, qc) if rc is not None and qc is not None else None
    if shape and re and qe:
        cov = min(dc)
    queries = sorted(qq)
    return dict(reference_expected=len(re), reference_observed=len(ro), query_expected=len(qe), query_observed=len(qo),
        reference_coverage=rc, query_coverage=qc, coverage=cov, dimension_reference_expected=de,
        dimension_reference_observed=do, dimension_query_expected=dqe, dimension_query_observed=dqo,
        dimension_coverage=dc, query_indices=queries,
        query_expected_mask=[[(q, d) in qe for d in range(dim)] for q in queries],
        query_observed_mask=[[(q, d) in qo for d in range(dim)] for q in queries],
        possible_query_cells=len(queries) * dim, possible_reference_cells=len(frames) * dim,
        expected_from_reference_only=True)


def arm_key(a, r, names, engine_sha, contract_sha):
    h = hashlib.sha256(b'arm-path-guard-v1' + engine_sha.encode() + contract_sha.encode())
    cols = [names.index(s + '_' + point + '_' + d) for s in ['L', 'R']
            for point in ['pose_wrist', 'elbow'] for d in ['x', 'y']]
    for seq in [a, r]:
        for v in [seq['x'][:, cols], seq['mask'][:, cols], seq['frame_id'], seq['timestamp_us']]:
            v = np.ascontiguousarray(v)
            h.update(str(v.shape).encode())
            h.update(v.dtype.str.encode())
            h.update(v.tobytes())
    return h.hexdigest()


def preserved(old, new):
    if isinstance(old, dict):
        for k, v in old.items():
            if k not in new:
                return False
            if k == 'status' and v == SIM and new[k] == UNK:
                continue
            if k == 'message' and old.get('status') == SIM and new.get('status') == UNK and new.get('guard_previous_message') == v:
                continue
            if not preserved(v, new[k]):
                return False
        return True
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(preserved(a, b) for a, b in zip(old, new))
    return old == new


def audit():
    report = read(U / 'GUARD_REPORT.json')
    execution = read(U / 'GUARD_EXECUTION.json', report['execution_sha256'])
    frozen = read(U / 'GUARD_FROZEN.json', report['contract_sha256'])
    contract_sha = A.sha(U / 'GUARD_FROZEN.json')
    engine_sha = frozen['source_pins'][str(U / 'engine.py')]
    pin(U / 'evidence_guard.py', report['code_sha256'])
    pin(U / 'run_evidence_guard.py', report['runner_sha256'])
    pin(Path(__file__).with_name('verify_guard_saved.py'))
    for p, h in execution['source_pins'].items():
        pin(p, h)
    check(len(execution['source_pins']) == 1118, 'execution1118originalpins')
    bank = read(U / 'REFERENCE_BANK.json')
    names = bank['feature_names']
    rows = {r['clip_id']: r for r in bank['rows']}
    refs = {k: arrays(r) for k, r in rows.items()}
    rmask = {k: A.independent_masks(a, names) for k, a in refs.items()}
    scope = {k: A.independent_scope(rows[k], rmask[k]) for k in rows}
    tasks = read(U / 'EVALUATION_TASKS.json')['tasks']
    oldreport = read(U / 'FEEDBACK_EVALUATION.json')
    original_cases = {r['id']: r for r in oldreport['case_index']}
    new_cases = {r['task_id']: r for r in report['case_index']}
    pair_index = {(r['task_id'], r['reference_id']): r for r in report['pair_index']}
    arm_index = {r['key']: r['source'] for r in report['arm_path_index']}
    check(len(tasks) == len(new_cases) == 96 and len(pair_index) == 864 and len(arm_index) == 648, 'exact_full_inventories')
    check(set(new_cases) == {t['id'] for t in tasks}, 'no_missing_or_added_case')
    arm = {}
    requests = []
    seen = set()
    tally = defaultdict(lambda: dict(cases=0, stages=Counter(), features=Counter(), stage_opportunities=0, feature_opportunities=0))
    changed = []
    feature_changes = []
    support_records = pair_stage_count = 0
    for task in tasks:
        tid = task['id']
        old = read(original_cases[tid]['path'], original_cases[tid]['sha256'])
        got = read(new_cases[tid]['path'], new_cases[tid]['sha256'])
        check(got['task_id'] == tid and got['query'] == task['query'] and got['kind'] == task['kind'], tid + ':identity')
        check(got['original_case'] == {k: original_cases[tid][k] for k in ['path', 'sha256']} and
              got['modified_input'] == old['modified_input'], tid + ':original_source_pointers')
        if old['modified_input']:
            ptr = old['modified_input']
            a = load(ptr['path'], ptr['sha256'])
            with np.load(ptr['path'], allow_pickle=False) as z:
                a['probabilities'] = z['probabilities']
        else:
            a = load(task['source_input'], task['source_input_sha256'])
            pin(task['source_probabilities'], task['source_probabilities_sha256'])
            with np.load(task['source_probabilities'], allow_pickle=False) as z:
                a['probabilities'] = z['probabilities']
        qm = A.independent_masks(a, names)
        paircalcs = {}
        keys = {}
        oldpointers = {r['reference']: r for r in old['comparisons']}
        for rid in task['references']:
            pp = pair_index[(tid, rid)]
            p = read(pp['path'], pp['sha256'])
            baseline = read(oldpointers[rid]['path'], oldpointers[rid]['sha256'])
            check(p['source_pair'] == oldpointers[rid] and p['query'] == task['query'] and
                  p['task_id'] == tid and p['reference_id'] == rid, tid + ':' + rid + ':source')
            g = p['guard']
            key = arm_key(a, refs[rid], names, engine_sha, contract_sha)
            keys[rid] = key
            check(g['arm_path_key'] == key and g['reference_id'] == rid and key in arm_index, tid + ':' + rid + ':arm_raw_identity')
            if key not in arm:
                arm[key] = read(arm_index[key]['path'], arm_index[key]['sha256'])
            ap = arm[key]
            check(g['arm_path_source'] == arm_index[key] and g['auxiliary_path'] == ap['path'], tid + ':' + rid + ':exact_saved_arm_path')
            check(ap['query_frame_ids'] == a['frame_id'].tolist() and ap['reference_frame_ids'] == refs[rid]['frame_id'].tolist() and
                  ap['query_timestamp_us'] == a['timestamp_us'].tolist() and ap['reference_timestamp_us'] == refs[rid]['timestamp_us'].tolist(), tid + ':' + rid + ':native_axes')
            check(A.complete_path(ap['path'], len(a['x']), len(refs[rid]['x'])) and
                  A.complete_path(baseline['path'], len(a['x']), len(refs[rid]['x'])), tid + ':' + rid + ':complete_unit_paths')
            expected_rules = [dict(stage_id=s['id'], frames=s['frames'], sides=s['confirmed_rest_sides'],
                evidence=s.get('reference_rest_evidence'), boundary_provenance=s.get('boundary_provenance', 'unknown'),
                linguistic_requiredness=None) for s in rows[rid]['stages'] if s.get('confirmed_rest_sides')]
            check(g['reference_scope_rules'] == expected_rules, tid + ':' + rid + ':reference_only_scope')
            check(len(g['stages']) == len(rows[rid]['stages']), tid + ':' + rid + ':stage_denominator')
            calcstages = []
            for si, (st, saved) in enumerate(zip(rows[rid]['stages'], g['stages'])):
                tag = f'{tid}:{rid}:{si}'
                bounds = st['frames']
                primary = [v for v in baseline['path'] if bounds[0] <= v[1] <= bounds[1]]
                auxiliary = [v for v in ap['path'] if bounds[0] <= v[1] <= bounds[1]]
                union = [list(v) for v in sorted(set(map(tuple, primary)) | set(map(tuple, auxiliary)))]
                queries = sorted({v[0] for v in union})
                check(saved['stage_index'] == si and saved['canonical_stage_id'] == st['canonical_stage_id'] and
                    saved['reference_bounds'] == bounds and saved['original_links'] == primary and
                    saved['auxiliary_links'] == auxiliary and saved['union_links'] == union, tag + ':exact_link_union')
                check(saved['query_frame_ids'] == a['frame_id'][queries].tolist() and
                      saved['query_timestamp_us'] == a['timestamp_us'][queries].tolist(), tag + ':union_query_axis')
                pose = support(qm['arm_path'], rmask[rid]['arm_path'] & scope[rid]['arm_path'], auxiliary, bounds)
                check(same(saved['auxiliary_arm_support'], pose), tag + ':auxiliary_only_pose_support')
                fs = {}
                for family in F:
                    actual = saved['families'][family]
                    rm = rmask[rid][family] & scope[rid][family]
                    ss = {}
                    for kind, links in [('original', primary), ('auxiliary', auxiliary), ('union', union)]:
                        ss[kind] = support(qm[family], rm, links, bounds, family.endswith('_shape'))
                        check(same(actual[kind], ss[kind]), tag + ':' + family + ':' + kind + ':all_support_cells')
                        support_records += 1
                    required = bool(scope[rid][family][bounds[0]:bounds[1]+1].any())
                    reasons = []
                    if required and (ss['union']['coverage'] is None or ss['union']['coverage'] < .65):
                        reasons.append('UNION_QUERY_OR_REFERENCE_FEATURE_SUPPORT_BELOW_0_65')
                    if required and (pose['coverage'] is None or pose['coverage'] < .65):
                        reasons.append('AUXILIARY_POSE_CORRESPONDENCE_SUPPORT_BELOW_0_65')
                    fs[family] = dict(required=required, sufficient=required and not reasons, reasons=reasons,
                                      coverage=ss['union']['coverage'], pose=pose['coverage'])
                    excluded = int(rmask[rid][family][bounds[0]:bounds[1]+1].sum() - rm[bounds[0]:bounds[1]+1].sum())
                    check(actual['comparison_required'] == required and actual['sufficient'] == fs[family]['sufficient'] and
                          actual['reasons'] == reasons and actual['reference_scope_excluded_expected'] == excluded,
                          tag + ':' + family + ':decision_scope')
                    raw = baseline['stages'][si]['families'][family]
                    check(ss['original']['reference_expected'] == raw['expected'] and
                          ss['original']['query_expected'] == raw['query_expected'] and
                          ss['union']['reference_expected'] + excluded == raw['original_expected'], tag + ':' + family + ':original_denominator')
                calcstages.append(fs)
                pair_stage_count += 1
            paircalcs[rid] = calcstages
        for mode, available in [('all9', task['references']), ('subset3', task['subset_references'])]:
            for rid in available:
                key = keys[rid]
                requests.append(dict(key=key, cache_hit=key in seen, query_id=task['query'], reference_id=rid))
                seen.add(key)
            before = old['modes'][mode]
            after = got['modes'][mode]
            selected = before['selected_whole_references']
            check(after['selected_whole_references'] == selected and len(selected) == 2, tid + ':' + mode + ':selected2_unchanged')
            check(set(after['guard_pair_sources']) == set(available), tid + ':' + mode + ':all_reference_pointers')
            for rid, ptr in after['guard_pair_sources'].items():
                check(ptr == {k: pair_index[(tid, rid)][k] for k in ['path', 'sha256']}, tid + ':' + mode + ':pair_pointer:' + rid)
            check(len(after['stages']) == len(before['stages']), tid + ':' + mode + ':stage_count')
            group = tally[(task['kind'], mode)]
            group['cases'] += 1
            changes = []
            for si, (oldstage, newstage) in enumerate(zip(before['stages'], after['stages'])):
                check(preserved(oldstage, newstage), tid + ':' + mode + ':' + str(si) + ':original_values_preserved')
                changed_names = []
                for j, (oldf, newf) in enumerate(zip(oldstage['features'], newstage['features'])):
                    family = F[j]
                    values = [paircalcs[rid][si][family] for rid in selected]
                    sufficient = all(v['sufficient'] for v in values)
                    expected_status = UNK if oldf['status'] == SIM and not sufficient else oldf['status']
                    check(newf['status'] == expected_status, f'{tid}:{mode}:{si}:{family}:independent_status')
                    check(same(newf['guard_support'], dict(reference_ids=selected, sufficient=sufficient,
                        union_coverage=[v['coverage'] for v in values], auxiliary_pose_coverage=[v['pose'] for v in values],
                        reasons=[v['reasons'] for v in values], threshold=.65, unchanged_original_coverage=True,
                        phase_correctness_verified=False)), f'{tid}:{mode}:{si}:{family}:guard_evidence')
                    transition = oldf['status'] + ' -> ' + expected_status
                    group['features'][transition] += 1
                    group['feature_opportunities'] += 1
                    if oldf['status'] != expected_status:
                        changed_names.append(oldf['name'])
                        feature_changes.append(dict(task=tid, kind=task['kind'], mode=mode, stage_index=si+1,
                            family=family, before=oldf['status'], after=expected_status))
                expected_status = UNK if oldstage['status'] == SIM and changed_names else oldstage['status']
                check(newstage['status'] == expected_status and newstage['guard_changed_features'] == changed_names,
                      f'{tid}:{mode}:{si}:stage_downgrade_only')
                group['stages'][oldstage['status'] + ' -> ' + expected_status] += 1
                group['stage_opportunities'] += 1
                changes.append(dict(before=oldstage['status'], after=expected_status, changed_features=changed_names))
                if oldstage['status'] != expected_status:
                    changed.append(dict(task_id=tid, kind=task['kind'], mode=mode, stage_index=si+1,
                        canonical_stage_id=oldstage.get('canonical_stage_id'), before=oldstage['status'], after=expected_status,
                        original_query_range_seconds=oldstage.get('query_range_seconds'),
                        selected_whole_references=selected, features=changed_names))
            check(after['guard_audit']['aggregate_changes'] == changes and after['guard_audit']['individual_changes'] == [],
                  tid + ':' + mode + ':change_ledger')
        if len(tally) and int(tid[1:]) % 16 == 0:
            print(json.dumps(dict(verified_cases=int(tid[1:]), checks=checks, failures=len(failures))), flush=True)
    summary = {}
    for (kind, mode), g in tally.items():
        summary.setdefault(kind, {})[mode] = {k: dict(v) if isinstance(v, Counter) else v for k, v in g.items()}
    check(summary == report['summary'], 'all_summary_counters')
    check(changed == report['changed_stage_records'], 'all_changed_stages')
    check(requests == report['arm_path_request_ledger'], 'cache_request_order_identity_hits')
    check(seen == set(arm_index), 'no_missing_extra_unique_arm_path')
    stages = sum(g['stage_opportunities'] for g in tally.values())
    features = sum(g['feature_opportunities'] for g in tally.values())
    check(stages == 624 == report['aggregate_stage_opportunities'] and features == 4368 == report['aggregate_feature_opportunities'], '624_stages_4368_features')
    check(pair_stage_count == 2808 and support_records == 58968, 'all864_pairs_2808_stages_three_support_views')
    check(len(requests) == 1152 == report['arm_path_requests'] and sum(r['cache_hit'] for r in requests) == 504 == report['arm_path_cache_hits'] and len(seen) == 648 == report['new_unique_arm_paths'], '648_new_504_reuse')
    remaining = {mode: summary['main_R_hand_missing'][mode]['stages'].get(SIM + ' -> ' + SIM, 0) for mode in ['all9', 'subset3']}
    check(report['acceptance']['remaining_missing_similar'] == remaining and
          report['acceptance']['no_missing_control_similar_remaining'] == all(v == 0 for v in remaining.values()), 'missing_acceptance')
    check(all(c['before'] == SIM and c['after'] == UNK for c in changed + feature_changes) and
          report['acceptance']['no_new_difference_candidate'], 'no_positive_promotion_or_new_difference')
    for path, digest in list(pins.items()):
        check(A.sha(path) == digest, 'end_source_pin:' + path)
    return dict(source_pins=pins, producer_report_sha256=A.sha(U / 'GUARD_REPORT.json'),
        original_execution_pins=len(execution['source_pins']), counts=dict(cases=96, modes=2, original_pairs=864,
            pair_stage_opportunities=pair_stage_count, independently_recomputed_support_blocks=support_records,
            aggregate_stages=stages, aggregate_features=features, unique_saved_arm_paths=len(seen),
            cache_requests=len(requests), cache_reuses=sum(r['cache_hit'] for r in requests)),
        summary=summary, changed_stages=changed, changed_features=feature_changes,
        remaining_missing_similar=remaining,
        limitations=['Both alignments are estimates. Guard support does not prove same phase, correct recurrence or linguistic correctness.',
            'The same96 exposed development cases diagnose the guard. Fixing their three missing-support counterexamples is not fresh held-out accuracy.',
            'All added abstentions are retained, including feature-only changes inside an already-insufficient stage.',
            'Original numeric metrics/source paths/selected references remain unchanged; only copied SIMILAR labels may become INSUFFICIENT.',
            'No live-integration behavior is audited here.'],
        execution=dict(producer_imports=0, new_DP=0, model_load=0, forward=0, training=0, RGB_reads=0))


if __name__ == '__main__':
    path = U / 'review/GUARD_RESULT_VERIFICATION.json'
    if path.exists():
        raise FileExistsError('Preserve first independent audit')
    report = {}
    try:
        report = audit()
    except BaseException:
        failure = U / 'review/GUARD_RESULT_VERIFICATION_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        failures.append(traceback.format_exc())
    report.update(status='PASS_INDEPENDENT_SAVED_GUARD_AUDIT' if not failures else 'FAIL_PRESERVED',
        checks=checks, failure_count=len(failures), failures=failures,
        checked_at=datetime.now(timezone.utc).isoformat(), verifier_sha256=A.sha(__file__))
    with path.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report.get(k) for k in ['status', 'checks', 'failure_count', 'counts', 'remaining_missing_similar']}))
    raise SystemExit(0 if not failures else 1)
