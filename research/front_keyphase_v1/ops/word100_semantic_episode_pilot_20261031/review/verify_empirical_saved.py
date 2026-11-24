"""Audit saved paths/results only. No model import, inference, detection or DTW search."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
U = HERE.parent
pins = {}
counts = Counter()
failures = []


def load(p):
    p = Path(p)
    b = p.read_bytes()
    pins[str(p.resolve())] = hashlib.sha256(b).hexdigest()
    return json.loads(b.decode('utf-8-sig'))


def archive(p):
    p = Path(p)
    pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ck(category, name, ok, detail=None):
    counts[category] += 1
    if not bool(ok):
        failures.append(dict(category=category, name=name, detail=detail))


def eq(category, name, got, expected):
    if isinstance(expected, (float, np.floating)):
        ok = got is not None and np.isclose(got, expected, atol=2e-7, rtol=2e-6)
    else:
        ok = got == expected
    ck(category, name, ok, dict(saved=got, recomputed=expected) if not ok else None)


auth = load(U / 'AUTHORIZATION.json')
frozen = load(U / 'FROZEN.json')
emp = load(U / 'EMPIRICAL_RANGE.json')
controls = load(U / 'CONTROLS.json')
ev = load(U / 'EVALUATION.json')
gate = load(U / 'FINAL_GATE.json')
prepared = load(U / 'PREPARED.json')
rows = {r['alias']: r for r in auth['rows']}
data = {a: archive(r['input']) for a, r in rows.items()}
auto = {(s, a): archive(U / 'runs' / str(s) / 'auto' / f'{a}.npz')
        for s in [42, 43, 44] for a in rows}
for a, r in rows.items():
    eq('input_integrity', a, pins[str(Path(r['input']).resolve())], r['input_sha256'])
for name in ['episode.py', 'experiment.py', 'review/EPISODE_CONTRACT_REVIEW.json',
             'review/SOURCE_USAGE_AND_RANGE.json', 'LABELS_FIT.json', 'LABELS_DEV.json']:
    p = U / name
    pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    ck('frozen_integrity', name, frozen['pins'].get(str(p)) == pins[str(p.resolve())]
       or frozen['pins'].get(str(p.resolve())) == pins[str(p.resolve())])

COLS = [28, 29, 36, 37, 30, 31, 32, 33] + list(range(39, 53))


def known_centers(a):
    valid = a['mask'][:, [28, 29]].all(1) & np.isfinite(a['x'][:, [28, 29]]).all(1)
    eligible = []
    for i in range(len(valid)):
        lo, hi = max(i - 6, 0), min(i + 7, len(valid))
        v = valid[lo:hi]
        continuous = (np.diff(a['frame_id'][lo:hi]) == 1).all() and (np.diff(a['timestamp_us'][lo:hi]) <= 150000).all()
        eligible.append(len(v) >= 3 and continuous and v.mean() >= .75
                        and v[:min(3, len(v))].any() and v[-min(3, len(v)):].any() and valid[i])
    return np.array(eligible, bool)


pair_summary = []


def verify_pair(tag, q, r, result, reference_state, query_state=None):
    category = 'saved_path_arithmetic'
    qlookup = {int(f): i for i, f in enumerate(q['frame_id'])}
    rlookup = {int(f): i for i, f in enumerate(r['frame_id'])}
    original_path = result['path_original_frames']
    path = np.array([(qlookup[i], rlookup[j]) for i, j in original_path], int)
    qi, ri = path.T
    steps = np.diff(path, axis=0)
    ck(category, tag + '/start_end', path[0].tolist() == [0, 0]
       and path[-1].tolist() == [len(q['x']) - 1, len(r['x']) - 1])
    ck(category, tag + '/monotone_steps', ((steps >= 0) & (steps <= 1)).all() and (steps.sum(1) >= 1).all())
    ck(category, tag + '/band', (np.abs(qi / max(len(q['x']) - 1, 1) - ri / max(len(r['x']) - 1, 1)) <= .25).all())
    ck(category, tag + '/all_query', np.array_equal(np.unique(qi), np.arange(len(q['x']))))
    ck(category, tag + '/all_reference', np.array_equal(np.unique(ri), np.arange(len(r['x']))))
    eq(category, tag + '/query_frame_count', result['all_query_frames_accounted'], len(q['x']))
    eq(category, tag + '/reference_frame_count', result['reference_frames'], len(r['x']))
    eq(category, tag + '/reference_original_ids', result['reference_frame_ids'], r['frame_id'].tolist())
    eq(category, tag + '/scope_retained', result['reference_scope_retained'], True)
    median = [int(q['frame_id'][int(np.median(qi[ri == j]))]) for j in range(len(r['x']))]
    eq(category, tag + '/display_median_only', result['query_map'], median)

    x, y = q['x'][:, COLS], r['x'][:, COLS]
    qm = q['mask'][:, COLS] & np.isfinite(x)
    rm = r['mask'][:, COLS] & np.isfinite(y)
    shared = qm[qi] & rm[ri]
    observed = np.minimum(np.abs(x[qi] - y[ri]), 1)
    max_delta = np.zeros((len(y), len(COLS)), np.float64)
    union = np.zeros_like(rm)
    # Independent aggregation over the frozen path; no path search is called.
    for j in range(len(y)):
        ix = ri == j
        max_delta[j] = np.max(np.where(shared[ix], observed[ix], 0), axis=0)
        union[j] = shared[ix].any(0)
    expected = int(rm.sum())
    compared = int(union.sum())
    path_missing = int((rm[ri] & ~shared).sum())
    pos = float(max_delta.sum() / compared) if compared else None
    qknown = query_state >= 0 if query_state is not None else known_centers(q)
    rknown = reference_state >= 0
    direction_expected = int(rknown[ri].sum())
    direction_compared = int((rknown[ri] & qknown[qi]).sum())
    direction_missing = direction_expected - direction_compared
    if query_state is not None:
        disagreements = int((rknown[ri] & qknown[qi] & (query_state[qi] != reference_state[ri])).sum())
        eq(category, tag + '/direction_disagreements', result['direction_disagreements'], disagreements)
    else:
        disagreements = result['direction_disagreements']
        ck(category, tag + '/saved_disagreement_bounds', 0 <= disagreements <= direction_compared)
    direction = disagreements / direction_compared if direction_compared else None
    distance = pos + .25 * (direction or 0) if pos is not None else None
    ref_missing = r['frame_id'][~rm.any(1)].tolist()
    ref_direction_missing = r['frame_id'][~rknown].tolist()
    complete = expected > 0 and not path_missing and not direction_missing and not ref_missing and not ref_direction_missing
    den = rm[ri].sum(1)
    geometric_sum = np.where(shared, observed, 0).sum(1) + (rm[ri] & ~qm[qi]).sum(1)
    geometric_cost = np.divide(geometric_sum, den, out=np.full(len(path), .5), where=den > 0)
    alignment_cost = float(geometric_cost.mean() + .25 * disagreements / len(path))
    values = dict(expected=expected, compared=compared, missing=expected - compared,
                  coverage=compared / max(expected, 1), path_missing=path_missing,
                  observed_position_shape_distance=pos, observed_direction_distance=direction,
                  distance=distance, alignment_cost=alignment_cost,
                  reference_unavailable_frames=ref_missing,
                  reference_unavailable_feature_cells=int(rm.size - expected),
                  reference_direction_unavailable_frames=ref_direction_missing,
                  direction_expected_pairs=direction_expected, direction_compared_pairs=direction_compared,
                  direction_missing_pairs=direction_missing,
                  uncompared_reference_frames=r['frame_id'][(rm & ~union).any(1)].tolist(),
                  query_frames_with_missing_expected=np.unique(q['frame_id'][qi[(rm[ri] & ~shared).any(1)]]).tolist(),
                  status='comparable' if compared / max(expected, 1) >= .75 and expected else 'insufficient',
                  evidence_status='complete_relative_to_reference' if complete else 'incomplete_observation')
    for k, v in values.items():
        eq(category, tag + '/' + k, result[k], v)
    pair_summary.append(dict(id=tag, query_frames=len(q['x']), reference_frames=len(r['x']),
                             path_pairs=len(path), expected=expected, compared=compared,
                             path_missing=path_missing, distance=result['distance'],
                             evidence_status=result['evidence_status'],
                             direction_labels_independently_available=query_state is not None))


thresholds = {}
for c in emp['calibration']:
    fit = {a for a, r in rows.items() if r['word_id'] == c['word_id'] and r['role'] == 'fit'}
    eq('calibration', c['word_id'] + '/three_fit', len(fit), 3)
    eq('calibration', c['word_id'] + '/lopo_ids', sorted(x['query'] for x in c['lopo']), sorted(fit))
    nearests = []
    for lopo in c['lopo']:
        q = lopo['query']
        eq('calibration', q + '/other_references', sorted(p['reference'] for p in lopo['pairs']), sorted(fit - {q}))
        for p in lopo['pairs']:
            r = p['reference']
            verify_pair(f'lopo/{q}/{r}', data[q], data[r], p, auto[(42, r)]['pred'], auto[(42, q)]['pred'])
        nearest = min(p['distance'] for p in lopo['pairs']) if all(p['status'] == 'comparable' for p in lopo['pairs']) else None
        eq('calibration', q + '/nearest', lopo['nearest'], nearest)
        nearests.append(nearest)
    threshold = max(nearests) if all(v is not None for v in nearests) else None
    eq('calibration', c['word_id'] + '/threshold', c['threshold'], threshold)
    thresholds[c['word_id']] = threshold

for d in emp['development']:
    q = d['alias']
    fit = sorted(a for a, r in rows.items() if r['word_id'] == d['word_id'] and r['role'] == 'fit')
    eq('development', q + '/role', rows[q]['role'], 'development')
    eq('development', q + '/all_reference_ids', sorted(p['reference'] for p in d['pairs']), fit)
    for p in d['pairs']:
        r = p['reference']
        verify_pair(f'development/{q}/{r}', data[q], data[r], p, auto[(42, r)]['pred'], auto[(42, q)]['pred'])
    best = min(d['pairs'], key=lambda p: p['distance'] if p['distance'] is not None else float('inf'))
    valid = all(p['status'] == 'comparable' for p in d['pairs']) and thresholds[d['word_id']] is not None
    proximity = ('within_empirical_range' if best['distance'] <= thresholds[d['word_id']] else 'outside_empirical_range') if valid else 'insufficient'
    eq('development', q + '/nearest_ref', d['nearest_reference'], best['reference'])
    eq('development', q + '/distance', d['nearest_distance'], best['distance'])
    eq('development', q + '/threshold', d['threshold'], thresholds[d['word_id']])
    eq('development', q + '/observed_proximity', d['observed_proximity'], proximity)
    eq('development', q + '/evidence_status', d['evidence_status'], 'complete_relative_to_references'
       if all(p['evidence_status'] == 'complete_relative_to_reference' for p in d['pairs']) else 'incomplete_observation')
    eq('development', q + '/linguistic_null', d['educational_verdict'], None)

kinds = ['timestamps_only_speed_0.75', 'timestamps_only_speed_1.25', 'raw_translation_uniform_scale',
         'all_missing', 'short_missing', 'reverse', 'delete_middle']
eq('controls', 'exact112rows', len(controls['rows']), 112)
eq('controls', 'saved_total', controls['total'], len(controls['rows']))
eq('controls', 'saved_passed', controls['passed'], sum(bool(x['pass_']) for x in controls['rows']))
for a in rows:
    eq('controls', a + '/seven_fixed_controls', sorted(x['test'] for x in controls['rows'] if x['alias'] == a), sorted(kinds))
for c in controls['rows']:
    a, kind = c['alias'], c['test']
    base = data[a]
    if 'result' not in c:
        if kind.startswith('timestamps'):
            eq('controls', a + '/' + kind + '/reference_frames', c['reference_frames'], len(base['x']))
            rm = base['mask'][:, COLS] & np.isfinite(base['x'][:, COLS])
            eq('controls', a + '/' + kind + '/expected', c['expected'], int(rm.sum()))
        continue  # No saved transformed predictions. Do not pretend to verify them.
    q = {k: base[k].copy() for k in ['x', 'mask', 'frame_id', 'timestamp_us']}
    n = len(q['x'])
    lo, hi = n // 2 - 2, n // 2 + 3
    if kind == 'all_missing':
        q['mask'][:] = False
    elif kind == 'short_missing':
        q['mask'][lo:hi] = False
    elif kind == 'reverse':
        q['x'], q['mask'] = q['x'][::-1].copy(), q['mask'][::-1].copy()
    elif kind == 'delete_middle':
        keep = np.r_[np.arange(lo), np.arange(hi, n)]
        q = {k: v[keep].copy() for k, v in q.items()}
    else:
        raise ValueError(kind)
    p = c['result']
    known = known_centers(q)
    verify_pair(f'control/{a}/{kind}', q, base, p, auto[(42, a)]['pred'],
                np.full(len(q['x']), -1) if kind == 'all_missing' else None)
    contract = p['reference_frames'] == len(base['x']) and p['expected'] == int((base['mask'][:, COLS] & np.isfinite(base['x'][:, COLS])).sum())
    if kind == 'all_missing':
        contract &= not known.any() and p['status'] == 'insufficient'
        eq('controls', a + '/all_missing_distance', p['distance'], None)
    if kind == 'short_missing':
        contract &= not known[lo:hi].any()
        eq('controls', a + '/short_missing_status', p['evidence_status'], 'incomplete_observation')
    eq('controls', a + '/' + kind + '/contract_flag', c['pass_'], bool(contract))


def metrics(truth, pred):
    truth, pred = np.array(truth), np.array(pred)
    cm = np.zeros((3, 4), int)
    for t, p in zip(truth, pred):
        cm[int(t), 3 if p < 0 else int(p)] += 1
    per = [float(2 * cm[k, k] / max(cm[k, :].sum() + cm[:, k].sum(), 1)) for k in range(3)]
    return dict(n=len(truth), correct=int((truth == pred).sum()), accuracy=float((truth == pred).mean()),
                unknown=int((pred < 0).sum()), macro_f1=float(np.mean(per)), class_f1=per,
                conditional_accuracy=float((truth[pred >= 0] == pred[pred >= 0]).mean()) if (pred >= 0).any() else None,
                confusion=cm.tolist())


events = ev['events']
for m in ev['known_windows']:
    selected = [e for e in events if e['role'] == m['role']]
    truth = [e['class_index'] for e in selected]
    pred = [e['candidate'][m['condition'].split('_')[-1]] if m['condition'].startswith('candidate') else e[m['condition']] for e in selected]
    values = metrics(truth, pred)
    for k, v in values.items():
        eq('known_window_metrics', m['condition'] + '/' + m['role'] + '/' + k, m[k], v)
for s in [42, 43, 44]:
    p = U / 'runs' / str(s) / 'window_predictions.npy'
    pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    pred = np.load(p, allow_pickle=False)
    ck('known_window_metrics', f'{s}/saved_prediction_matches_event_rows',
       np.array_equal(pred, [e['candidate'][str(s)] for e in events]))
    for a in rows:
        ck('automatic_support', f'{s}/{a}/eligible_equals_observed_support',
           np.array_equal(auto[(s, a)]['pred'] >= 0, known_centers(data[a])))

for m in ev['automatic_core']:
    seed = int(m['condition'].split('_')[-1]) if m['condition'].startswith('candidate') else 42
    key = 'pred' if m['condition'].startswith('candidate') else m['condition']
    pred, truth = [], []
    for e in events:
        if e['role'] != 'development':
            continue
        frame = data[e['alias']]['frame_id']
        state = auto[(seed, e['alias'])][key]
        part = state[(frame >= e['core_start_frame']) & (frame <= e['core_end_frame'])]
        p = int(np.bincount(part[part >= 0], minlength=3).argmax()) if (part >= 0).sum() >= .75 * len(part) else -1
        pred.append(p)
        truth.append(e['class_index'])
    for k, v in metrics(truth, pred).items():
        eq('automatic_core_metrics', m['condition'] + '/' + k, m[k], v)

dev = {x['condition']: x for x in ev['known_windows'] if x['role'] == 'development'}
autos = {x['condition']: x for x in ev['automatic_core']}
candidates = [dev[f'candidate_{s}'] for s in [42, 43, 44]]
mean_candidate = float(np.mean([c['macro_f1'] for c in candidates]))
mean_auto = float(np.mean([autos[f'candidate_{s}']['macro_f1'] for s in [42, 43, 44]]))
parts = dict(old_plus_005=mean_candidate >= dev['old']['macro_f1'] + .05,
             geometry_no_worse=mean_candidate >= dev['geometry']['macro_f1'],
             other_count_no_worse=all(c['confusion'][2][2] >= dev['old']['confusion'][2][2] for c in candidates),
             automatic_core_no_worse=mean_auto >= autos['old']['macro_f1'])
quality = all(parts.values())
eq('gate', 'quality', ev['provisional_quality_gate_pass'], quality)
eq('gate', 'final_quality', gate['physical_core_quality'], quality)
eq('gate', 'all_contract_controls_saved_arithmetic', gate['all_contract_controls'], all(c['pass_'] for c in controls['rows']))
eq('gate', 'final_development_pass', gate['physical_core_development_pass'], quality and all(c['pass_'] for c in controls['rows']))
for key in ['automatic_boundary_adoption', 'educational_validation', 'production_replacement']:
    eq('gate', key, gate[key], False)

short = [c for c in controls['rows'] if c['test'] == 'short_missing']
reverse = [c for c in controls['rows'] if c['test'] == 'reverse']
deletion = [c for c in controls['rows'] if c['test'] == 'delete_middle']
summary = dict(
    threshold_by_word=thresholds,
    development=[{k: v for k, v in d.items() if k != 'pairs'} for d in emp['development']],
    original_comparisons=36,
    original_comparisons_incomplete=sum(p['evidence_status'] == 'incomplete_observation' for p in pair_summary if not p['id'].startswith('control')),
    controls_rows=112, controls_per_clip=7, full_control_paths=64,
    total_frozen_paths_verified=len(pair_summary),
    all_missing=dict(n=16, null_distance=sum(c['result']['distance'] is None for c in controls['rows'] if c['test'] == 'all_missing')),
    short_missing=dict(n=16, nonzero_observed=[dict(alias=c['alias'], distance=c['result']['distance']) for c in short if c['result']['distance'] > 0], all_incomplete=all(c['result']['evidence_status'] == 'incomplete_observation' for c in short)),
    reversal=dict(n=16, all_positive_observed=all(c['result']['observed_position_shape_distance'] > 0 for c in reverse), min_distance=min(c['result']['distance'] for c in reverse), max_distance=max(c['result']['distance'] for c in reverse)),
    deletion=dict(n=16, all_positive_observed=all(c['result']['observed_position_shape_distance'] > 0 for c in deletion), min_distance=min(c['result']['distance'] for c in deletion), max_distance=max(c['result']['distance'] for c in deletion), all_reference_coverage_one=all(c['result']['coverage'] == 1 for c in deletion)),
    known_core=dict(candidate_mean_accuracy=float(np.mean([x['accuracy'] for x in candidates])), old_accuracy=dev['old']['accuracy'], geometry_accuracy=dev['geometry']['accuracy'],
                    candidate_mean_macro_f1=mean_candidate, old_macro_f1=dev['old']['macro_f1'], geometry_macro_f1=dev['geometry']['macro_f1'],
                    candidate_correct=7, total=16, unknown=8, candidate_conditional_accuracy=.875,
                    candidate_minus_old_macro_f1=mean_candidate-dev['old']['macro_f1'], candidate_minus_geometry_macro_f1=mean_candidate-dev['geometry']['macro_f1']),
    automatic_core=dict(candidate_correct=7, old_correct=3, geometry_correct=4, total=16, unknown=9, candidate_mean_macro_f1=mean_auto, old_macro_f1=autos['old']['macro_f1'], geometry_macro_f1=autos['geometry']['macro_f1'], candidate_conditional_accuracy=1.0),
    quality_gate_parts=parts, final_adoption=False)

report = dict(
    schema='wis_empirical_saved_result_verification_v1',
    created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    reviewer='Codex /root/guide_stage_map; independent arithmetic of saved outputs, same model family, not independent human truth',
    status='PASS_SAVED_ARITHMETIC_WITH_EXPLICIT_UNVERIFIED_CONTROL_COMPONENTS' if not failures else 'FAIL_SAVED_ARITHMETIC',
    execution=dict(model_inference=0, training=0, detector_calls=0, DTW_path_search=0, new_feature_extraction=0,
                   video_access=0, source_changes=0, old_artifact_changes=0,
                   method='Independent NumPy arithmetic on saved arrays and saved paths; never imports episode.py/experiment.py'),
    check_count=sum(counts.values()), checks_by_category=dict(counts), failure_count=len(failures), failures=failures,
    input_pins=[dict(path=p, sha256=h) for p,h in sorted(pins.items())],
    summary=summary, pair_summary=pair_summary,
    verified_scope=[
        'All24 fit LOPO pairs and12 development pairs use the correct same-expression fit references, all3pseudoqueries and fixedfit-only maximum thresholds.',
        'All100 saved original/control paths preserve endpoints, legal monotone steps, the normalized band, every query frame and every original reference frame.',
        'All100 paths: reference expected/compared/coverage, missing and reference-unavailable cells, observed position/shape maxima, display median and arithmetic distance/alignments rederived.',
        'Original36 comparisons: saved query/reference motion states independently rederive all direction disagreement counts and completeness statuses.',
        'Controls64 detailed paths: reference motion states and query eligibility independently rederive known/missing direction denominators. All-missing direction labels are known-1 without a model.',
        'Development known/automatic core scores and final no-adoption gate independently recomputed from saved predictions and target rows.',
    ],
    verification_limits=[
        'Speed32 and translation/scale16 control rows save pass flags without transformed features/predictions/distances. Their presence, count and saved flags are verified; prediction equality cannot be independently proved from these artifacts without prohibited reinference.',
        'Short-missing/reverse/delete48 controls do not save transformed learned classes. Their support denominators, coordinate arithmetic and disagreement-rate arithmetic are verified, but the disagreement numerator is only range-checked against support and retained as reported. No guessed predictions were inserted.',
        'This audit validates the frozen paths and arithmetic, not an independent recomputation of globally optimal DTW paths.',
        'Control112/112 is implementer-recorded contract success, not112 independently established model-behavior successes. Source/detector correctness and human sign correctness are outside this arithmetic audit.',
    ],
    interpretation=[
        'Known supplied cores improve over old4/16 to7/16, but geometry obtains8/16; macroF1 falls below the predeclared geometry gate. Candidate remains unadopted despite automatic-core improvement.',
        'All3seeds have the same aggregate scores on the same consumed16clips. They are not independent examples or evidence of unseen-person/expression generalization.',
        'Conditional87.5% known-core /100% automatic-core accuracy excludes8/9 unknown cases respectively. Full denominators remain16; these conditional percentages cannot be used alone.',
        'Observed empirical range is within for3dev clips and outside forP048. All4dev and all36fit/dev pair comparisons are incomplete. None is a complete normal-performance verdict.',
        'P048 is a nominal public reference demonstration outside the small fit3range:1/4potential warning on nominal development examples, not a validated linguistic false-positive rate. Its form/source/observability differences and source mismatch remain unresolved.',
        'Mask-only short_missing changes4of16 observed DTW distances slightly abovezero because missing support changes correspondence. The source coordinates themselves were not physically changed. These residuals are alignment sensitivity, not proof of a performed movement error.',
        'Reversal and central5-frame deletion produce positive observed distance in16/16 each. Central deletion can still havecoverage1 because DTW covers reference cells with nearby query frames. Original query gaps/direction support and incomplete status must remain visible.',
        'Synthetic reversal/deletion are not expert real-error examples; central5-frame deletion is not necessarily deletion of a whole required episode. The range does not establish tolerance to arbitrary body proportions, camera changes or normal sign variation.',
    ],
    disposition='Keep candidate unadopted; display the empirical range and partial evidence only in the authorized isolated research replay. No retrospective threshold/seed/metric tuning from these results.',
)
for p,h in pins.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, f'Input changed during audit: {p}'
dest=HERE/'EMPIRICAL_RESULT_VERIFICATION.json'
with dest.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
memo=(f'저장 결과 독립 산술검산: {sum(counts.values())}개 확인, 실패 {len(failures)}개.\n'
      '새 모델 추론/학습/검출/DTW 경로 재탐색/원본 변경 없음. 저장100개 경로의 원본 프레임, 분모, 누락과 관측거리 계산을 확인했다.\n'
      'fit3 LOPO 경험범위와 dev 결과 산술은 일치한다. 4dev 중3은 관측범위 안, P048은 밖이며 모두 관측부족이다. 전체 정상/오류 판단이 아니다.\n'
      '후보는 주어진 구간7/16로 기존4/16보다 높지만 기하8/16보다 낮아 채택 실패가 맞다. 자동 core7/16의 조건부100%는 unknown9개를 제외한 값이다.\n'
      '통제112/112는 구현자 저장 상태다. speed/이동확대48행은 변형 예측을 저장하지 않아 독립 재증명 불가. 상세64경로의 좌표/분모는 검산했고 변형 분류48행의 불일치 분자는 저장 집계에 한정된다.\n'
      '마스킹만 한4/16에서 작은 관측거리 잔차가 생겼다. DTW 대응 민감도이며 수행한 동작의 오류가 아니다.\n')
with (HERE/'EMPIRICAL_RESULT_VERIFICATION_KO.txt').open('x',encoding='utf-8') as f:f.write(memo)
print(json.dumps(dict(status=report['status'],checks=report['check_count'],failure_count=len(failures),failures=failures[:10],report=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),summary=summary),ensure_ascii=False))
