"""Pure NumPy contract counterexamples. Does not import observer or call models."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
ROOT = UNIT.parents[3]
code = (UNIT / 'episode.py').read_bytes()
experiment = (UNIT / 'experiment.py').read_bytes()
allowed = {'support', 'encode', 'geometry', 'old_predict', 'automatic', 'runs', 'compare'}
tree = ast.parse(code.decode('utf-8-sig'))
ns = {'np': np, 'COLS': [28, 29, 36, 37, 30, 31, 32, 33], 'SUPPORT': .75,
      'WINDOW': 13, 'CLASSES': ['up', 'down', 'other']}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name in allowed], type_ignores=[]),
             'reviewed_episode_snapshot', 'exec'), ns)
# A constant stub tests support gating only; this is not a learned model call.
ns['infer'] = lambda model, items: np.tile(np.array([[1., 0., 0.]]), (len(items), 1))


def arr(n):
    return dict(x=np.zeros((n, 69), np.float32), mask=np.ones((n, 69), bool),
                frame_id=np.arange(n), timestamp_us=np.arange(n) * 33333)


def comp(q, r, qs=None, rs=None):
    return ns['compare'](q, r, np.full(len(q['x']), 2) if qs is None else qs,
                         np.full(len(r['x']), 2) if rs is None else rs)


checks = []


def check(name, ok, actual):
    checks.append(dict(name=name, status='PASS' if bool(ok) else 'FAIL', actual=actual))


a = arr(13)
a['mask'][6, [28, 29]] = False
z = ns['automatic'](a, None)
check('center_hand_missing_is_unknown', z['pred'][6] == -1 and z['geometry'][6] == -1,
      dict(pred=int(z['pred'][6]), geometry=int(z['geometry'][6]), coverage=float(z['coverage'][6])))

q, r = arr(10), arr(10)
q['mask'][2:4] = False
v = comp(q, r)
check('missing_not_physical_discrepancy', v['observed_position_shape_distance'] == 0
      and v['evidence_status'] == 'incomplete_observation' and v['path_missing'] > 0, v)

q, r = arr(7), arr(3)
q['x'][1] = 1
v = comp(q, r)
check('short_query_spike_retained', v['distance'] > 0 and v['all_query_frames_accounted'] == 7, v)

q, r = arr(7), arr(3)
q['mask'][1] = False
v = comp(q, r)
check('missing_query_outside_median_retained', v['path_missing'] > 0
      and v['evidence_status'] == 'incomplete_observation' and v['all_query_frames_accounted'] == 7, v)

q, r = arr(5), arr(5)
q['x'][:] = 1
r['mask'][1:] = False
v = comp(q, r)
check('reference_unknown_not_distance_dilution', v['observed_position_shape_distance'] == 1
      and len(v['reference_unavailable_frames']) == 4 and v['reference_frames'] == 5
      and v['evidence_status'] == 'incomplete_observation', v)

q, r = arr(5), arr(5)
q['mask'][:] = False
v = comp(q, r, np.full(5, -1), np.full(5, 2))
check('all_missing_never_passes', v['status'] == 'insufficient' and v['distance'] is None
      and v['compared'] == 0 and v['path_missing'] > 0, v)

q, r = arr(5), arr(5)
v = comp(q, r, np.full(5, -1), np.full(5, 2))
check('query_direction_unknown_separate', v['direction_missing_pairs'] > 0
      and v['evidence_status'] == 'incomplete_observation' and v['observed_direction_distance'] is None, v)

q, r = arr(5), arr(5)
v = comp(q, r, np.full(5, -1), np.full(5, -1))
check('reference_direction_unknown_not_complete', v['evidence_status'] == 'incomplete_observation'
      and len(v['reference_direction_unavailable_frames']) == 5, v)

v = ns['runs'](np.array([0, 0, 0]), np.array([0, 1, 5]), np.array([0, 33333, 166667]))
check('runs_split_original_gap', len(v) == 2, v)
v = ns['runs'](np.array([0, 0, 0]), np.array([0, 1, 2]), np.array([0, 33333, 250000]))
check('runs_split_pts_gap', len(v) == 2, v)
a = arr(5)
a['frame_id'][3:] += 2
check('window_original_gap_unknown', not ns['encode'](a)[1], dict(eligible=ns['encode'](a)[1]))
a = arr(5)
a['timestamp_us'][3:] += 250000
check('window_pts_gap_unknown', not ns['encode'](a)[1], dict(eligible=ns['encode'](a)[1]))

q, r = arr(5), arr(5)
v = comp(q, r)
check('fully_supported_identical_sequence', v['distance'] == 0
      and v['evidence_status'] == 'complete_relative_to_reference' and v['coverage'] == 1, v)
v = comp(q, r, np.zeros(5, int), np.ones(5, int))
check('observed_direction_disagreement_preserved', v['observed_direction_distance'] == 1
      and v['distance'] == .25 and v['direction_disagreements'] == 5, v)


def pin(p, content=None):
    p = Path(p)
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes() if content is None else content).hexdigest())


prepared_path = UNIT / 'PREPARED.json'
prepared = json.loads(prepared_path.read_text(encoding='utf-8-sig'))
snapshots = [pin(UNIT / 'episode.py', code), pin(UNIT / 'experiment.py', experiment),
             pin(UNIT / 'AUTHORIZATION.json'), pin(UNIT / 'review/SOURCE_USAGE_AND_RANGE.json'),
             pin(UNIT / 'LABELS_FIT.json'), pin(UNIT / 'LABELS_DEV.json'), pin(prepared_path),
             pin(UNIT.parent / 'word100_guide_video_feedback_20261028/motion_observer.py'),
             pin(Path(__file__))]
findings = [
    dict(id='F01', severity='high', status='RESOLVED', location='episode.py automatic',
         before='Window coverage 12/13 could label an unobserved center hand as up/down/other.',
         fix='Actual center hand support now required for candidate, geometry and old comparator.',
         evidence='center_hand_missing_is_unknown'),
    dict(id='F02', severity='high', status='RESOLVED', location='episode.py compare',
         before='A purely masked query acquired a physical difference penalty; reference rows with zero support diluted the observed mean.',
         fix='Alignment still penalizes missing query evidence; ref-zero-support alignment is neutral 0.5. Final observed discrepancy uses shared evidence only and returns expected/compared/path missing/original scope separately.',
         evidence=['missing_not_physical_discrepancy', 'reference_unknown_not_distance_dilution', 'all_missing_never_passes']),
    dict(id='F03', severity='high', status='RESOLVED', location='episode.py compare',
         before='Median query per reference omitted short extra query motion; q7/r3 spike at q1 gave distance0 and mapping[0,3,5].',
         fix='All DTW path points contribute. Per-reference-channel maximum observed discrepancy retains short query changes. Median mapping is display only.',
         evidence=['short_query_spike_retained', 'missing_query_outside_median_retained']),
    dict(id='F04', severity='high', status='RESOLVED', location='episode.py compare',
         before='Unknown reference/query motion could return complete comparison with observed direction distance null.',
         fix='Query direction missing pairs and reference direction unavailable frames are explicit; reference unknown direction prevents complete status.',
         evidence=['query_direction_unknown_separate', 'reference_direction_unknown_not_complete']),
    dict(id='F05', severity='medium', status='RESOLVED', location='episode.py encode/runs',
         before='Same motion label bridged deleted original frame IDs or a long PTS gap.',
         fix='Runs split at frame discontinuity or PTS gap above150ms; windows crossing either are ineligible.',
         evidence=['runs_split_original_gap', 'runs_split_pts_gap', 'window_original_gap_unknown', 'window_pts_gap_unknown']),
    dict(id='F06', severity='medium', status='RESOLVED_BY_EXPLICIT_SCOPE', location='experiment.py evaluate/empirical',
         before='Core majority classification and independently overlapping runs do not establish automatic event boundary quality.',
         fix='Saved evaluation and final gate explicitly set boundary adoption false; research-only diagnostic claim remains.',
         residual='Matching permits one proposed run to overlap several annotated events; no exhaustive false-positive/split/merge metric. A future boundary-adoption claim requires a separate matching/evaluation contract.'),
]
report = dict(
    schema='wis_episode_contract_review_v1',
    created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    reviewer='Codex /root/guide_stage_map; separate same-model code review, not independent human/KSL truth',
    result='PASS_BOUNDED_CODE_CONTRACT' if all(x['status'] == 'PASS' for x in checks) else 'FAIL_COUNTEREXAMPLE',
    scope='Static source/DTW empirical-range contract plus synthetic pure NumPy counterexamples. No model performance assessment.',
    authorization='Latest six-stage development authorization; empirical range development may proceed before expert release validation.',
    activities=dict(training=0, learned_model_calls=0, detector_calls=0, new_video_extraction=0,
                    downloads=0, private_data_access=0, old_file_changes=0, root_document_changes=0,
                    synthetic_function_execution=True, automatic_infer_replaced_by_constant_stub=True),
    input_snapshots=snapshots,
    checks=dict(total=len(checks), passed=sum(c['status'] == 'PASS' for c in checks), rows=checks),
    pre_review_findings_and_resolutions=findings,
    static_contract_assessment=[
        dict(topic='reference expected denominator and original scope', status='PASS_BOUNDED',
             evidence='compare fixes expected=reference mask sum independently of query; all original reference frame IDs retained. All query frames appear in full DTW path.',
             limit='expected is known reference feature cells, not all22 columns times all frames. Reference-unavailable cells/frames and reference direction unknown must remain visible.'),
        dict(topic='whole coherent reference', status='PASS',
             evidence='empirical compares each of3 fit references separately, preserves all pairs and chooses one whole-reference nearest scalar; no stage-wise splice.'),
        dict(topic='fit-only range', status='PASS_BOUNDED',
             evidence='3 fit LOPO pseudoqueries; both comparisons must be comparable for each pseudoquery, all3 nearest values required for max threshold. Dev does not choose range or seed; seed42 is fixed.',
             limit='Comparable means coverage>=.75, not complete. A supported observed-only range can be computed with missing evidence, but cannot establish a whole-performance pass or linguistic allowance.'),
        dict(topic='alignment cost versus observed discrepancy', status='PASS_BOUNDED',
             evidence='Missing penalty used only to find alignment/support. observed_position_shape_distance is maximum shared per-ref-channel discrepancy divided by count of channels with at least one shared path pair; observed_direction_distance is known path-pair disagreement rate; combined distance adds0.25 direction.',
             limit='Coverage can equal1 while path_missing>0 because a ref channel may have both observed and missing query partners. evidence_status must accompany the range indicator. Full-path maxima are sensitive to outliers/path fan-out and are a frozen descriptive metric, not an established linguistic distance.'),
        dict(topic='gap and missing unknown', status='PASS_BOUNDED',
             evidence='No frame interpolation/gap filling. Actual hand wrist support gates the center. Invalid windows/centers return-1; other is a distinct observed physical class.',
             limit='Reference-local complete status is not complete69-feature, bilateral, 3D, visual, or linguistic evidence.'),
        dict(topic='camera transform and normalization', status='PASS_BOUNDED',
             evidence='raw_features copies stored raw observations, retains original ROI eligibility, transforms all xy coordinates in float64, then calls unchanged contemporaneous shoulder-center/width normalization.',
             limit='One positive uniform scale+translation with fixed masks is a numerical invariance check. It does not test new camera detection, rotation, anisotropic body proportions, depth, clipping, occlusion or environment generalization. Raw source pixels are not overwritten.'),
        dict(topic='time and speed', status='PASS_BOUNDED',
             evidence='Frame order/IDs retained; sourcePTS kept. Uniform timestamp-only speed controls preserve all samples, adjust velocity fields consistently, and do not claim blur/camera resampling simulation.',
             limit='Normalized local time and omitted absolute velocity make invariance expected by construction. compare DTW band is normalized row index, not physical elapsed time; duration/speed tolerance is not learned or validated.'),
        dict(topic='side and existing shape', status='PASS_STATIC',
             evidence='New model uses fixed right-side columns; no anatomical-side swapping/normal-cross repair. compare keeps right projected14 angles separate from motion class. Old learned shape probabilities and old C/T3F files are read/pinned, never overwritten.',
             limit='Right-only model does not verify left-hand, bilateral alternation, actual finger contact, 3D shape, or whole sign. Preservation of old files after execution still requires final hash validation by root.'),
        dict(topic='padding', status='PASS_STATIC_IMPLEMENTER_CHECK_RECORDED',
             evidence='Head applies valid mask after every convolution and activation, removing padded hidden leakage. PREPARED records padding batch invariance pass.',
             limit='Reviewer did not execute torch or a model; the recorded padding numeric check is implementer evidence, not independently rerun here.'),
    ],
    source_connection=dict(
        ground_truth_separation='RGB event labels are same-Codex physical drafts. Dictionary clauses are never copied into video timestamps or learned labels. Full source form and linguistic verdict remain unverified/null.',
        words=[
            dict(word_id='WORD0739', label='순종', supported='R up before R down is a proxy for one described sequence component.', unsupported='Fist, lateral direction, whole form and mandatory extent not proved.'),
            dict(word_id='WORD1133', label='잘하다', supported='R down may proxy the forearm-stroke component of candidate9543.', unsupported='Image-plane down is not arm-relative contact/progress; alternate41 must remain separate.'),
            dict(word_id='WORD1567', label='조상', supported='R up is one component of the described rise.', unsupported='Thumb, L movement, bilateral alternation and count not established.'),
            dict(word_id='WORD2523', label='이집트', supported='R down is a shared fragment of23905/5990 and should count once.', unsupported='Oblique upright pose is not up motion. Fingertip contact, gathering, both-hands and candidate-specific suffix remain unsupported.'),
        ],
        implementation_boundary='episode.py/experiment.py do not yet implement source-conditioned order/arm-relation verdicts; later replay may link source proxies without changing observer output.'),
    prepared_evidence=dict(path=str(prepared_path.resolve()), provenance='Implementer-saved checks read only; not repeated',
                           counts_by_role=prepared['counts_by_role'], eligible_fit_counts=prepared['fit_supported_counts'],
                           excluded_fit_count=len(prepared['excluded_fit']), saved_check_count=len(prepared['checks']),
                           saved_checks_all_pass=all(x['pass_'] for x in prepared['checks']),
                           all_dev_events_remain_denominator=prepared['all_dev_events_remain_denominator']),
    residual_limits=[
        'No new generalization:16 development-used public clips; ID9, held and fresh excluded. Fit3 LOPO ranges are empirical extrema, not population confidence intervals.',
        'Automatic core accuracy is not automatic event boundary accuracy. Main run offsets are diagnostics only; boundary adoption staysfalse.',
        'Reverse/deletion controls currently require scope retention and report response; pass does not itself establish robust sensitivity to physical changes or real learner errors.',
        'The support and gap constants75%,13frames,150ms are research settings, not linguistically validated acceptance limits.',
        'Coordinates/angles being present does not certify detector visual accuracy or correct side/3D handshape; score agreement cannot fill missing observation.',
        'compare assumes nonempty aligned-schema arrays and valid monotone timelines. A future general external API needs explicit empty/nonmonotone/no-path output validation; current prepared16 validates its inputs.',
    ],
    next_allowed='Freeze the corrected code, run the authorized bounded experiment once, independently audit saved distances/masks/paths/results, and expose partial evidence in isolated research replay.',
    not_established=['model improvement', 'automatic boundary readiness', 'linguistic correctness', 'expert normality', 'arbitrary body/camera robustness', 'production readiness'],
)
for p, b in [(UNIT / 'episode.py', code), (UNIT / 'experiment.py', experiment)]:
    assert p.read_bytes() == b, f'Code changed during review: {p}'
assert report['result'] == 'PASS_BOUNDED_CODE_CONTRACT'
dest = HERE / 'EPISODE_CONTRACT_REVIEW.json'
with dest.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
memo = ('episode.py / experiment.py 코드 계약 검토 완료\n'
        f'순수 NumPy 반례 {len(checks)}개 모두 통과. 모델 호출·학습·재추론·원본 변경 없음.\n'
        '미관측 중심 손의 자동 판정, 누락의 차이점수 혼합, median 경로에서 짧은 동작/누락 삭제, '
        '참고 방향 미관측의 완전 판정, 원본/PTS 단절 연결 문제를 지적했고 수정 반례를 확인했다.\n'
        'fit3 참고를 각각 유지하고 fit LOPO로 경험적 범위를 만든다. 부분관측에서도 관측된 차이를 기록할 수 있으나 '
        'evidence_status와 누락 목록을 함께 표시해야 한다. coverage=1만으로 완전한 관측이 아니다.\n'
        '정규화 좌표의 이동·등방 확대 및 timestamp 속도 통제는 새 카메라/신체/환경이나 실제 수어 허용차이 검증이 아니다.\n'
        '구간 내용 점수와 자동 경계 검증은 분리되어 경계 채택은 false다. 수어 정오·전체 표현형·출시 준비를 확인한 검토가 아니다.\n'
        '다음은 승인된 한 번의 실행과 저장 결과 산술 검산이다. 소스 연결은 기존 SOURCE_USAGE_AND_RANGE 계약의 부분 proxy 범위를 따른다.\n')
with (HERE / 'EPISODE_CONTRACT_REVIEW_KO.txt').open('x', encoding='utf-8') as f:
    f.write(memo)
print(json.dumps(dict(report=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                      tests=len(checks), passed=len(checks), episode_sha256=hashlib.sha256(code).hexdigest(),
                      experiment_sha256=hashlib.sha256(experiment).hexdigest()), ensure_ascii=False))
