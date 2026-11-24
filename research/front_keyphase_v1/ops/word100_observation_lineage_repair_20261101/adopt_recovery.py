"""Research-only geometry gate from saved independent and original-RGB checks."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

math=read(U/'review/RECOVERY_MATH_VERIFICATION_FINAL.json')
source=read(U/'review/RECOVERY_SOURCE_VERIFICATION.json')
visual=read(U/'review/RECOVERY_VISUAL_REVIEW.json')
assert math['status'].startswith('PASS') and source['status']=='PASS'
assert not math.get('failures',[]) and not source['failures']
assert visual['counts']['reviewed_added_hand_frames']==18
assert visual['counts']['executed_frames_reviewed']==20
added=[a for f in visual['frames'] for a in f['additions']]
assert len(added)==18
for a in added:
    assert a['coarse_hand_identity']=='CONSISTENT_WITH_VISIBLE_ARM_CHAIN_IN_THIS_FRAME'
    assert a['new_opposite_hand_duplicate']=='NOT_SEEN_IN_THIS_QUALITATIVE_REVIEW'
    assert a['gross_displaced_or_collapsed_hand_overlay']=='NOT_SEEN_AT_REVIEWED_BODY_SCALE'
    assert a['fine_finger_landmark_accuracy']=='NOT_VALIDATED'
pins={str(p):sha(p) for p in [U/'RECOVERY_FROZEN.json',U/'RECOVERY_RESULT_MANIFEST.json',
    U/'review/RECOVERY_MATH_VERIFICATION_FINAL.json',U/'review/RECOVERY_SOURCE_VERIFICATION.json',
    U/'review/RECOVERY_VISUAL_REVIEW.json',U/'review/BRIDGE_METHOD_REVIEW.json']}
write(U/'RECOVERY_ADOPTION.json',{
    'status':'RESEARCH_GEOMETRY_BRIDGE_ONLY','at':datetime.now(timezone.utc).isoformat(),
    'geometry_bridge_allowed':True,'pins':pins,
    'scope':'All saved candidate additions in the fixed303 source frames; no favorable-slot filtering.',
    'basis':'Independent saved-array/formula checks and original-RGB review of all20 attempted frames/18 added hand slots.',
    'root_visual_spotcheck':['P048 f72','P066 f57','P072 f70'],
    'old_supported_features_preserved':16154,'new_supported_features':604,
    'added_hand_frames':18,'eligible_hand_frames':22,'remaining_unresolved_hand_frames':4,
    'unsupported_added_joint_slots':6,
    'coordinates_role':'MediaPipe estimated observations with same-frame arm/full-crop evidence; not landmark ground truth.',
    'fine_finger_accuracy_validated':False,'same_detector_twice_is_independent_truth':False,
    'visual_review_is_independent_human_truth':False,
    'old_DTW_mismatch_fixed':False,'instructional_correctness_validated':False,
    'classification_accuracy_measured':False,'deployment_allowed':False,
    'model_training_performed':False,'user_or_mentor_data_used':False,
    'limits_ko':['양손·팔 연결 수준의 큰 오류는 새로 보이지 않았으나 가려진 손가락의 정확한 위치는 미검증이다.',
                 'P048 f74, P063 f57/f72, P066 f71은 근거가 부족하여 미관측을 유지한다.',
                 '기존 학습 판정부 점수와 위치 경로 결과는 원입력 결과로 보존하며 새 근거로 승격하지 않는다.',
                 '기존 DTW의 서로 다른 동작 시점 대응은 이번 재검출로 해결됐다고 판단하지 않는다.']})
(U/'CURRENT_EXECUTION.json').write_text(json.dumps({
    'phase':'RECOVERY_VERIFIED_BRIDGE_READY','at':datetime.now(timezone.utc).isoformat(),
    'active':'Recovery candidate completed and separately verified; next freeze/run changed-parent geometry bridge once.',
    'new_perception_inference':42,'new_classifier_inference':0,'new_dtw':0,'new_training':0,
    'recovery_completed':True,'recovery_locks_absent':not (U/'RECOVERY_RUNNING.json').exists()
},ensure_ascii=False,indent=2),encoding='utf-8')
print('RECOVERY_ADOPTION research geometry only, all18 additions; bridge ready')
