import json,hashlib,re,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;G=U/'guarded';ROOT=U.parents[3];O=U.parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');g=read(G/'RESULT.json');gv=read(G/'INDEPENDENT.json');ctx=read(G/'CONTEXT.json');events=read(G/'EVENTS.json')
assert v['numeric_status']=='PASS' and v['acceptance_status']=='NOT_MET' and gv['status']=='PASS'
assert g['changed_hands']==1 and g['normal441_preserved']==441 and g['non_known_error_flags_unchanged']==314
assert g['remaining9_correct']==8 and g['remaining9_withheld']==1 and g['remaining9_wrong_hint']==0
assert not (U/'RUNNING.json').exists()
visual={
 'at':now,'reviewer':'Same Codex implementer. Separate numeric code, no second human/agent.',
 'scope':'Both V1changed frames with previous/current/next actual detector FOV, plus12frame broader source context for the new overlap case. V2changes only airport; V2veto preserves extract. Prior441labels and13contour changes reused by exact output checks.',
 'airport':{'clip_id':'NIA_SL_WORD2177_REAL09_F','view':'sensor','index':40,'raw':0,'actual':'One visible anatomical Right hand; Left arm is lowered elsewhere. Two skeleton hypotheses cover the same right hand. Raw0wrist falls into palm/finger region, raw1wrist is nearer visible forearm connection.','before':'provisional Left','after':'withheld anatomical hint; raw21points retained','judgment':'Beneficial withholding of known wrong Left hint. Does not certify the alternative fingers/wrist or reconstruct the real wrist.','source':'evidence/change_02.jpg'},
 'extract':{'clip_id':'NIA_SL_WORD0920_REAL05_F','view':'masked','index':56,'raw':1,'actual':'Both physical hands are present and overlap; one palm is nearly edge-on/foreshortened. Previous frame detected only1candidate despite2physicalhands. Current Pose wrists usable/in-frame on both sides.','V1':'Left hint withheld; safe duplicate discrimination not established, exclude from improvement count.','V2':'Keep previous [Right,Left] hints, protect both raw observations.','judgment':'Risk prevented by generic body-evidence veto. Do not claim every joint of raw1 has independently verified truth.','source':'evidence/change_01.jpg','context':'evidence/extract_context.jpg'},
 'terminology_correction':'The old315crossing set is geometric selection, not315independent anatomically verified normal labels. It includes known airport error and1uncertain frame. Actual changed old315item is airport, not extract. Extract was newly found by fullscan outside785sample. Prior interim wording that extract was a confirmed normal-assignment regression was too strong; it is an unsafe suppression in a genuine two-hand overlap scene.',
 'checked_images':['evidence/change_01.jpg','evidence/change_02.jpg','evidence/extract_context.jpg'],
 'limits':['Wrist projection ratio is not a correctness score','V1 and V2 developed with known publicTRAIN cases','No new signer, classifier accuracy, RGB fullscene annotation, or finaltrainingtensor evaluation']}
write(U/'VISUAL_REVIEW.json',visual)
nexttext='공항의 잘못된 왼손 임시 배정 1건 보류와 실제 양손 겹침 보호 결과를 사용자와 확인한다. 기존 C/웹은 유지하고 학습은 대기한다. 다음은 손 관측 좌표·좌우 잠정/확정·중복 의심·손목 신뢰를 최종 입력에 연결하는 계약부터 정한다. 미확정 손을 삭제하거나 중복 두 후보를 각각 양손의 확정 근거로 넣지 않는다. 보류를 실제 손목 복원으로 보고하지 않는다. 공개 TRAIN의 보이지 않는 두 번째 손/가림·손바닥 투영 조건을 별도 검증하고, 승인 후에만 C 대비 비교 조건을 동결한다. 사용자·멘토 촬영은 학습/튜닝/선택에 사용하지 않는다. 완료한 전수 계산·검산은 재실행하지 않는다.'
result={**g,'status':'VALIDATED_GUARDED_WITHHOLDING_AWAITING_REVIEW','at':now,
 'public_train_videos':899,'detector_conditions':2,'frames':226362,'hand_candidates':305602,
 'raw_points_preserved':6292471,'unchanged_associations':305601,'prior13corrections_preserved':13,
 'V1_status':'REJECTED_AUTOMATIC_SUPPRESSION','V2_status':'RESEARCH_CANDIDATE_VALIDATED_NOT_DEPLOYED',
 'V1_changed_hands':2,'V2_withheld_wrong_hint':1,'V2_corrected_side_or_wrist_coordinates':0,
 'wrist_collapse_alone_flags':4127,'joint_suspicion_frames_V1':2,'protected_real_overlap_frames':1,
 'normal_known_labels':441,'crossing_selection_note':'315geometricflags include1known airport error and1uncertain.314otherflags unchanged;knownerror improved by withholding. Not315normalgroundtruth.',
 'remaining9_in_candidate':{'corrected_by_prior_contour':8,'known_wrong_hint_now_withheld':1,'reconstructed_wrist':0},
 'candidate_preserves_all_points':True,'identity_promotion':False,'production_applied':False,
 'training':0,'classifier_inference':0,'user_mentor_training_tuning_selection':0,'DEV_heldout_analysis':0,
 'numeric_verification':'PASS','protected_pins':27690,'contracts_V1':19,'contracts_V2':12,
 'order_and_hidden_point_checks':786,'visual_reviewer':'SameCodex','report':str(U/'RESULT_KO.html'),
 'next_action':nexttext}
write(U/'RESULT.json',result)
write(U/'NEXT_PLAN.json',{'status':'AWAIT_USER_REVIEW_NO_TRAINING','next_single_action':nexttext,'remaining':['Actual wrist/finger reconstruction unresolved','Stable physical-hand tracks through long overlap not solved','Sparse side patches are research output, not final model input','Recognition/generalization improvement not measured'],'method_to_consider':'Separate observed point quality from anatomical identity; use existing own-side arm evidence and pre-overlap two-hand history to avoid treating detector count as physical hand count. Freeze before further validation, no word exceptions.'})
report=f'''중복 검출·손목 위치 보완 결과
{now}

결론
보호 규칙을 포함한 별도 후보에서 ‘공항’의 잘못된 왼손 임시 배정 1건을 보류했다.
정상 배정 441건과 앞선 팔 윤곽 후보의 13개 정정은 유지했다. 기존 C/웹에는 적용하지 않았다.
손목을 올바른 좌표로 복원한 것은 아니다. 손 관측은 모두 남기고 잘못된 좌우 라벨만 막았다.

무슨 문제였나
공항의 실제 오른손 하나 위에 관절 후보 두 개가 겹쳐 검출되었다.
그중 첫 후보의 손목이 손바닥/손가락 쪽에 잘못 잡혀 있고, 다른 후보와 손목 위치도 크게 달랐다.
하나를 왼손, 하나를 오른손이라고 배정하면 실제로 없는 왼손 관측처럼 쓰일 수 있었다.

공통 보완
손목 한 점 대신 관측된 손바닥 5점과 손 전체 영역을 사용한다.
손목-손바닥 중심 거리 / 손가락 시작 관절들 사이 최대거리가 매우 작은지, 두 손 후보 영역이 크게 겹치는지,
손목 후보가 서로 멀리 떨어졌는지, 직전 단일 검출과 공간적으로 연결되는지 함께 확인한다.
손바닥이 카메라 쪽으로 향하면 정상적으로도 이 비율이 작아지므로 이 값 하나만으로 제거하지 않는다.
확정된 기존 좌우는 보호하고, 모든 원래 관절 좌표와 관측 마스크를 보존한다.
새 단어별 예외와 공항 전용 문턱은 없다. 안 보이는 손목/손가락을 만들어 넣지 않는다.

첫 후보에서 발견한 위험과 추가 보호
첫 후보 V1은 공항뿐 아니라 ‘추출’의 겹침 장면도 보류했다. 원본에는 실제 두 손이 있다.
직전 프레임에서 검출 후보가 하나였다는 사실은 실제 손이 하나였다는 뜻이 아니다.
추출에서는 양쪽 Pose 손목이 현재 화면 안에서 높은 신뢰도로 관측된다(약0.985,0.988).
V2는 이 기존 몸 관절 근거가 양쪽 모두 있으면 중복 추정만으로 손 후보를 보류하지 않도록 보호한다.
Pose 신뢰 기준을 낮추지 않았다. 이 근거를 정답으로 확정하는 데 쓰지 않고 기존 배정을 보호하는 데만 쓴다.
공항은 양쪽 Pose 손목이 기존 관측 조건을 충족하지 못한다(약0.423,0.495).
최종 V2는 추출의 기존 양손 배정을 유지하고 공항의 오배정만 보류했다.
V1과 최초 검산 실패는 그대로 보존했다. V1 결과에 맞춰 기하 문턱을 바꾸지 않고, V2의 추가 근거와 기준을 별도로 동결했다.

검증 범위와 결과
- 공개 학습 영상 899개, 2가지 검출 조건. 서로 다른 영상1798개라는 뜻이 아니다.
- 총226,362프레임·손 후보305,602개에 대한 저장 관측을 조사했다. 사용자/멘토/DEV/보류 자료를 분석·학습·튜닝하지 않았다.
- 첫 후보: 변경2건. 최종 보호 후보: 공항1건만 잘못된 왼손→보류, 나머지305,601개 배정 유지.
- 알려진 정상 배정441/441, 앞선 팔윤곽 정정13/13 유지.
- 기존 기하학적 교차 후보315개 중314개 유지, 나머지1개는 이미 오류로 확인된 공항이다.
  이315개를 모두 독립 검증된 정상 교차라고 부르는 것은 부정확하다. 알려진 오류1개와 불확실1개가 포함되어 있다.
- 이전 남은9건 기준: 앞선8개 정정 유지 + 공항1개는 오배정을 보류. 9개를 모두 정확한 손으로 복원한 결과가 아니다.
- 관측 좌표6,292,471점 모두 보존. 공항의 문제 후보21점도 삭제하지 않았다.

검산
V1 합성계약19개, V2 보호계약12개 통과. 후보순서/숨긴좌표 교란786프레임 확인.
OpenCV를 쓰지 않는 별도 다각형/스칼라 코드로 전수226,362프레임,130,888손영역 쌍,1798개 결과를 검산했다.
다각형 겹침 비율 최대 차이는약0.00000327, 좌우 결정은 전부 동일했다.
별도 V2 코드로 모든1798배열에 보호 규칙과 최종1건 패치를 대조했고 정상441·나머지교차314·앞선13정정을 확인했다.
첫 검산은315개 모두 불변이라는 수용조건에서 멈췄다. 수치 계산 불일치가 아니었다.
뒤 검산은 수치PASS와 V1수용실패를 분리해 저장했다. 기존 수용실패를 성공으로 바꾸지 않았다.
V2에서는 이전부터 알려진 오류 장면과 정상 보존 대상을 구별한 판정 조건을 먼저 기록했다.
27,690개 보호 해시 유지. 같은 Codex의 구현·영상 검토와 별도 수치 코드 검증이며 독립된 사람의 평가가 아니다.

남은 문제
실제 손목 좌표와 가려진 손가락을 복원하지 못했다. 겹친 두 손을 항상 완벽히 구분한다는 증거도 없다.
Pose도 다른 검출기 출력이므로 잘못될 수 있다. 이 경우 안전하게 보호할 뿐 정답이라고 단정하지 않는다.
현재 후보는 원래 관측과 좌우 확실성을 구분한 연구 산출물이다. 최종 학습 입력 연결과 인식률 개선은 미검증이다.
기존 C·TCN·DTW·웹, 사용자 촬영 자료는 유지했다. 학습/분류 추론0회.

다음 한 작업
{nexttext}
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
css='''*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e9f0fa;font:17px/1.7 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:48px 24px 80px}h1{font-size:36px;line-height:1.35;margin:15px 0}h2{font-size:25px;margin:0 0 18px}p{margin:12px 0}section{background:#142136;border:1px solid #29405b;border-radius:18px;padding:28px;margin:24px 0}.tag{color:#67d7d0;letter-spacing:1px}.lead{font-size:20px;color:#bdcde1}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.stat{background:#172a42;padding:22px;border-radius:14px}.stat strong{display:block;font-size:31px;color:#81e6c7}.muted{color:#bdcce1;font-size:15px}.notice{border-left:4px solid #f4bf69;padding:16px 22px;background:#262a33;border-radius:8px}img{display:block;width:100%;height:auto;border-radius:12px;margin:18px 0}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:13px;text-align:left;border-bottom:1px solid #33445b}th{color:#9ee4dc}.flow{background:#0b192b;padding:20px;border-radius:12px;font-weight:bold;color:#a9e8dc}a{color:#80cbff}summary{cursor:pointer;font-weight:600;padding:12px 0}ul{padding-left:24px}@media(max-width:700px){.stats{grid-template-columns:1fr}h1{font-size:29px}section{padding:20px}table{font-size:14px}}'''
page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>중복 검출·손목 위치 검증</title><style>__CSS__</style></head><body><main><div class="tag">WORDS IN SILENCE · 공개 자료 검증 · 2026.10.18</div><h1>‘공항’의 잘못된 왼손 배정을 막았습니다.<br>손목 좌표 자체를 복원한 것은 아닙니다.</h1><p class="lead">손 관측은 그대로 남기고, 잘못 붙은 좌우 이름만 보류하는 별도 후보입니다.</p><div class="stats"><div class="stat">추가 개선<strong>오배정 1건 보류</strong>공항 · 같은 오른손의 중복 검출</div><div class="stat">알려진 정상 배정<strong>441 / 441 유지</strong>앞선 팔 윤곽 정정13건도 유지</div><div class="stat">원래 관측 좌표<strong>모두 보존</strong>6,292,471점 · 삭제/재작성 없음</div></div><div class="notice"><b>기존 C와 웹은 그대로입니다.</b> 사용자 촬영을 학습·튜닝·선택에 사용하지 않았고, 학습은 시작하지 않았습니다. 이번 숫자는 수어 인식 정확도가 아닙니다.</div>
<section><h2>1. 실제 문제는 무엇이었나</h2><p>영상에는 오른손 하나가 움직이는데, 그 손 위에 <b>서로 다른 관절 후보 두 개</b>가 잡혔습니다. 첫 후보의 손목은 손가락 쪽에 잘못 놓였습니다. 두 후보에 왼손·오른손 이름을 하나씩 붙이면 실제로 없는 왼손 관측처럼 사용할 위험이 있습니다.</p><img src="evidence/change_02.jpg" alt="공항의 원본 장면과 중복 관절: 문제 손목과 정상에 가까운 다른 손목"><p class="muted">왼쪽은 실제 검출 범위 전체입니다. 오른쪽은 검토용 손 확대 화면입니다. 흰 네모가 손목이며, 이 확대 크기로 다시 학습하거나 관절을 추출한 것이 아닙니다.</p></section>
<section><h2>2. 손목 하나만 보고 결정하지 않게 했습니다</h2><div class="flow">손바닥·손 영역 확인 → 중복 가능성과 손목 불일치 확인 → 현재 팔 근거로 보호 → 확실하지 않은 좌우만 보류</div><p>손목과 손바닥 중심이 비정상적으로 가까운지, 두 후보의 손 영역이 크게 겹치는지, 손목 위치는 서로 멀리 떨어져 있는지, 이전 검출과 연결되는지를 함께 봅니다.</p><p><b>이 중 한 가지가 해당한다고 손을 버리지 않습니다.</b> 손이 카메라를 향하면 정상 손바닥도 납작하게 보일 수 있습니다. 실제로 이런 작은 비율은4,127개 손·프레임에서 관찰됐지만, 이를 모두 오류로 취급하지 않았습니다.</p><p>단어 이름이나 사람ID를 판단 조건에 넣지 않았습니다. 기존 확정 배정, 원래 좌표와 관측 마스크는 유지합니다. 좌우 이름을 보류해도 손 자체가 사라진 것은 아닙니다.</p></section>
<section><h2>3. 실제 두 손 겹침을 보호하는 규칙도 추가했습니다</h2><p>첫 후보는 ‘추출’에서도 한 손 후보를 보류했습니다. 하지만 원본에는 <b>실제 두 손이 겹쳐 있고, 현재 양쪽 팔의 손목도 관측</b>됩니다. 바로 앞 프레임에서 손이 하나만 검출됐다는 이유로 실제 손도 하나라고 생각하면 안 됩니다.</p><img src="evidence/change_01.jpg" alt="추출의 실제 양손 겹침에서 첫 후보가 손목 투영을 잘못 해석할 위험"><p>보호 후보에서는 <b>양쪽 Pose 손목이 기존 신뢰 조건을 충족하고 화면 안에 있으면, 중복 의심만으로 기존 손 배정을 보류하지 않습니다.</b> 신뢰 문턱을 낮추거나 특정 단어 예외를 넣지 않았습니다.</p><table><thead><tr><th>장면</th><th>첫 후보</th><th>보호 규칙을 포함한 최종 후보</th></tr></thead><tbody><tr><td>공항 · 동일 오른손 중복</td><td>잘못된 왼손 이름 보류</td><td>오배정 보류 유지</td></tr><tr><td>추출 · 실제 양손 겹침</td><td>왼손 후보 보류 위험</td><td>기존 양손 배정 유지</td></tr></tbody></table><details><summary>‘추출’의 더 긴 원본 흐름 보기</summary><img src="evidence/extract_context.jpg" alt="겹치기 전후로 실제 두 손이 확인되는 원본 연속 장면"><p>이 장면은 검토용입니다. 뒤 프레임의 정보를 현재 후보 계산에 넣지 않았습니다. 해당 관절 후보의 모든 점이 정확하다고 판정한 것도 아닙니다.</p></details></section>
<section><h2>4. 검증 결과</h2><p><b>공개 학습 영상899개를 두 검출 조건에서 조사</b>했습니다. 총226,362프레임·305,602개 손 후보입니다. 새 사용자 시험이 아니라 저장된 공개 관측의 검증입니다.</p><table><thead><tr><th>항목</th><th>결과</th></tr></thead><tbody><tr><td>최종 배정 변경</td><td>공항1건: 잘못된 왼손 → 좌우 보류</td></tr><tr><td>나머지 손 배정</td><td>305,601개 유지</td></tr><tr><td>알려진 정상 배정</td><td>441개 전부 유지</td></tr><tr><td>앞선 팔 윤곽 후보의 정정</td><td>13개 전부 유지</td></tr><tr><td>교차 후보에서 알려진 오류를 제외한 장면</td><td>314개 전부 유지</td></tr><tr><td>앞서 남아 있던9건</td><td>기존8건 정정 유지 + 공항1건 오배정 보류</td></tr></tbody></table><div class="notice"><b>9건 모두를 복원한 것은 아닙니다.</b> 공항의 잘못된 왼손 이름을 막았지만, 그 후보의 실제 손목·손가락을 정확한 좌표로 바꾸지는 못했습니다.</div><details><summary>교차315개라는 기존 설명의 정확한 의미</summary><p>이 목록은 기하학적으로 교차처럼 보이는 장면을 모은 것입니다. 모두 정상이라고 사람이 독립 확인한315개가 아닙니다. 이미 오류로 확인한 공항1개와 불확실1개도 포함합니다. 최종 결과는 나머지314개 유지, 공항 오배정1개 보류입니다. 추출은 이315목록에 없으며 이번 전수 조사에서 추가로 발견됐습니다.</p></details></section>
<section><h2>5. 검산과 남은 한계</h2><p>별도 코드로 전체 수치와 배정을 다시 계산했습니다. 첫 후보의 합성검사19개, 보호규칙 검사12개,786프레임의 순서·숨긴좌표 대조를 통과했습니다. 기하 계산130,888쌍과1798개 결과 배열을 대조했고, 기존 자료27,690개 해시를 확인했습니다.</p><p class="muted">구현·영상 검토는 동일 Codex, 계산 검산은 별도 코드입니다. 독립된 사람의 평가나 새 환경 성능 검증은 아닙니다. 첫 후보와 검산 중 수용조건 실패 기록은 보존했습니다.</p><ul><li>가려진 손가락과 실제 손목 좌표 복원은 남아 있습니다.</li><li>두 손이 오래 겹쳐 있다가 다시 나올 때의 추적이 완전히 해결된 것은 아닙니다.</li><li>Pose 자체도 틀릴 수 있습니다. 보호 근거로 활용하며 정답으로 단정하지 않습니다.</li><li>공개 개발 자료로 보완한 결과입니다. TCN 인식률 향상이나 새 사람 일반화는 아직 측정하지 않았습니다.</li></ul></section>
<section><h2>다음에는 무엇을 확인하나</h2><p>이번 결과를 함께 확인한 뒤, <b>관측 좌표·좌우 확실성·중복 의심·손목 신뢰를 학습 입력에 어떻게 연결할지</b> 먼저 정해야 합니다. 좌우가 불확실한 손을 통째로 지우거나, 중복 후보 둘을 실제 양손이라고 넣어서는 안 됩니다.</p><p>기존 C·TCN·DTW·웹과 사용자 촬영은 유지했습니다. 학습과 C 성능 비교는 결과를 확인한 다음 진행합니다.</p></section><p class="muted"><a href="RESULT_KO.txt">상세 설명</a> · <a href="RESULT.json">최종 집계</a> · <a href="guarded/INDEPENDENT.json">최종 별도 검산</a> · <a href="VISUAL_REVIEW.json">영상 검토 및 정정 기록</a> · <a href="NEXT_PLAN.json">다음 작업</a></p></main></body></html>'''.replace('__CSS__',css)
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
summary='ops/word100_duplicate_wrist_quality_20261018. 공개 TRAIN899영상×2조건226362프레임/305602손후보 전수 기하 검증. 손 전체 겹침·손목 투영 붕괴·직전 단일 관측 V1은 공항 오배정과 추출 실제 양손 겹침의 후보2건을 보류하여 미채택. 현재 양쪽 Pose 손목이 기존 usable/화면안 조건을 충족하면 기존 배정을 보존하는 V2를 별도 동결·검증. 최종 공항sensor40raw0의 잘못된 L1건만 보류, 다른305601배정/정상441/앞선F정정13/관측6292471점 유지. 기존 남은9=앞선8정정+공항1오배정보류(손목복원/확정승격0). 기존315는 정상정답집합이 아닌 기하교차flag이며 공항오류1·불확실1포함; 나머지314유지. 추출은785선정범위외 추가 발견, V2가 이전 양손 배정보호. V1기하4127낮은손목비율만으로 오류판정금지. 합성19+12/순서·숨긴점786/별도226362전수·130888다각형·1798배열 검산PASS/27690보호핀 유지. V1원본/첫수용검사실패보존(수치불일치아님),V1/V2분리. 동일Codex시각검토+별도수치검산. C/TCN/DTW/웹/사용자자료 유지·학습/분류0·학습입력미연결.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf8')
for head,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','### 2026-10-18 중복·손목 보호 후보 검증 완료 — 공항 오배정1건 보류',summary),('## 9. 다음 단일 작업','### 2026-10-18 중복·손목 후보 결과 검토·학습 입력 연결 대기',nexttext)]:
    assert text.count(head)==1;text=text.replace(head+'\n',head+'\n\n'+title+'\n'+body+'\n',1)
p.write_text(text,encoding='utf8')
p=O/'STATE.json';state=read(p);state['updated_at']=now;state['updated_date_kst']=now;state['current_stage']='WORD100_DUPLICATE_WRIST_GUARDED_VALIDATED_AWAITING_REVIEW';state['next_single_action']=nexttext;state[U.name]=result;write(p,state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}

### 현재 목표·마지막 완료
{summary}

### 다음 단일 작업
{nexttext}

### 실행·검증·미적용
- 새 단위 candidate.py: 16점 이상·5점 손바닥 관측, 손목-중심/MCP폭<.25·다른후보>=.5·작은영역겹침>=.75·손목차>=.5폭·직전<=100ms단일충분관측과 양쪽MCP거리<=.4어깨. 단어/사람/미래 특징없음. 손목좌표 변경0.
- V1 exec2639종료0:899영상×2검출조건226362프레임305602후보. 공항sensor40raw0 L→보류, 추출masked56raw1 L→보류 두변경. 추출 실제 양손 겹침에서 안전성미입증→V1미채택,원본산출물보존.
- V1 verify.py exec93560종료1은 마지막315불변수용조건. 수치계산실패아님. FIRST_VERIFY_ERROR/VERIFICATION_NOTE 보존. verify_numeric.py exec5235종료0:수치PASS/수용NOT_MET를별도로저장. 다각형130888쌍,최대비율차3.27e-6,배정정확일치.
- guarded/CONTEXT 별도동결 후 guarded_run exec2802종료0. guarded_candidate는 현재양쪽Pose손목이 기존usable+finite+화면안이면 보류취소. 신뢰문턱완화/좌우강제확정없음. 추출 양목conf.985/.988 보호,공항.423/.495로미충족. V2는V1변경의부분집합이며계산재활용.
- guarded_verify exec28690종료0: 별도guard계산+1798배열전수패치재구성PASS. 최종변경1,정상441·교차중오류제외314·기존F정정13보존.6,292,471관측점 유지. 보호27690핀.
- 교차315는기하선정이며알려진공항오류1·불확실1포함. 기존보고의315정상이라는표현정정. 실제315내변경은공항이고추출은아님. 추출raw1의모든관절정답을확정한것은아니며실제양손겹침에서안전한중복판정이안되는사례로기록.
- VISUAL_REVIEW:V1변경2개전후입력+추출12프레임원본연속확인. 최종공항은동일오른손의중복/첫손목왜곡. 잘못된L보류이지Right복원/실제손목복원/인식률개선아님.
- 최종출력 guarded/BASELINE_MANIFEST + SIDE_PATCHES.json(1건), 기존U/F원본불변. 최종학습텐서나운영에미연결. 연구후보로만보존.
- 현재실행/잠금없음. 학습/분류추론0,사용자·멘토·DEV·보류 분석/튜닝0. C/웹/TCN/DTW유지.원본W:/sources/과거결과읽기전용.카메라/다운로드/외부전송/Git없음.st-gcn자동화삭제유지.
''',encoding='utf8')
for p,h in ctx['pins'].items():assert sha(p)==h,p
for p in list(U.glob('*.json'))+list(G.glob('*.json')):read(p)
links=re.findall(r'(?:src|href)="([^"]+)"',page)
for link in links:assert (U/link).is_file(),link
assert state[U.name]['changed_hands']==1 and state['next_single_action']==nexttext
write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now,'protected_pins':len(ctx['pins']),'report_links':len(links),'documentation_synchronized':True,'first_candidate_preserved':True,'normal441':True,'other_geometric_crossing314':True,'known_airport_error_withheld':True,'no_active_process':True,'no_training':True,'production_changed':False,'files':{str(p):sha(p) for p in list(U.glob('*'))+list(G.glob('*')) if p.is_file() and p.name!='FINAL_AUDIT.json'}})
print('FINAL PASS',U/'RESULT_KO.html')
