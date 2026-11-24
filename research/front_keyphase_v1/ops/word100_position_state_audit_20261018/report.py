"""Publish verified audit findings, proposal only, and current handoff."""
import json,hashlib,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 assert not p.exists(),p;p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');assert v['status']=='PASS'
visual={
 'reviewer':'Codex local, same operator as implementation; not independent human annotation',
 'reviewed_panels':12,'unique_clip_frame_side_scenes':10,'source_videos':7,
 'selection':'Prespecified observation strata, first lexical clips, longest run midpoint. No predictions or user data used.',
 'coverage_limit':'Representative scenes concern two words because lexical selection was fixed. Numerical coverage is all TRAIN899 / 100 words. This is not visual truth annotation of all frames.',
 'findings':[
  {'panels':[1,2],'finding':'Same RGB time has different usable assigned hand observations. Panel1 raised right hand has 0 vs21 usable points; panel2 lower-boundary hand has6 vs19 and only the latter has all palm5. Detection/crop/assignment pipeline differs; state relabeling cannot restore missing Hand21.'},
  {'panels':[3,7],'finding':'Lowered right hand is below lower border. Both have0 usable Hand21. Native Pose wrist y889.7 vs border888 is removed in masked policy; sensor wrist y902.2 is extrapolated and retained as evidence. Unknown vs boundary/outside does not establish one is a detector error.'},
  {'panels':[4],'finding':'Near lower border, masked has2 usable points while sensor has0 and an outside wrist estimate. Real partial visibility and extrapolation-policy effects coexist.'},
  {'panels':[5,8],'finding':'Left hand visible in RGB but not assigned usable Hand21 in either view. Both Pose arm chains support interior wrist. Sensor raw unassigned candidate triggers assignment_uncertain before arm-chain branch; removing priority only reveals existing interior evidence, not recovered hand shape.'},
  {'panels':[6],'finding':'Both hands visibly raised. Masked right Hand21 observed, sensor right candidate withheld. Assignment uncertainty and actual observation loss coexist; do not promote withheld candidate to real hand.'},
  {'panels':[9],'finding':'Both target Hand21 absent while hand is raised in RGB. Projected forearm/shoulder ratio .10395 vs .09555 crosses frozen .1 minimum. Small projected geometry difference changes interior to unknown; no threshold changed. 2D foreshortening may matter; physical correctness not independently established.'},
  {'panels':[10],'finding':'Lowered target hand near/below border; native wrist y921.9 outside914 while sensor y905.9 is inside914. The difference remains even with common visible-only policy because actual Pose estimates fall on opposite sides.'},
  {'panels':[11,12],'finding':'Normal observed-palm control stays observed in both views and all diagnostic policies.'}
 ],'no_model_or_coordinate_change':True}
save(U/'VISUAL_REVIEW.json',visual)
plan={'status':'PROPOSED_AWAITING_USER_REVIEW_NO_IMPLEMENTATION_OR_TRAINING',
 'objective':'Separate actual hand visibility, arm-based location evidence, and identity uncertainty without forcing agreement between different observations.',
 'minimal_next_unit':[
 'Keep C280 input coordinates/masks/timestamps, model weights, web and DTW unchanged. Implement an isolated sidecar candidate only after review.',
 'Represent observed Hand21/palm5 availability separately from arm-based wrist location. Observed palm5 must not be labeled full21 visibility.',
 'Retain assignment uncertainty as a separate flag. It must not erase otherwise valid arm location evidence, and must not promote a withheld hand to observed coordinates.',
 'Keep outside-frame Pose estimates explicitly inferred/unobserved. Do not copy full-source hidden wrists into masked synthetic view or zero all useful sensor extrapolation to force agreement.',
 'Record wrist edge distance, pose confidence and arm-geometry reason alongside validity for diagnostic review. Do not relax .1/.15/.5 thresholds using failures or tune per-word.',
 'Verify TRAIN899 with existing good palm controls unchanged, no coordinate/mask promotion, no hidden/future leakage, exact same-evidence consistency, and representative boundary/occlusion/assignment cases. Different real evidence is allowed to produce different states.',
 'After user review of sidecar reliability, separately freeze any minimal training comparison against preserved C and necessary capacity control. No training is authorized by this audit completion.'
 ],'learning':0,'no_accuracy_gain_guarantee':True,'user_mentor_data':'not read / not used for training, tuning, or selection','heldout_14_18':'not read'}
save(U/'NEXT_PLAN.json',plan)
text='''공개 학습 자료의 위치 상태 표시 차이 분석

결론
실제 관절 관측 차이와 처리 규칙 차이가 함께 있었다. 현재 추가된 여섯 가지 '위치 상태'에는 손이 보이는지, 손목 위치가 어디인지, 좌우 배정을 확신하는지가 한 항목에 섞여 있다. 특히 배정 보류가 팔의 위치 근거를 가리는 처리와, 화면 밖 Pose 손목 추정을 사용하는 정책 차이가 확인됐다. 다만 이것만이 이전 인식률 저하의 전부라는 증명은 아니다. 기존 C를 유지하며 새 학습은 하지 않았다.

1. 분석 범위와 숫자의 뜻
AI Hub 공개 TRAIN899영상, 100단어, 촬영자1~9만 읽었다. DEV/보류14~18/사용자/멘토 촬영과 분류 점수는 이번 감사에서 읽거나 사용하지 않았다. 전체 영상 검출 후 상반신 범위만 남긴 masked 보기와, 상반신으로 잘린 영상 자체를 검출한 sensor 보기를 비교했다.
서로 같은 원본 프레임 ID·시각이며 두 보기의 whole 학습 구간에 모두 속하는58,320프레임×좌우2손=116,640손-프레임 중26,108(22.38%)에서 위치 상태가 달랐다. 이는 단어 인식 오답률이나 독립 표본의 비율이 아니다. 인접 프레임과 같은 사람의 관측은 서로 연관된다.

2. 실제 검출·관측 차이
손바닥5점의 사용 가능 여부가 다른 경우는6,770손-프레임이었다. 잘린 영상 검출에서 손바닥이 추가로 잡힌 경우4,403, 반대로 사라진 경우2,367이었다. 이는 저장된 실제 검출·좌우배정·범위제한 후 관측의 차이이며, 전부를 검출기만의 오류 또는 정확한 좌우 정답이라고 판정한 것은 아니다.
같은 촬영 순간에도 '수어' 공개영상에서 한쪽 방식은 오른손 관절0개, 다른 방식은21개가 남았다. 다른 장면은6개 대19개였다. 상태 이름을 통일해도 없는 손가락 좌표가 복원되지는 않는다.

3. 화면 밖 손목 추정의 영향
sensor는 잘린 화면에서 Pose가 추정한 화면 밖 손목을 '관측된 손'과 구분하면서 위치 단서로 사용한다. masked는 원본 전체 영상에만 있던 화면 밖 좌표를 지워 정보 유입을 막는다. 따라서 실제 사용 가능한 Hand21이 양쪽 모두0개여도 '모름'과 '경계/밖 추정'으로 달라질 수 있다.
동일한 sensor 저장 관측에서 화면 밖 손목 추정만 일시적으로 제한하자8,693손-프레임이 경계/밖 추정에서 모름으로 변했다. 기존 불일치5,898개는 사라졌지만, 원래 같던1,935개는 새로 달라졌다. 이 계산은 원인 확인용이며 운영·학습 입력에 적용하지 않았다. 모든 화면 밖 추정을 삭제하는 방안은 유용한 단서도 버린다.

4. 같은 근거를 처리하는 순서의 영향
sensor에는 미배정 손 후보가 있으면 팔 연결을 보기 전에 '좌우 배정 보류'를 출력하는 우선순위가 있다. masked에는 숨긴 원본 손 후보가 새어 들어오지 않도록 이 표시를 전달하지 않는다.
sensor의 배정 보류1,814손-프레임에서 해당 우선순위만 분리해 보면 기존 팔 근거로 안쪽533, 경계/밖1,207, 근거 부족74로 나뉜다. '어느 손인지 불확실함'과 '그쪽 팔의 손목이 어디에 있을지 추정 가능함'은 동시에 기록할 수 있다. 좌우 배정 불확실성을 없애거나 미검출 손을 실제 검출로 올리자는 뜻은 아니다.
같은 sensor 관측을 고정하고 화면 안 근거로 제한하는 두 정책을 함께 적용한 진단에서는 기존 차이6,795개가 없어졌다. 실제 두 보기의 모든 숫자가 동일했다는 뜻이 아니라, 동일 sensor 입력에서 정책만 바꾼 비교다.

5. 원래 불일치26,108개를 순서대로 구분한 결과
화면 안 근거 정책을 맞췄을 때 사라진 차이:6,795(26.03%).
그 후에도 남는 실제 손 관측·부분관측 상태 차이:11,860(45.43%).
그 후에도 남는 팔·손목 위치 추정 근거 차이:7,453(28.55%).
합계는 원래 불일치26,108이다. 여러 원인은 한 장면에 함께 존재한다. 이것은 명시된 진단 순서에 따른 분해이며, 세 가지 물리적 원인이 서로 완전히 독립이라는 뜻은 아니다. 정책 통일로 새로 생긴1,935차이는 이 원래 불일치 집합 밖이다.

6. 추가 확인 사항
실제 손 관절이0개인데 '일부 검출'이라고 표시된 사례는 이번 공통 비교 범위에서는0개였다. 앞서 가능성으로 의심한 문제는 확인되지 않았다.
'관측됨'은 손바닥5점 기준이다. 손가락21점 전부가 보인다는 보장이 아니다. 손바닥 관측으로 표시됐지만21점 미만인 경우 masked3,779 / sensor4,247손-프레임이었다.
팔꿈치가 카메라를 향하는 장면에서는2D 팔 길이가 짧아진다. 예시에서 팔뚝/어깨 비율0.10395와0.09555가 고정 기준0.1의 양쪽에 놓여 '안쪽'과 '모름'으로 달라졌다. 작은 검출 차이와 단단한 범주 기준이 표시를 바꿀 수 있다. 이 예시에 맞춰 기준을 바꾸지 않았다.
이 여섯 위치 상태는 과거 움직임 이력 때문에 달라진 것이 아니었다. 과거 이력으로 inside_missing과 overlap_likely가 달라져도 둘은 같은 wrist_interior 위치값으로 묶인다. full/whole/context 저장 위치를 같은 현재 근거만으로 전수 재구성했을 때 모두 일치했다. 최근 근접 단서 등 다른 항목까지 이력이 불필요하다는 뜻은 아니다.

7. 권장 보완 방향 — 아직 실행하지 않은 제안
추가 위치 상태를 더 복잡하게 한 칸에 넣기보다, ①실제로 보이는 손 관절, ②팔로 추정한 손목 위치와 근거, ③좌우 배정 불확실성을 분리한다. 팔의 위치 단서가 있어도 손가락 관측은 없음으로 유지할 수 있어야 한다. 화면 밖 손목은 추정임을 유지하고, 원본에서만 보이는 숨긴 좌표를 학습 입력으로 넘기지 않는다.
기존 C의280개 입력과 모델은 그대로 둔 채 먼저 별도 진단 출력에서 공통 규칙을 검증하는 것이 다음 단계다. 정상 관측 보존, 미관측 좌표 승격 없음, 숨긴 좌표·미래 정보 누출 없음, 같은 근거·같은 정책의 일관성을 확인한다. 특정 단어 예외나 실패에 맞춘 임계값 조정은 하지 않는다.
그 결과를 함께 확인한 뒤 학습 입력에 무엇을 넣을지 확정한다. 이번 결과는 보완할 지점을 찾았다는 근거이고, 인식률 개선을 보장하는 결과는 아니다. 이전 실험에서 위치 묶음을 뺐을 때 회복한 부분에는 학습 경로·조기 종료 차이도 영향을 줄 수 있다.

8. 검증과 보존
별도 벡터 계산으로 저장된 TRAIN full/whole/context 위치값을 전수 재구성했다. 별도 scalar 원래 코드로899영상에서 각3프레임·양손·6조건=32,364위치 상태를 재검산해 일치했다. 저장 통계도 다시 계산했다. source/code/input10,793해시는 유지됐다.
원본 RGB12패널(중복 제외10장면/7영상)을 검토했다. 사전 정한 관측 조건·파일순서로 선정하여 그림은 두 단어에 집중됐다. 전체 프레임의 물리적 손 위치·좌우 정답을 사람이 주석한 평가가 아니다. 수치는100단어899영상 전체에서 계산했다.
구현·시각검토·별도 계산 검산 담당: Codex 로컬. 다른 사람/에이전트의 독립 검토라고 부르지 않는다. 새 검출·모델 추론·학습·다운로드·카메라 사용0. 기존 C/웹/DTW 변경0. 사용자 촬영 학습·튜닝·후보 선택0.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
cards=''.join(f'<article><h3>{title}</h3><p>{desc}</p><img src="evidence/{num:02}.png" alt="{html.escape(title)}"></article>' for num,title,desc in [
 (1,'① 실제 손 관절이 다르게 잡힌 장면','같은 촬영 시점인데 사용 가능한 오른손 관절이0개와21개다. 실제 관측 차이는 표시 이름만 바꿔 해결할 수 없다.'),
 (3,'② 화면 밖 손목 추정','실제 손 관절은 양쪽 모두0개다. 한쪽은 숨긴 손목 근거를 지우고, 다른 쪽은 잘린 영상에서 추정한 손목을 위치 단서로 남긴다.'),
 (5,'③ 좌우 보류가 위치 단서를 가린 장면','팔 근거로 손목이 화면 안쪽이라고 추정할 수 있지만, 좌우 미배정 후보가 있으면 보류 표시가 먼저 선택된다.'),
 (9,'④ 작은 좌표 차이와 경계 기준','짧게 투영된 팔뚝의 비율이0.1 기준 양쪽으로 나뉘어 위치 표시가 달라졌다. 이 사례에 맞춰 기준을 바꾸지 않았다.')])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>위치 표시 차이 분석</title><style>
body{margin:0;background:#edf2f7;color:#132438;font:17px/1.65 'Malgun Gothic',sans-serif}main{max-width:1080px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{font-size:25px}h3{font-size:22px}article,.box{background:white;border-radius:14px;padding:26px;margin:24px 0}img{width:100%;display:block;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:14px;border-bottom:1px solid #d9e1ea}.metric{font-size:32px;color:#146b9b;font-weight:bold}a{color:#146b9b}pre{font:16px/1.8 'Malgun Gothic',sans-serif;white-space:pre-wrap}</style><main>
<p>WordsInSilence · 공개 TRAIN899 분석 · 새 학습 없음</p><h1>위치 표시가 다른 이유는<br>관측 차이와 처리 규칙 차이가 함께 있기 때문</h1>
<div class="box"><span class="metric">22.38%</span><p>동일 시점의116,640손-프레임 중26,108에서 표시가 달랐다.<br>단어 오답률이 아니라 두 입력 방식의 위치 표시 불일치다.</p>
<table><tr><th>원래 차이를 구분한 결과</th><th>개수</th><th>비중</th></tr><tr><td>근거 사용 정책을 맞추자 사라진 차이</td><td>6,795</td><td>26.03%</td></tr><tr><td>그 후에도 남는 손 관측·부분관측 차이</td><td>11,860</td><td>45.43%</td></tr><tr><td>그 후에도 남는 팔·손목 추정 차이</td><td>7,453</td><td>28.55%</td></tr></table><p>고정한 진단 순서에 따른 분해다. 같은 장면에 여러 원인이 함께 있을 수 있다.</p></div>
<div class="box"><h2>다음 보완은 정보의 역할을 분리하는 것</h2><p>실제로 보이는 손 관절 / 팔로 추정한 위치 / 좌우 배정 불확실성을 별도로 기록한다. 불확실한 손을 실제 관측으로 바꾸지 않고, 기존 C는 유지한다.</p><p>화면 밖 손목 추정을 무조건 지우면 원래 같던1,935표시도 새로 달라진다. 단순 삭제나 강제 일치는 해결책이 아니다.</p></div>
'''+cards+'<div class="box"><h2>전체 분석·한계·다음 계획</h2><pre>'+html.escape(text)+'</pre></div></main></html>'
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
body='ops/'+U.name+'. 공개TRAIN899/100단어만 감사, 새학습·추론0. 공통116,640손프레임 중 위치26,108(22.38%)불일치. 실제팔꿈치/손목/손관측 차이+정책차이 공존. 동일sensor의화면안근거정책으로기존6,795차이해소되나새1,935차이발생; 강제일치/추정일괄삭제금지. 배정보류1,814가팔위치근거우선차단(안쪽533/경계밖1207/모름74). 0관절일부관측표시0, 관측됨은palm5이지21전부아님. 저장위치전수재구성+scalar32,364검산PASS/10,793핀보존, RGB12패널10고유장면7영상검토. 같은담당자별도계산검산. 결과RESULT_KO.txt/html, 다음NEXT_PLAN.json 제안만. 기존C/웹/DTW유지; 사용자/멘토/DEV/보류관측읽기0.'
nextbody='word100_position_state_audit_20261018/RESULT_KO.txt와NEXT_PLAN.json을사용자와확인. 다음제안은실제Hand21관측/팔위치추정/좌우배정보류를분리한별도진단출력검증이며아직구현·학습승인아님. 동일근거일관성/정상보존/숨긴정보누출방지검증후별도학습계획검토. 기존C 유지, 사용자자료학습/튜닝/선택금지. 완료감사재실행금지, 새학습은결과함께확인후.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';doc=p.read_text(encoding='utf-8-sig')
for heading,title,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-18 공개 TRAIN 위치 상태 차이 원인 감사 완료',body),('## 9. 다음 단일 작업','2026-10-18 위치·관측·배정 근거 분리 보완 제안 검토',nextbody)]:
 assert doc.count(heading)==1;doc=doc.replace(heading,heading+'\n\n### '+title+'\n'+b+'\n',1)
p.write_text(doc,encoding='utf-8')
state=read(O/'STATE.json');state['current_stage']='WORD100_POSITION_STATE_AUDIT_COMPLETE';state['updated_date_kst']=at;state['next_single_action']=nextbody
state[U.name]={'at':at,'status':'COMPLETE_AWAITING_REVIEW','train_videos':899,'shared_hand_frames':116640,'position_mismatch':26108,'mismatch_percent':s['mismatch_percent'],'independent':'PASS','new_training':0,'user_mentor_heldout_used':False,'dev_observations_used':False,'production_changed':False,'result':str(U/'RESULT_KO.txt'),'next_plan':str(U/'NEXT_PLAN.json')}
(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 最新 인수인계\n\n갱신 '+at+'\n\n### 현재 목표·마지막 완료\n'+body+'\n\n### 다음 단일 작업\n'+nextbody+'\n\n현재 실행 중 명령·잠금 없음. st-gcn 자동화 삭제 상태 유지. 원본/W:/sources/과거동결결과 읽기전용, 카메라/다운로드/외부전송/Git 없음.\n',encoding='utf-8')
status={'stage':'COMPLETE_VERIFIED_AWAITING_REVIEW','at':at,'training':0,'model_inference':0,'production_changed':False}
(U/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pins={str(p):sha(p) for p in U.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'}
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:pins[str(p)]=sha(p)
save(U/'FINAL_AUDIT.json',{'status':'PASS','at':at,'verification':str(U/'INDEPENDENT.json'),'visual':str(U/'VISUAL_REVIEW.json'),'output_pins':pins,'training':0,'next_is_proposal_only':True})
print('Report and three current-state documents synchronized. No training or production changes.')
if __name__!='__main__':raise RuntimeError('Run as script only')
