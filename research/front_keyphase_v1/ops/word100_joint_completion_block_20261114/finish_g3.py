from pathlib import Path
import json,hashlib,datetime,shutil,importlib.util
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
v=read(U/'VERIFIED.json');assert v['passed']and v['g3_development_pass'];assert not(U/'RESULTS_REVIEW.json').exists()
write(U/'FAILURES_AND_REPAIRS.json',dict(
 body_candidate='First body/forearm MLP: region23/27 (3unknown,1wrong), forearm3/5 (2wrong). Not admitted; original forearm head unchanged.',
 extent_candidate='First face-extent linear model25/27,2unknown. Failed original all-known-probe criterion; not admitted.',
 repair='Reviewed24 FIT source frames selected by low confidence.17 clear upper-body labels added,7 ambiguous excluded. Same linear architecture/optimizer/confidence. Final96/96 FIT and27/27 original exposed probe.',
 region_scope='Face-front vs upper-body projected position only; no face contact, depth, precise forehead/cheek region, or expression correctness.',
 integration_attempts=[
 'V1 stopped before output: exact coordinate matching failed for resampled input; normalization differed by ~1e-5.',
 'V2 stopped before output: repeated body poses did not uniquely identify facial frames.',
 'V3 stopped before output: historical edited-hand controls used cropped RTMW source, not the full-view RTMW cache.',
 'V4 completed: authoritative stored edit maps and correct full/crop sources; body anchors crosschecked. Failed attempts repeated partial NEW region forwards during recovery, not completed old TCN/DTW or G2 experiments.',
 'Saved assembly then separated unsupported requirements from visibility; prevented unconfirmed ownership from generating order-coaching. No neural rerun.'],
 all_failed_result_files_preserved=True,old_code_hashes_are_historical=True,
 remaining='Forearm across/upright labels overlap on oblique poses. Keep uncertain; do not relax angle/timing limits or change probe truth.',
 no_expert_review_claim=True))
report='''G3 관측 근거 통합 — 개발 검증 완료

쉽게 말하면
손모양 하나가 맞았다고 행동 전체를 완료했다고 하지 않도록 바꿨다.
같은 행동에서 손모양·움직임 방향·신체 위치·순서를 함께 보고, 확인된 차이와 확인할 수 없는 부분을 나눴다.
현재 완료는 연구 경로의 통합 단계다. 웹 교체와 실제 사용자 교정 검증은 아직 남아 있다.

얼굴 점에 대한 답
RTMW의 저장 결과에는 얼굴 점도 있다. 이전 신체 위치 부품은 코·어깨 중심이었고 이번에는 얼굴 윤곽·턱과 손 전체의 상대 위치를 사용했다.
손이 턱보다 아래인지, 얼굴 앞에 겹치는지, 얼굴 옆 어깨 부근인지 구분하는 데 활용한다.
정면 2D 투영 관계이며 실제 접촉·깊이·정확한 이마/광대 위치나 표정까지 판별했다는 뜻은 아니다.

이전 → 현재
1) 턱·어깨 가까운 정상 손 위치를 얼굴 앞이라고 오판: 기존23/27 → 새27/27 일치.
   27은 공개 기록08/09의 세 표현에서 이미 검토한 손 위치 정답 시점이다. 영상27개/사용자27명/전체 수어 정확도가 아니다.
   FIT02~07의 정답만 추가했고, 평가 정답이나 신뢰도 문턱은 바꾸지 않았다. 이 결과는 개발 비교다.
2) 펼친 손→접은 손만 확인 → 얼굴 앞→상체 쪽 위치 변화도 같은 구간에서 함께 확인.
   공개2영상에서 정상·짧은 충분수행·두 속도조건8개는 함께 확인했다. 중간에 자른5조건과 끝 자세만 보여준 경우는 완료로 올리지 않았다.
   전체27조건에는 가림·역재생·고정자세도 포함된다. 손가림과 실제 오류를 같은 말로 표현하지 않는다.
3) 손모양 오류/방향 오류를 따로 저장 → 같은 행동의 여러 근거를 함께 보존.
   자동82조건×3모델=246결과를 연결했다. 손모양+역방향 동시 시험3원본에서 손모양 차이는3원본, 방향 차이는2원본의 근거를 유지했다. 나머지1원본의 방향은 여전히 확정하지 않는다.
   정상 속도차이81순서쌍을 보존했고 정상 자료에 새 교정 안내를 추가하지 않았다.
4) 사용자 문장 초안도 근거와 연결했다.
   차이: ‘오른손의 네 손가락 펴기 부분을 다시 확인해 보세요.’
   반대 방향: ‘이동 방향이 참고와 반대로 관측됐어요.’ + 해당 설명 문장.
   끝 자세만 보임: ‘끝 자세는 보이지만, 그 자세로 바뀌는 과정은 확인하지 못했어요.’
   연결 불확실: ‘이 움직임과 설명을 아직 확실히 연결하지 못했어요.’
   이는 저장 결과용 문장 초안이며 운영 웹에 활성화하지 않았다.

검증 범위와 남은 문제
- G3 원래 네 기준(기존 근거 보존, 복합 차이, 관측 부족의 합격 방지, 원본 시간 연결)을 통과했다.
- 코드/저장 확률/입력 정규화/양손 공통 처리/보호477파일을 별도 검산했다. 같은 Codex의 검산이며 독립 한국수어 전문가 검증은 아니다.
- 팔뚝이 비스듬한 자세의 가로/세로 구분 후보는 실패하여 적용하지 않았다. 기존 동작 방향 근거와 손모양은 보존하고, 정확한 끝 자세는 불확실로 둔다.
- ‘조상’의 반복 전체, ‘이집트’의 손끝 오므림, 정확한 접촉·엄지 모양은 여전히 충분히 검증되지 않았다. 필요한 항목이 빠졌는데 단어 전체를 합격으로 표시하지 않는다.
- 부분 수행이 실제 오답인지, 가림 때문인지 애매하면 교정 명령으로 바꾸지 않는다. 편집 시험은 실제 학습자 오류 정확도가 아니다.
- 최초 두 부품 후보 실패와 세 번의 프레임 출처 확인 중단은 보존했다. 마지막 통합 결과와 별도 검산이 최종 근거다.

다음 작업
G4: 이번 모델·판정·문장 범위를 고정한 뒤, 봉인된 미사용 공개8자료의 노출 여부와 평가계약을 확인하고 비교한다.
통과하면 G5에서 확인된 근거만 웹의 색상·구간·짧은 문장으로 연결한다. 불확실/미지원 항목은 별도로 표시한다.
G6에서 사용자가 시작하는 실제 촬영으로 자동종료·전체 지연·재시도 흐름을 확인한다.
실패하면 원인별 FIT 보완으로 돌아간다. 최종자료에 맞춰 임계값을 고치고 같은 자료를 미사용 시험이라고 다시 부르지 않는다.

기존 C·T3-F·공통100 학습모델·운영 웹 유지. 사용자/멘토자료 접근·학습·튜닝 없음.
Final8 미개봉. 카메라·다운로드·외부전송·서버 재시작·Git 없음. 모든 이번 작업 프로세스 종료.
근거: extent_augmented/EVALUATION.json, JOINT_FINAL_V2.json, VERIFIED.json, FAILURES_AND_REPAIRS.json.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
write(U/'RESULTS_REVIEW.json',dict(status='G3_PASS_DEVELOPMENT_INTEGRATION_G4_PENDING',at=stamp,block_complete=True,whole_project_complete=False,
 implementer='Codex',verifier='same Codex separate saved-output and source audit; no independent human KSL verification',
 graph_box='G3',g3_checks=v['g3_checks'],counts=v['counts'],training_runs=9,
 training_scope='3 body/region candidate versions x3seeds; final region has96 FIT labelled observations, original27 exposed probe labels. No common100 retrain.',
 web_changed=False,final8_opened=False,private_access=False,process_running=False,
 limits=['Only validated observed features, never whole-sign correctness','Forearm posture definition unresolved; original head kept','Alternation, thumb, fingertips/contact and some clause ownership unsupported','Prior exposure means no independent accuracy claim'],
 next_box='G4',evidence={str(U/n):sha(U/n)for n in['RESULT_KO.txt','VERIFIED.json','JOINT_FINAL_V2.json','FAILURES_AND_REPAIRS.json']}))
(U/'before_finish').mkdir()
for p in[G/'TASK_GRAPH.json',G/'CURRENT_EXECUTION.json',O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']:
 if p.exists():shutil.copy2(p,U/'before_finish'/p.name)
spec=importlib.util.spec_from_file_location('g3_graph_api',G/'graph.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.record('G3','PASS',v['g3_checks'],[U/'RESULTS_REVIEW.json',U/'VERIFIED.json',U/'JOINT_FINAL_V2.json'],
 'Fixed G3 development integration gates passed. Joint shape/body goal verified within exposed source controls; preserved existing features/order. Forearm/fine-shape/repetition limitations remain explicit, no whole-sign verdict. G4 sealed evaluation and G5/G6 web/capture remain pending.')
next_action='G4: 이번 관측 근거·판정·문장 범위를 고정하고 Final8 노출감사/평가계약 확인 후 미사용 공개자료 비교·별도 검산. 실패 시 원인별 FIT 보완, 최종자료 재튜닝 금지. G4 통과 후 G5 웹 시각화, G6 사용자 시작 촬영·자동종료·전체지연 검증. 정확한 팔 끝 자세/반복/손끝 등 미검증 항목을 합격으로 올리지 않음.'
write(U/'CURRENT_EXECUTION.json',dict(status='G3_COMPLETE_DEVELOPMENT_SCOPE',at=stamp,running=False,training_running=False,next_box='G4'))
write(G/'CURRENT_EXECUTION.json',dict(status='G3_PASS_G4_NEXT',at=stamp,active_unit=str(U),running=False,next_box='G4'))
s=read(O/'STATE.json');s['current_stage']=G.name;s['current_followup']=U.name;s['next_single_action']=next_action
s[U.name].update(status='G3_PASS_DEVELOPMENT_INTEGRATION_G4_PENDING',completed=stamp,block_complete=True,training_running=False,training_runs=9,report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFIED.json'),next_box='G4',web_changed=False)
s[G.name].update(status='G3_PASS_G4_PENDING',at=stamp,active_unit=str(U),latest_evidence=str(U/'RESULTS_REVIEW.json'),training_running=False,web_changed=False,progress='G3 fixed development integration gates passed; no whole-sign correctness or web rollout',remaining=next_action)
write(O/'STATE.json',s)
title='### 2026-11-14 G3 동일 행동 근거 통합 개발 검증 완료 / G4 미사용 공개평가 다음'
entry='얼굴 윤곽·턱과 손 전체 상대 위치를 추가하고 FIT 경계17관측을 보강해 기존23/27→27/27 일치(노출 probe08/09의 정답 시점). 두 실패 후보 보존. 자동82조건×3모델246결과의 모양·방향·순서·같은구간 끝 상태를 통합하고 정상속도81순서쌍 유지. 짧은 충분수행 포함8긍정 대조 유지, 중간잘림5와 끝자세만은 완료 금지. 복합오류3원본의 모양3/방향2 근거 유지. 477보호파일/저장검산 통과. G3 개발 통합 기준만 PASS이며 전체수어 정오/반복/손끝·접촉/비스듬한 팔 끝자세는 미해결·명시 보류. 문장 초안 준비, 웹/Final8 미실행. C·T3-F/공통100 유지, 개인자료 접근 없음.'
context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),('## 9. 다음 단일 작업',next_action+' 이번9학습/82조건 완료 결과 재실행금지. 모든 프로세스 종료. 사용자 요청 G3칸 마무리와 그래프 보고까지 완료.')]:
 assert text.count(header)==1;text=text.replace(header+'\n',header+'\n\n'+title+'\n\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `VERIFIED.json`, `RESULTS_REVIEW.json`.\n\n',1)
context.write_text(text,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# G3 개발 통합 완료, 다음 G4 — {stamp}
현재 단위: {U}
전체 목표: 현재4단어 웹 피드백. 공통100/C·T3-F/개인자료 금지 유지.
최종 근거: RESULT_KO.txt / RESULTS_REVIEW.json / VERIFIED.json / JOINT_FINAL_V2.json.
G3 원래4개 개발 기준 PASS. 전체 단어 정오/실사용 교정/웹 적용 완료 아님.
얼굴점은 RTMW에 저장됨. 기존 코/어깨 기반에서 얼굴 윤곽·턱 대 손 전체 관계로 보완. 표정/물리접촉/세부얼굴부위 판별 아님.
3후보x3seed=9회 학습: 최초 몸관계/팔자세 후보 실패, 손-얼굴범위25/27 실패, FIT 경계17정답 보강 후27/27. 평가정답/문턱 변경 없음.
기존 학습모델/공통100/끝모양/웹 불변. 자동82조건x3모델246결과 통합, 정상속도81순서쌍 유지. 긍정대조8/8, 부분잘림5는 완료 안 함. 복합3원본 모양3/방향2 유지.
프레임 연결 실패 V1~V3와 수정 내용 FAILURES_AND_REPAIRS.json 확인. 최종 V4는 저장 편집 매핑과 원래 full/crop 관절 출처 사용. 이후 저장조립만 수정, 신경망 재실행 없음.
팔 가로/세로 경계, 조상 반복전체, 이집트 손끝 모으기, 정확 접촉/엄지, 일부 행동소유권은 미검증. 모르는 것을 오답·완료로 올리지 않음.
다음 한 작업: {next_action}
Final8 미개봉. 카메라/다운로드/외부전송/Git/서버재시작 없음. 사용자·멘토자료 접근/학습/튜닝 금지.
모든 학습·검증 종료. 완료 결과 재실행하지 말고 재사용. 독립 사람/KSL전문가 검토 아님.
''','utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');navy='#172D49';light='#EDF1F5'
fig,ax=plt.subplots(figsize=(14,7),dpi=160);fig.patch.set_facecolor('white');ax.set(xlim=(0,14),ylim=(0,7));ax.axis('off')
ax.text(.4,6.5,'피드백 기능 작업 그래프',fontproperties=font,fontsize=24,weight='bold',color=navy)
ax.text(.4,6.03,'이번 완료: 같은 행동의 근거를 모아, 확인된 차이와 판단 보류를 구분',fontproperties=font,fontsize=13,color=navy)
boxes=[(.45,4.05,'G0·G1  완료','자료·설명·학습 정답',True),
 (4.95,4.05,'G2  완료','행동 구간 ↔ 설명 연결',True),
 (9.45,4.05,'G3  이번 완료','모양·위치·방향·순서 통합',True),
 (.45,1.75,'G4  다음','미사용 공개자료 비교',False),
 (4.95,1.75,'G5  예정','영상 강조 + 짧은 교정 안내',False),
 (9.45,1.75,'G6  예정','촬영·자동종료·전체 지연',False)]
for x,y,t,b,done in boxes:
 ax.add_patch(FancyBboxPatch((x,y),4.05,1.35,boxstyle='round,pad=.08,rounding_size=.12',facecolor=navy if done else light,edgecolor=navy if t.startswith('G4')else'none',linewidth=2))
 ax.text(x+.2,y+.9,t,fontproperties=font,fontsize=16,weight='bold',color='white'if done else navy)
 ax.text(x+.2,y+.34,b,fontproperties=font,fontsize=13.5,color='white'if done else navy)
for y in[4.72,2.42]:
 for a,b in[(4.58,4.84),(9.08,9.34)]:ax.add_patch(FancyArrowPatch((a,y),(b,y),arrowstyle='->',mutation_scale=18,lw=1.6,color=navy))
ax.annotate('',xy=(.8,3.2),xytext=(13.1,3.87),arrowprops=dict(arrowstyle='->',color=navy,connectionstyle='angle,angleA=-90,angleB=0,rad=8',lw=1.3))
ax.text(.45,.93,'완료 = 정해 둔 개발 검증 통과. 웹 적용·실제 사용자 교정 검증은 아직 남아 있습니다.',fontproperties=font,fontsize=12,color=navy)
ax.text(.45,.43,'남은 한계: 비스듬한 팔 끝 자세 · 반복 전체 · 손끝/엄지/접촉  →  판정 불가를 오답으로 표시하지 않음',fontproperties=font,fontsize=11,color=navy)
fig.savefig(U/'PROGRESS_GRAPH.png',bbox_inches='tight',pad_inches=.15);plt.close(fig)
write(U/'RECORDS_SYNCED.json',dict(at=stamp,graph_status=next(n['status']for n in read(G/'TASK_GRAPH.json')['nodes']if n['id']=='G3'),paths={str(p):sha(p)for p in[G/'TASK_GRAPH.json',O/'STATE.json',context,R/'NEW_CHAT_HANDOFF.md',U/'RESULT_KO.txt',U/'PROGRESS_GRAPH.png']},process_running=False))
print(json.dumps(dict(status='G3_PASS_DEVELOPMENT',report=str(U/'RESULT_KO.txt'),graph=str(U/'PROGRESS_GRAPH.png'),next='G4'),ensure_ascii=False))
