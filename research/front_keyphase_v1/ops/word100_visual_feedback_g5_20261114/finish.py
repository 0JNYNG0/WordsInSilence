from prepare import U,O,read,write,sha
from pathlib import Path
import datetime,shutil,importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=O.parents[2];G=O/'word100_web_completion_graph_20261111';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
v=read(U/'VERIFIED.json');b=read(U/'BROWSER_VERIFIED.json');t=read(U/'CONTRACT_VERIFIED.json');assert v['status']==b['status']==t['status']=='PASS'
for p in (O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',G/'TASK_GRAPH.json'):
    dest=U/'before'/p.name
    if not dest.exists():shutil.copy2(p,dest)
remaining=[
 'Live capture still uses the prior research consumer. Latest G3/G4 qualified automatic inference is not yet attached to new camera attempts; do not imply this saved-result viewer proves live feedback.',
 'One source still has contradictory interval/direction decisions; it is gray uncertain, not repaired model accuracy.',
 'Alternating repetition entirety, fingertip gathering, thumb, physical contact and whole-sign correctness remain unsupported.',
 'Actual user-started capture, automatic stop and total waiting time are untested in this G5 task.'
]
summary=dict(status='G5_PUBLIC_EVIDENCE_VISUALIZATION_COMPLETE',at=now,url='http://127.0.0.1:8821/feedback-g5/',
 implemented=['Generic evidence-to-display adapter: exact source interval, anatomical hand, guide clause and correction','Side-by-side real public original/test media with hand/arm emphasis, selected intervals, slow/focused playback and native one-frame stepping','Difference, measured agreement and unknown/conflict are separate; whole-sign pass never emitted','Current 8821 practice-room navigation integration and rollback backups'],
 data=dict(source_public_videos=8,saved_scenarios=40,display_cards=74,correction_cues=17,meaning='Same G4 saved outputs rendered; scenarios include speed, prefix, missing hands, continuous reverse. Not 40 new videos or accuracy denominator.'),
 verification=dict(evidence_media_http_checks=v['checks'],synthetic_display_contract_checks=len(t['tests']),browser=True,protected_files=v['protected_files']),
 remaining=remaining,model_training_runs=0,neural_inference_runs=0,camera_opened=False,private_data_access=False,C_T3F_unchanged=True,common100_unchanged=True,
 alignment='Viewer matches playback duration only. No new DTW scores; current live DTW branch preserved.',
 live_runtime='prior existing runtime unchanged; new qualified runtime integration remains required',
 author='Codex',verifier='same Codex separate source/media/API and browser verification; not independent KSL expert')
write(U/'RESULTS_REVIEW.json',summary)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[
 'First local HTTP audit was blocked by sandbox WinError10013. Same read-only localhost audit ran in normal environment and passed.',
 'Focused playback overshot when browser animation callbacks were delayed. Added media timeupdate enforcement and clamped both videos to final evidence frames; browser verified.',
 'Order correction initially replayed only one reference action. Reference spans now include both action IDs supplied by exact order evidence, no word-specific exception.',
 'Narrow layout caused short text wrapping and uneven video caption heights. Updated mobile layout and visually verified.',
 'Desktop viewport DOM geometry was checked; full desktop screenshot API could not capture that override. Narrow screenshot inspection was completed; no full-desktop visual claim.'
],model_changes=0,training=0))
report='''G5 완료: 확인한 분석 근거를 손·팔 강조와 짧은 안내로 표시

화면: http://127.0.0.1:8821/feedback-g5/
기존 연습실 오른쪽 위의 “새 피드백 · 공개 검증 결과”에서도 들어갈 수 있습니다.

이전 → 이번
저장된 수치·문장 중심 → 해당 순간의 실제 영상과 손·팔 강조, 짧은 교정 안내.
예: 주황색 손·팔과 “오른손을 펴고 왼팔을 따라 손목 쪽으로 쓸어 보세요.”
판단 충돌 → 회색 “확인 필요 · 오답 아님”. 원시 차이값이 남아 있어도 교정 명령으로 표시하지 않습니다.
순서 차이 → 설명의 두 행동을 문장으로 연결하고, 참고 영상도 그 두 행동을 함께 보여줍니다.
구간 재생 끝 지점을 지나치던 문제를 고쳤고, 느린 재생·실제 1프레임 이동·확대를 추가했습니다.

무엇으로 확인했나
G4에서 이미 분석한 공개 8영상의 40조건을 사용했습니다. 40은 새로운 영상 수가 아니라 원본·속도·중간 잘림·손 관측 제거·전체 역재생 조건 수입니다.
원본과 편집 시험을 화면에서 구별합니다. 주황색 오류 예시는 실제 사용자 오답이 아니라 공개 영상을 역재생한 시험의 실제 분석 결과입니다.
문장·시점·손·원본 대응 및 로컬 API 검산 통과. 기존 보호477파일 유지. 실제 브라우저에서 정상/방향/순서/불확실/관측 누락/부분 수행과 재생을 확인했습니다.
별도의 가상 입력으로 좌우 손모양 안내 연결만 검사했습니다. 이것은 손모양 모델 성능 시험이나 학습 정답 추가가 아닙니다.

아직 남은 부분
G5는 공개 저장 결과의 웹 표시와 검증 완료입니다. 새 촬영은 아직 이전 판정부를 사용합니다. 최신 G3/G4 통합 판정부가 새 촬영에도 자동 적용됐다는 뜻은 아닙니다.
다음 G6에서 촬영 중 뽑은 관절을 최신 판정부에 연결하고, 공개 입력 재생으로 이전 결과와 일치하는지 확인한 후 사용자 시작 촬영·자동 종료·결과 표시·전체 대기시간을 확인해야 합니다.
이 공개 결과 화면은 재생 길이만 맞춰 나란히 보여줍니다. DTW를 새로 계산한 것이 아니며, 기존 촬영 화면의 DTW는 유지했습니다.
반복 전체, 세밀한 손끝·엄지·실제 접촉 및 단어 전체 합격은 여전히 미완성입니다. 판단이 엇갈리는 사례의 모델 자체 오류도 남았습니다.

보호와 복구
기존 C·T3-F와 공통100 가중치 유지. 새 학습·모델 추론·사용자 자료 접근·튜닝·촬영 없음.
서버·첫 페이지에 공개 결과 경로와 링크만 추가했습니다. 복구는 이 폴더 rollback.py로 두 파일을 되돌린 후 서버를 재시작합니다. 이후 편집이 있으면 복구 스크립트가 덮어쓰기를 거절합니다.
구현과 별도 검산 담당은 Codex이며 독립 한국수어 전문가 검토는 아닙니다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(16,9));fig.patch.set_facecolor('white');ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off');navy='#203650';muted='#607186';light='#eef2f7'
ax.text(.55,8.45,'피드백 기능 · 현재 작업 흐름',fontsize=25,weight='bold',color=navy)
ax.text(.55,7.88,'G5 공개 결과 화면 완료   →   새 촬영에서의 연결·작동 확인이 다음 단계',fontsize=16,color=muted)
def box(x,y,w,h,title,body,done=True):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035,rounding_size=.1',facecolor=navy if done else light,edgecolor=navy,lw=1.3));ink='white' if done else navy
 ax.text(x+.17,y+h-.39,title,fontsize=15,weight='bold',color=ink);ax.text(x+.17,y+.24,body,fontsize=12,color=ink,linespacing=1.6)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color=muted,lw=1.6))
titles=[('G0·G1 완료','범위와 설명\n정답 주석 정리'),('G2 개발 검증','행동 구간 찾기\n설명과 연결'),('G3 개발 검증','모양·방향·순서\n같은 행동에서 비교'),('G4 검증 완료','새 공개자료 평가\n불확실한 결과 분리'),('G5 화면 완료','손·팔 강조 + 교정문\n공개 저장 결과 검증')]
for i,(title,body)in enumerate(titles):box(.55+i*3.1,5.3,2.85,1.85,title,body)
for i in range(4):arrow((3.44+i*3.1,6.2),(3.6+i*3.1,6.2))
box(.55,2.4,7.15,2.05,'지금 화면에서 볼 수 있는 것','확인된 차이 → 주황색 손·팔 + 짧은 교정\n근거가 부족하거나 판단 충돌 → 회색 확인 필요',False)
box(8.15,2.4,7.1,2.05,'G6 남음 · 새 촬영에서 끝까지 확인','최신 통합 판정 연결 → 공개 입력 재검산\n사용자 시작 촬영 → 자동 종료 → 표시·대기시간',False)
arrow((14.15,5.2),(12.1,4.56));arrow((7.77,3.4),(8.08,3.4))
ax.text(.6,1.48,'전체 단어의 정답 판정까지 완료된 상태는 아닙니다.',fontsize=17,weight='bold',color=navy)
ax.text(.6,.83,'남은 한계: 구간 판단 충돌 · 반복 전체 · 손끝·엄지·접촉     |     C·T3-F와 공통100 유지',fontsize=13,color=muted)
fig.savefig(U/'PROGRESS_GRAPH.png',dpi=145,bbox_inches='tight',facecolor='white');fig.savefig(U/'PROGRESS_GRAPH.svg',bbox_inches='tight');plt.close(fig)
spec=importlib.util.spec_from_file_location('g5graph',G/'graph.py');graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
graph.record('G5','PASS',dict(evidence_text_visual_agree=True,no_unqualified_claim=True,rollback=True),[str(U/'RESULTS_REVIEW.json'),str(U/'VERIFIED.json'),str(U/'BROWSER_VERIFIED.json'),str(U/'CONTRACT_VERIFIED.json'),str(U/'CONNECTION.json')],
 'G5 qualified public saved-evidence viewer integrated into current web, exact source/card/time/hand checked; unknown never orange; reversible. No model accuracy gain. Live capture remains prior consumer and latest qualified runtime attachment/capture acceptance explicitly remains next G6; not claimed live-ready.')
next_action='G6: 새 촬영의 RTMW 관절 재사용 경로에 최신 G3/G4 통합 판정부와 G5 표시 계약을 연결한다. 먼저 공개 입력 재생으로 저장 결과와 동등성·충돌 보류·처리시간 검증 후, 사용자 시작 촬영의 자동종료·TCN/DTW·피드백·전체지연을 확인한다. 현재 실제 촬영은 이전 판정부이며 G5는 공개 저장결과 표시 완료다. C·T3-F/공통100 유지, 사용자자료 학습·튜닝 금지.'
state=read(O/'STATE.json');state.update(current_stage=G.name,current_followup=U.name,next_single_action=next_action,last_updated=now)
state[U.name]=dict(status=summary['status'],path=str(U),result=str(U/'RESULTS_REVIEW.json'),verified=str(U/'VERIFIED.json'),graph=str(U/'PROGRESS_GRAPH.png'),url=summary['url'],training_running=False,web_changed=True,private_access=False,remaining=remaining)
state[G.name].update(status='G5_PUBLIC_VISUALIZATION_COMPLETE_G6_PENDING',active_unit=str(U),at=now,latest_evidence=str(U/'RESULTS_REVIEW.json'),progress='G0-G5 limited-scope stages complete; public evidence viewer done. New capture inference attachment and G6 acceptance pending.',remaining=next_action,training_running=False,web_changed=True)
state['web_tutorial_live_integration_20261113']['g5_public_viewer']=dict(url=summary['url'],result=str(U/'RESULTS_REVIEW.json'),live_inference_changed=False)
write(O/'STATE.json',state)
heading='### 2026-11-14 G5 공개 결과 시각화 완료 / 새 촬영 최신 판정 연결·G6 남음'
latest='G4 공개8영상의 저장40조건을 기존8821웹에 연결. 74설명 카드에 원본 시점·좌우 손·교정문·불확실을 연결하고 역재생 시험5조건의 차이를 주황, 판단충돌1조건을 회색으로 표시. 실제 손·팔 강조/구간·느린재생/1프레임 이동/확대 제공. 순서 비교의 양쪽 참고구간 누락과 구간재생 종료초과 보완. 저장근거·영상·HTTP1144검사 및 표시계약12검사·브라우저 확인 통과. 새 학습/추론/사용자자료 접근 없음, 보호477파일 유지. 이번은 공개 저장결과 웹표시 완료이며 새 촬영의 최신 G3/G4 판정은 아직 미연결(기존 촬영판정부 유지). 기존DTW/C·T3-F/공통100 불변. 근거: `research/front_keyphase_v1/ops/word100_visual_feedback_g5_20261114/RESULTS_REVIEW.json`, `VERIFIED.json`, `BROWSER_VERIFIED.json`. 화면 http://127.0.0.1:8821/feedback-g5/ . 서버PID25556, 카메라 꺼짐.'
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',latest),('## 9. 다음 단일 작업',next_action+' G4의8자료는 평가 노출됨. 재학습하면 독립 새평가 필요. 완료학습·G4추론 반복금지.')]:
 assert marker in s;s=s.replace(marker,marker+'\n\n'+heading+'\n\n'+body+'\n',1)
p.write_text(s,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# G5 공개 시각화 완료, G6 새 촬영 연결·검증 다음 — '+now+'\n현재 단위: '+str(U)+'\n'+latest+'\n\n다음 한 작업: '+next_action+'\n\nG5 구현은 adapter.py/prepare.py/site/ 및 g5_routes.py. 기존8821 server.py와 site/index.html은 경로·링크만 변경; CONNECTION.json과before에 복구근거. 공개8영상은 C:에 있어 현재 결과뷰는 HDD 없이 열린다. 전체 데모 이식성 보장은 별도.\nG4 유효결과 CONFLICT_REGRESSION.json 그대로 읽었고 모델 변경 없음. 40조건/74카드/17안내는 표시 수이지 새 정확도가 아니다. 역재생 주황 예시는 실제 사용자 오류 아님. 최신 G3/G4를 새 촬영에도 적용했다고 보고하지 않는다.\nG0-G5 제한범위 PASS, G6/DONE PENDING. 엄지/손끝/반복전체/접촉/전체단어정오 미지원, 구간충돌1 미해결. 카메라를 자동으로 켜지 않는다. G6 공개 재생을 먼저 검산하고 실제 촬영은 사용자 시작으로 진행.\n서버PID25556, 127.0.0.1:8821. 실행중 학습/추론 없음. Codex가 구현 및 별도 검산, 독립 전문가 검토 아님.\n','utf-8')
for p in (U/'CURRENT_EXECUTION.json',G/'CURRENT_EXECUTION.json'):
 write(p,dict(status=summary['status'],at=now,active_unit=str(U),running=False,server_running=True,server_pid=25556,port=8821,camera_opened=False,next=next_action))
print('G5 public visualization completed and recorded. G6/live attachment explicitly pending.')
