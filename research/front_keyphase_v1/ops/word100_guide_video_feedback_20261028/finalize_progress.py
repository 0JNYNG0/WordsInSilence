"""Record honest partial completion and exact resume point; do not claim AI complete."""
from pathlib import Path
import json,hashlib,ast
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary=read(U/'SUMMARY.json');verify=read(U/'review/INDEPENDENT_VERIFY.json');ui=read(U/'UI_VERIFY.json')
assert ui['pass'] and summary['technical_invariance_pass']==summary['technical_invariance_cases']
for p in U.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
goal=read(U/'AUTHORIZATION.json')['goal_ko']
stages=[
 {'step':1,'name':'한국수어 설명과 공개 영상 연결','status':'SOURCE_INDEX_AND_PARTIAL_LINKS_DONE',
  'done':'12표현36영상 경로/출처 감사,5사전항목6문구→7특징63시점의 부분 연결 초안',
  'remaining':'완전한 표현형·시간단계·필수/허용 규칙 연결은 미확인'},
 {'step':2,'name':'사전학습 모델의 실제 동작 관찰 시험','status':'SAVED_MEDIAPIPE_REPLAY_DONE_VLM_NOT_RUN',
  'done':'저장 MediaPipe 4351프레임에서69종 관측값 추출,가이드와 분리;영상언어시험8영상128이미지/실행기준비',
  'remaining':'Qwen3-VL-2B 약4.27GB 구체적 다운로드 확인 대기; 실제 모델 추론0회'},
 {'step':3,'name':'학습할 단계·관계와 평가 조건 정리','status':'CONTRACT_DONE_TRAINING_NOT_READY',
  'done':'손별 상태·관계·변화·순서·관측부족 출력과 사람/표현 분리,가이드 정답누설 차단 계약',
  'remaining':'검증된 학습 정답과 실제 정상/오류 근거 없음;새 의미/교정학습0회'},
 {'step':4,'name':'전체 피드백 비교 검증','status':'GEOMETRIC_REPLAY_DONE_SEMANTIC_VALIDATION_PENDING',
  'done':'공개ID03의12영상을 ID01/02와24쌍비교,미관측처리수정,72불변+24범위/보류계약검사',
  'remaining':'중심의미모델/중요오류검출/정상허용경계/교육정확도/전체CPU지연미검증'}
]
save(U/'STAGE_STATUS.json',{'at':now,'overall':'NOT_ALL_FOUR_STEPS_COMPLETE','stages':stages,
    'primary_next_gate':'USER_CONFIRMATION_FOR_SPECIFIED_EXTERNAL_MODEL_ACQUISITION',
    'new_training':0,'new_vlm_inference':0,'external_downloads':0,'operating_changes':0})
text='''한국수어 가이드·영상 기반 피드백 — 현황 보고

목표
'''+goal+'''

결론
목표와 방향성을 프로젝트 기준 문서 §2·§8·§9, 현재 STATE, 인수인계에 고정했다. 로컬 자료 준비와 관측·숫자 비교 검증은 진행했지만 요청한 1~4 전체와 중심 피드백 AI가 완성된 것은 아니다. 사전 설명을 이해하고 의미 단계와 허용/오류를 판정하는 모델은 아직 실행·학습하지 않았다.

실제로 한 일
1. 기존 공개 한국수어12표현·기록ID3명·36영상과576RGB표본의 출처/연결을 감사했다. 사전5항목6문구에서7공통특징을63표본에 부분연결했다. 관찰지지50/다른표본상태4/불확실9를 보존했다. 전체 의미 단계의 정답 연결이 아니다.
2. 저장 MediaPipe 관측에서 손위치·팔각도·손가락각도·양손관계·방향·속도를69종으로 분리했다. 단어와 기대가이드는 관측 계산에 넣지 않았다. 원본4351프레임을 보존하고 기존 주석구간1500프레임으로 비교했다. 새로운 MediaPipe 검출을 다시 실행한 것은 아니다.
3. 공개기록ID03의12영상을 각각 ID01과02참고에 연결해24쌍을 재생했다. 한 영상의 관측이 사라져도 원래 시간대와 비교범위를 남긴다. 참고영상들은 하나로 평균내지 않았다.
4. 새 영상언어 후보 Qwen3-VL-2B의 시험입력8영상128이미지와 로컬실행기를 준비했다. 다운로드 확인이 없으므로 새 가중치/자료 다운로드와 추론은 실행하지 않았다.
5. 어떤 정보를 학습하고 어떤 정보를 미확인으로 남길지 계약을 작성했다. 검증되지 않은 가이드나 AI 초안을 독립 정답으로 변경하지 않았다.

실제 수정한 문제
- 참고와 비교 양쪽에서 모두 확인하지 못한 항목이 동작차이 비용에 섞였다. 그 정보는 측정차이와 분리하고 참고의미관측으로 남겼다.
- DTW가 다른프레임을 대신 연결하면 비교영상의 짧은 미관측이 대표대응값에서 사라질 수 있었다. 원래 비교영상의 직접미관측도 전체상태에 반영했다.
- 통제변환 float32 반올림오류는 최초결과를 보존한 후 float64수학변환으로 수정했다. 임계값을 완화해 통과시키지 않았다.

검증 결과의 의미
72개 소프트웨어 불변조건과24개 중간삭제/역순의범위·보류계약이 통과했다. 별도Codex가2060개산술/코드조건,가이드부분연결177개조건,브라우저24개조건을 검산했다. 검사수는 영상수나 인식정답수가 아니다. 독립 사람/한국수어 전문가의 정답확인이 아니다. 실제 수어 정상/오류 정확도는 측정하지 않았고 운영성능향상을 주장하지 않는다.

중요하게 남은 문제
- '멀다'는 확보한사전의두손시작과현재영상의한손시작이달라,같은이름이라는이유로강제연결하지않았다. 다른표현도세부변이·방향·접촉·필수반복횟수는미확인이다.
- DTW가시간을맞췄다고같은의미단계가맞은것은아니다. 참고미관측쪽으로과도하게정렬될가능성도남아있다.
- 구간삭제에서수치가변해도그부분이수어의필수동작인지판정할정답은없다. 현재코드는누락판별성공을주장하지않는다.
- 기존RGB초안과관절추정은정답데이터와다르다. 정상동작의차이분포만으로언어적허용범위를확정할수없다.

이어갈 순서
다운로드확인후 준비한영상으로Qwen후보의가이드없는관찰부터시험한다. 실제손·모양·순서·근거시간과미관측을잘구분하는지,잘못된가이드에끌려영상에없는행동을만들지않는지확인한다. 그결과에서신뢰가능한학습대상을정하고,작은검증된한국수어자료로중심학습·피드백비교를이어간다. V4최적화나새검출기추가로목표를바꾸지않는다.

유지
C·T3-F/운영웹/DTW/기존결과 보존. 사용자·멘토영상의내용열람·학습·튜닝·선택 없음. 새다운로드/학습/영상외부전송/카메라/Git작업 없음.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
msg='사용자 확정 목표를 §2에 기록하고 공개12표현36영상 자료감사·부분가이드연결·69관측값분리·24쌍재생을 수행했다. 양쪽미관측을차이와분리하고DTW대표선택이숨기는query미관측을복원표시했다. 72불변/24변형계약 및 별도2060산술/177가이드/24UI검사통과. 이는소프트웨어검증이며의미단계AI/교육정확도개선이아니다. 1~4전체미완료. 새VLM8영상입력과offline실행기준비;Qwen3-VL-2B 약4.27GB다운로드의구체적확인대기. 새학습/추론0·C/T3F/운영보존·개인자료제외.'
nxt='사용자가 pending Qwen3-VL-2B 약4.27GB(Apache-2.0,공식HF) 다운로드를 확인하면 프로젝트별도폴더에서가중치/필요라이브러리를확보하고 run_local_vlm.py로8공개영상의가이드없는관찰부터시험한다. 확인전새다운로드금지. 관측주장/시간/잘못된가이드/역순대조를검토한뒤유효학습목표를정한다. 학습/교육정오를미리완료라고하지말고단계별STAGE_STATUS를갱신. 운영C/T3F와개인자료유지. 기존완료기하재생은불필요재실행하지않는다.'
p=R/'WordsInSilence_PROJECT_CONTEXT.md';ctx=p.read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-28 가이드·공개영상 연결과 비교 검산 완료 — 중심 AI 시험 미실행',msg),('## 9. 다음 단일 작업','2026-10-28 준비된 영상언어 관찰 시험 이어가기',nxt)]:
    if '### '+title not in ctx:ctx=ctx.replace(marker,marker+'\n\n### '+title+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.html, STAGE_STATUS.json.\n',1)
p.write_text(ctx,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: '+goal+'\n\n'+msg+'\n\n다음: '+nxt+'\n\n작업단위: '+U.name+'\n현재 학습/추론/다운로드 프로세스 없음. 1~4 전체완료아님. MODEL_TRIAL_PLAN/VLM_INPUTS/LEARNING_CONTRACT/WORK_PLAN을읽고재개. DOWNLOAD_AUTHORIZATION.json은아직없음. 사소한문제는자율해결하되신규외부확보확인은받기로한사용자기준유지.\n',encoding='utf-8')
s=read(O/'STATE.json');s.update(current_stage='GUIDE_VIDEO_REPLAY_VERIFIED_VLM_CONFIRMATION_PENDING',updated_at=now,next_single_action=nxt)
s[U.name].update(status='PARTIAL_COMPLETE_VLM_ACQUISITION_CONFIRMATION_PENDING',at=now,summary=msg,next=nxt,steps=stages,
    all_four_steps_complete=False,report=str(U/'RESULT_KO.html'),model_training_runs=0,vlm_inference_runs=0)
save(O/'STATE.json',s)
save(U/'CURRENT_EXECUTION.json',{'status':'NO_ACTIVE_PROCESS_AWAITING_DOWNLOAD_CONFIRMATION','at':now,'training_active':False,'download_started':False,'all_four_steps_complete':False})
files=[U/f for f in ['RESULT_KO.html','RESULT_KO.txt','STAGE_STATUS.json','SUMMARY.json','GUIDE_CLAIM_LINKS.json','VLM_INPUTS.json','LEARNING_CONTRACT.json','review/INDEPENDENT_VERIFY.json','review/GUIDE_LINK_VERIFY.json','UI_VERIFY.json']]
for f in files:assert f.exists()
save(U/'CLOSEOUT.json',{'status':'AVAILABLE_LOCAL_WORK_VERIFIED_OVERALL_REQUEST_NOT_FINISHED','at':now,
    'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    'root_documents_synchronized':True,'outstanding':['Specified model download confirmation','Actual VLM observation trial','Validated semantic/educational learning targets','Meaning-aware feedback evaluation'],
    'verification':'Separate Codex arithmetic/source/UI checks, not independent human ground truth'})
print('Partial progress, exact goal, and remaining execution gates synchronized.')
