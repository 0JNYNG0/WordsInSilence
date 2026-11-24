import json,shutil,hashlib
from pathlib import Path
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 assert read(U/'VERIFICATION.json')['count']==54
 assert read(U/'BRIDGE_COMPILE.json')['exit_code']==0
 assert read(U/'LOCAL_REPLAY.json')['status']=='PUBLIC_CONSUMER_REPLAY_PASS'
 graph=dict(A_source_review='COMPLETE_WITH_ONE_INVALID_SOURCE_AND_ONE_UNRESOLVED_VARIANT',B_input_and_learning='COMPARISON_COMPLETE_NO_PROMOTION',C_unity_existing_word_bridge='CSHARP_COMPILE_PASS_EDITOR_LICENSE_BLOCKS_PLAYER_BUILD',D_portable_demo='CONSUMER_REPLAY_PASS_FULL_DEVICE_AND_PACKAGE_PENDING',greeting='MULTISIGNER_RGB_NOT_READY',prior_C7='PRESERVED_NOT_REPLACED')
 write(U/'GRAPH.json',graph)
 write(U/'CURRENT_EXECUTION.json',dict(status='FIXED_COMPARISONS_COMPLETE_NO_RUNNING_JOBS',sessions_finished=[30290,55932,96372,71282,25319],source_failure_preserved=True,training_complete=True,models_promoted=False,unity_exit=198,unity_license_needed=True,camera_opened=False,next='New fixed FIT-only handshape transfer/contrast design; actual Unity build after license activation. Do not repeat completed comparison.',prior_C7='Retained; numeral probe does not clear G2'))
 summary='공식ㄴ 수형문장·손모양보조지도논문검토. 공개60계획중59관측(원본손상1),숫자35영상105RGB앵커검수;사과ID05뜻변이보류. 숫자26학습/9개발,이름만vs손가락보조지도각3seed모두9/9로추가개선미입증. 별도104후보새12/12,기존300의293→292회귀로웹미채택. 기존C·T3-F/100공통학습보존. Unity코드컴파일PASS,실제빌드는라이선스exit198. W:Python읽기차단현재잘하다공개소비부추론PASS;전체HDD분리시연미검증. 같은Codex54별도검산.'
 nxt='공통 손모양 전이용 학습 대비를 먼저 정리한다: 기존100 FIT에서 숫자와 같은 모양·다른 방향, 검지 접힘/펴짐, 정지/이동 중 같은 모양을 구분하여 부족한 정답을 준비하고 새 고정 비교를 설계한다. 완료한6개수형모델/104후보 재학습·개발문턱튜닝금지. Unity 실제빌드는Hub 편집기라이선스확인 후 저장public fixture로재개. C7의기존구간/유지연구재개점은보존하며이숫자시험을G2통과로간주하지않는다.'
 before=U/'before';before.mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
  q=before/p.name
  if not q.exists():shutil.copy2(p,q)
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig')
 title='### 2026-11-13 지숫자·설명 특징 지도 비교 및 시연 연결 준비'
 assert title not in s
 h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 s=s.replace(h8,h8+'\n\n'+title+'\n'+summary+'\n근거: `research/front_keyphase_v1/ops/demo_handshape_learning_20261113/RESULT_KO.txt`, `VERIFICATION.json`.\n',1)
 s=s.replace(h9,h9+'\n\n'+title+'\n'+nxt+'\n근거: `research/front_keyphase_v1/ops/demo_handshape_learning_20261113/GRAPH.json`.\n',1);p.write_text(s,encoding='utf-8')
 state=read(O/'STATE.json');state['active_task']='demo_handshape_learning_20261113';state['next_single_action']=nxt;state['demo_handshape_learning_20261113']=dict(status='FIXED_RESEARCH_COMPLETE_NO_PROMOTION_UNITY_LICENSE_PENDING',summary=summary,report=str(U.relative_to(ROOT)/'RESULT_KO.txt'),graph=graph,shape_runs=6,classifier_runs=1,original_model_replaced=False,user_training=False,new_downloads=0,camera_opened=False,git_actions=False,reviewer='Codex implementation and separate arithmetic; not independent human/KSL expert',next=nxt);write(O/'STATE.json',state)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 지숫자·사전설명 보조지도 비교\n\n'+summary+'\n\n다음: '+nxt+'\n\nU=research/front_keyphase_v1/ops/demo_handshape_learning_20261113\nPY=research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8\n\n'+
 '숫자 사전 원문을 자동 연결한 실험이 아니다. 공식 ㄴ 원문은 구조화 가능성의 근거이며 숫자105시점은 실제 공개 RGB로 검수했다. 손모양 두조건 같은32은닉63입력/600단계/3seed. 입력에서 단어/사람/몸절대위치/화면내손방향제거. 표본9개개발영상/27상관시점, 정적3모양밖 전이효과미확인. index모두펴짐,숫자6 ID08 ring/little6칸미확인마스크. 모델선택·전체피드백성능주장금지.\n\n'+
 'RTMW 새47개/감사기존12개재사용. 숫자4 ID05 ZIP원본moov없음,최초실패·SHA대조보존. 사과ID05 움직임변이뜻미확정으로classifier제외,오답판정X. TCN옛몸체+100출력고정,추가4출력만FIT899+34=933로80epoch학습. 개발기존300/새12. 진주시지부ID12가숫자2로추가혼동,104웹미채택. 원가중치/기존4단어웹유지.\n\n'+
 'Unity별도소스: Assets/_Project/Scripts/Practice/LocalFeedbackBridgeDemo.cs 및 Assets/WordsInSilenceValidation/Editor/LocalFeedbackBridgeBuild.cs. 기존씬/EditorBuildSettings불변. compile_bridge.ps1 standalone컴파일PASS,첫중복참조오류보존. Editor실제빌드exit198 No valid Unity Editor license;활성화는사용자UnityHub확인필요. 공개PUBLIC_RESULT_FIXTURE.json준비,실제Unity통신/카메라실행은아직. 설치된편집기6000.3.21f1.\n\n'+
 'LOCAL_REPLAY: W:Python open차단새프로세스에서현재FeedbackConsumer+잘하다공개RTMW좌표→TCNtop3/카드2PASS. 약0.083초는소비부만. 물리HDD분리/네이티브파일차단/촬영/전체패키지시험아님. 사용자자료·멘토자료미접근. 모든본단위학습/실행완료;실제웹서버/카메라새로시작안함.\n\n'+
 '기존본연구: word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization. C6 36개진단모델완료;전체움직임/유지개선이나팔따라쓸기seed42 174→169/180회귀,G2REWORK. 다음C7 새움직임손실15고유FIT시점(경계8)원인분리. 그모델재실행/최종8사용/DEV문턱튜닝금지. 이번숫자학습이C7/G2대신통과한것아님.\n',encoding='utf-8')
 files=['RESULT_KO.txt','SOURCE_REVIEW.json','ADMISSION.json','SHAPE_RESULTS.json','TCN_RESULTS.json','VERIFICATION.json','LOCAL_REPLAY.json','BRIDGE_COMPILE.json','GRAPH.json','CURRENT_EXECUTION.json']
 write(U/'COMPLETION.json',dict(status='RESEARCH_COMPARISON_COMPLETE_NOT_PRODUCT_COMPLETION',outputs={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in files},protected_pins_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in read(U/'FROZEN.json')['protection'].items()),state_key='demo_handshape_learning_20261113',unity_build_blocked=True,next=nxt))
 print('REPORT AND HANDOFF SYNCHRONIZED')
if __name__=='__main__':main()
