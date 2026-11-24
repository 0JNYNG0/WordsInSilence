from pathlib import Path
import json,sys,hashlib

X=Path(__file__).resolve().parent; O=X.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

browser=read(X/'SOURCE_DEMO_BROWSER_VERIFIED.json')
parent=read(X/'RUNTIME_SAVED_VERIFIED.json')
review=read(X/'NEXT_STEP_REVIEW.json')
assert browser['status']=='PASS' and parent['status']=='PASS'
assert browser['actual_fresh_public_posts_across_attempts']==1
assert browser['actual_upload_posts']==0 and browser['camera_calls']==0
report=X/'RESULT_KO.txt'; assert report.exists()
completion=dict(status='SOURCE_LINKED_PARTIAL_DEMO_VERIFIED',whole_goal_complete=False,
  implementer='Codex root',separate_reviewers=['Codex retention_direction_review','Codex predicate_contract_review'],
  independent_human_KSL_ground_truth=False,
  browser_cases=12,independent_saved_bridge_checks=96,parent_runtime_checks=104,
  fresh_public_jobs=1,fresh_job_id=browser['fresh']['job_id'],
  fresh_server_analysis_seconds=browser['fresh']['latency']['total_analysis_seconds'],
  latency_distribution_or_click_to_display_measurement=False,
  accuracy_gain_claim=False,new_training=0,new_download=0,camera_calls=0,
  private_footage_read=False,user_or_mentor_training_tuning_selection=False,
  old_models_and_demos_replaced=False,isolated_demo='http://127.0.0.1:8808/',
  evidence={str(X/n):sha(X/n) for n in ['SOURCE_DEMO_BROWSER_VERIFIED.json','RUNTIME_SAVED_VERIFIED.json','BRIDGE_REVIEW_V2_CHECKS.json','NEXT_STEP_REVIEW.json','RESULT_KO.txt']})
(X/'FINAL.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=X
summary=(
 '새8808 설명연결 실제데모 검증완료.4표현 normal/cut/mainreverse 저장12화면·양영상재생·원시간·입력/결과일치·추정구분 통과. '
 '별도bridge96검산/root104검산 불일치0. 공개잘하다 새분석정확히1회 job fd785d173921436492c65e1703e46589, '
 'GPU.0/119프레임/서버9.800894초(관측9.280915·RGB6.119403병행/DTW.013366/설명연결.000514). '
 '개별참고대조0·학습prototype1.새학습0/이번정확도향상주장없음. 기존16관측/DTW·C/T3F·8804보존. '
 '전체순서·필수반복·누락정오·새사용자신뢰성미완료/goal active. '
 '순서다음검토: 기존576유도질문은45원관계/27FIT영상 계보로재사용가능하나완료·누락정답아님; 기존order도RGB입력포함,단순RGB추가재실험금지.')
nxt=(
 'X/NEXT_STEP_REVIEW.json과기존word100_guided_event_order_20261102/order_model.py·FROZEN·ORDER_SUPERVISION확인후 '
 '기존FIT순서감독을재사용하는공통readout비교를별도단위로준비. 기존물리관측/좋은RGBMotion고정·병렬보존; '
 '기존role/physical×occurrence곱readout과직접RGB시간표현에서질의한부분행동의발생을배우는readout을동일조건으로비교할계약을먼저고정. '
 '576질문=독립576사례아님/54보류는unknown유지/미관측=부재금지. 실제reverse접합부와남은정상움직임을무조건오답학습금지. '
 '정상보존·중요앞뒤관계·짧은역순·영상누락·원출력동일성preflight후필요한새학습/개발비교만1회실행. '
 '전체완료/필수반복감독은이번에만들어내지않는다. H14/15·ID16·개인·멘토금지. '
 'X새publicjob/저장12/완료U/R/T학습평가재실행금지. 새8808서버PID35040 유지(DEMO_LAUNCH/실제상태확인);현재훈련없음. '
 '9.8초는1회서버분석값이며이후관측병목을동일조건분해하되단축위해짧은동작프레임을버리지않는다.')
sync_flow.sync('설명 연결 실제 데모 검증 완료·순서 판정부 다음 비교 준비',summary,nxt,
 'SOURCE_LINKED_PARTIAL_DEMO_VERIFIED_ORDER_READOUT_NEXT',
 dict(server_port=8808,recorded_server_pid=35040,training=False,fresh_public_job_complete=browser['fresh']['job_id']))
assert read(O/'STATE.json')[X.name]['status']=='SOURCE_LINKED_PARTIAL_DEMO_VERIFIED_ORDER_READOUT_NEXT'
print('Demo milestone and next source-conditioned order task synchronized; whole goal remains active.')
