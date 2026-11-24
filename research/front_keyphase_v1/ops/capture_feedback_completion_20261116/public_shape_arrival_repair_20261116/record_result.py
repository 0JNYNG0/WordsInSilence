from pathlib import Path
import json,hashlib,datetime,shutil
H=Path(__file__).resolve().parent;G=H.parent;O=G.parent;ROOT=O.parents[2];V=G/'completion_v1';WEB=O/'web_tutorial_live_integration_20261113'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();normal=read(H/'validation_v2/RESULT.json');controls=read(V/'distinction_combined_v2/RESULT.json');camera=read(H/'web/CAPTURE_CHECK.json');web=read(H/'web/CONSUMER_CHECK.json');http=read(H/'web/HTTP_CHECK.json');confirm=read(H/'confirmation_views/RESULT.json');adm=read(H/'web/ADMISSION.json');contract=read(H/'CONTRACT.json')
 assert all(sha(p)==h for p,h in adm['pins'].items());assert camera['passed']and confirm['passed']and http['repair_active']and all(r['passed']for r in http['rows'])
 protected={p:sha(p)==h for p,h in contract['pins'].items()if not p.endswith('server.py')};assert all(protected.values())
 fold=[r for r in controls['rows']if r['kind'].startswith('flex')]
 assert len(fold)==25 and all(any('shape'in x for x in r['differences'])for r in fold)
 assert normal['controls_passed']==54 and controls['failed']==0 and normal['shape']['lost_correct']==0
 source_review=dict(reviewer='Codex direct native RGB and skeleton review; not independent human expert',
  shape_false_fold=[dict(clip='NIA_SL_WORD1290_REAL07_L',frames=[68,78],split='FIT source',evidence='Projected edge-on open hand; shortened/overlapping chains led to confident curl.'),dict(clip='NIA_SL_WORD1290_REAL11_R',frames=[62,66],split='Known DEV, not optimization',evidence='Open fingers aimed toward camera. Legacy folded span is inconsistent with source RGB.')],
  repeat=dict(clip='NIA_SL_WORD1290_REAL13_F',split='Known DEV used for diagnosis',observed_ends=[58,67],missed_reversal=61,old=1,new=2,reason='Old bridge required two adjacent separating-motion centers. A seven-frame separating window plus independently observed separated posture now rearms.'),
  nearby_anchor_audit='WORD2551 and WORD0071 neighbouring frames include genuine closing; not relabelled open because nearby anchors were open. WORD1139 short projection uncertainty recorded separately.',
  train=dict(people=[3,7],cameras=['L','R'],frames=[55,65,75],hands=['L','R'],new_coarse_anchors=24,individual_finger_labels=None),
  images=[str(H/'public_views/fit_source_review.jpg'),str(H/'public_views/shape_observation_review.jpg'),str(H/'public13_repeat_review.jpg')])
 write(H/'SOURCE_REVIEW.json',source_review)
 result=dict(status='PUBLIC_REPRODUCTION_REPAIRED_AND_WEB_ADMITTED_NEW_USER_CHECK_PENDING',at=now,
  implementation='Codex root',verification='Same Codex separate source/functional/hash/HTTP checks; no independent expert or new subagent',
  graph=dict(A_public_reproduction='complete',B_isolated_shared_repair='complete',C_public_regression_and_capture='complete',D_local_web_connection='complete',physical_webcam_followup='not_run'),
  changes=dict(shape='Fine-tuned shared local Conv+GRU projection observer. Confident disagreement from majority of fixed observers makes the old curled claim unknown, never automatically open/normal.',
   repeat='Common learned separating motion + observed separated posture rearm, existing observed near endpoint required. No expected-count fill, DTW count, or word-name threshold.',
   ui='Unknown shape uses its actual disputed interval and neutral sentence, not curl correction.',
   deployment='Validated gratitude assembly. Shared head/bridge have no word/person ID inputs. Apple and original C/T3-F paths unchanged; all-word improvement not claimed.'),
  training=dict(runs=3,seeds=[42,43,44],new_source_hand_anchors=24,fit_samples=552,fit_clips=105,core100_training='preserved, not retrained this task',user_training=False,user_tuning=False,user_candidate_selection=False),
  evidence=dict(gratitude_inputs=39,distinct_source_videos=34,camera640_variants=5,new_camera_confirmation=confirm,shape_regression=normal['shape'],movement_controls_passed=54,movement_controls_total=54,other_controls_passed=40,other_controls_total=40,reverse_diagnostic_only=5,synthetic_folded_pose_controls_with_warning=len(fold),apple_unchanged_public_clips=7,public_camera=camera,http=http),
  scope='Known development plus fresh camera-view regression. Synthetic keypoint controls are not learner-error accuracy; public camera fixture excludes physical webcam and browser paint.',
  retained_issues=['Ambiguous shape is now unknown, not guaranteed correctly recognized open.','Existing one source-anchor error and some unknowns remain.','Side-camera pose/plane/arrival failures remain outside qualified frontal lesson scope.','No new private webcam test; old private captures not re-evaluated.'],
  rejected_or_corrected=['Initial unanimous-open disagreement guard failed DEV11_R and was not deployed.','Single-model opposing vote would lose one previously correct DEV anchor; not deployed. Majority opposing votes used for abstention, same .85 model confidence; known DEV policy selection documented.',
   'Initial extra-view archive lookup failed; corrected from ZIP member index without downloading.','First consumer check had no candidate admission and stopped before results.','Combined distinction script resolved through existing module search path; results written to new completion_v1/distinction_combined_v2, not overwritten frozen outputs.'],
  protected=protected,web_pid=read(WEB/'SERVER.json')['pid'],web_url='http://127.0.0.1:8821/',admission=str(H/'web/ADMISSION.json'),camera_opened=False,new_downloads=0)
 write(H/'RESULT.json',result)
 report='''공개 자료에서 두 문제를 재현했고, 검증을 통과한 보완을 웹에 적용했습니다.

1. 편 손을 굽혔다고 경고하는 문제
공개 감사의 옆 카메라 2영상에서 손가락이 짧게 겹쳐 보일 때 굽힘 경고가 나왔습니다.
공개 수행자3·7의 4영상에서 확인한 양손24시점 정답을 기존 공개 정답에 더해 작은 공통 손모양 신경망을 추가 학습했습니다.
기존 판정과 새 관측이 충돌하면 굽힘이라고 단정하지 않고, 그 순간을 회색 ‘손모양 확인 부족’으로 표시합니다.
학습 사례7_L과 학습에 넣지 않은 개발 사례11_R에서 잘못된 굽힘 경고가 확인 부족으로 바뀌었습니다. 편 손을 항상 정확히 맞히게 됐다는 뜻은 아닙니다.
기존 손모양 검증237시점에서 추가 오판·정답 소실0. 기존 오판1은 남아 있습니다.
실제 공개 굽힌 손 자세를 넣은 합성 시험25개는 모두 손모양 차이 경고를 유지했습니다. 실제 사람의 오류 행동25영상이라는 뜻은 아닙니다.

2. 반복의 마지막을 놓치는 문제
공개 수행자13은 실제 두 번 움직였지만, 짧은 손떼기 구간을 연결하지 못해 한 번으로 셌습니다.
학습된 움직임과 손이 떨어진 자세를 함께 확인해 다시 움직이기 시작한 것으로 연결합니다. 실제 도착 위치가 관측돼야만 횟수를 인정합니다.
같은 원본과640해상도 처리본에서 1회→2회로 개선됐습니다. 서로 다른2명의 개선 결과로 세지 않습니다.
중간 종료·움직이지 않음·누락·시간척도·크기 변경 등을 포함한54조건, 굽힘·손면 변경 등의40조건 검사를 통과했습니다. 반복을 뒤집은5조건은 오류 정답으로 단정하지 않고 진단만 했습니다.

3. 사과·자동 종료·속도
사과 공개 영상7개에서 정상7항목, 교정 카드, 비교재생 대응, TCN 결과가 이전과 동일했습니다.
같은 공개 영상을 웹캠처럼 공급해 RTMW→자동종료→TCN·피드백까지 실제 경로를 비교했습니다.
사과: 종료 후 계산0.313→0.321초. 감사:0.342→0.372초. 두 단어 모두 자동종료·정상통과, 프레임 누락0, 촬영후 재검출0.
이 시간은 공개영상 재생시험의 결과 계산까지이며 실제 웹캠·브라우저 표시까지의 시간은 아닙니다. 횟수별 부하 변동을 포함한 정밀 벤치마크도 아닙니다.

4. 적용 범위와 남은 확인
특정 단어 이름·수행자 ID로 예외를 넣지 않은 공통 손모양·구간 연결 보완입니다. 이번에 검증한 감사 경로에 연결했고 사과 및 기존 C·T3-F는 보존했습니다. 모든 수어 단어 성능을 보장하지 않습니다.
새 관측이 불확실하면 통과시키지 않습니다. 손가락 겹침, 옆 시점의 방향·도착 판정에는 확인 부족이 남을 수 있습니다.
실제 웹캠에서 감사 동작이 개선됐는지는 새 촬영으로 확인해야 합니다. 이전 사용자 촬영은 이번 학습·튜닝·후보 선택·재평가에 사용하지 않았습니다.
최신 화면: http://127.0.0.1:8821/ (새로고침 후 02 연습 → 감사)

작성·실행·검증: Codex. 독립 수어 전문가 검수 아님. 새 다운로드·외부 전송·실제 카메라 실행 없음.
상세 근거: RESULT.json, SOURCE_REVIEW.json, validation_v2/RESULT.json, web/CONSUMER_CHECK.json, web/CAPTURE_CHECK.json, web/HTTP_CHECK.json.
실패 후보/정정도 RESULT.json에 보존했습니다. 검증 통과를 전체 피드백 기능 완성으로 해석하지 않습니다.
'''
 (H/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 next_action='새 웹캠 감사 정상 수행을 사용자가 할 때 자동종료·회색 확인부족/교정/정상통과·표시시간을 진단한다. 사용자 촬영은 학습·튜닝·후보선택 금지. 추가 문제가 있으면 공개 자료로 재현한 뒤 공통 보완한다. 실제 오류→교정→재시도와 HDD분리 시연 확인은 아직 남음.'
 state=read(O/'STATE.json');stage=state['capture_feedback_completion_20261116'];stage.update(status=result['status'],updated_at=now,next_single_action=next_action,report=str(H/'RESULT_KO.txt'))
 stage['public_shape_arrival_repair_20261116']=dict(status=result['status'],result=str(H/'RESULT.json'),admission=str(H/'web/ADMISSION.json'),training_runs=3,web_applied=True,private_training=False,private_tuning=False,private_candidate_selection=False,physical_webcam_recheck=False,pid=result['web_pid'])
 state['current_stage']='capture_feedback_completion_20261116';state['next_single_action']=next_action;state['updated_at']=now;write(O/'STATE.json',state)
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 entry8='\n\n### 2026-11-16 공개 손모양·반복 문제 보완 / 웹 적용, 새 사용자 확인 대기\n공개 감사7_L·11_R의 편손 굽힘 오판을 재현. FIT3·7의4다시점영상24손시점으로 작은 공통 Conv+GRU 손모양 관측부 추가학습(3seed), 기존굴곡판정과 충돌하는 구간은 회색 확인부족으로 표시. 13_F의 짧은 손떼기 연결을 공통 보완해 원본·640처리에서1→2회. 기존237손모양 지점 추가오판/정답소실0, 기존오판1남음;54이동조건·40구분조건통과(합성25굽힘경고유지). 사과7영상 결과동일. 공개실제캡처경로 자동종료 유지, 종료→계산 사과0.313→0.321초/감사0.342→0.372초(웹캠·화면표시제외). 8821 새모듈 HTTP검증완료, PID'+str(result['web_pid'])+'. C·T3-F/촬영코드/사과 보존, 사용자자료학습·튜닝·후보선택·재평가0. 공통코드이나 검증된 감사조립에 적용, 전체단어·정확한3D 판정 보장아님. 근거 capture_feedback_completion_20261116/public_shape_arrival_repair_20261116/RESULT.json·RESULT_KO.txt.\n'
 entry9='\n\n### 2026-11-16 공개 보완 적용 후 실제 사용 확인\n'+next_action+'\n'
 assert h8 in s and h9 in s;s=s.replace(h8,h8+entry8,1).replace(h9,h9+entry9,1);context.write_text(s,encoding='utf-8')
 handoff=ROOT/'NEW_CHAT_HANDOFF.md';shutil.copyfile(handoff,H/'before/NEW_CHAT_HANDOFF.md')
 handoff.write_text(f'''# 최신 인수인계 — {now}
## 현재 결과
공개 자료에서 편손 굽힘 오판과 반복 끝 연결 실패를 재현하고 보완, 검증 후8821에 적용. 손모양은 애매한 편손을 정상 확정한 것이 아니라 잘못된 굽힘 경고→회색 확인부족. 반복은 공개13_F 원본/640의1→2회 개선. 실제 사용자 새촬영 검증은 아직 없음.
## 보완과 검증
FIT3·7 다시점4영상의24손시점+기존공개정답으로 작은 손모양 Conv+GRU3seed만 추가학습. 공통100백본/TCN/사과/촬영코드 불변. 237손모양기준 추가오판·정답소실0(기존오판1남음),54움직임조건+40기타조건 통과, 합성25굽힘 모두 경고유지. 사과7공개영상 카드·판정·대응·TCN동일. 새4카메라뷰는 기존정면으로 보던 DEV인물14·15의 미사용시점이며 완전새인물시험 아님; 추가악화없지만 일부unknown 유지.
공개캡처 전후 시험: 사과0.313→0.321초, 감사0.342→0.372초, 두단어 자동종료·완료유지. 실제웹캠/브라우저시간 아님.
## 파일/웹
{H}
RESULT.json/RESULT_KO.txt/SOURCE_REVIEW.json/validation_v2/RESULT.json/web/ADMISSION.json/web/CAPTURE_CHECK.json/web/HTTP_CHECK.json
합성통합검사 결과는 import경로 때문에 새 {V/'distinction_combined_v2/RESULT.json'} 에 생성됨. 이전 동결파일 덮어쓰기 없음.
서버 {WEB/'server.py'} 에 새 어댑터 import2줄. 새프로세스 PID{result['web_pid']}, http://127.0.0.1:8821/ ready, 실제카메라안열음. 사용자 새로고침 필요. 3D 시범/원본 비교 유지.
## 한계·보존
단어이름 예외없는 공유 관측부/연결로직이나 이번 적용은 검증된 감사 조립범위. 모든단어/시점/3D 접촉/새사용자 정확도 보장하지 않음. 아직 방향/관측unknown·기존오판 남음.
사용자3촬영은 이전 진단만, 이번 학습·튜닝·후보선택·재평가0. 새다운로드/외부전송/Git stage commit push/실제카메라0. Codex 자체 검증이며 독립 전문가 검수 아님. 실험프로세스 전부종료, 웹서버만유지.
## 다음
{next_action}
''',encoding='utf-8')
 for p in[O/'STATE.json',H/'RESULT.json',H/'web/ADMISSION.json']:read(p)
 assert all(sha(p)==h for p,h in adm['pins'].items());print(json.dumps(dict(status=result['status'],pid=result['web_pid'],report=str(H/'RESULT_KO.txt'),protected=protected),ensure_ascii=False))
if __name__=='__main__':main()
