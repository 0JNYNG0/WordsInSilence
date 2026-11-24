from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=LAB.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
v=read(U/'VERIFICATION.json');assert v['status']=='PASS' and v['checked_training_clips']==899
assert read(U/'RESULT.json')['summary']['classes']==100
assert not (U/'COMPLETION.json').exists()
before=U/'before';before.mkdir(exist_ok=True)
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json']
for p in paths:shutil.copy2(p,before/p.name)
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
status='WORD100_CROP_STAGE1_COMPLETE_HAND_ASSIGNMENT_FIX_NEXT'
unit='ops/word100_crop_readiness_20261016'
next_action='사용자와1단계결과검토후 연구용 좌우손배정·미검출처리 후보를 구현하고 공개 교차손/한쪽pose손목 미검출 사례로 검증한다. 아직 재추출전체/학습/운영교체하지 않는다.'
summary='기능별7단계TODO 작성,1단계완료. 공개학습899개/100단어/9명 저장관절 전수 경계검산,기존실제crop12쌍 재검토. 고정box899/899,가로경계이탈0. 가운데구간의기존손점전체밖54영상/14단어 중48영상은1프레임; 시각검토3개는수행손이안에보이나기존검출이잠깐끊긴경우. 핵심동작손실54건으로해석금지. 수영crop60좌우역전,수어crop33한쪽pose손목미유효시반대손강제배정 확인. 잘하다56은원본배정오류. 상반신방향유지,좌우배정안정화를학습전우선한다. 코드검산899개/2753파일해시보존PASS,동일Codex별도알고리즘및시각검토,새에이전트/전문가검증아님. 최초경로가정오류는기록후재사용records로해결. 새추출/추론/학습/운영/DTW/카메라변경0.'
context=paths[0].read_text('utf-8-sig')
heading='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert context.count(heading)==1
context=context.replace(heading,heading+'\n### 2026-10-16 웹캠 보완1단계 완료 — 좌우배정 보완 우선, 사용자 검토 대기\n'+summary+'\n근거 '+unit+'/COMPLETION.json, RESULT_KO.txt, Docs/WEBCAM_IMPROVEMENT_TODO.md.\n',1)
heading9='## 9. 다음 단일 작업\n';assert context.count(heading9)==1
context=context.replace(heading9,heading9+'\n### 2026-10-16 다음 — 1단계 결과 상호 검토 후 입력 안정화\n'+next_action+'\n사용자 최신 요청은 기능별 단위 완료보고→서로검토→다음단계 진행. 단순계획에머물지않고1단계를실행완료했다. 사용자자료학습/튜닝/후보선택금지와기존모델보존유지.\n',1)
paths[0].write_text(context,'utf-8')
state=read(paths[2]);state['updated_date_kst']=now;state['current_stage']=status;state['next_single_action']=next_action
state['word100_crop_readiness_20261016']={'status':'STAGE1_COMPLETE_AWAITING_USER_REVIEW','unit':unit,'report':unit+'/RESULT_KO.txt','completion':unit+'/COMPLETION.json','todo':'Docs/WEBCAM_IMPROVEMENT_TODO.md','training_clips_checked':899,'classes':100,'stored_actual_crop_pairs':12,'guarded_files':2753,'geometry_flags':{'clips':54,'classes':14,'single_middle_frame_clips':48,'meaning':'Not54essential-motion-loss clips;3visualsamples show visible active hand with transient original detector absence'},'confirmed_hand_assignment_issues':['swimming_crop_frame60','sign_language_crop_frame33_forced_opposite_side','well_done_native_frame56'],'next':next_action,'new_training':0,'new_extraction':0,'new_inference':0,'production_changed':False,'running_processes':[]}
write(paths[2],state)
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}. 현재 목표는 웹캠의 상반신/화면밖 손 조건에 맞춘 인식 보완이다.

## 사용자 작업 방식
기능별TODO를 하나씩 실행→정확한 결과와 문제 즉시보고→사용자와검토→다음단계. 2026-10-16 첫단계 직접승인으로실행완료. 지금은1단계결과검토대기이며다음학습자동시작아님.

## 마지막 완료
{summary}
보고: `{unit}/RESULT_KO.txt`, 완료: `{unit}/COMPLETION.json`.
목록: `Docs/WEBCAM_IMPROVEMENT_TODO.md` (1완료,2좌우배정/미검출안정화,3조건고정/추출,4TCN,5DTW,6웹재시험,7확대/게임).

## 다음 한 작업
{next_action}
배정수정과crop효과를한꺼번에인식개선으로합치지않는다. 운영추출기를직접덮어쓰지말고연구후보로분리한다. 단순X정렬·현재유효pose손목하나로강제배정·장시간임의보간금지. 필수손에는정적인받침손포함. 100단어의언어학적필수손정답라벨은미확보이며동작량만으로생성하지않음.

## 보존 기준
TCN `runs/word100_tcn_v1/whole_context/best.pt`, SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 100단어 학습01–09의899영상/개발11–13의300영상,96.67%는개발검증. 보류10/14–18미사용.
사용자12회시험Top1 6/12(50%),Top5 8/12(66.67%)는개발참여자1인실사용평가로보존. 사용자촬영은학습/튜닝/후보선택/임계값조정금지.
DTW기준·서버·모델변경없음. 원본ZIP/관절/기존동결산출물보존. 새작업프로세스모두종료. 카메라/장치/다운로드/외부전송/Git변경없음.

## 발표/문서
8장발표및대본포함8페이지PDF는완료상태유지. 이번연구단위에서발표내용갱신없음. 한컴독스서식2줄바꿈수정과중간보고문안작성은별도완료; 연구모델과무관.
'''
paths[1].write_text(handoff,'utf-8')
completion={'status':'COMPLETE_AWAITING_USER_REVIEW','finished_at_kst':now,'unit':unit,'author_executor_verifier':'Codex; separate local verifier, no new independent agent','verification':'899 training observations recomputed;2753 input/code/model files unchanged;7public source clips visually inspected (4assignment/detection+3boundary)','training':0,'inference':0,'mediapipe_extraction':0,'production_changes':False,'user_arrays_used':False,'scope':'Stage1 diagnosis only; not crop model performance or all-class semantic approval','next':next_action,'running_processes':[],'artifacts':{p.name:sha(p) for p in U.iterdir() if p.is_file()}}
write(U/'COMPLETION.json',completion)
assert read(paths[2])['word100_crop_readiness_20261016']['training_clips_checked']==899
assert all((U/f).exists() for f in ['RESULT.json','VERIFICATION.json','RESULT_KO.txt','DECISION.json'])
assert '2026-10-16 웹캠 보완1단계 완료' in paths[0].read_text('utf-8')
print('Stage1 completion and 4 authoritative todo/state/handoff documents synchronized.')
