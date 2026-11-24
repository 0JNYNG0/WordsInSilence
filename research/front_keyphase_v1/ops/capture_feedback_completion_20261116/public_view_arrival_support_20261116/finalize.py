from pathlib import Path
import json,hashlib,datetime,ast
K=Path(__file__).resolve().parent;J=K.parent/'public_view_observation_repair_20261116';G=K.parent;O=G.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 now=datetime.datetime.now().astimezone().isoformat();comp=read(K/'preserved_checks/RESULT.json');reg=read(K/'regression/RESULT.json');fresh=read(K/'confirmation/RESULT.json');capture=read(K/'web/CAPTURE_CHECK.json');consumer=read(K/'web/CONSUMER_CHECK.json');http=read(K/'web/HTTP_CHECK.json');parent=read(J/'web/ADMISSION.json');pins=read(K/'web/ADMISSION.json')['pins']
 assert comp['new_warning']==comp['lost']==comp['control_failed']==0 and fresh['passed']and read(K/'preserved_checks/BOUNDARY_PRESERVATION.json')['passed']
 assert reg['shape_new_wrong']==reg['shape_lost_correct']==reg['distinction_failed']==reg['temporal_failed']==0 and not reg['known_error_loss']
 assert read(K/'regression/confirmation/PATTERN_CHECK.json')['passed']and capture['passed']and consumer['passed']and http['arrival_support_active']and not http['camera_active']
 assert read(K/'web/CARD_CHECK.json')['passed']and read(K/'web/PARTIAL_CARD_CHECK.json')['passed']
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())and all(pins[p]==h for p,h in parent['pins'].items())
 for p in list(K.glob('*.py'))+list((K/'runtime').glob('*.py')):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 stats=dict(known_public_conditions=len(comp['rows']),preserved_normal_and_intervals=True,additional_observed_cycles=sum(r['after']['events']-r['before']['events']for r in comp['rows']),additional_full_completions=sum(all(v=='supported'for v in r['after']['criteria'].values())and not all(v=='supported'for v in r['before']['criteria'].values())for r in comp['rows']),new_normal_warning=0,lost_supported=0,shape_anchors=237,distinction_controls=40,pattern_controls=6,temporal_controls=48,side_partial_hold_controls=8,new_public_camera_views=2,consumer_pairs=len(consumer['rows']))
 timing=[{k:r[k]for k in['phase','word','auto_end','frames','observed','redetection','stop_to_result_seconds']}for r in capture['rows']]
 def time_pair(word):
  a,b=[r for r in timing if r['word']==word];return f"{a['stop_to_result_seconds']:.3f} → {b['stop_to_result_seconds']:.3f}초"
 result=dict(status='COMPLETE_VIEW_ARRIVAL_SUPPORT_WEB_APPLIED',at=now,stats=stats,previous_repair=str(J/'RESULT.json'),training=dict(actual_runs=6,admitted_runs=3,admitted_fit_rows=162,admitted_fit_people=9,admitted_fit_clips=15,features='Existing51 body/palm features plus8 actual observed symmetric hand-edge/palm region distance features, normalized by body and projected palm diameter.',initialization='51-feature head weights copied,8 added columns initializedzero; three fixed-seed supervised fine-tunes, backbone frozen.',not_admitted='New2/5 FIT side views reviewed independently and28 extra state anchors added (190rows/19clips),3seed candidate failed to improve and shifted an existing interval into a normal-posture warning. Not deployed.'),correction='Preserve every deployed event and its exact bounds; candidate may only append non-overlapping independently observed approaches ending in source-trained near state. Original movement, handshape, posture and target checks still required.',benefit='11_L: zero observed cycles toone; shape/plane/target for that observed cycle supported, repetition remains unknown and overallpassfalse.',remaining='First cycle in11_L and other side-view finger/endpoint/target uncertainties remain. New6L/R both findone rather thantwo; do not call complete. No false completion relaxation.',checks=dict(comparison='preserved_checks/RESULT.json',boundaries='preserved_checks/BOUNDARY_PRESERVATION.json',regression='regression/RESULT.json',unselected_views='confirmation/RESULT.json',partial_card='web/PARTIAL_CARD_CHECK.json',capture='web/CAPTURE_CHECK.json',http='web/HTTP_CHECK.json'),web=dict(url='http://127.0.0.1:8821/',pid=http['server']['pid'],entrypoint=str(K/'serve.py'),launcher='START_WEB_DEMO.cmd -> arrival_demo_launcher.py',camera_active=False),capture_timing=timing,pins=len(pins),preserved_parent_pins=len(parent['pins']),private_data_read_or_trained=False,verification='Codex source/native image, independent serialized output/hash and executable runtime checks; no separate reviewer or KSL expert.',not_claimed=['All side-view issues solved','Actual physical webcam or browser paint timing','User-independent KSL accuracy','Physical3Dcontact or hiddenfinger truth'])
 write(K/'RESULT.json',result);write(K/'PRESERVATION_VERIFIED.json',dict(passed=True,pins=len(pins),parent_pins=len(parent['pins']),all_hashes_match=True,private_data_used=False))
 graph=[dict(id='A',work='옆시점 원본·관측·판정 원인 분리',status='complete'),dict(id='B',work='손 자세 분류 학습 및 오답 경고 보완',status='complete',evidence=str(J/'RESULT.json')),dict(id='C',work='상대 접근 특징 학습·기존 구간 보존·부분 수행 구분',status='complete'),dict(id='D',work='정상/오류/미사용 공개시점·웹 자동종료/속도 확인',status='complete'),dict(id='REMAIN',work='가린 손가락과 일부 도착/반복 근거 부족',status='unresolved_no_false_pass'),dict(id='USER',work='실제 웹캠 정상→오류→교정 재시도 확인',status='pending_user_capture_not_training')]
 write(K/'GRAPH.json',dict(at=now,nodes=graph,scope='이번 원인별 보완 루프. 전체 수어 모델 완성률 아님.'))
 report=f'''이번 작업 전체 결과 (손 자세 보완 + 추가 도착 근거 보완)

옆시점 문제는 공개 자료를 옆 카메라에서 봤을 때, 정상 손모양이 겹쳐 보이거나 손이 가까워진 끝 자세를 놓치는 문제입니다.

개선한 것
1. 공개 정면/좌우 자세 정답102개로 작은 손 자세 판정부를 미세조정했습니다. 확인한46조건에서 정상 자세를 잘못 지적한 경고5개가0개로 줄었습니다. 모든 요구 항목 확인은31→34조건, 정면17개 정상 판정은 유지했습니다.
2. 손 사이 접근 정보를 몸 크기와 보이는 손바닥 크기에 맞춰 넣고, 공개162정답으로 작은 도착 판정부를 미세조정했습니다. 한 옆시점에서 동작을 전혀 못 찾던 상태에서 한 번을 확인하게 됐습니다. 두 번 완료로 속이지 않고 반복은 여전히 확인부족으로 표시합니다.
3. 새 부품이 기존 구간까지 바꾸는 문제도 발견해 고쳤습니다. 기존에 검증된 시작·끝은 그대로 두고 근거가 확인된 새 동작만 추가합니다.48개 기존 공개조건 모두 기존 구간 보존을 확인했습니다.
4. 학습용 두 사람의 추가 좌우 원본4개를 검토하고28정답을 더한190정답 후보도 시험했습니다. 더 나아지지 않아 이 후보는 배포하지 않았습니다. 사람/단어에 맞춘 예외나 임계값 완화는 넣지 않았습니다.

검증
- 기존 손모양237정답: 추가오판/정답소실0. 기존오판을 전부 해결했다는 뜻은 아닙니다.
- 기존 오류구분40조건과 다른손가락모양6합성조건 유지. 기존에 잡던 오류를 확인부족으로 숨기지 않았습니다.
- 속도·크기·부분수행·관절누락48조건과 옆시점부분/정지8조건 통과.
- 후보에 쓰지 않았던 공개6번 수행자 좌우2영상에서도 기존 판정을 유지. 두 번 중 한 번만 확인되므로 완료로 처리하지 않습니다. 새 사람 일반화 성능으로 해석할 수 없습니다.
- 웹 소비부12쌍에서 기존 TCN 결과/DTW 연결 유지. 손모양 오류는 해당 손 강조와 짧은 문장, 불확실은 별도 회색 안내. 추가로 찾은11_L한번동작은 반복확인부족/전체통과false를 확인.
- 공개카메라 재생시험에서 종료→결과계산 사과 {time_pair('WORD0099')}, 감사 {time_pair('WORD1290')}. 기존 자동종료·3초시작·정상통과 유지,123/125수신프레임 전부관측,재검출0. 실제사용자웹캠/브라우저화면 표시시간은 아닙니다.

남은 한계
일부 옆시점의 손가락 가림과 첫 도착 순간은 아직 확정하지 못합니다. 정상동작을 틀렸다고 단정하거나 확인하지 못한 반복을 완료로 만들지 않았습니다. 현재 시연의 정면 사용을 보존하면서 각도 변화에 대한 판정을 개선한 결과이지, 모든 각도/단어가 해결됐다는 뜻은 아닙니다.
웹: http://127.0.0.1:8821/ 새로고침. START_WEB_DEMO.cmd도 최신 연결. 카메라는 꺼져 있습니다.
C·T3-F/100단어 TCN/공통Conv+GRU본체/기존 손가락·손모양 모델 유지. 사용자촬영 읽기·학습·튜닝·후보선택0. 부모153핀/최신163핀검증. 실제 새 촬영 정상→오류→교정 재시도는 사용자 시작이 필요합니다.
'''
 (K/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 next_action='최신8821 사과·감사의 실제 새촬영 정상→오류→교정 재시도를 사용자가 시작하면 진단한다(학습·튜닝·후보선택 금지). 공개 측면의 남은 가림/도착 문제는 학습된 관측만으로 확정 못한 상태이며 완료로 강제하지 않는다. 추가연구는 여러FIT사람의 실제 도착·떼기 전후와 가림 신뢰도 정답을 보강한 후 기존구간보존/정상/오류/부분/속도 검증을 적용한다. 현재102자세/162도착 보완의 검증을 불필요하게 반복하지 않는다. 실제브라우저 대기·HDD분리시연은 미검증.'
 state=read(O/'STATE.json');parentstate=state['capture_feedback_completion_20261116'];already=parentstate.get('public_view_arrival_support_20261116',{}).get('status')==result['status'];state.update(current_stage='PUBLIC_VIEW_ARRIVAL_SUPPORT_WEB_APPLIED',last_updated=now,next_single_action=next_action);parentstate.update(status=state['current_stage'],updated_at=now,report=str(K/'RESULT_KO.txt'),training_runs=parentstate['training_runs']+(0 if already else 6),training_runs_current_task=6,training_runs_current_task_admitted=3,current_work_graph=str(K/'GRAPH.json'),current_checkpoint=str(K/'PROGRESS.json'),next_single_action=next_action,web_admitted=True,private_data_used=False)
 parentstate['public_view_arrival_support_20261116']=dict(status=result['status'],at=now,path=str(K),report=str(K/'RESULT_KO.txt'),result=str(K/'RESULT.json'),graph=str(K/'GRAPH.json'),stats=stats,pid=http['server']['pid'],session_id=None,private_data=False,next_single_action=next_action);write(O/'STATE.json',state)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig');heading='### 2026-11-16 옆시점 도착 근거 추가 / 기존 동작 경계 보존 적용';assert heading not in text
 update=f'\n\n{heading}\n앞선손자세보완(46조건정상경고5→0/전체확인31→34)은유지. 기존51몸·손바닥특징에8개관측접근거리특징추가,기존head복사후162공개FIT정답(9명15영상)3seed미세조정.11_L동작0→1,그구간5항목확인하되반복unknown/전체미통과유지.2/5좌우4영상28추가정답(190행)후보3seed는추가효과없어미적용.새후보가기존경계를당겨정상경고생기는문제도발견해기존48조건모든구간보존후추가동작만연결하도록보완.237손모양/40구분/6패턴/56보호조건유지.미사용6좌우는여전히1회만확인(완료아님).웹12쌍TCN/DTW유지,1회부분카드/오류·불확실구분확인.공개캡처사과{time_pair("WORD0099")},감사{time_pair("WORD1290")}·자동종료유지,실제웹캠표시시간아님.8821 PID{http["server"]["pid"]} ready/cameraoff/arrival_support_active,arrival_demo_launcher.부모153/총163핀불변,C·T3-F/100본체/개인자료0.근거 {K.relative_to(R).as_posix()}/RESULT.json·RESULT_KO.txt·GRAPH.json.남은관측/끝판정불확실은성공으로강제하지않음.\n'
 marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert marker in text;text=text.replace(marker,marker+update,1);marker='## 9. 다음 단일 작업';assert marker in text;text=text.replace(marker,marker+'\n\n### 2026-11-16 옆시점 자세·도착 보완 후 실제 사용 확인\n'+next_action+'\n',1);p.write_text(text,encoding='utf-8')
 handoff=f'''# 최신 인수인계 — {now}

## 완료한 작업
옆시점 원인분리→자세학습→도착근거보완→회귀수정→공개 정상/오류/미사용 시점→웹 통합까지 완료.
최신근거: {K} (RESULT.json / RESULT_KO.txt / GRAPH.json / preserved_checks / regression / confirmation / web).
직전 손자세 보완: {J}/RESULT.json. 이전 결과 동결 유지.

## 실제 적용
PreservedArrivalLesson: 배포J의모든기존이벤트와경계를보존. 관측51몸/손바닥특징+8개양방향손날-손바닥영역거리(몸/관측손바닥지름정규화),162공개FIT정답(9명15영상),기존head51가중치복사+새8열0초기화,3seed미세조정으로추가near후보. 기존접근시계열·관측유효성과함께새비중복도착만연결.3D접촉이나가린손가락진실아님. 정면정상/TCN/C·T3-F/공통Conv+GRU본체/기존손모양/손가락 유지.
초기단순head교체는기존구간을앞당김. 추가FIT2/5좌우4영상검토28정답을더한190행3seed후보는17L기존정상자세경고+추가효과없어미적용. 최종162행head + 기존구간보존으로수정.새조정은신경망학습과공통구간연결코드양쪽이며단어/사람ID예외없음.
앞선J는102한손자세정답/3seed 적용(정상경고5→0,전체확인31→34/46).J의palm190/관계162추가후보는효과없어미적용.현재K훈련6run중3적용,누적STATE66run(J포함총이번15run중6적용).

## 검증·남은 한계
48개알려진공개조건정상/기존구간유지.11L0→1동작과그구간모양·방향·목표5항목회복,2회완료로처리하지않음.237손모양추가오판/정답소실0(기존오판해결주장아님);40오류/6합성패턴/48속도·크기·부분·누락/8측면부분·정지유지,기존오류지적손실0.역재생5진단만.
후보선택전미사용6L/R공개영상확인(정면knownFIT,새사람일반화검증아님).두영상모두기존처럼1회만확인/반복unknown;가림/끝상태/목표위치한계여전.정면17정상보존.옆시점전체해결아님.
웹12쌍TCN/DTW유지.1.83~2.30초다른손모양강조/교정과확인부족회색구분,11L1회부분수행/전체통과false카드검증.공개카메라종료후사과{time_pair('WORD0099')},감사{time_pair('WORD1290')},123/125수신프레임관측/재검출0/3초시작·자동종료·정상통과유지.실제웹캠/브라우저표시시간아님.
Codex구현+원본/직렬화결과/해시/실행검증,별도에이전트/수어전문가없음.개인촬영읽기/학습/튜닝/선택0.새다운로드/외부전송/커밋0.

## 실행/재개
http://127.0.0.1:8821/ PID{http['server']['pid']},ready/cameraoff,일반(require_escalated)실행.
entrypoint: {K/'serve.py'}
START_WEB_DEMO.cmd -> arrival_demo_launcher.py. 이전view_demo_launcher/motion_demo_launcher/demo_launcher는163핀안에서동결. 부모153핀모두보존.학습/검증프로세스종료,서버만유지.교체내역web/ACTIVATION.json.

## 다음 한 작업
{next_action}
'''
 (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8');write(K/'PROGRESS.json',dict(status=result['status'],at=now,running_sessions=[],server_pid=http['server']['pid'],next_single_action=next_action));print('FINALIZED',stats,'pid',http['server']['pid'],flush=True)
if __name__=='__main__':main()
