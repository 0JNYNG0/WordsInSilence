from probe_front import *
from datetime import datetime
from zoneinfo import ZoneInfo
import statistics,struct

def main():
 root=O.parents[2];at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();http=read(N/'web/HTTP_CHECK.json');assert http['passed']
 checks={k:read(N/'web'/(k+'.json'))for k in ['CONSUMER_CHECK','CARD_CHECK','CAPTURE_CHECK']};assert all(q['passed']for q in checks.values())
 admission=read(N/'web/ADMISSION.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
 old=read(M/'web/ADMISSION.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old['pins'].items())
 assets=[Path(p)for p in admission['pins']]
 for base in [G/'apple_v1',V]:
  assets +=[Path(read(base/'web/REFERENCE.json')['local_video']),base/'web/reference_features.npz',Path(read(base/'web/SOURCES.json')['normal']['video'])]
 web=O/'web_tutorial_live_integration_20261113/site';preview=O/'web_tutorial_stage_preview_20261113/site'
 glbs=[web/'sign-demos/WORD0099.glb',web/'sign-demos/WORD1290.glb',preview/'assets/robot.glb',preview/'assets/npc.glb'];assets+=glbs+[preview/'assets/NeoDunggeunmoPro-Regular.ttf']
 assets+=list((O/'word100_rtmw_mediapipe_compare_20261108/models').glob('rtmw_l_384/end2end.onnx'))+list((O/'word100_rtmw_mediapipe_compare_20261108/models').glob('person_yolox_m/**/end2end.onnx'))
 for p in assets:assert p.is_file()and p.resolve().drive.upper()=='C:',p
 embedded=[]
 for p in glbs:
  with p.open('rb')as f:
   magic,version,length=struct.unpack('<III',f.read(12));n,kind=struct.unpack('<II',f.read(8));q=json.loads(f.read(n))
  uris=[a['uri']for key in ['buffers','images']for a in q.get(key,[])if 'uri'in a];assert all(u.startswith('data:')for u in uris)
  embedded.append(dict(path=str(p),external_uri_count=0))
 write(N/'web/LOCAL_ASSET_CHECK.json',dict(passed=True,files=len(assets),all_on_C=True,glb=embedded,physical_hdd_disconnect_test=False,scope='Current feedback pins, pose models, references and demo GLB/font local. Not an exhaustive native filesystem interception or physical disconnected-device test.'))
 graph=read(N/'GRAPH.json')
 for node in graph['nodes']:
  if node['id']in['F1','F2','F3','F4']:node['status']='complete_public_verified'
  elif node['id']=='F5':node['status']='local_assets_verified_physical_user_test_pending'
  else:node['status']='pending_after_user_demo'
 graph['updated_at']=at;graph['scope_note']='Front-camera feedback only; avatar multi-view unchanged. F4 is public/contract integration, not physical webcam acceptance.';write(N/'GRAPH.json',graph)
 summary=dict(status='FRONTAL_SCOPE_RESTORED_WEB_APPLIED_USER_DEMO_PENDING',at=at,implementation='Codex local',verification='Same Codex separate source/hash, source-regression, runtime/capture/HTTP checks; no independent human expert review',normal_public_clips=32,normal_preserved=32,thanks_normal=17,apple_normal=15,shape_anchors=237,shape_new_wrong=0,shape_lost_correct=0,synthetic_distinctions_passed=40,synthetic_diagnostic_only=5,temporal_controls_passed=48,web_card_and_quest_checks=7,training_runs=3,fit_people=9,fit_posture_labels=114,common100_retrained=False,old_pins_preserved=190,active_pins=len(admission['pins']),physical_camera=False,user_training=False,user_tuning=False,user_candidate_selection=False,http=http,graph=str(N/'GRAPH.json'),local_assets=str(N/'web/LOCAL_ASSET_CHECK.json'),initial_failures=['Direct rollback lost 2 normal front handshape decisions; reused existing front-only palm head.','Web distinct-pattern test lost inspection when handshape changed; trained front-only body posture proposal.','Uncertain missing-finger card had no interval; preserved actual missing evidence interval.'],remaining=['Physical seated normal/error/corrected retry for apple/thanks, automatic stop and perceived latency.','Physical HDD-unplugged startup/capture/replay; declared dependencies verified local.','Freeze demonstrated version and record actual demo images/video with bounded claims.'],future=['S-CURT: more words/people and source-labelled short shape changes; expert verification.','Unity integration after stable web demo; five-view avatar animation independent.'],limits=['Public sources reused for development, not unseen user accuracy.','Synthetic keypoint edits are not naturally recorded learner errors.','7 prior additional orientation warnings on shape/translation edits not retained as unverified orientation truth; explicit shape/plane test targets retained.','2D proximity and image-plane pose do not certify hidden fingers, exact 3D contact or full KSL correctness.'])
 timing=[]
 for word in ['WORD0099','WORD1290']:
  rr=[r for r in checks['CONSUMER_CHECK']['rows']if r['word']==word];capture=next(r for r in checks['CAPTURE_CHECK']['rows']if r['word']==word)
  timing.append(dict(word=word,before_median=statistics.median(r['before_seconds']for r in rr),after_median=statistics.median(r['after_seconds']for r in rr),public_capture_stop_to_result=capture['stop_to_result_seconds'],auto_stop=capture['auto_end'],redetection=capture['redetection']))
 summary['timing']=timing;write(N/'RESULT.json',summary);write(N/'PROGRESS.json',dict(status=summary['status'],at=at,live_web=str(N/'serve.py'),pid=http['server']['pid'],training_runs=3,next='F5: user-seated normal/error/corrected retry and physical HDD-disconnected demo check'))
 write(N/'SUPERSEDED_SCOPE.json',dict(at=at,decision='Public side-camera feedback experimentation is excluded from current demo and frontal capability claims. Preserve history; do not resume side-view loops.',excluded_runtime=['H projection_head mixed source','D relation side continuation','E geometry/palm relation mixed source','J side-palm/plane','K side-arrival','L temporal visibility','M cycle composition with mixed parents'],retained_shared_code=['H linked_arrivals independent movement linking','P apple hand-role observability','display DTW and evidence cards','original100 context and exact finger observers'],avatar='Five-view motion reconstruction remains authorized and unchanged.',new_front_head='114 original frontal FIT posture labels, 9 people, 3 fixed seed models; random initialization, no mixed parent weights'))
 report='''정면 피드백 복구 결과

잘못 넓어진 방향
정면 웹캠을 평가하는 프로젝트에서 옆카메라 영상의 문제까지 피드백 학습/평가 대상으로 확장했던 것은 원래 시연 범위를 벗어났다. 5방향 자료의 용도는 3D 캐릭터 동작 제작이다. 과거 측면 개선 수치를 정면/새 사용자 정확도로 사용하지 않는다. 과거 원본·가중치·실험은 이력으로 보존하고 현재 실행 경로에서 분리했다.

이번 변경
1. FEEDBACK_DATA_POLICY.json과 공개 입력 검사로 정면(F)만 피드백 학습·평가에 받는다. 개인 촬영과 아바타 자료를 학습 입력에서 거부한다.
2. 기존 정면용 몸/손 관측과 손바닥 형태 모델을 조합했다. 단순 롤백에서 다시 불확실했던 정면 2장면도 유지했다. 손 전체가 편 모양이라는 판단과 각 손가락이 완전히 펴졌다는 판단은 다르게 취급한다.
3. 손모양을 바꾸면 행동 검사 구간이 뒤로 밀리는 문제를 공개 시험에서 재현했다. 정면 9명의 몸·팔 위치 정답 114개로 작은 판정부를 3개 고정 시드로 학습했다. 이 판정부는 손모양을 검사할 구간만 제안하며, 반복 완료를 만들어내거나 손이 가려진 동안 모양을 정답으로 채우지 않는다. 단어·사람별 예외와 임계값 변경은 없다.
4. 정면용 소비부를 직접 시작하게 하여 측면용 모델을 로드했다가 교체하던 경로도 없앴다. 기존 100단어 본체와 C·T3-F는 보존했다. 손·팔 강조, 짧은 교정, 회색 확인 부족, DTW 비교 재생과 사과→감사 퀘스트를 유지했다.

검증 결과
공개 정면 사과15/감사17영상에서 기존 정상 판정·TCN 결과 유지. 손모양 정답237개에서 새 오판/정답 손실0. 공개 관절을 편집한 오류 시험40개, 부분·속도·누락·크기 시험48개 통과. 원영상 역재생5건은 실제 오류 정답이 아니므로 성공률 분모에 넣지 않았다. 손모양/손면을 바꾼 시험의 명시적 차이는 유지했다. 과거 부수적으로 나온 손면 경고7개는 해당 편집의 검증된 정답이 아니어서 보존 대상으로 강제하지 않았다.
현재 입력은 개발 중 이미 사용한 자료다. 위 숫자는 새 사용자 정확도나 전체 수어 정확도가 아니다. 자연 촬영된 다양한 실제 실수 전부를 검증한 것도 아니다.

속도와 연결
공개 영상 카메라 모의 입력에서 두 단어 모두 3초 카운트다운, 손 내림 자동 종료, 전체 프레임 관절 재사용, TCN 1위·필수 관측 통과. 자동 종료 뒤 결과 계산은 사과0.23초/감사0.24초였다. 자동 종료 판정 대기, 실제 웹캠, 브라우저 그리기까지의 체감 시간은 별도다. 소비부 감사 중앙값은 이전0.421초에서 현재0.206초; 같은 입력을 사용했으나 여러 번 반복한 통계적 속도시험은 아니다.
웹8821 정면용 서버 적용, 공개 HTTP 재생 두 단어 통과·퀘스트 자동 진도 없음 확인. 실제 카메라는 켜지 않았다. 시작 파일 START_WEB_DEMO.cmd 갱신.

현재 작업 그래프
F1 자료/모델 분리 [완료] → F2 정면 정상/오류 검증 [완료] → F3 필요한 보완/학습 [완료] → F4 공개 촬영/교정/퀘스트 연결 [완료] → F5 실제 앉은 촬영/HDD 분리 확인 [남음] → F6 시연본 고정/발표 근거 [남음]
문제가 나오면 무조건 새로운 모델로 바꾸지 않는다. 관절 관측·모델 분류·검사 구간·화면 표시를 나누고, 공개 정면 자료로 재현한 원인만 수정하여 관련 정상/오류 검증을 다시 한다.

다음 마감 순서
1. 최신8821에서 사과·감사를 평소 앉은 자세로 정상 수행, 명확한 오류, 교정 후 재시도로 확인한다. 촬영 시작은 사용자만 한다. 개인 촬영은 진단 전용으로 학습/튜닝/후보 선택에 쓰지 않는다.
2. 필요한 모델/참고영상/캐릭터/폰트가 C:에 있는 것은 검사했다. HDD를 실제 분리한 실행·촬영은 아직 검증하지 않았다. 이 조건에서 시작·3D 시범·촬영·결과를 확인한다.
3. 확인된 버전을 고정하고 정상 통과와 오류→교정의 실제 시연 화면을 발표 자료로 남긴다. 아직 확인하지 않은 기능은 완성이라고 표현하지 않는다.
이후 S-CURT에서 단어·사람·손가락 조합 확대와 전문가 검증을 진행한다. Unity 병합은 웹 시연을 고정한 다음 단계다.

담당/제한
작성·학습·실행·별도 해시/회귀/소비부검증: Codex. 독립 사람/수어 전문가 검증은 아니다. 기존190개 동결파일 해시 유지. 새 외부 다운로드·업로드·Git 커밋 없음. 사용자 자료는 이번 학습/튜닝/선택에 사용하지 않았다.
'''
 (N/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 next_action='F5: 정면용8821에서 사용자가 사과·감사 정상→명확한 오류→교정 재시도를 시작하면 진단한다. HDD 분리 실물 시연을 확인하고 F6버전/발표 근거를 고정한다. 개인 학습/튜닝/선택과 옆카메라 피드백 확대 금지.'
 statepath=O/'STATE.json';state=read(statepath);state.update(current_stage=summary['status'],updated_at=at,next_single_action=next_action)
 task=state['capture_feedback_completion_20261116'];task.update(status=summary['status'],updated_at=at,current_checkpoint=str(N/'RESULT.json'),report=str(N/'RESULT_KO.txt'),current_graph=str(N/'GRAPH.json'),training_runs=75,web_admitted=True,next_single_action=next_action)
 task['frontal_scope_recovery_20261117']=dict(summary);task['active_feedback_runtime']=str(N/'serve.py');task['side_feedback_candidates_current_status']='Archived research; superseded for current frontal demo. Older web_applied flags are historical.';write(statepath,state)
 context=root/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig')
 old8='### 2026-11-17 정면 피드백/아바타 자료 용도 정정'
 # Insert newest verified state above the earlier audit, preserving the record.
 h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 new8=f'''\n\n### 2026-11-17 정면 기준 복구 및 웹 적용 완료
정면 피드백은 정면(F) 공개자료만, 5방향은 아바타용으로 복구했다. 최근 측면 모델은 현재 실행에서 제외하고 과거 실험으로 보존한다. 정면9명114자세정답으로 검사구간용 작은 모델3개를 새로 학습; 100단어 본체/C·T3-F 동결. 사과15+감사17정면 정상/TCN 유지,237손모양 추가오판/손실0,40오류/48시간·부분조건 및7문장/퀘스트 검사 통과. 공개 캡처 자동종료/재검출0,종료후0.23/0.24초(실제카메라·브라우저시간아님). 같은개발자료의 회귀검증이지 새사용자정확도가 아니다. 기존190파일해시 유지. 전체과거학습72+이번3=75(옆시점이력포함),현재본체재학습없음.
현재8821 {N/'serve.py'}, PID{http['server']['pid']}, 일반환경·카메라꺼짐. START_WEB_DEMO.cmd→frontal_demo_launcher.py. F1~F4공개검증완료;F5앉은실제촬영/물리HDD분리→F6실제시연·발표고정 남음. 필요한모델·영상·GLB·폰트 C:점검완료,실제HDD분리미실시. 사용자촬영학습/튜닝/후보선택없음. 상세 {N/'RESULT_KO.txt'};그래프 {N/'GRAPH.json'};정책 FEEDBACK_DATA_POLICY.json. 아래 과거 옆시점 후속계획은 폐기된 방향의 이력이며 재개 지시가 아니다.
'''
 text=text.replace(h8,h8+new8,1).replace(h9,h9+'\n\n### 2026-11-17 정면용 시연 마감\n'+next_action+'\n',1);context.write_text(text,encoding='utf-8')
 (root/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
목표: 정면 웹캠 사과·감사 촬영→자동종료→빠른 교정/확인부족 구분→재시도/실전퀘스트. 5방향은3D아바타 전용. 옆카메라 피드백 후속작업으로 다시 넓히지 않는다.
최신결과 {N/'RESULT_KO.txt'} / RESULT.json. F1자료계보분리,F2정면회귀,F3필요학습,F4공개촬영/웹연결 완료. 다음F5 실제사용자/물리HDD분리,F6시연고정.
정면9명114자세정답만으로 작은 검사구간판정부3seed학습;본체/C-T3F그대로. 정면32정상/TCN유지,237손모양새오판/손실0,40오류/48시간·부분시험,7문장/퀘스트검사통과. 초기 단순롤백 손모양2건/오류구간누락/불확실위치누락은보완후통과. 7부수손면경고는편집에대한정답이없으므로억지보존안함. 알려진개발자료/합성오류시험이지새사용자정확도아님.
웹 {N/'serve.py'}, PID{http['server']['pid']},포트8821일반환경,카메라꺼짐. START_WEB_DEMO.cmd→frontal_demo_launcher.py --check-only검증. 공개캡처사과/감사3초카운트다운·자동종료·재검출0·종료후0.23/0.24초. 실제웹캠/화면체감은미검증. 실행파일·참고·GLB C:점검;HDD실제분리미검증. 진행중학습/평가세션없음(서버만실행).
옆시점혼합가중치로드금지worker가드,정면자료policy,63핀검사. 이전190핀보존. 예전M등web_applied상태는당시이력이며현재활성이아니다. 과거학습72+이번3=75. 사용자학습/튜닝/선택없음. 독립전문가/사용자실물검증없음. 작성/실행/검증Codex.
다음단일작업: {next_action}
''',encoding='utf-8')
 print('FINALIZED',summary['status'],'PID',http['server']['pid'],'LOCAL_ASSETS',len(assets))
if __name__=='__main__':main()
