from prepare import U,O,read,sha,exclusive
from sync_status import sync
from datetime import datetime,timezone

verified=read(U/'SAVED_VERIFIED.json')
assert verified['checks']==280 and verified['result_sha256']==sha(U/'RESULT.json')
assert read(U/'RUN_FINISHED.json')['result_sha256']==sha(U/'RESULT.json')
decision=read(U/'NEXT_LEARNING_DECISION.json')
for p,h in decision['pins'].items():assert sha(p)==h
prototype=O/'word100_local_event_learning_20261103'
checks=read(prototype/'STRUCTURAL_CHECKS_v2.json')
assert checks['checks']==18 and checks['code_sha256']==sha(prototype/'local_events.py')
assert checks['real_video_inferences']==checks['optimizer_steps']==0
exclusive(U/'FINAL.json',dict(at=datetime.now(timezone.utc).isoformat(),
    status='FIXED_SCOPE_DIAGNOSTIC_COMPLETE_TASK_MISMATCH_CONFIRMED_NOT_ADOPTED',
    new_scopes=6,selected_fit_record_ids=[1,4,7],new_training=0,existing_weights_modified=False,
    broad_motion_confirmation=0,broad_motion_queries=3,late_motion_false_confirmation=0,late_queries=3,
    metrics_are_not_accuracy=True,code_structural_checks=61,separate_saved_checks=280,
    new_head_prototype=str(prototype),new_head_structural_checks=18,new_head_trained=False,
    live_web_changed=False,whole_goal_complete=False,
    pins={str(p):sha(p) for p in [U/'RESULT.json',U/'SAVED_VERIFIED.json',U/'RESULT_KO.txt',
        U/'INTERPRETATION.json',U/'NEXT_LEARNING_DECISION.json',prototype/'local_events.py',
        prototype/'STRUCTURAL_CHECKS_v2.json',prototype/'PLAN_RESOLUTION.json']}))
summary='''AF 직접 맥락 검토 완료: 공개4표현×ID01/04/07의12영상,129질문(고유86시점)에서 해당52/다름12/불확실65,별도3004검산. 불확실은검출실패가아니며검사할움직임으로비교범위를고르면누락이범위밖으로밀리는문제가있었다. 실제12영상PTS/저장원시각일치,옛2시트평균FPS자막만최대0.011921초다름. 이어AG 공통episode-query 구현·61구조검사·공개FIT3의broad3+late3단회추론·280별도저장검산완료. 기존좁은쓸기구간p약1이지만새넓은구간3개모두미확인,실제원RGB14frame의짧은유지구간3개는움직임으로표시안함(07연결불확실). 이는정확도0%나원모델열화가아니다:원학습target은주어진구간의순이동방향이고필요한과제는전체흐름안의동작존재/위치라서서로다르다. scope확장만의후보미채택,문턱조정/재학습/웹교체없음. 6구간head7.9395ms는캐시계산만이며전체대기시간아님. ID07windowmetadata누락첫오류보존후검증된원생성계보로분리;ID01/04window값정확재구성. RGB지원범위/원시간/누락별도보존. AH 시간별발생판정부local_events.py 구현:기존915입력·projection전이함수,5물리방향출력,전구간laststate없는finite15row TCN,39013param. 초기14검사보존후관절/RGB둘다없는fragment분리등18합성검사PASS. 이새판정부는아직가중치로드/실영상추론/학습0이며성능개선주장없음. NEXT_LEARNING_DECISION과AH PLAN_RESOLUTION에지도학습·비교순서고정. C/T3F/기존웹/원본/사용자·멘토자료보존. 동일Codex별도검산이며독립수어전문가정답아님. 전체학습형피드백goal active·미완료.'''
next_action='''AH PLAN_RESOLUTION.json/AG NEXT_LEARNING_DECISION.json과local_events.py의현재18합성검사를먼저읽고직접학습준비로이어간다. 기존45FIT ordered positive를dense1없는positivebag으로,AG late3을정확한국소정의의negative로사용. 나머지FIT최대33clip와DEV최대12clip의기존stable위치를라벨없는후보로한번고정하고원시연속RGB로실제발생/비발생/null만준비;oldstable0·partial0·다른단어를자동음성화금지/원하는0나올때까지scope변경금지. registeredID/sourceSHA분리·source-query의존그룹·known/unknown·관측범위검증후지원target을고정(양·음각최소3FIT ID),부족한head는mask하고전체가완벽할때까지멈추지않는다. AH915입력/projectiontransfer/RF15와계획예시887/RF7차이는PLAN_RESOLUTION근거사용. negativebag의누락eligibility와RGB바깥문맥/nominal차이를코드검토한뒤seed42/final80epoch1회(정확steps·loss·.5/.8/.2·DEV분모학습전동결)학습,원시간후보/정상hold오표시/짧은변화/관측부족/공통head비교. 모델은한fullsequence forward이며동일score열의고정bag집계만;반복subwindow재추론탐색금지. 조건통과시같은승인범위에서구현·검증계속,사용자재승인때문에훈련보류금지. AG6완료추론/AF검토/AE16/ABAC완료학습평가재실행·재채점금지. 새후보는미학습이며웹미적용;현재실행중모델프로세스없음. C/T3F/기존데모·원본보존,개인/멘토/잠금평가·외부전송/다운로드/운영교체금지.'''
state=dict(stage='LOCAL_EVENT_TIME_READOUT_IMPLEMENTED_UNTRAINED_DATA_GATE_NEXT',summary=summary,next_action=next_action)
(U/'STATUS_SUMMARY.json').write_text(__import__('json').dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
synced=sync(summary,next_action,state['stage'])
(U/'DOC_SYNC.json').write_text(__import__('json').dumps(synced,ensure_ascii=False,indent=2),encoding='utf-8')
print('AG finalized; AH implemented but untrained; authoritative documents synchronized; full goal active')
