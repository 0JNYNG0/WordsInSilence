"""Synchronize verified milestone and next task without changing experimental results."""
import study as S
from pathlib import Path
import json,sys,traceback

def main():
    verified=S.read(S.U/'VERIFIED.json'); responses=S.read(S.U/'RESPONSES_VERIFIED.json'); render=S.read(S.U/'HTML_RENDER_VERIFIED.json')
    assert verified['tests']=={'train_only_bounds':899,'verified_pairs':2697}
    assert responses['counts']['responses']==900 and render['status']=='PASS_HEADLESS_CHROME'
    f=S.read(S.U/'FROZEN.json');S.check(f['protected']);S.check(S.read(S.U/'MANIFEST.json')['pins']);S.check(S.read(S.U/'REPORT_MANIFEST_FINAL.json')['files'])
    assert not(S.U/'RUNNING.json').exists() and not(S.U/'PIPELINE_RUNNING.json').exists()
    now=S.now();unit=S.U.name;root=S.O.parents[2];context=root/'WordsInSilence_PROJECT_CONTEXT.md';handoff=root/'NEW_CHAT_HANDOFF.md';state=S.O/'STATE.json'
    summary='사용자의 재개 승인에 따라 공개DEV300→TRAIN899 개별참고2697쌍 검증 완료. TRAIN ID1~9만 참고범위 계산,DEV ID11~13 제외. 최소5명·최대잔차 규칙 유지. 원본관측/좌우/시간/기준분모/C·T3F·E3/운영웹 보존. 순간각도34.38%,연속각도19.15%,순간위치77.33%,연속위치66.42%가 현재 근거기준 충족(인식률·교정정확도 아님). 899개범위·2697개경로/차이/상태 별도산술검산,단일각도100/손전체200고정경로통제 및10단어×7종70실제재정렬 통제,900개모델별재생응답 검산 완료. 단일각도누락은이웃유지,손전체누락은그손잘맞음금지/반대손팔보존. slow.75 새연속위치범위밖0/7008·근거손실0;fast1.25는367/7008(5.24%)새범위밖·근거손실2칸 남음. 왼손누락/짧은누락 무신호2영상은원래손관측0이라 성공/실패로세지않음. 자연속도/의미상짧은필수동작/정밀손가락교정은미검증. 연구재생과HTML표시완료,실제웹캠미적용·새학습/검출/TCN추론/개인자료접근0. Chrome headless desktop/mobile 및표선택검증PASS. 최초sandbox쓰기오류와읽기검토module이름충돌보존·안전복구. 이번단계완료이며전체피드백완성아님.'
    nextstep='다음은word100_train_reference_dev_replay_20261026의저장DONE/CONTROLS/VERIFIED/RESPONSES_VERIFIED와기존S/T시간진단을재사용하여fast1.25표본손실과다른사람간정렬변화를분리한다. 완료2697/899/370/900재실행금지. 목표는정상속도에새경고를만들지않으면서짧은누락을놓치지않는위치·팔비교기준. 먼저저장70통제의기존경로와알려진원본시점대응을진단용으로만대조하고,새입력후보에는정답시점/단어별예외/DEV맞춤임계값을넣지않는다. 원래손관측0사례를새검출실패로세지않음. 손가락은34.38/19.15%지원에그치므로기준완화나추정확정금지. 근거부족·시간정렬불확실·실제관측차이를구분하고기준범위밖을사용자오류로지시하지않는다. 기존C·T3F·E3/운영/개인·멘토/추가199·200자료보존. 별도단어확장·실제웹운영교체는이번단계에포함되지않음. 현재활성실행없음.'
    backup=S.U/'before';backup.mkdir(exist_ok=True)
    for path in (context,handoff,state):
        out=backup/path.name
        if not out.exists():out.write_bytes(path.read_bytes())
    title='### 2026-10-26 공개 DEV300→TRAIN899 다중 참고 피드백 검증 완료'
    old=context.read_text('utf-8-sig');m8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';m9='## 9. 다음 단일 작업';assert title not in old
    evidence='근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html,VERIFIED.json,CONTROLS_DONE.json,CONTROL_EDGE_REVIEW.json,RESPONSES_VERIFIED.json,HTML_RENDER_VERIFIED.json.'
    old=old.replace(m8,m8+'\n\n'+title+'\n'+now+'\n'+summary+'\n'+evidence+'\n',1).replace(m9,m9+'\n\n'+title+'\n'+now+'\n'+nextstep+'\n'+evidence+'\n',1);context.write_text(old,encoding='utf8')
    st=S.read(state);st['current_stage']='DEV300_TRAIN899_REFERENCE_REPLAY_VERIFIED_RESEARCH_ONLY';st['next_single_action']=nextstep;st[unit]=dict(status='COMPLETE_VERIFIED_RESEARCH_REPLAY_NOT_LIVE_CORRECTION',at_kst=now,unit='ops/'+unit,report='ops/'+unit+'/RESULT_KO.html',summary=summary,next=nextstep,query_count=300,reference_count=899,pair_count=2697,responses_verified=900,fixed_mask_controls=300,realignment_controls=70,training=0,classifier_inference=0,detector_inference=0,personal_data_access=0,operating_changed=False,active_sessions=[],implementation_and_verification='Codex implementation plus separate saved arithmetic and browser checks; not independent human groundtruth')
    state.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음 단일 작업: '+nextstep+'\n\n'+evidence+'\n\n이번단계exec25471(freeze/calibrate/run/verify)·52070(controls)·62346(build)·56734(responseverify) 모두EXIT0. report 및headlessChrome 검사EXIT0. 원본공개범위외자료접근없음. 사용자촬영은평가용이며학습/튜닝/선택금지. 이전파일의goalactive표시는중단전이력;이번재개는사용자의명시승인으로진행했으며완료는이번단계만뜻한다.\n',encoding='utf8')
    S.write(S.U/'CURRENT_EXECUTION.json',dict(at=now,status='EXIT0_VERIFIED_REPORTED',active_sessions=[],pipeline_session=25471,controls_session=52070,build_session=62346,response_verifier_session=56734,first_sandbox_freeze_session=35520,first_error_preserved=True))
    S.write(S.U/'NEXT_PLAN.json',dict(at=now,next_single_action=nextstep,milestone_complete=True,overall_feedback_goal_complete=False))
    S.write(S.U/'SYNC_CHECK.json',dict(at=now,status='PASS',pins={str(p):S.sha(p) for p in (context,handoff,state)},state_key=unit,preserved_protected_count=len(f['protected']),active_sessions=[]))
    assert S.read(state)[unit]['pair_count']==2697 and title in context.read_text('utf8') and nextstep in handoff.read_text('utf8')
    print('SYNC_COMPLETE_CURRENT_MILESTONE_ONLY',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=S.U/'FIRST_ERROR_finalize.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
