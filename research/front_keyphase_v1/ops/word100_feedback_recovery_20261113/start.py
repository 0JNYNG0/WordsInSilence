from common import *
import shutil

def main():
    assert not (U/'AUTHORIZATION.json').exists()
    auth=dict(status='USER_AUTHORIZED_6STEP_EXECUTION_AND_AVATAR_PILOT',user_quote='작업 모두 진행해주고, 3D 움직임으로 캐릭터에 적용해서 사용자에게 동작을 보여주는 것도 계획해서 진행해줘.',
        clarification='중간 중간 불확실한 내용이나, 확인이 필요한 것들에 대해서는 나한테 질문을 해. 근거 없는 상상으로 내용을 임의로 채우지 말고 타당한 근거가 뒷받침되도록 진행해줘.',
        approved_plan=str(P/'PLAN_KO.txt'),plan_sha256=sha(P/'PLAN_KO.txt'),
        protected=read(PREV/'FROZEN.json')['protected'],
        scope=['Public FIT cause/truth/observation audit','Paired fine-head training max6','Motion recovery max2conditions36foldruns after cause and pretrain review','Native guide/motion integration','Eligible unused public evaluation after exposure audit and preceding gate','Passed research web integration/public replay; actual camera only user initiated','One public same-record calibrated3D/avatar retarget/baked preview pilot'],
        prohibited=['private user/mentor training tuning selection','sealed10/14-18 or final8 premature access','original source mutations','unreviewed production model replacement','Git','external upload','new downloads without explicit acquisition authorization','invented source labels/calibration'],
        actors=dict(root='Coordinator, paired training, integration and independent checks',shape='expansion_labels_review_1113',motion='c7_graph_recovery_1113',avatar='avatar_multiview_review_1113'),
        question_policy='Ask when evidence cannot resolve required user-specific facts or meaningful scope decisions; continue independent work while answer pending.')
    for p,h in auth['protected'].items():assert sha(p)==h,p
    write(U/'AUTHORIZATION.json',auth)
    write(U/'GRAPH.json',dict(status='ACTIVE',original_graph=read(P/'PLAN.json')['original_graph'],
        nodes=dict(P1_cause='RUNNING',P2_truth='RUNNING',P3_train='WAIT_CAUSE_AND_LABEL_GATE',P4_link='READ_ONLY_AUDIT',P5_fresh='WAIT_PRIOR_GATES',P6_web='WAIT_PRIOR_GATES',AV_avatar='SOURCE_RECOVERY_AND_RIG_INSPECTION'),
        no_production_change=True,running_owners=auth['actors']))
    write(U/'CURRENT_EXECUTION.json',dict(status='SOURCE_CAUSE_AUDIT',training_running=False,root_pid=None,subagents=auth['actors']))
    before=U/'before';before.mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        shutil.copy2(p,before/p.name)
    title='2026-11-13 복합원인 보완6단계·3D 아바타 시험 실행 승인'
    status='사용자가 검토한6단계 전체와 실제3D캐릭터 적용 시험을 승인했다. root는학습/통합,별도에이전트는공개FIT손가락정답·움직임일반화·아바타원본복구를병렬진행. 기존승인대기해제. 불확실한원본/의미/촬영보정은추측하지않고질문. C·T3-F/운영TCN·웹/공통100보존,개인자료금지. 새성과는아직없고원인감사부터진행.'
    nxt='현재단위 word100_feedback_recovery_20261113의 AUTHORIZATION/GRAPH를따른다. 공개FIT실패-성공대조와부족B/양손직접정답→원인근거기반동결비교. motion_recovery와shape_recovery 작업자중복금지. 아바타는동일클립3D/cam/5뷰복구→재투영→리그변환·bake;자료없으면질문하고가짜보정금지. 완료과거9/36재실행금지. 앞단계검증전웹승격/봉인최종평가금지.'
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status),('## 9. 다음 단일 작업',nxt)]:
        s=s.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/AUTHORIZATION.json`, `GRAPH.json`.\n',1)
    p.write_text(s,'utf-8')
    state=read(O/'STATE.json');state['active_task']=U.name;state['next_single_action']=nxt;state[P.name]['status']='USER_APPROVED_EXECUTION_IN_SUCCESSOR'
    state[U.name]=dict(status='RUNNING_CAUSE_TRUTH_AVATAR',authorization=str(U/'AUTHORIZATION.json'),graph=str(U/'GRAPH.json'),training_running=False,private_used=False,production_changed=False)
    write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {title}

{status}
현재단위 {U}
{nxt}

손모양 agent: {U/'shape_recovery'}, 기존861정답중B26/오른손편중의FIT원본검토. 새학습전정답검증.
움직임 agent: {U/'motion_recovery'}, C8잔여11과C9새손실·성공대조, 원인분리 후 최대2조건 고정비교. root사전검토후학습.
아바타 agent: {O/'avatar_motion_pilot_20261113'}, 동일공개클립원본3D/카메라복구와리그실행준비. 원본없으면사용자에필요파일질문,추정카메라로성과조작금지.
root: 손가락동결기반vs미세조정조건 준비, 설명구간시간계약·전체통합검증. 루트문서편집은root만.
이전완료 {PREV/'RESULT_KO.txt'} — 9학습과C9의36완료미채택,새사용자성능아님. 원본/실패/동결읽기전용.
승인범위내루틴작업은계속하되사용자확인필요사항질문. private/sealed·다운로드·외부전송·Git 금지. 카메라사용자시작전접근금지. 기존공통100모델에보인FIT기록을새독립사람으로칭하지않음.
''','utf-8')
    event('authorized_started')
if __name__=='__main__':main()
