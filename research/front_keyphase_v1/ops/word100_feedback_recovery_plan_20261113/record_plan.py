"""Record a reviewed plan awaiting user confirmation. No experiments."""
from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone,timedelta

P=Path(__file__).resolve().parent;O=P.parent;ROOT=P.parents[3]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')

def main():
    previous=O/'word100_feature_expansion_20261113'
    sources=[previous/'COVERAGE.json',previous/'EVALUATION_COMPLETE.json',previous/'RESULT_VERIFIED.json',
        previous/'feedback_contract/INTEGRATION_DECISION.json',O/'word100_web_completion_graph_20261111/TASK_GRAPH.json',
        O/'word100_web_completion_graph_20261111/GATES.json',O/'avatar_multiview_review_20261113/VERIFIED.json']
    c9=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization/c7_local_cause_recovery_20261113/c9_sequence_context'
    sources += [c9/'COMPARISON.json',c9/'FIT_AUDIT.json']
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    plan=dict(status='REVIEWED_WAITING_USER_CONFIRMATION',created=now,report=str(P/'PLAN_KO.txt'),
        objective='Existing four-word web feedback completed with preserved common100; transferable attributes; game merge later.',
        original_graph=str(O/'word100_web_completion_graph_20261111/TASK_GRAPH.json'),
        nodes=[
            dict(id='P1',original='G2',title='원인 분리',depends=[],exit='Source/observation/model/link errors and overlaps documented with matched successful controls.'),
            dict(id='P2',original='G1/G2',title='부족한 정답·관측 보강',depends=['P1'],exit='Reviewed native truth, unknown masks, provenance, source/encoder/head exposure audit; only supported contrasts admitted.'),
            dict(id='P3',original='G2',title='원인에 맞는 제한 학습',depends=['P2'],exit='Paired frozen protocol, all seeds, hands/states/normal-speed/crossing/direction and coverage preserved; no gain by abstention only.'),
            dict(id='P4',original='G2/G3',title='설명·방향·순서·누락 통합',depends=['P3'],exit='Correct native evidence for each required action; no expected-order forcing; simultaneous errors included.'),
            dict(id='P5',original='G4',title='미사용 공개자료 확인',depends=['P4'],exit='Eligible exposure-audited data only; one frozen evaluation; otherwise independent scope remains incomplete.'),
            dict(id='P6',original='G5/G6',title='웹 연결·촬영 흐름',depends=['P5'],exit='Text/time/highlight agree; public replay then user-initiated camera acceptance, true end-to-end wait measured.')],
        loops={'observation':'P2','truth_conflict':'P2','generalization':'P1/P2/P3_new_contract','misbinding':'P1/P4','no_new_evidence':'report_before_more_candidates'},
        model_budget=dict(finger_candidates=2,finger_seeds=3,finger_max_runs=6,movement_candidates_max=2,movement_seeds=3,movement_record_folds=6,movement_max_runs=36,
            note='Upper limits, not mandatory run targets. Cause audit precedes any new run. No completed runs repeated.'),
        avatar=dict(role='parallel_secondary_non_blocking_for_web',next='Recover same-record raw3D/cameras locally before reprojection/retarget pilot',download_authorized=False),
        no_execution_this_turn=True,user_private_training=False,existing_C_T3F_preserved=True,source_pins={str(p):sha(p) for p in sources})
    write(P/'PLAN.json',plan)
    write(P/'REVIEW_MOTION.json',dict(status='REVIEWED_CHANGES_INCORPORATED',reviewer='Codex c7_graph_recovery_1113, separate from root',
        requested_fixes=['C8 residual11 not C9 entire error population','same observation and predictions for link-only causal comparison','new36 maximum after cause review; no candidate proliferation','boundary tolerance does not replace legacyGATES','Codex RGB review is not independent humanexpert truth','one bounded comparison then evidence-based next loop'],
        applied=True,no_training_or_inference=True))
    title='2026-11-13 다음 작업계획 검토 완료 — 사용자 확인 대기'
    summary='기존 작업 그래프 G2 보완→G3 통합→G4 미사용 공개평가→G5/G6 웹연결·촬영검증을 유지한6단계 계획. 원본/관절/판정/설명연결의 복합원인 분리, 부족한 굽힘·양손·움직임 정답 보강, 제한된 동결기반vs미세조정 비교. 원인·정답→고정비교→보존검사1회 단위이며 새근거없는후보반복금지. 기반모델이본FIT기록을전체미노출수행자로칭하지않음. 애매경계는기존정상보존기준을완화하지않음. 아바타동일클립원본복구는별도가지. 이번은문서/검토만, 새학습·추출·추론·웹변경없음.'
    nextaction='사용자가 이번 PLAN_KO.txt의6단계 계획을 확인할 때까지 실행 대기. 승인 후 첫작업은 공개FIT의 실패와 성공 대조를 원본·관측·판정·구간연결로 나눈 원인표 작성. 기존완료9/36 및C6~C8재실행금지. C·T3-F/공통100/운영웹보존, 사용자·멘토자료학습·튜닝·선택금지, 봉인자료임의개봉금지.'
    before=P/'before';before.mkdir(exist_ok=True)
    files=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    for p in files:
        if not (before/p.name).exists():shutil.copy2(p,before/p.name)
    cp=files[0];ct=cp.read_text('utf-8-sig');assert title not in ct
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nextaction)]:
        assert marker in ct;ct=ct.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+P.name+'/PLAN_KO.txt`, `PLAN.json`.\n',1)
    cp.write_text(ct,'utf-8')
    s=read(O/'STATE.json');s['active_task']=P.name;s['next_single_action']=nextaction
    s[P.name]=dict(status='WAIT_USER_CONFIRMATION',plan=str(P/'PLAN.json'),report=str(P/'PLAN_KO.txt'),execution_started=False,training_running=False,previous_completed_unit=previous.name)
    s['word100_web_completion_graph_20261111']['next_plan']=str(P/'PLAN.json')
    write(O/'STATE.json',s)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {title}

최신 사용자 요청: 다음 작업의 타당하고 자세한 계획을 보고받고 확인한 뒤 체계적으로 진행. 현재 새 실행 승인 대기다. 이번에는 저장 결과 검토·에이전트 계획 검토·문서 정리만 했다.
계획 {P/'PLAN_KO.txt'} / {P/'PLAN.json'}

{summary}

직전 완료: {previous/'RESULT_KO.txt'}
공통100 유지, 추가7표현74영상으로 실제9개 미세조정 비교 완료. 손가락861학습항목 중굽힘26, 추가정답오른손편중. 일반DEV389에서전체합계일부개선이나굽힘·오른손·일부기존항목회귀로미채택. 설명문장계약만준비/웹미연결.
기존그래프C9 고정36비교완료/G2REWORK. 학습FIT동작/유지오답0이나제외기록일반화실패. 완료결과재학습·재추론금지.
아바타공식5뷰/3D/카메라근거확인,남은3D기하급변. 과거3뷰삼각측량의도지만저장계산은2.5D근사. 원본keypointZIP옛경로없어동일클립자료복구/재투영이먼저. 새다운로드미승인.

{nextaction}
학습프로세스없음. 기존연구서버를이번턴에시작/중지하지않음. 카메라는사용자시작전접근하지않음. 동일Codex별도검산과독립사람/KSL전문가검수구분.
''','utf-8')
    with files[3].open('a',encoding='utf-8') as f:f.write('\n\n### '+title+'\n'+summary+'\n'+nextaction+'\n근거: '+str(P/'PLAN_KO.txt')+'\n')
    write(P/'DOCUMENT_SYNC.json',dict(status='PLAN_ONLY_SYNCHRONIZED',files={str(p):sha(p) for p in files},training_or_inference=False))
    print('PLAN_ONLY_SYNCHRONIZED_WAIT_USER_CONFIRMATION')

if __name__=='__main__':main()
