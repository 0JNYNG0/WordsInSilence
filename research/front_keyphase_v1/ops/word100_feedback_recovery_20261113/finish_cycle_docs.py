from common import *

def main():
    av=O/'avatar_motion_pilot_20261113'; ex=av/'proportion_examples'
    for p in [ex/'RESULT_KO.txt',ex/'SOURCE_STYLIZED_MANNEQUIN_KO.png',ex/'MANNEQUIN_PUBLIC_PREVIEW/APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4',U/'RESULT_KO.txt',U/'NEXT_PLAN_KO.txt',U/'shape_recovery/MOTION_SAVED_RESULT_VERIFIED.json']:
        assert p.exists() and p.stat().st_size>0,p
    title='2026-11-13 손모양·물리이동 고정비교 검산 완료 및 공개5뷰 아바타·사람비율 예시'
    status='손가락 직접정답200개 보강 후 기반고정/미세조정 각3seed 총6학습 완료. 일반DEV389항목 이전354/359/356 대비 기반고정343/347/347, 미세조정338/337/334로 미채택. FIT1061항목 자체는1009~1044정답으로 학습사례와다른기록일반화를분리했다. 물리이동과설명소속을구분한 새24영상115창230손(M142/H86/U2) 정답,353가중치 공통판정부18학습 완료. 원본정순 규칙221/228→225/226/225이나 고유2손항목의정역/짧은이동회귀로미채택. 실제crop106보류중88은준비/복귀이며전부주요동작실패가아니다. 손모양별도286889·움직임10906검산PASS(모델채택PASS아님). 설명앵커59개, 출처/손/시간/설명ID/엄격bool계약19검사완료. 자동구간통합/새평가/웹승격은미완료. 공개조상1클립5뷰127프레임30fps→추정3D→기존캐릭터FBX/영상 제작,몸통축오류수정. 사용자사람비율요청에 보유모델검토와로컬생성마네킹같은동작예시추가. 정확3D/얼굴접촉/엄지전체/교육정답은미검증. C·T3-F/운영TCN·웹/공통100 보존,개인자료학습튜닝없음.'
    next_action='현재단위 고정6+18학습/평가/검산은종료했으므로재실행하지않는다. NEXT_PLAN_KO.txt에따라공개FIT의같은손모양-다른표현/기록 관측대조와 작은실제이동-정상흔들림대조를먼저감사한다. 물리이동유무의정역불변성과방향/순서보존을구분하고새원인근거·조건동결전추가후보금지. G2보완/G3실제통합후에만G4최종평가·G5/G6웹. 아바타는사람비율/손가락가독성선호확인됐고예시완료,얼굴·가슴부위관계와엄지/접촉/충돌검증을다음공통리타게팅과제로남긴다. 기존공개5뷰추정카메라의한계유지. 사용자자료/원본/봉인평가보호.'
    g=read(U/'GRAPH.json');g['status']='FIXED_CYCLE_VERIFIED_NOT_ADOPTED_AVATAR_TECHNICAL_EXAMPLES_COMPLETE';g['nodes']['AV_avatar']='PUBLIC_5VIEW_ONE_CLIP_AND_PROPORTION_EXAMPLES_COMPLETE_NOT_TEACHING_READY';g['whole_project_complete']=False;write(U/'GRAPH.json',g)
    write(U/'CURRENT_EXECUTION.json',dict(status='FIXED_CYCLE_COMPLETE_NOT_ADOPTED',training_running=False,root_pid=None,root_session=None,shape_sessions_completed=[13217,52084,73587],motion_session_completed=64405,no_completed_run_restart=True,avatar_examples=str(ex),next_plan=str(U/'NEXT_PLAN_KO.txt')))
    s=read(O/'STATE.json');s['next_single_action']=next_action;s[U.name].update(status='FIXED_CYCLE_VERIFIED_NOT_ADOPTED_AVATAR_EXAMPLES_COMPLETE',training_running=False,root_exec_session=None,report=str(U/'RESULT_KO.txt'),next_plan=str(U/'NEXT_PLAN_KO.txt'),motion_verified=str(U/'shape_recovery/MOTION_SAVED_RESULT_VERIFIED.json'),avatar_examples=str(ex),whole_project_complete=False);write(O/'STATE.json',s)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';v=p.read_text('utf-8')
    for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status),('## 9. 다음 단일 작업',next_action)]:
        start=v.index(head)+len(head);mark='### '+title
        if mark not in v[start:start+len(body)+800]:v=v[:start]+'\n\n'+mark+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_feedback_recovery_20261113/RESULT_KO.txt`, `GRAPH.json`, `NEXT_PLAN_KO.txt`; `research/front_keyphase_v1/ops/avatar_motion_pilot_20261113/proportion_examples/RESULT_KO.txt`.\n'+v[start:]
    p.write_text(v,'utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — '+title+'\n\n'+status+'\n\n다음: '+next_action+'\n\n현재단위: '+str(U)+'\n현재root/학습실행없음. 별도세션완료기록: shape13217/52084/73587, motion64405 exit0. 저장모델/평가재실행금지.\n새아바타질문답변: 사람캐릭터가아니어도사람과비슷한체형과손/팔가독성을원함. 이선호확인됐으므로이전질문대기상태를재적용하지않음. 예시폴더: '+str(ex)+'\nroot가문서/학습통합, motion= c7_graph_recovery_1113, 별도검산=expansion_labels_review_1113, avatar=avatar_multiview_review_1113. 같은Codex검산을독립사람/수어전문가정답으로부르지않음.\nC·T3-F와기존웹원본보존. private/봉인자료/다운로드/외부전송/Git/운영교체금지. 실제카메라는사용자가시작. 아바타예시는C:에있어HDD없이볼수있지만전체프로젝트HDD분리보장을뜻하지않음.\n','utf-8')
    p=ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt';v=p.read_text('utf-8')
    if title not in v:v+='\n\n### '+title+'\n'+status+'\n판단변화: 데이터량만늘리는대신 물리이동과설명행동소속의의미를분리했고, 학습자료를맞히는것과다른기록을맞히는것을분리했다. 운영성능향상으로포장하지않고고정비교미채택과원인을보존했다. 아바타는원본카메라ZIP미확보상태에서사용자승인5뷰추정시험으로전환,구현오류수정과캐릭터체형에따른의미위치차이를구분했다.\n'+next_action+'\n근거: '+str(U/'RESULT_KO.txt')+'\n'
    p.write_text(v,'utf-8')
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
        assert title in p.read_text('utf-8')
    assert read(O/'STATE.json')[U.name]['training_running'] is False
    write(U/'CLOSEOUT_VERIFIED.json',dict(status='REPORT_AND_CHECKPOINT_SYNC_VERIFIED',whole_project_complete=False,
       report=str(U/'RESULT_KO.txt'),graph=str(U/'GRAPH.json'),next_plan=str(U/'NEXT_PLAN_KO.txt'),
       pins={str(p):sha(p)for p in [U/'RESULT_KO.txt',U/'NEXT_PLAN_KO.txt',U/'GRAPH.json',U/'PROTECTED_VERIFIED.json',ex/'RESULT_KO.txt',ex/'SOURCE_STYLIZED_MANNEQUIN_KO.png']},no_model_execution=True))
    print('Evidence report, graph, context, STATE, handoff and research history synchronized.')

if __name__=='__main__':main()
