from condition_model import *
ROOT=E.parents[3]
def main():
    at=clock();progress=read(E/'TRAIN_PROGRESS.json')
    summary='공개FIT36영상의 추가 시간순 검토로 준비·복귀174시점/대기42시점을 추가했고96애매경계는 보류했다. 기존850희소정답은 그대로 보존했다. 조건부 관측정보 유무×준비·복귀 감독 유무의 동일구조4조건을seed42/80epoch로 고정하고 학습을 시작했다. 主guide1_context1_42이며 기존 D6학습·528비교는 재실행하지 않는다. 아직 새 평가결과/웹적용 없음. 전체goal active.'
    nxt='E word100_guide_conditioned_episode_20261102의 실제 학습session55688/PID20724와 TRAIN_PROGRESS를 확인하여 중복 실행하지 않는다. 완료 후 evaluate_condition.py로88조건×4모델352비교를 한 번 수행하고 verify_condition.py로 별도산술/관측보존/순서/짧은역순/이동삭제/영상누락을 검산한다. 통과시만 안정성·전체처리시간·격리데모로 진행. 실패시 고정결과 원인분리 보고를 마감하고 의미없는 문턱완화/주요프레임삭제를 하지 않는다. 사용자·멘토자료/H14·15/기존C·T3F·웹 보존.'
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','설명조건·준비복귀4조건 비교 시작',summary),('## 9. 다음 단일 작업','설명조건4학습과 고정 비교 이어받기',nxt)]:
        header=f'### 2026-11-02 {title}'
        if header not in text:text=text.replace(marker,marker+'\n\n'+header+'\n\n'+body+'\n',1)
    path.write_text(text,encoding='utf8')
    state=read(E.parent/'STATE.json');state['current_stage']='GUIDE_CONTEXT_FACTORIAL_TRAINING';state['next_single_action']=nxt;state['updated_date_kst']='2026-11-02';state[E.name]=dict(at=at,status='TRAINING',active=progress,plan=str(E/'FROZEN.json'),review=str(E/'TARGETS_VERIFIED.json'),whole_goal_complete=False)
    write(E.parent/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(E/'FROZEN.json')+'\n'+str(E/'TARGETS_VERIFIED.json')+'\n\n같은Codex검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    write(E/'CURRENT_EXECUTION.json',dict(at=at,status='TRAINING',session=55688,pid=20724,progress=progress,next=nxt,whole_goal_complete=False))
if __name__=='__main__':main()
