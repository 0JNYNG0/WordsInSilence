from model import *
ROOT=F.parents[3]
def main():
    at=clock();progress=read(F/'TRAIN_PROGRESS.json');summary='F分리구조의 입력계약·사전검증을 완료했다. 직접관측한 동일손자세가 다른행동과정에 속하는공존사례를 확인했고, D행동정답과E준비복귀정답을 서로대체하지않는 행동내용/과정역할 두출력으로 나눴다. 主frozen_encoder42는 학습된embed/RGB/GRU/delta특징을 고정하고 행동해석readout/별도역할은 학습한다. joint_encoder42와 같은입력216·80epoch로2조건 학습진행중. 기존관측/원래시간보존. 새결과미확인/웹미변경/전체goal active.'
    nxt='F word100_separate_action_context_20261102의 train.py exec58891/PID41180과TRAIN_PROGRESS실제실행을 확인하고 중복실행하지않는다. 완료후 evaluate_split.py로88개×2조건176새비교1회,verify_split.py로별도검산한다. 정확한행동이 역할판정때문에빠진것까지오류로계산한다. 통과한경우만안정성·목표불일치·전체처리시간·격리데모로이어간다. 실패시고정결과를보존하고다시점수/문턱조정하지않는다. E4학습352비교와D완료결과재실행금지. H14/15·사용자/멘토자료/C·T3F·기존웹보존.'
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','행동내용·과정역할 분리학습 시작',summary),('## 9. 다음 단일 작업','분리구조2학습과176비교 이어받기',nxt)]:
        h=f'### 2026-11-02 {title}'
        if h not in text:text=text.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    path.write_text(text,encoding='utf8');state=read(O/'STATE.json');state['current_stage']='SEPARATE_ACTION_CONTEXT_TRAINING';state['next_single_action']=nxt;state[F.name]=dict(at=at,status='TRAINING',active=progress,plan=str(F/'FROZEN.json'),summary=summary,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(F/'FROZEN.json')+'\n'+str(F/'INPUT_CONTRACT.json')+'\n\n같은Codex검토/검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    write(F/'CURRENT_EXECUTION.json',dict(at=at,status='TRAINING',session=58891,pid=41180,progress=progress,next=nxt,whole_goal_complete=False))
if __name__=='__main__':main()
