from prepare_coherent import *
ROOT=K.parents[3]
def main():
    ex=read(K/'CURRENT_EXECUTION.json');at=clock()
    summary='J2학습176비교미채택을마감하고 K동일전체센서입력조건 비교를동결했다. 공개개발ID11~13원본12개를FIT와같은capture/RGB/시간/몸상대좌표로별도추출중이며GPU.0확인. 모델가중치추가학습없음/기존완료결과변경없음. 최초role명오류·모듈import충돌은추출전보존후복구,첫freeze와정정EXECUTION_FREEZE둘다보존. 전체goal active.'
    nxt='K prepare_coherent.py exec57498/PID39068·PREP_PROGRESS확인후중복실행금지. 완료하면diagnose_coherent.py로새입력12개×고정J2모델24진단,verify_coherent.py별도산술/출처/시간검산. 새동일입력계약진단이지옛점수재채점아님. ROOT§8/§9/STATE/HANDOFF동기화. 사용자/멘토/H14·15/C/T3F/운영웹보존. 같은조건에서도남는문제를확인전추가학습문턱변경금지.'
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','동일 전체센서 공개12입력 준비 시작',summary),('## 9. 다음 단일 작업','동일입력 준비12개·24고정모델진단 이어받기',nxt)]:
        h='### 2026-11-02 '+title
        if h not in txt:txt=txt.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    p.write_text(txt,encoding='utf8');state=read(O/'STATE.json');state['current_stage']='COHERENT_SENSOR_INPUT_PREPARING';state['updated_at']=at;state['next_single_action']=nxt
    state[K.name]=dict(at=at,status='PREPARING_NEW_INPUTS',active=ex,session=57498,plan=str(K/'EXECUTION_FREEZE.json'),summary=summary,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(J/'RESULT_KO.txt')+'\n'+str(K/'EXECUTION_FREEZE.json')+'\n',encoding='utf8')
if __name__=='__main__':main()
