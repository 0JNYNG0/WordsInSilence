from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sync(title,summary,nxt,status,active=None):
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf8')
    for marker,h,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,summary),('## 9. 다음 단일 작업',title+' 이어받기',nxt)]:
        h='### 2026-11-02 '+h
        if h not in s:s=s.replace(marker,marker+'\n\n'+h+'\n\n'+b+'\n',1)
    p.write_text(s,encoding='utf8');state=read(O/'STATE.json');state.update(current_stage=status,next_single_action=nxt,next_single_task=nxt,updated_at=at)
    state[U.name]=dict(at=at,status=status,summary=summary,unit=str(U),active=active,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(U)+'\n\n완료V2학습176비교/L/J/K/F/E/D/U감사 재실행금지. 새대용량은 W파생폴더만. C/T3F/기존웹/사용자/멘토/H14·15보존. 같은Codex검산≠독립수어전문가정답. 전체goal active.\n',encoding='utf8')
    write(U/'CURRENT_EXECUTION.json',dict(at=at,status=status,active=active,next=nxt,whole_goal_complete=False))
    assert read(O/'STATE.json')[U.name]['summary']==summary and summary in p.read_text(encoding='utf8')

if __name__=='__main__':
    sync('학습한 기준 흐름을 구간 판단에 연결하는 입력 준비',
        'V2학습176비교완료: 원판독589고정/제외대조9·후보10/순서6·짧은역순1로새연결미채택. 후속 GRAPH_SCOPE_AUDIT에서 FIT4468프레임 중 직접과정정답1066·주요706, sparse앞뒤근거가전체동일모양의비중첩규칙을증명하지않음을정리. 새flow역할후보는기존P학습prototype1개DTW의위치·거리·가이드구간힌트·압축정보를역할판정부에만연결. 정렬로새정답을만들지않고원관측/짧은근거보존. 공개FIT252/DEV88정렬준비중.',
        'prepare_flow.py session82712 실제세션 확인후중복금지. 완료후 train_flow.py 80epoch 1후보, evaluate_flow.py 88개한번,verify_flow.py검산. V role_only는완료된동일조건대조라재학습/재추론없음. 초기L가중치·학습입력/정답/seed/optimizer동일,역할에만학습prototypeDTW힌트추가. 원판독exact589유지,주요제외<9/순서>6/짧은역순>=2/대기>=96등고정. 학습한기준흐름힌트가정답순서강제/짧은오류숨김이되지않는지평가. 전체goal미완료/웹미적용.',
        'LEARNED_FLOW_ROLE_PREPARING',82712)
