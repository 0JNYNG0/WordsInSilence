from pathlib import Path
import json
from datetime import datetime,timezone,timedelta
V=Path(__file__).resolve().parent;O=V.parent;ROOT=V.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sync(title,summary,nxt,status,active=None):
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
    for marker,head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,summary),('## 9. 다음 단일 작업',title+' 이어받기',nxt)]:
        h='### 2026-11-02 '+head
        if h not in text:text=text.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    p.write_text(text,encoding='utf8');state=read(O/'STATE.json');state.update(current_stage=status,next_single_action=nxt,next_single_task=nxt,updated_at=at)
    state[V.name]=dict(at=at,status=status,summary=summary,unit=str(V),active=active,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(V)+'\n\n완료L/J/K/F/E/D/U감사재실행금지. W파생폴더만새출력/원본과이전결과보존. 같은Codex검산≠독립수어전문가정답. 전체goal active.\n',encoding='utf8')
    write(V/'CURRENT_EXECUTION.json',dict(at=at,status=status,active=active,next=nxt,whole_goal_complete=False))
    assert read(O/'STATE.json')[V.name]['summary']==summary and summary in p.read_text(encoding='utf8')
if __name__=='__main__':
    sync('관측 보존·내용연결 역할판정부 2학습 시작',
      'V공통계약검사통과: 손동작확률원본보존,준비복귀에동작모양이있어도대기손으로변경X,짧은반대근거/영상누락/전체시간유지. L主시작가중치의행동판독전체를고정하고구간역할판정부만학습. 같은FIT252입력/80epoch에서역할추가학습대조와9개판독확률을참고하는zero-init연결후보2조건진행중. 목표순서강제없음/새성능미확인/웹미적용.',
      'train_bridge.py session85123 실제세션/TRAIN_PROGRESS확인후중복금지. 완료후 evaluate_bridge.py 새2모델x88=176한번,verify_bridge.py 별도저장배열검산. L저장원판독/관측확률과exact일치검사. raw정답개선으로포장금지;주요자격손실/이벤트전체/순서/짧은역순/누락/정상대기별도평가. C/T3F/웹/H14·15/개인/멘토보존. W분리영향PORTABILITY_NOTE참조.',
      'CONTENT_ROLE_BRIDGE_TRAINING',85123)
