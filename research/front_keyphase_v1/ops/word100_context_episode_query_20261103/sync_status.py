"""Root-owned status synchronization; preserve historical entries and results."""
from pathlib import Path
import json,shutil,re
from datetime import datetime,timezone,timedelta
from prepare import U,O,read,sha
ROOT=O.parents[2]
TITLE='2026-11-03 동작 맥락 검토와 구간별 움직임 질문 분리'

def sync(summary,next_action,stage):
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
    backup=U/'before';backup.mkdir(exist_ok=True)
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
    handoff=ROOT/'NEW_CHAT_HANDOFF.md'
    state=O/'STATE.json'
    for p in [context,handoff,state]:
        dest=backup/p.name
        if not dest.exists():shutil.copy2(p,dest)
    text=context.read_text(encoding='utf-8-sig')
    for section,body,heading in [(8,summary,TITLE),(9,next_action,TITLE+' 이어받기')]:
        start=re.search(r'^## '+str(section)+r'\..*$',text,re.M)
        assert start
        end=re.search(r'^## ',text[start.end():],re.M)
        stop=start.end()+end.start() if end else len(text)
        content=text[start.end():stop]
        item='\n\n### '+heading+'\n\n'+body+'\n\n'
        pattern=r'\n*### '+re.escape(heading)+r'\n.*?(?=\n### |\Z)'
        if re.search(pattern,content,re.S):content=re.sub(pattern,item,content,count=1,flags=re.S)
        else:content=item+content.lstrip('\n')
        text=text[:start.end()]+content+text[stop:]
    context.write_text(text,encoding='utf-8')
    s=read(state)
    s.update(updated_date_kst=stamp[:10],current_stage=stage,next_single_action=next_action,next_single_task=next_action)
    s['word100_direct_event_context_20261102']=read(O/'word100_direct_event_context_20261102/FINAL.json')
    s[U.name]=dict(updated_at=stamp,status=stage,summary=summary,next_action=next_action,
                   directory=str(U),whole_goal_complete=False,latest_result=str(U/'RESULT.json') if (U/'RESULT.json').exists() else None)
    if (U/'FINAL.json').exists():
        s[U.name]['status']=read(U/'FINAL.json')['status']
        s[U.name]['final']=str(U/'FINAL.json')
    prototype=O/'word100_local_event_learning_20261103'
    if (prototype/'STATUS.json').exists():
        s[prototype.name]=dict(read(prototype/'STATUS.json'),directory=str(prototype),
                              implementation_plan=str(prototype/'PLAN_RESOLUTION.json'),updated_at=stamp)
    state.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+stamp+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n'+str(U)+'\n\n전체 goal active·미완료. 기존 C/T3F/8804/8808/8809·원본·기존 동결결과 보존. 사용자·멘토·잠금평가 자료 학습/튜닝/선택 금지. 동일 Codex 검산과 독립 수어 전문가 검증은 다르다. 현재 단위에 기록된 실행 상태/실제 프로세스 확인 후 이어받고 완료추론 재실행 금지.\n',encoding='utf-8')
    assert read(state)[U.name]['summary']==summary
    assert summary in context.read_text(encoding='utf-8') and next_action in handoff.read_text(encoding='utf-8')
    return dict(stamp=stamp,pins={str(p):sha(p) for p in [context,handoff,state]})

if __name__=='__main__':
    data=read(U/'STATUS_SUMMARY.json')
    result=sync(data['summary'],data['next_action'],data['stage'])
    (U/'DOC_SYNC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result['stamp'],data['stage'])
