from prepare_seated import *
ROOT=L.parents[3]
def sync(title,summary,nxt,status,active=None):
    at=clock();p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
    for marker,heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,summary),('## 9. 다음 단일 작업',title+' 이어받기',nxt)]:
        h='### 2026-11-02 '+heading
        if h not in text:text=text.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    p.write_text(text,encoding='utf8');s=read(O/'STATE.json');s.update(current_stage=status,updated_at=at,next_single_action=nxt,next_single_task=nxt)
    s[L.name]=dict(at=at,status=status,active=active,unit=str(L),summary=summary,whole_goal_complete=False);write(O/'STATE.json',s)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(L)+'\n\n완료J/K/F/E/D 재실행금지. 같은Codex 검산은 독립수어전문가정답이 아니다. 전체goal active.\n',encoding='utf8')
    assert summary in p.read_text(encoding='utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
if __name__=='__main__':
    sync('공개 웹캠시야36입력 준비 재개',
      'L공개FIT01~09의36영상 크롭·원래시각검증완료. 19개추출후 C여유1GiB안전장치로중단(exit1/15537). 완료19·최초오류보존하고나머지17의파생입력만별도W:/WordsInSilence_derived에이어추출중(session74271). 원본W:/sources불변/삭제이동없음. 표본시각검토9ID의중요손·팔동작이화면내보존됨을확인;손내림의화면밖손모양을정답강제하지않는다. 새학습아직없음/전체목표미완료.',
      'resume_storage.py session74271 및EXTRACT_PROGRESS/CURRENT_EXECUTION실제상태확인,중복금지. 36입력완료후시간·해시·학습정답/화면근거검증. 같은원본재생36추가대조 대 웹캠시야36추가로입력변이효과만분리한미세조정조건을결과전동결한다. FIT01~09,DEV11~13은기존노출개발. H14/15/사용자/멘토/C/T3F/웹보존. W별도파생경로만새출력;원본/옛결과삭제금지.',
      'SEATED_FIT_EXTRACTION_RESUMED_LOCAL_W',74271)
