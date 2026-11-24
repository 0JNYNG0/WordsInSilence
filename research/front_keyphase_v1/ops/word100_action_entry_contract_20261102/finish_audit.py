from audit_entries import *
ROOT=U.parents[3]
def replace_json(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not (U/'REVIEW.json').exists();a=read(U/'AUDIT.json');plan=read(U/'FROZEN.json')
    for p,h in plan['pins'].items():assert sha(p)==h
    for r in a['cards']:assert sha(r['image'])==r['sha256']
    n=sum(r['main_points'] for r in a['rows']);assert n==706
    review=dict(at=clock(),status='SPARSE_BOUNDARY_CONTRACT_AUDITED',reviewer='same Codex visual analysis, not independent KSL expert',
        cards_reviewed=8,fit_ids_reviewed=[1,5,9],words=['WORD0739','WORD1133','WORD1567','WORD2523'],
        findings=['All36FIT main labels remain in E main supervision; no main-versus-context label overwrite.',
          'Entry witnesses have unlabelled predecessors. Unknown frames were already masked, not wrongly labelled nonmain by the data loader.',
          'Adjacent RGB shows gradual arrival and departure; first scored main frame is not an independently annotated exact action onset.',
          'A terminal hand configuration remains visually similar during initial return. Content and process role can differ without making observed handshape false.',
          'Review panels support coarse motion/position continuity, not precise finger contact or dense new labels.'],
        result='Do not add guessed dense boundaries. Separate native physical content from learned role; score event coverage and point mistakes separately without hiding short contrary evidence.',
        original_labels_unchanged=True,new_training=0,new_model_inference=0,whole_goal_complete=False)
    save(U/'REVIEW.json',review)
    summary='L2학습176비교후 U공개FIT36의706주요정답과준비복귀감독일치감사완료. 학습정답덮어쓰기0/미주석을준비로가르치는로더오류0. ID1·5·9의4표현진입/종료8연속표본검토: 첫채점시점은정밀시작경계가아니고동일손모양이복귀초기에도남음. 새정답/학습/모델추론없음. L은원동작587→589/615이나준비해석이14시점(원본5곳)을가려최종584→575라미채택. 전체목표active.'
    nxt='U REVIEW/AUDIT를따라,원관측과과정역할을별도로보존하는계약을구현·행동시험부터한다. 역할불확실/충돌을대기손모양으로덮지않으며,시점분류손실과행동전체누락을별도계산한다. 정상복귀와짧은역순을둘다살린고정비교를FIT에서정의후필요한공동감독후보만학습. L/J/K/F/E/D완료재실행금지,DEV문턱튜닝/강제순서금지. H14·15/개인/멘토/C/T3F/기존웹보존. 현재active없음. W분리시현재연구와8804일부참고재생/시작영향있음;PORTABILITY_NOTE참조. 최종배포전W없는시작·추론·재생검증필요,지금파일이동없음.'
    txt='주요 동작 시작 구간의 학습 정답 점검\n\n'+summary+'\n\n주요동작의정답자체가준비동작정답으로덮인오류는없었다. 다만일부시점에만정답이있어정밀한시작경계를배웠다고말할수없다. 새모델이맞게읽은동작을준비로제외하는문제는관측과해석연결에서도보완해야한다.\n\n'+nxt+'\n'
    (U/'RESULT_KO.txt').write_text(txt,encoding='utf8')
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','주요진입 감독 감사·저장장치 의존 확인',summary),('## 9. 다음 단일 작업','관측 보존과 과정 역할 계약',nxt)]:
        head='### 2026-11-02 '+title;assert head not in text;text=text.replace(marker,marker+'\n\n'+head+'\n\n'+body+'\n',1)
    p.write_text(text,encoding='utf8');state=read(O/'STATE.json');state.update(current_stage='ACTION_ENTRY_CONTRACT_AUDITED',next_single_action=nxt,next_single_task=nxt,updated_at=clock())
    state[U.name]=dict(at=clock(),status='SPARSE_BOUNDARY_CONTRACT_AUDITED',active=None,summary=summary,report=str(U/'RESULT_KO.txt'),whole_goal_complete=False);replace_json(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+clock()+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(U/'REVIEW.json')+'\n'+str(U/'PORTABILITY_NOTE.json')+'\n'+str(L/'VERIFIED.json')+'\n'+str(L/'CAUSE_AUDIT.json')+'\n\nL추출재개74271exit0/학습22885exit0/평가176·검산완료. U감사24348exit0/8카드시각검토완료. active없음. 같은Codex검산≠독립수어전문가정답. 전체goal active.\n',encoding='utf8')
    replace_json(U/'CURRENT_EXECUTION.json',dict(status='AUDIT_COMPLETE',active=None,new_training=0,next=nxt,whole_goal_complete=False))
    assert read(O/'STATE.json')[U.name]['summary']==summary and summary in p.read_text(encoding='utf8')
    save(U/'VERIFIED.json',dict(at=clock(),status='PASS_AUDIT_DOCUMENTS',pins_unchanged=True,major_labels=706,review_cards=8,whole_goal_complete=False))
    print('U audit complete; no new training; goal remains active')
if __name__=='__main__':main()
