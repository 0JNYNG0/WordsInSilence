from prepare_plan import *

def main():
    plan=read(U/'PLAN.json');assert not plan['execution_authorized']
    title='### 2026-11-13 공통 특징 학습 선별 확장 계획 — 사용자 승인 대기'
    summary='사용자 요청에 따라 실행없이 100단어 밖 선별확장 계획만 검토. 목록3000단어/54000정면기록,학습후보26945(2945단어9기록·55단어8기록) 확인;전영상디코딩보증아님. 기존100유지+부족특징을채울최대20외부단어(기확보숫자포함)부터,세부정답→RTMW재사용/추가→TCN+GRU A기존100/B추가단어/C추가세부지도 각3seed→일반단어손가락/정상오경고/방향·순서·속도회귀→설명구간교정문장.20은작업상한이지최적수치아님. 기존TCN100/웹4단어범위유지. 새추출/학습/추론없음.'
    next_action='사용자가 이번6단계 선별확장 계획을 확인하고 승인할 때까지 대기한다. 승인후 첫작업은 기존100의누락특징표와HDD중그대비를채우는단어목록(최대20,기존숫자포함)확정. 대규모추출/새학습자동재개금지. C·T3-F/현재웹/100공통/사용자자료금지및C7재개점보존.'
    rel=U.relative_to(ROOT).as_posix();before=U/'before';before.mkdir(exist_ok=True)
    paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    for p in paths:
        assert not (before/p.name).exists()
        (before/p.name).write_bytes(p.read_bytes())
    text=paths[0].read_text('utf-8-sig')
    for h,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
        assert text.count(h)==1
        text=text.replace(h,h+'\n\n'+title+'\n'+body+'\n근거: `'+rel+'/PLAN_KO.txt`, `PLAN.json`, `PLAN_REVIEW.json`.\n',1)
    paths[0].write_text(text,'utf-8')
    state=read(paths[2]);state['active_task']=U.name;state['next_single_action']=next_action
    state[U.name]=dict(status=plan['status'],summary=summary,plan=rel+'/PLAN.json',report=rel+'/PLAN_KO.txt',review=rel+'/PLAN_REVIEW.json',training_runs=0,inference_calls=0,new_video_extraction=0,production_change=False,requires_user_plan_approval=True,pid=None)
    write(paths[2],state)
    paths[1].write_text(f'''# 최신 인수인계 — 2026-11-13 선별확장 계획 승인 대기

최우선: 사용자가 계획을 보고 이상없다고 판단한 뒤 작업을 진행하자고 요청했다. 지금은 계획/자료목록/기존결과 검토만 완료. 새관절추출·학습·추론·카메라·운영변경을 시작하지 않는다.
현재단위={rel}
상태=PLAN_REVIEWED_WAITING_USER_APPROVAL
계획: {summary}
다음: {next_action}
계획검토는같은Codex의저장근거/원논문/설계대조이며독립전문가검증이나새성능검증아님.20단어는상한이며부족한대비를더적게채우면적게선정.기존반복DEV11~13은새독립시험이아님.기존봉인자료그대로,최종평가적격자료는사용이력/중복부터확인.주석/추정DTW/모델예측구분.

직전완료 word100_handshape_transfer_20261113: 고정9손모양부품학습.공통100손모양묶음중12표현60학습영상860특징,36DEV영상503특징. 일반만495/496/496,숫자이름494/496/494,손가락추가495/496/495;명확한전이이득없음/미채택.503은세가지큰수형의알려진항목,전체피드백/개별손가락정확도아님. 숫자2/4/6검지모두펴짐.0/1/3/5/9/10 FIT01 원본6개·12시점후보검토,5검지접힘대비후보. 다인정답미확정/새학습미투입. 손가락교정문장 FEEDBACK_CONTRACT는명세,웹미적용. 12입력/84별도검산PASS. 완료실험재실행/DEV튜닝금지.

목표: 숫자/문자이름을일반단어의미로붙이지않고공통손가락·움직임특징을배워설명구간에맞는교정.100단어는기존공통기반이며확대가능.단어수증가만으로학습자오류/교육적허용범위가배워지는것아님.부분관측/확인안된손가락은오답으로쓰지않음.공통학습어휘와TCN인식어휘·웹피드백지원어휘는별도.

기존demo_handshape_learning_20261113:숫자6모델9/9는숫자내시험만.104TCN후보기존300의293→292로미채택.숫자4ID05원본moov손상,사과ID05뜻변이보류. Unity standalone컴파일PASS,실제빌드편집기라이선스exit198;Hub확인후공개fixture로재개. W Python open차단소비부공개재생PASS,물리HDD분리/전체카메라시연미검증.

본연구 C7: word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization. C6고정36모델완료,팔따라쓸기seed42 174→169/180회귀로G2REWORK. 다음15고유FIT새손실(경계8)원인분리. 새수형실험으로이문제를해결했다고하지않음. 기존C·T3-F/100TCN/웹모델보존;W원본읽기전용;사용자/멘토학습·튜닝·선택금지.
''','utf-8')
    with paths[3].open('a',encoding='utf-8') as f:f.write('\n\n'+title+'\n'+summary+'\n'+next_action+'\n근거: '+rel+'/PLAN_KO.txt\n')
    checks=dict(state_waiting=read(paths[2])[U.name]['requires_user_plan_approval'],context_record=title in paths[0].read_text('utf-8'),handoff_record='승인 대기' in paths[1].read_text('utf-8'),plan_hash=read(U/'PLAN_REVIEW.json')['plan_sha256']==sha(U/'PLAN.json'),previous_complete_preserved=all(sha(PREV/p)==h for p,h in read(PREV/'COMPLETION.json')['hashes'].items()),protected=all(sha(p)==h for p,h in read(PREV/'FROZEN.json')['protected'].items()))
    assert all(checks.values());write(U/'DOCUMENT_CHECK.json',dict(status='PASS',checks=checks,new_experiments=0))
    print('Plan recorded, evidence preserved, awaiting user approval; no training or extraction started')
if __name__=='__main__':main()
