from prepare import *
from datetime import datetime, timezone

def main():
    root=U.parents[3];unit=U.name;rel=U.relative_to(root).as_posix()
    complete=read(U/'COMPLETION.json');assert all(sha(U/f)==h for f,h in complete['hashes'].items())
    # Preserve completed evidence; correct a bibliographic short title in an addendum.
    evidence=read(U/'RESEARCH_EVIDENCE.json')
    evidence['references'][0]['title']='Handshape-Aware Sign Language Recognition: Extended Datasets and Exploration of Handshape-Inclusive Methods'
    evidence['bibliographic_review']='Exact first-paper title checked against ACL metadata after initial report finalization; numerical conclusions unchanged. Original evidence is preserved.'
    write(U/'RESEARCH_EVIDENCE_REVIEWED.json',evidence)
    summary='공통손모양 전이근거 검토와 고정9학습 완료. 감사한100공통 손모양 묶음은12표현/학습60영상860특징,개발36영상503특징. 일반수어만495/496/496,숫자이름추가494/496/494,손가락보조추가495/496/495로명확한이득없음. 숫자2/4/6의검지는모두펴짐. 추가0/1/3/5/9/10 FIT01원본6개·12시점확인,5의검지접힘대비후보확보. 개별손가락교정문장근거계약작성,미배포. 입력12/별도검산84PASS. C·T3-F/공통100/웹보존,사용자자료미접근.100밖확장은누락특징기반선별로판단,3000일괄학습이나피드백향상입증아님.'
    next_action='기존 일반단어와 추가 숫자5·3·1에서 손가락별 펴짐/접힘/굽힘·엄지 관계 대비를 FIT 다인RGB로 보강한다.0·10의둥근모양은완전접힘으로억지라벨하지않는다. 일반단어 개별손가락 평가정답을 점수보기전동결하고 같은공통100기반의보조학습유무를비교한다. 검증된손가락·구간만설명문장과연결한다. 이번9모델/과거완료모델재실행·DEV튜닝금지. C7움직임/유지G2REWORK와Unity라이선스재개점별도보존.'
    paths=[root/'WordsInSilence_PROJECT_CONTEXT.md',root/'NEW_CHAT_HANDOFF.md',O/'STATE.json',root/'docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    before=U/'before';before.mkdir(exist_ok=True)
    for p in paths:
        q=before/p.name
        assert not q.exists(),'Do not repeat document sync'
        q.write_bytes(p.read_bytes())
    ctx=paths[0].read_text('utf-8-sig')
    title='### 2026-11-13 공통 수형 전이 비교·100단어 확장 근거·세부 문장 계획'
    for head,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
        assert ctx.count(head)==1
        ctx=ctx.replace(head,head+'\n\n'+title+'\n'+text+'\n근거: `'+rel+'/RESULT_KO.txt`, `VERIFICATION.json`, `NEXT_PLAN.json`.\n',1)
    paths[0].write_text(ctx,'utf-8')
    state=read(paths[2]);state['active_task']=unit;state['next_single_action']=next_action
    state['updated_at']=datetime.now(timezone.utc).isoformat();state['updated_date_kst']='2026-11-13'
    state[unit]=dict(status=complete['status'],summary=summary,report=rel+'/RESULT_KO.txt',verification=rel+'/VERIFICATION.json',evidence=rel+'/RESEARCH_EVIDENCE_REVIEWED.json',next_plan=rel+'/NEXT_PLAN.json',feedback_contract=rel+'/FEEDBACK_CONTRACT.json',training_runs=9,production_changed=False,common100_preserved=True,private_used=False,downloads=0,reviewer='Codex implementation + separate arithmetic; no independent KSL expert',pid=None,next=next_action)
    write(paths[2],state)
    handoff=f'''# 최신 인수인계 — 2026-11-13 공통 수형 전이·확장 검토

현재 목표: 지숫자·지문자를 숫자/글자 이름으로 일반단어에 붙이는 것이 아니라 공통 손가락 특징을 배워 일반 수어의 손모양 피드백을 개선한다. 2/4/6은 예시다. 새 사용자 추가요청: 근거가 있으면 손가락별로 구체적인 교정 문장도 제공하고,100밖 공통학습의 효용을 검토한다.

현재 작업 {unit}: {summary}
U={rel}
PY=research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8
실행 모두 종료. CURRENT_EXECUTION/COMPLETION 저장. 서버·카메라·운영코드·기존모델 변경 없음.

다음 한 작업: {next_action}
세부조건 NEXT_PLAN.json A~D. 이번 비교는손모양부품(63→32 MLP)이며웹시간합성곱+GRU전체를재학습한것아님. 기준도같은부품의일반단어만학습조건이지현재웹성능이아님.503은36개역사DEV영상의알려진3종특징합계,개별손가락/사용자교정/전체피드백정확도아님. 높은기본점수에서추가이득없음. 지문자전체자료확보주장금지.6신규영상은FIT01후보시각검토이며다인정답미확정/학습미투입.원본W읽기전용.

개별손가락 문장: FEEDBACK_CONTRACT.json은명세만. 목표설명구간과보이는손가락상태둘다확인된때만지시. 가려짐/애매한구간/허용변이불명은오답으로쓰지않음. 엄지/벌림/구부러짐/겉으로가까움/실제접촉은다른항목. RGB정답·DTW추정·모델예측을혼동하지않음. 부위방향/움직임/순서는보존.

근거: 손모양보조지도2023,SignBERT+2023데이터규모실험,정적+시간수형2025,지문자혼합효과가혼재한2025논문검토. RESEARCH_EVIDENCE_REVIEWED.json이정확서지제목판.기존공통100의추가8단어손모양119~120→121~122/133은작은개선근거이나단계개선아님.100→3000실험은없음. 더많은단어는필요특징/기록ID다양성을채울때선별확대.단어이름만으로개별교정기준학습완료라하지않음.

기존 demo_handshape_learning_20261113 완료보존:60계획중59관측(숫자4ID05원본moov손상),숫자35영상105앵커. 숫자26FIT/9DEV의6모델모두9/9(숫자내시험뿐).104TCN후보기존300의293→292로미채택.사과ID05뜻변이보류.과거완료결과재실행/개발문턱튜닝금지.

Unity분기: LocalFeedbackBridgeDemo.cs 및 LocalFeedbackBridgeBuild.cs standalone컴파일PASS,실제Unity6000.3.21f1빌드exit198편집기라이선스필요.사용자Hub확인후공개fixture로재개. 공개잘하다REAL02현재소비부W Python open차단통과;물리HDD분리/전체촬영/빌드성공아님.

본연구 C7 재개점보존: word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization. C6고정36모델종료. 전체움직임/유지개선이나팔따라쓸기seed42 174→169/180회귀,G2REWORK. 다음15고유FIT움직임손실중경계8 원인분리. 새수형시험이G2통과아님. 최종8/DEV/개인자료튜닝금지. 기존C·T3-F와현재웹보존.
'''
    paths[1].write_text(handoff,'utf-8')
    with paths[3].open('a',encoding='utf-8') as f:
        f.write('\n\n'+title+'\n사용자 의도: 숫자/지문자 자체의 어휘 인식보다 공통 수형 특징 전이,100밖 다양성,손가락별 구체적 피드백.\n'+summary+'\n설계 판단: 더 많은 단어만으로 교정능력을 보장하지 않는다. 큰 세 가지 수형 비교에서 추가이득 없음을 보존하고,누락된 검지접힘/엄지/굽힘 대비와 일반단어 개별손가락 정답을 우선 보강한다.\n다음: '+next_action+'\n근거: '+rel+'/RESULT_KO.txt\n')
    checks=dict(completed_artifacts_preserved=all(sha(U/f)==h for f,h in complete['hashes'].items()),state_key=read(paths[2])['active_task']==unit,context=title in paths[0].read_text('utf-8'),handoff=unit in paths[1].read_text('utf-8'),history=title in paths[3].read_text('utf-8'),protected=all(sha(p)==h for p,h in read(U/'FROZEN.json')['protected'].items()))
    assert all(checks.values())
    write(U/'DELIVERY_CHECK.json',dict(status='PASS',checks=checks,document_sha256={str(p):sha(p) for p in paths},reviewed_evidence_sha256=sha(U/'RESEARCH_EVIDENCE_REVIEWED.json')))
    print('Synced context8/9, STATE, handoff, project history; original artifacts preserved')
if __name__=='__main__':main()
