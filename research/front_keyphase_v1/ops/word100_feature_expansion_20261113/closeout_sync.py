"""Synchronize reviewed, bounded results. Does not train, infer, or change production."""
from finish import *
import shutil

def main():
    c=read(C9/'COMPARISON.json');sv=read(C9/'SAVED_VERIFICATION.json');cl=read(C9/'CLOSEOUT_LEARNING.json')
    assert sv['status']=='PASS' and sv['comparison_sha256']==sha(C9/'COMPARISON.json')
    assert cl['trained36'] and not cl['training_running'] and not cl['diagnostic_pass']
    checks=0
    for a in c['aggregate']:
        folds=[r for r in c['by_fold'] if r['seed']==a['seed'] and r['baseline']==a['baseline']]
        assert len(folds)==6
        for cat,z in a['categories'].items():
            for key,val in z.items():
                assert sum(r['categories'][cat][key] for r in folds)==val;checks+=1
            assert z['after']==z['before']+z['recoveries']-z['regressions'];checks+=1
    fa=read(C9/'FIT_AUDIT.json');assert len(fa['rows'])==36
    for r in fa['rows']:
        for k in ('movement','own_hold'):assert r[k]['n']==r[k]['correct'];checks+=1
    write(U/'C9_ROOT_AGGREGATE_REVIEW.json',dict(status='PASS',actor='root separate from C9 author',checks=checks,
        method='Sum six held-record fold tables and compare paired totals/recoveries/regressions; read fixed saved FIT predictions audit. No model execution.',
        source_sha256=sha(C9/'COMPARISON.json'),diagnostic_pass=False))
    c9text='''

기존 그래프 작업도 함께 완료한 비교
C7·C8에서 누락 좌표를0으로 평균내는 문제와 짧게 투영된 팔길이로 나누어 값이 커지는 문제를 확인하고 보완 후보를 비교했습니다. 일부 혼동은 줄었지만 움직임·유지·교차 단서를 함께 보존하지 못했습니다.
이어 C9에서 앞뒤 관측을 평균내는 방법과 실제 순서대로9시점을 보여주는 방법을 같은 작은 시간합성곱+GRU로 비교했습니다. 기록ID 하나를 학습에서 빼는6분할, 초기값3개씩 총36개 학습과 저장결과 검산을 마쳤습니다.
평균대조→순서보존, 초기값42·43·44 순:
실제 움직임:713·687·685 →706·693·700 /822항목
자세 유지:461·469·472 →454·450·465 /522항목
양손 상대높이:264·264·264 →267·271·270 /294항목
양손 높이 단서는 조금 나아졌지만 전체 움직임·유지 보존에 실패했습니다. /822 등은 화면조건과 정역을 반복한 진단 항목 수이며 독립 영상 수가 아닙니다. 저장된 기존 C6 기준에도 미치지 못했습니다.
학습에 쓴 움직임·유지 항목은 두 조건 모두 오류가 없었지만, 제외한 기록ID에서는 위와 같이 실패했습니다. 단순히 반복 학습이 부족한 것보다는 다른 수행자에게 적용하는 일반화 문제가 남았다는 증거입니다. 시간 순서가 불필요하다는 뜻도, 평균이 모든 오류의 원인이었다는 뜻도 아닙니다.
기존 그래프의 G2는 재보완 상태입니다. 동작의 구간 연결·완료·누락·실제 교정이 모두 해결됐다고 올리지 않았습니다. 이36개 모델을 반복하거나 결과가 좋은 초기값만 고르지 않습니다.

아바타 검토와 사용자 삼각측량 의견
예전3방향 시도는 삼각측량으로3D를 만들려던 의도였음을 기록했습니다. 남은 계산 코드는 카메라 보정값을 쓰는 정밀 삼각측량이 아니라 정면·측면을 조합한2.5D 근사였습니다.
공식5방향 동시 촬영과3D·카메라 정보 제공 기록을 확인했고, 공개 조상1기록의5영상을 실제 대조했습니다. 다른 기록에 남아있는 공식3D 손 좌표도 확인했지만 두 기록을 섞지 않았습니다. 손가락 길이가 갑자기 달라지는 구간이 있어 그대로 캐릭터에 복사하지 않습니다.
우선순위: 같은 촬영의 공식3D·카메라 원본 확보 →5방향 화면에 다시 겹쳐 검증 →기존 Mixamo의 손가락뼈·기준자세에 맞춰 변환 →FBX/AnimationClip으로 저장.
제공3D가 부족한 구간에서만 보정된5뷰 관절로 삼각측량하는 경로가 타당합니다. 현재 원본keypoint ZIP의 옛 경로가 없어 복구가 먼저입니다. 새 다운로드·아바타 제작은 이번 검토에 포함하지 않았습니다. 게임에는 완성 애니메이션을 담아 재생하므로 아바타 재생 때문에 매번HDD나5뷰 분석을 요구할 필요는 없습니다.
'''
    rp=U/'RESULT_KO.txt';txt=rp.read_text('utf-8');assert '기존 그래프 작업도 함께 완료한 비교' not in txt;rp.write_text(txt+c9text,'utf-8')
    np=read(U/'NEXT_PLAN.json');np['steps'][2].update(
        work='C9 is complete and not adopted. Compare reviewed public FIT source/geometry against repeated fold failures, then target cross-record variation and compact representations rather than adding epochs.',
        reason='All36 training-fit movement/hold tables have zero misses while held-record scores regress; this supports a generalization gap, not proof of one unique cause.',
        gate='Do not rerun completed C9 or choose a seed/threshold. New bounded FIT-only design must preserve C6 movement/hold/crossing before full automatic interval/order/omission comparison. Historical DEV remains historical, sealed final8 stays closed.')
    write(U/'NEXT_PLAN.json',np)
    g=read(U/'GRAPH.json');g['nodes'].update(G4_train='COMPLETE_FIXED_9',G5_regression='COMPLETE_AUDITED_ADOPTION_FAIL',
        G6_feedback='CONTRACT_COMPLETE_DEPLOYMENT_WITHHELD',C7_motion_hold='C7_C8_C9_COMPARED_G2_REWORK',
        C9_sequence='COMPLETE_FIXED_36_NOT_ADOPTED',AV_avatar='REVIEW_AND_GEOMETRY_AUDIT_COMPLETE_REPROJECTION_PENDING')
    g['evidence'].update(comparison=str(U/'EVALUATION_COMPLETE.json'),verification=str(U/'RESULT_VERIFIED.json'),
        C9=str(C9/'GRAPH_PATCH.json'),feedback=str(U/'feedback_contract/INTEGRATION_DECISION.json'))
    g['bounded_comparison_complete']=True;g['project_complete']=False;g['running_jobs']=[];write(U/'GRAPH.json',g)
    write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_COMPARISONS_NOT_ADOPTED',pid=None,training_running=False,
        training_runs=9,C9_runs=36,report=str(rp),next_plan=str(U/'NEXT_PLAN.json')))
    title='2026-11-13 선별 확장9학습·기존 그래프 C9·5뷰 아바타 검토 완료'
    summary='''기존100단어 FIT899/DEV300 유지, 숫자1~6·사과 추가74(FIT56/DEV18), 총1273입력. 공개RGB186프레임검토와 기존숫자주석으로 FIT손가락861항목/일반DEV389항목 준비. 실제시간합성곱+GRU A/B/C×3seed 고정9학습·1회평가·별도저장검산 완료. 추가세부지도C의총정답356/359/360 대 A354/359/356이나 굽힘9/10/10→6/9/8·오른손·일부기존항목회귀로미채택. 펴짐/접힘 합산개선이지각상태일관개선아님. 문장계약51앵커·합성16검사완료, 성능/구간검증미통과로웹미연결.
기존C7/C8에이어C9 평균vs실제9시점순서36학습완료. 움직임713/687/685→706/693/700(/822), 유지461/469/472→454/450/465(/522); 전체보존실패/G2REWORK. 학습FIT움직임·유지오류0이나제외기록ID실패로일반화문제근거. 전체진단은화면조건/정역반복이고새사용자평가아님.
아바타공식5뷰동시촬영/3D·camparam근거와공개5영상실물확인. 과거삼각측량의도는기록했으나저장구현은2.5D근사. 다른기록의남은3D캐시는손마디길이급변을확인. 동일클립3D+카메라원본복구/재투영후Mixamo리타게팅권고;원본ZIP옛경로없음.애니메이션제작아님.
C·T3-F/운영TCN·웹/공통100보존,사용자·멘토·봉인자료미사용. 새다운로드/카메라/Git없음.학습종료. 작성/검산은root와Codex하위에이전트이며독립사람전문가검수와구분.'''
    nexttext='''다음은 일반 동작의 굽힘/접힘 대비를 다인·양손·움직이는 상태로 보강하고, 원본에는 보이나 관절이 틀린 경우와 모델 해석 실패를 나눈다. 공통100유지·기존방향/순서보존을 위해 동결기반 손가락별도분기와 미세조정분기를 새 FIT-only 계약으로 비교설계한다. C9는추가epoch보다제외기록의일반화실패감사와표현/자료균형을우선한다. 완료9/36 및C6~C8재학습·DEV점수튜닝·웹교체금지. 전체G2통과전최종8/출시검증으로넘기지않는다. 아바타는동일클립원본위치복구→재투영이첫작업. 새외부확보는현재검토범위에포함되지않는다.'''
    before=U/'before_closeout';before.mkdir(exist_ok=True)
    for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt'):
        dest=before/p.name
        if not dest.exists():shutil.copy2(p,dest)
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ct=context.read_text('utf-8-sig');assert title not in ct
    evidence='근거: `research/front_keyphase_v1/ops/word100_feature_expansion_20261113/RESULT_KO.txt`, `GRAPH.json`, `NEXT_PLAN.json`.'
    for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nexttext)]:
        assert section in ct;ct=ct.replace(section,section+'\n\n### '+title+'\n'+body+'\n'+evidence+'\n',1)
    context.write_text(ct,'utf-8')
    state=read(O/'STATE.json');state['active_task']=U.name;state['next_single_action']=nexttext
    state[U.name].update(status='COMPLETE_COMPARED_NOT_ADOPTED',training_runs=9,C9_runs=36,training_running=False,
        report=str(rp),verification=str(U/'RESULT_VERIFIED.json'),next_plan=str(U/'NEXT_PLAN.json'),
        C7_G2='REWORK',C9_status='COMPARED_REWORK_NOT_ADOPTED',feedback='CONTRACT_ONLY_NOT_DEPLOYED',
        avatar_review_complete=True,project_complete=False,production_changed=False,private_used=False)
    state['word100_web_completion_graph_20261111'].update(status='C7_C8_C9_COMPARED_G2_REWORK_NOT_ADOPTED',
        active_unit=str(C9),latest_evidence=str(C9/'GRAPH_PATCH.json'),training_running=False,
        progress='C7/C8/C9 fixed comparisons complete; C9 ordered9-time inputs versus matched mean control failed movement/hold preservation. Training-fit misses0, held-record generalization unresolved. No G2 pass or web replacement.',
        remaining=nexttext)
    write(O/'STATE.json',state)
    handoff=f'''# 最新 인수인계 — {title}

현재 사용자 승인: 선별 확장1~6전체, 기존그래프 병행, 공개5뷰 아바타 병렬 검토. 최신 단위 {U}
이번 고정 비교는 완료했고 실행 중 학습은 없다. 프로젝트/웹피드백 완성은 아니다.

{summary}

완료결과 재학습·재추론 금지. 기존개발ID11~13은새독립평가가아님. C9는공개FIT ID02~07의6분할이며DEV/최종8미사용.
검증: RESULT_VERIFIED.json, C9/SAVED_VERIFICATION.json, C9_ROOT_AGGREGATE_REVIEW.json. 코드작성자와저장결과검산자를구분한다.
보존한오류: FROZEN_PREFLIGHT_V1(정규화축누출/캐시해시검사수정전), TIMING_FIRST_FAILURE(소비부import common충돌,추론전수정), 옛손상원본/뜻변이제외기록. 타이밍원자료와범위정정은TIMING_CONTRACT_REVIEW.json을함께본다.

다음: {nexttext}
세부 {U/'NEXT_PLAN.json'}
아바타 {O/'avatar_multiview_review_20261113/RESULT_KO.txt'}
주의: 기존C·T3-F/운영웹/TCN100 보존. 사용자·멘토자료 학습/튜닝/선택금지. W:·sources·과거동결읽기전용. 새다운로드/외부전송/카메라/Git없음.
'''.replace('# 最新','# 최신')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
    history=ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with history.open('a',encoding='utf-8') as f:f.write('\n\n### '+title+'\n사용자 의도: 숫자 자체를 더 맞추기보다 여러 수어에 공유되는 손가락 특징을 배우고 설명에 근거한 교정으로 연결. 기존그래프와5방향아바타삼각측량검토를함께진행.\n'+summary+'\n다음 판단: '+nexttext+'\n'+evidence+'\n')
    write(U/'DOCUMENT_SYNC.json',dict(status='SYNCHRONIZED',title=title,files={str(p):sha(p) for p in [context,ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',history]},
        no_training_or_inference=True,project_complete=False))

if __name__=='__main__':main()
