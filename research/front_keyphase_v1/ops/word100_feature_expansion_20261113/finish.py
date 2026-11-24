"""Close the bounded comparison using saved results only. No model execution."""
from common import *
from datetime import datetime

C7=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization/c7_local_cause_recovery_20261113'
C9=C7/'c9_sequence_context'

def main():
    e=read(U/'EVALUATION_COMPLETE.json');v=read(U/'RESULT_VERIFIED.json')
    assert v['status']=='PASS_SAVED_RESULT_ARITHMETIC'
    assert len(e['models'])==9 and not e['all_pass']
    rows=[]
    for cond in 'ABC':
        for seed in (42,43,44):
            m=e['models'][f'{cond}_{seed}'];n=m['finger_by_variant']['normal'];cm=n['confusion']
            assert sum(map(sum,cm))==389 and sum(cm[i][i] for i in range(3))==n['correct']
            rows.append(dict(condition=cond,seed=seed,total=n['total'],correct=n['correct'],macro_f1=n['macro_f1'],
                             extended_correct=cm[0][0],bent_correct=cm[1][1],folded_correct=cm[2][2],
                             withheld=n['withheld'],accepted=n['accepted'],accepted_correct=n['accepted_correct']))
    write(U/'READABLE_COMPARISON.json',dict(rows=rows,unit='One finger at one reviewed ordinary-sign native time point',
        truth_support='36 clips / 12 expressions / historical development IDs11-13; 389 known finger cells',
        states=dict(extended=161,bent=27,folded=201),not_independent_new_user_accuracy=True,
        old_baseline_has_no_fine_finger_head=True,legacy=e['retention_all_conditions'],gates=e['gates']))
    write(U/'feedback_contract/INTEGRATION_DECISION.json',dict(status='WITHHELD_NOT_DEPLOYED',
        comparison_sha256=sha(U/'EVALUATION_COMPLETE.json'),reason='All three C seeds fail fine-macro-F1, per-hand and per-state gates. Legacy retention passes seed42 and fails seeds43/44. Every seed fails overall adoption.',
        contract_ready=True,model_attribute_validated=False,phase_binding_validated=False,web_changed=False,
        ready_scope='51 source-linked RGB anchor components; formatter synthetic and source contract checks only',
        supported_dictionary_finger_clause='Four reviewed ancestor-thumb-extended anchors only; other anchors are RGB observation evidence',
        unresolved_binding='Egypt REAL01 frame59 both hands outside stored guide endpoint55; never attach forcibly',
        requirement='A fine-grained correction requires validated observation, correct hand, source-supported expected attribute and correct native action interval. Unknown is not wrong.',
        not_completed=['Reliable individual-finger web correction','Expert-validated acceptable KSL variation','New camera acceptance testing']))
    now=datetime.now().astimezone().isoformat()
    write(U/'PROGRESS_AFTER_EVALUATION.json',dict(time=now,training_runs=9,evaluation_complete=True,
        independent_saved_result_audit='PASS',adoption=False,feedback_contract='COMPLETE_NOT_CONNECTED',
        c9_pending=not (C9/'COMPARISON.json').exists()))
    table='\n'.join(f"{r['condition']} / {r['seed']}: {r['correct']}/389 · 세 상태 균형점수 {r['macro_f1']:.3f} · 굽힘 {r['bent_correct']}/27 · 보류 {r['withheld']}" for r in rows)
    report=f'''공통 손가락 학습 확장과 기존 그래프 작업 — 2026-11-13

이번에 확인한 결론
숫자·사과 자료를 추가하면 일부 손가락 구별은 좋아졌습니다. 그러나 살짝 구부린 모양과 기존에 잘하던 일부 판단이 나빠져, 새 후보를 웹에 넣지 않았습니다. '단어 수를 늘렸으니 피드백이 좋아졌다'고 결론 내릴 수 없는 결과입니다.

1. 어떤 자료를 늘렸나
기존100단어의 학습899영상과 개발300영상은 유지했습니다. 부족한 손가락 대비를 채우기 위해 숫자1~6과 과일 사과, 7표현의74영상을 추가했습니다. 그중56은 학습,18은 개발용입니다. 합계1273영상 입력을 준비했으며, 1273영상 전부에 세밀한 정답이 있다는 뜻은 아닙니다.
새로 필요한 숫자1·3·5의36영상,3347프레임만 RTMW로 추출하고 기존 캐시는 재사용했습니다. 숫자4의 손상 원본1개와 뜻이 다른 사과 사례는 기록을 남겨 제외했습니다. 공통 학습 범위를 늘린 것이며 웹의 단어TCN100이나 피드백4단어를 확대한 것은 아닙니다.

2. 무엇을 정답으로 가르쳤나
공개 원본186프레임을 직접 확인하고, 이전에 확인한 숫자2·4·6의 주석도 연결했습니다. 각 손가락을 '펴짐 / 구부림 / 접힘 / 확인 어려움'으로 나눴습니다. 이름이 숫자5라서 특정 손모양이라고 추정한 정답은 아닙니다. 보이지 않는 곳은 학습 정답에서 제외했습니다.
일반 수어 학습 정답427개와 추가 자료434개, 총861개의 손가락 항목이 있습니다. 여기서 한 항목은 '어떤 시점의 어느 손가락 상태'입니다. 영상 수나 전체 수어 정답 수가 아닙니다. 굽힘 정답은26개로 상대적으로 적습니다. 이것이 성능 저하의 유일한 원인이라고 증명된 것은 아닙니다.
자료 판독과 별도 검산은 Codex와 하위 에이전트가 수행했습니다. 독립된 한국수어 전문가의 정답 검수와는 다릅니다.

3. 실제로 어떻게 학습했나
RTMW 관절을 입력하는 기존 시간 합성곱+GRU 연구 신경망에 양손 공통 손가락 판정부를 붙였습니다. 처음에는 새 판정부를 배우게 하고, 이후 기존 신경망도 낮은 학습률로 미세조정했습니다. 공통100단어 학습을 함께 유지했습니다. RGB 모델·SignRep을 새로 추가한 것은 아닙니다.
A: 기존100단어 + 일반 단어의 세밀한 정답
B: A + 추가7단어의 이름 학습
C: B + 추가 자료의 손가락별 정답 학습
각 조건을 같은 자료 순서·학습 횟수·구조로, 초기값3개씩 총9번 비교했습니다. 개발 점수를 보고 가중치나 기준을 다시 조정하지 않았습니다.

4. 결과를 어떻게 읽어야 하나
일반 수어12표현의 개발36영상에서 확인한389개 손가락 항목을 비교했습니다. 이전부터 사용한 개발자료이며 새 사용자 실전 정확도가 아닙니다. 세 상태 균형점수는 흔한 펴짐·접힘에 가려 굽힘 실패가 숨지 않게 함께 보는 점수입니다.
{table}

C는 펴짐과 접힘을 합친 정답 수가 모든 초기값에서 증가하고 왼손도 개선됐습니다. 두 상태가 각각 모두 좋아진 것은 아닙니다. 초기값43의 펴짐은154개에서152개로 감소했습니다. 굽힘도 A의9·10·10개에서 C의6·9·8개로 감소했고, 오른손과 판단 가능한 범위도 줄었습니다. 한두 개 정답 증가만으로 채택할 수 없는 이유입니다.
속도를 바꾸거나 상반신만 남긴 조건, 짧고 긴 관측 누락도 비교했습니다. C의 일부 초기값에서 기존 손모양·동작 단계 판정이1개씩 감소했습니다. 기존 공통 모델의 저장 결과와 비교한 것이며, 실제 운영 TCN을 바꿔 정확도가 떨어진 것이 아닙니다.

5. 결과 문장은 어디까지 준비했나
원본의 손가락 근거와 설명 구간을 연결하고 '관측됨 / 참고와 같음 / 차이 있음 / 확인 어려움'을 구분하는 문장 부품을 만들었습니다. 다만 새 모델의 성능 조건을 통과하지 못했으므로 실제 웹 교정에 연결하지 않았습니다.
사전이 손가락 상태를 직접 요구하는 경우와 단지 참고 영상에서 관측한 경우도 구분했습니다. 이집트의 일부 손모양 시점은 저장된 설명 구간 밖에 있어 억지로 연결하지 않았습니다. 자세히 보았다는 이유만으로 '틀렸으니 펴세요'라는 말을 내보내지 않습니다.

6. 처리시간의 의미
저장된 관절 입력에서 새 신경망을 계산하는 시간은 이번 CPU 측정에서 영상당 중앙값 약0.005~0.008초였습니다. 관절 검출·촬영·화면 표시를 포함한 대기시간이 아닙니다.
공개4영상의 저장 관절을 현재 웹과 같은 소비부에 넣는 별도1회씩 측정도 했습니다. 기존 계산은0.022~0.046초, 후보 분기를 추가한 전체 계산은0.040~0.082초였습니다. 추가 분기만 직접 잰 시간은0.011~0.013초입니다. 다른 학습 작업이 동시에 실행되던 조건이라 전체 시간 차이를 그대로 후보 비용이라고 보지 않습니다. 후보 출력의 실제 문장·구간 연결과 브라우저 대기는 포함하지 않았습니다. 정확한 범위는 TIMING_CONTRACT_REVIEW.json에 기록합니다. 이 결과로 전체 촬영 대기가 개선됐다고 주장하지 않습니다.

남은 핵심
더 많은 단어를 무작정 넣는 것보다, '조금 구부림'과 '완전히 접힘'이 잘 구별되는 일반 동작·다른 손·다른 사람의 정답을 균형 있게 확보해야 합니다. 또 눈에는 보이는데 RTMW 점이 잘못 잡힌 경우와, 점은 충분한데 신경망이 잘못 읽은 경우를 나눠야 합니다. 기존 방향·순서를 지키면서 손가락 부분을 개선하는 학습 구조도 다음 비교에서 분리합니다.

기존 그래프의 움직임/유지 결과와 5방향 아바타 검토는 아래 별도 보고서를 함께 확인합니다. 이번 손가락 시험만으로 동작 구간 연결 문제가 해결됐다고 판단하지 않았습니다.
기존 C·T3-F, 운영 TCN·웹, 원본 자료는 보존했습니다. 사용자·멘토 촬영의 학습·튜닝·후보 선택, 봉인자료 사용, 카메라, 새 다운로드, Git 작업은 하지 않았습니다.

근거
READABLE_COMPARISON.json / EVALUATION_COMPLETE.json / RESULT_VERIFIED.json
annotation_review/ / feedback_contract/INTEGRATION_DECISION.json
기존 그래프: {C9}
아바타: {O/'avatar_multiview_review_20261113/RESULT_KO.txt'}
'''
    (U/'RESULT_KO.txt').write_text(report,'utf-8')
    write(U/'NEXT_PLAN.json',dict(status='NEXT_BOUNDED_RECOVERY_DESIGN',
        completed_comparisons_are_frozen=True,new_results_require_new_contract=True,
        steps=[
            dict(id=1,work='Audit only public FIT curved/folded/extended transitions, both hands, moving and held appearances; distinguish visible source truth from extracted geometry.',
                 gate='At least3 recorded FIT IDs and2 ordinary expressions per admitted contrast; ambiguous projection stays unknown; no DEV-error-directed source selection.',
                 reason='Current fine FIT has26 bent vs427 extended and408 folded cells; numeric frames predominantly right-handed.'),
            dict(id=2,work='If geometry contains the distinction, compare a frozen-backbone dedicated temporal finger branch against the prior fine-tuned design under matched supervision and compute.',
                 gate='Freeze from FIT checks before development; retain all seeds and all known-item denominators; cannot claim existing outputs preserved if a shared latent changes.',
                 reason='Fine-tuning helped common states but regressed bent/right-hand/legacy items. Data imbalance, observation, and loss interference remain separate hypotheses.'),
            dict(id=3,work='Carry existing motion/hold graph through C9 stored-result verification, then return to the stage indicated by its diagnostic result.',
                 gate='Component gain alone cannot close whole G2; automatic interval/normal-speed/crossing/order/omission comparison must pass before release.'),
            dict(id=4,work='Bind only independently validated finger/motion evidence to native-time guide clauses; unknown or unbound moments remain observation requests.',
                 gate='No educational correction based only on confidence, DTW match, or requested word label. Before runtime use, share internal guarded arrays/masks rather than display sentinels and use the actual target index; timing-only sidecar is not this integration.'),
            dict(id=5,work='Avatar: recover same-clip official3D plus camera parameters, verify reprojection before Mixamo retarget and FBX baking.',
                 gate='No calibration/record mixing; missing local raw source may require separate acquisition decision. This is offline reference-asset work, not live feedback inference.')],
        no_automatic_production_switch=True,no_new_large_model_required_by_evidence=True))

if __name__=='__main__':main()
