"""Assemble a review from completed saved reports; never run a verifier/model."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent
def read(n):return json.loads((U/n).read_text('utf-8-sig'))
def digest(n):return hashlib.sha256((U/n).read_bytes()).hexdigest()
comparison=read('training_COMPARISON.json')
scores=read('training_scores_INDEPENDENT.json')
trace=read('training_trace_INDEPENDENT.json')
context=read('training_CONTEXT.json')
history_a15=read('training_A_20261015_history.json')
result_c13=read('training_C_20261013_RESULT.json')
names=['training_COMPARISON.json','training_scores_INDEPENDENT.json','training_trace_INDEPENDENT.json','training_CONTEXT.json','FROZEN_PROTOCOL.json','READY.json','READINESS_VISUAL_REVIEW.json','training_A_20261015_history.json','training_C_20261013_RESULT.json']
report={
 'status':'CROSS_REVIEW_COMPLETE_NO_BLOCKING_CONTRADICTION_FOUND',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'reviewer':'Codex subagent roi_training_protocol',
 'method':'Read completed comparison, independent score recount, independent saved-trace report, frozen context and selected saved history. No verifier rerun and no model/array/checkpoint load in this cross-review.',
 'provenance':{str(U/n):digest(n) for n in names},
 'authorship_and_independence':{
  'training_driver_core_author':'Codex roi_training_protocol; subsequently cross-reviewed by roi_missing_protocol',
  'score_trace_verifier_author':'Codex roi_training_protocol; distinct calculation code, not training_core aggregation; subsequently independently statically reviewed',
  'execution':'Root orchestrated the training and final verifier commands; this reviewer did not rerun them.',
  'independence_limit':'Algorithmically separate saved-evidence checks, not a claim of completely independent authorship or an external third-party replication.',
 },
 'saved_report_status':{
  'implementation_comparison':comparison['status'],
  'scores_independent':scores['status'],
  'trace_independent':trace['status'],
  'interpretation':'The comparison file retains its creation-time PENDING_INDEPENDENT_RECOUNT status. The two later PASS reports resolve that workflow stage; the immutable comparison was not rewritten.',
 },
 'frozen_conditions_as_recorded':{
  'train_clips':899,'train_person_ids':[1,2,3,4,5,6,7,8,9],
  'development_clips':300,'development_person_ids':[11,12,13],
  'classes':100,'parameter_count':result_c13['parameter_count'],
  'model':context['model'],'training':context['training'],'seeds':context['seeds'],
  'arms':{'A':'new full-frame native observations','B':'new full-frame observations with fixed upper-body mask','C':'masked and independently detected sensor-crop observations in the frozen 50:50 alternating schedule'},
  'exposure':'Each original TRAIN clip exactly once per epoch; 899 examples and 29 batches. Original whole/context time schedule alternates each epoch; C spatial view alternates every two epochs with per-clip hash offsets, covering all four combinations over four epochs.',
  'selection':'0.5 masked DEV CE + 0.5 sensor DEV CE; strict improvement >1e-6, patience12, max80. Input-failed case remains in denominator300 and receives ln(100) CE penalty.',
  'actual_input_failures':context['fixed_input_failures'],
  'user_data_used':False,'heldout_final_test_used':False,
 },
 'epoch_and_trace_summary':{
  run:{'epochs_completed':v['history']['epochs'],'selected_epoch':v['history']['selected_epoch'],'selected_ce':v['history']['selected_ce'],'steps':v['history']['steps'],'source_exposure_count_as_recorded':v['history']['expected_source_exposures'],'strict_best_and_early_stop_match':v['history']['strict_best_and_early_stop_match'],'selected_tensors_equal_committed_best':v['checkpoint_link']['selected_tensors_equal_committed_best'],'saved_scores_equal_committed_best':v['checkpoint_link']['saved_scores_equal_committed_best']}
  for run,v in trace['per_run'].items()
 },
 'score_summary_reused_without_recount':scores['three_seed_summaries'],
 'M0_reference':{
  view:{k:v[k] for k in ('total','top1_correct','top5_correct','top1_percent','top5_percent','input_failures','by_signer')}
  for view,v in scores['M0'].items()
 },
 'effect_interpretation_ko':[
  '주된 방법 비교는 같은 sensor 평가 보기에서 B 97.56% → C 98.67%, 평균 +1.11%p이다. 세 seed의 정답 수 차이는 0,+5,+5건이므로 2개 개선·1개 동률이며 모두 개선한 것은 아니다.',
  'C의 sensor 3seed 정답 수는 295,299,294/300, 평균296/300이다. 3seed 평균은 단일 모델의 실사용 정확도가 아니며 동일한 300개 개발 자료를 반복한 결과다.',
  'M0 sensor 159/300=53.00% → C sensor 평균98.67%는 전체 입력·추출기·학습 파이프라인 개선 +45.67%p이다. 이를 sensor 혼합 자체의 효과나 모델 구조 변경 효과로 표현하면 안 된다.',
  '기존 M0 원본입력 96.67%와 C masked 평균98.44%의 +1.78%p는 원래 전체 시스템과 새 ROI 시스템의 비교다. C를 원본 무마스크 입력으로 평가한 결과가 아니다.',
  'C가 모든 지표에서 B를 이기지는 않았다. masked Top1은 seed별 -2,+6,-1건, sensor Top1은 0,+5,+5건이다. C Top5 평균99.67%는 B99.78%보다 약0.11%p 낮다.',
  '촬영자13의 masked 평균은 M0 100% → C99.33%로 약0.67%p 회귀했다. 사전 허용범위 이내이지만 일부 회귀가 없었다고 보고하면 안 된다.',
  '기존 모델에 대한 사용자 실사용 오류가 해결됐다는 결론은 아직 낼 수 없다. 이번 sensor 보기는 공개 영상의 오프라인 상반신 절단·재검출 조건이며 실제 웹캠과 다른 새 사람 시험이 아니다.',
 ],
 'A_arm_caution':{
  'masked_top1_mean_percent':29.777777777777775,'sensor_top1_mean_percent':33.22222222222222,
  'A_20261015_first_history_row':history_a15[0],
  'A_20261015_last_history_row':history_a15[-1],
  'interpretation_ko':'A는 전체 관측으로 학습하고 절단된 두 DEV 보기의 CE로 선택했다. seed20261015는 epoch1이 선택되어 sensor14/300이고, epoch13은 sensor117/300이지만 가중 CE가 개선되지 않아 선택되지 않았다. 동결 규칙에 부합하며 임의로 마지막 epoch나 최고Top1 epoch를 대신 택하면 안 된다. A의 낮은 절단조건 점수는 입력분포 불일치와 CE 선택 영향을 포함하며, 원본 입력 자체의 정확도 저하 또는 학습 실패의 확정 증거는 아니다. 근본 원인은 이번 교차검토로 확정하지 않는다.',
 },
 'preregistered_progression':scores['progression'],
 'gate_interpretation_ko':'사전 11개 개발 검토 기준을 통과했다. 이는 다음 사용자 결과 검토로 진행할 자격이며, 통계적 일반화 증명·최종시험 통과·운영 자동채택을 뜻하지 않는다.',
 'seed_policy':{
  'highest_scoring_seed_selection_allowed':False,
  'predefined_future_deployment_seed':comparison['predefined_future_deployment_seed'],
  'predefined_C_20261013':{'selected_epoch':77,'masked_top1_correct':294,'masked_top1_percent':98.0,'sensor_top1_correct':295,'sensor_top1_percent':98.33333333333333,'top5_both_views_correct':297,'top5_both_views_percent':99.0},
  'interpretation_ko':'최고 점수인 seed20261014(299/300)를 사후 선택할 수 없다. 향후 사용은 사전 지정20261013이며 현재 운영 교체는 하지 않았다. 이 seed에서 C sensor는 B와295/300 동률, masked는 B보다2건 낮다.',
 },
 'retained_input_risks_ko':[
  '전멸 REAL02 핵심구간 오른손35프레임 미관측과 초대권 받침손 등 sensor 검출·배정 문제는 이번 높은 단어분류 점수로 해결되거나 무효화되지 않는다.',
  '입력검증과 838조건 표본 시각검토 통과는 모든 연속 프레임 및 언어학적 의미보존 보장을 뜻하지 않는다.',
  '관측 누락 그룹은 고정된 관측 패턴이지 한손/양손 수어의 전문가 정답 라벨이 아니다.',
 ],
 'trace_limits_from_existing_verifier':trace['scope_limits'],
 'score_limits_from_existing_verifier':scores['limitations'],
 'review_findings':{
  'blocking_inconsistencies_found':[],
  'conditions_changed':False,'results_reselected':False,'new_verifier_execution':False,
  'new_model_inference':False,'new_training':False,'new_detector':False,
  'production_change':False,'automatic_adoption':False,'generalization_proven':False,
 },
}
out=U/'FINAL_CROSS_REVIEW_training.json'
if out.exists():raise RuntimeError('Do not overwrite an existing review')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(str(out))
