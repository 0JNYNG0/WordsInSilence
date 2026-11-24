"""Interpret saved aggregate results only; no prediction, model load or re-scoring."""
from pathlib import Path
import hashlib
import json
import statistics

U = Path(__file__).resolve().parent.parent
def read(name): return json.loads((U/name).read_text(encoding='utf8'))
def mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None
def digest(name): return hashlib.sha256((U/name).read_bytes()).hexdigest()
S = read('SCORES.json')
P = read('PREVIOUS_STORED_COMPARISON.json')
D = read('DIAGNOSTICS.json')
roles = ['development','new_feature_expression_evaluation']
conds = ['pose_only','pose_plus_signrep']
seeds = [42,43,44]
role_results = []
target_results = []
family_results = []
for role in roles:
    sm = {c:next(r for r in S['summary'] if r['role']==role and r['condition']==c) for c in conds}
    paired = []
    for seed in seeds:
        rr = {c:next(r for r in S['rows'] if r['role']==role and r['condition']==c and r['seed']==seed) for c in conds}
        paired.append(dict(seed=seed,pose=rr[conds[0]]['macro_balanced_accuracy'],rgb=rr[conds[1]]['macro_balanced_accuracy'],
                           delta_percentage_points=100*(rr[conds[1]]['macro_balanced_accuracy']-rr[conds[0]]['macro_balanced_accuracy'])))
    role_results.append(dict(role=role,stored_summaries=sm,delta_percentage_points=100*(sm[conds[1]]['mean']-sm[conds[0]]['mean']),
                             paired_seed_results=paired,
                             mean_known_cell_accuracy={c:mean([r['known_cell_accuracy'] for r in S['rows'] if r['role']==role and r['condition']==c]) for c in conds}))
    for t in S['targets']:
        rr = {c:[next(tg for tg in r['targets'] if tg['target']==t) for r in S['rows'] if r['role']==role and r['condition']==c] for c in conds}
        first=rr[conds[0]][0]
        values={c:[r['balanced_accuracy'] for r in rr[c]] for c in conds}
        means={c:mean(values[c]) for c in conds}
        target_results.append(dict(role=role,target=t,known=first['known'],positive=first['tp']+first['fn'],negative=first['tn']+first['fp'],
                                  seed_order=seeds,balanced_accuracy_by_seed=values,balanced_accuracy_means=means,
                                  delta_percentage_points=100*(means[conds[1]]-means[conds[0]]) if means[conds[0]] is not None else None,
                                  saved_confusions=rr))
    for family in ['shape','height','motion']:
        chosen=[r for r in target_results if r['role']==role and
                (('_motion_' in r['target']) if family=='motion' else
                 (r['target'].endswith('higher') if family=='height' else r['target'] in P['targets'])) and
                r['balanced_accuracy_means'][conds[0]] is not None]
        m={c:mean([r['balanced_accuracy_means'][c] for r in chosen]) for c in conds}
        family_results.append(dict(role=role,family=family,targets=[r['target'] for r in chosen],means=m,
                                   delta_percentage_points=100*(m[conds[1]]-m[conds[0]])))

old_target_results=[]
for t in P['targets']:
    means={model:mean([tg['balanced_accuracy'] for r in P['rows'] if r['role']=='development' and r['model']==model
                      for tg in r['targets'] if tg['target']==t]) for model in ['previous_pose_signrep','new_pose_signrep']}
    old_target_results.append(dict(target=t,means=means,delta_percentage_points=100*(means['new_pose_signrep']-means['previous_pose_signrep'])))
old_summaries={r['model']:r for r in P['summary'] if r['role']=='development'}
old_seed_deltas=[]
for seed in seeds:
    v={model:next(r['macro_balanced_accuracy'] for r in P['rows'] if r['role']=='development' and r['seed']==seed and r['model']==model)
       for model in ['previous_pose_signrep','new_pose_signrep']}
    old_seed_deltas.append(dict(seed=seed,**v,delta_percentage_points=100*(v['new_pose_signrep']-v['previous_pose_signrep'])))

subgroups=[]
keys=sorted({(r['role'],r['group_key'],str(r['group_value'])) for r in D['subgroups'] if r['role'] in roles})
for role,key,value in keys:
    grouped={c:[r for r in D['subgroups'] if r['role']==role and r['group_key']==key and str(r['group_value'])==value and r['condition']==c] for c in conds}
    means={c:mean([r['macro_balanced_accuracy'] for r in grouped[c]]) for c in conds}
    r=grouped[conds[0]][0]
    subgroups.append(dict(role=role,key=key,value=value,windows=r['windows'],known_cells=r['known_cells'],balanced_supported_targets=r['balanced_supported_targets'],
                          means=means,delta_percentage_points=100*(means[conds[1]]-means[conds[0]]) if means[conds[0]] is not None else None,
                          caution='Within-subgroup descriptive comparison only. Target support and class counts differ across subgroups; do not average these to reconstruct the pooled macro.'))

persistent=[]
for condition in conds:
    for role in roles:
        details=[r for r in D['details'] if r['condition']==condition and r['role']==role]
        cells=[dict(sample_id=r['sample_id'],target=t,truth=v['truth']) for r in details for t,v in r['targets'].items() if v['all_three_wrong']]
        persistent.append(dict(condition=condition,role=role,known_cells_wrong_in_all_three_seeds=len(cells),cells=cells,
                               unit='window-target cells, not unique clips or three independent observations'))

out=dict(
    status='COMPLETED_SAVED_RESULT_INTERPRETATION_NO_ADOPTION',
    reviewer='Codex subagent A; same system, not independent human KSL expert assessment',
    main_conclusion_ko='개발 묶음의 이동 특징에는 개선이 있지만 추가 표현과 기존 손모양 비교에는 회귀가 있어 전반적 향상이나 운영 채택 근거로 삼을 수 없다.',
    production_adoption=False,
    sources={n:digest(n) for n in ['SCORES.json','PREVIOUS_STORED_COMPARISON.json','DIAGNOSTICS.json']},
    role_results=role_results,target_results=target_results,family_decomposition=family_results,
    previous_three_static_comparison=dict(stored_development_summaries=old_summaries,per_target=old_target_results,paired_seed_results=old_seed_deltas,
        rgb_delta_percentage_points=100*(old_summaries['new_pose_signrep']['mean']-old_summaries['previous_pose_signrep']['mean']),
        interpretation_ko='같은 개발12영상 중앙프레임·같은 새RGB라벨78셀·같은3특징으로 이전 저장확률과 비교한 결과다. 예전 발표점수를 바꾸거나 새표현 일반화를 비교한 것이 아니다. R_open_four는 이전10양성을 모두 맞혔으나 이번에는 매 seed1개씩 놓쳤다. L_open_four는 양성1개를 이전1seed/이번2seed가 놓쳐 BA 변화가 크게 난다. 이것은 산술적 기여 설명이며 학습 회귀의 원인 확정은 아니다.'),
    denominator_warning=dict(development=dict(windows=36,known_cells=276,balanced_targets=10),
        additional_expression=dict(windows=24,known_cells=175,balanced_targets=7,
            no_positive_targets=['L_open_four','L_higher','L_motion_still'],negative_only_counts=[14,14,15]),
        interpretation_ko='175개 known cell 모두가7특징 BA에 들어가는 것은 아니다. 양성0인3특징은 confusion과 known-cell accuracy에는 남고 BA에서는 제외된다. DEV10특징과 추가표현7특징의 절대 BA를 직접 비교하거나 차이를 전부 일반화 손실로 해석하면 안 된다. 각role안 두조건 비교의 분모는 동일하다.'),
    rare_positive_fragility=dict(development_L_open_four=dict(positive=1,negative=13,pose_seed_recall=[1,1,1],rgb_seed_recall=[0,0,1]),
        additional_R_open_four=dict(positive=1,negative=14,pose_seed_recall=[1,0,0],rgb_seed_recall=[0,0,0]),
        interpretation_ko='균형정확도에서 양성1개를 놓치면 그특징 BA가50%p 내려갈 수 있다. 작은양성분모는 결과를 무시할 이유가 아니라 불확실성과 추가관측 필요성을 보고할 이유다. seed3개는 그양성을3개로 늘리지 않는다.'),
    subgroup_results=subgroups,
    contradiction_diagnostics=[r for r in D['contradictions'] if r['role'] in roles],
    persistent_errors=persistent,
    normal_and_crossing_scope=dict(
        can_claim=['Saved feature-level performance on the prescribed centre/window labels.','Observed limited motion gains on the consumed development cohort.','Raw inputs, masks and production pipeline remain a separate preservation claim, not an accuracy claim.'],
        cannot_claim=['Normal KSL accepted more reliably.','Natural hand crossing preserved or improved by the new head.','Reduced false correction of signers.','Whole-video stage boundaries, repetitions or linguistic correctness learned.'],
        reason_ko='정상수어/오류수어의 전문가 판정이나 crossing 전용 사전고정 층화라벨·분모가 없다. 원본 관측을 보존한 것과 판정부가 정상 교차를 잘 처리하는 것은 별개다. 새head에는상호배타특징동시양성도남아있고 unknown셀출력을 오답이라고 셀수없다.'),
    overfitting_assessment=dict(possible=True,proved=False,
        evidence_ko='두조건·세seed 모두 fit의825 known셀/10특징에서100%다. 개발과추가표현에서는오류와seed차이가남는다. 적은영상의겹치는창/드문양성/44,250파라미터에서과적합가능성을살필이유는있다.',
        alternative_explanations=['Small and imbalanced positive support','Provisional labels and uncertain boundaries','Different people and expressions/distribution shift','Shared static/motion head and different time support than the prior head','Noncausal16-frame RGB interpolation'],
        no_causal_claim='No ablation or untouched replicated evaluation isolates any of these causes. Do not assert that RGB, architecture, or overfitting caused the regression.'),
    timing_interpretation=dict(stored_latencies=[{k:v for k,v in r.items() if k!='raw_milliseconds'} for r in D['latencies']],
        note_ko='두조건모두동일구조의작은head3준비창시간이다. RGB조건의더작은측정중앙값이RGB추출포함전체속도가더빠르다는뜻은아니다. 디코드·MediaPipe·SignRep·피드백·촬영대기시간은제외되어있다.'),
    diagnostics_exit_note='Parent reports diagnostics serialized DIAGNOSTICS.json, then failed only in final stdout because json was not imported. This review successfully parsed the saved content; guide owns numerical completion verification. No diagnostic rerun was performed.',
    next_single_task=dict(
        recommendation_ko='같은관측정의로희소양성·전이·교차가포함된다음공개주석묶음을준비하고학습용/새평가용을점수전에고정한다.',
        priority_ko='왼손네손가락펼침및기존정지손모양을유지하면서짧은이동을구별할수있는표본지원부터보강한다. 이는특정단어예외나이번오류샘플맞춤수정이아니다.',
        bounded_contract=['Keep the current10 target meanings and frozen results; preserve unsupported targets explicitly.','Choose additional public clips by predefined source coverage, not by whether current model predictions succeed.','Use direct13-frame RGB observation; separate static-centre labels, whole-window motion, crossing/occlusion visibility and unknown.','Fix sufficient positive and negative evaluation support across people/expressions before any next model scores.','Treat the current evaluated8 clips as consumed diagnostic evidence; no checkpoint/threshold/seed selection on them.','Only after that readiness, pre-register one small comparison; no architecture or operating-model replacement in this task.'],
        not_an_execution_authorization=True),
    execution=dict(model_loads=0,new_inference=0,training=0,score_function_calls=0,tuning=0,prior_result_writes=0,
                   arithmetic='Means/differences of already-saved aggregate metrics and saved diagnostic counts only.'),
    limitations=['Same-Codex provisional RGB labels, not human expert KSL ground truth.','56 public clips and168 selected windows are not dense all-video GT.','Additional8 clips have prior public TCN TRAIN lineage, not universally unseen data.','Three-seed ranges do not establish statistical significance.','This is interpretation; independent saved-numeric verification is separate.'])
with (U/'review/RESULT_INTERPRETATION.json').open('x',encoding='utf8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(out['status'])
