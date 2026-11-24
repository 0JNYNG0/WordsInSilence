"""Independent arithmetic audit of completed files; never imports study code.

Run only after root confirms all six runs, scores, comparison and diagnostics
are complete. No inference, optimizer, model construction, or score-file writes.
"""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import itertools
import json
import math
import traceback

import numpy as np

U = Path(__file__).resolve().parents[1]
OUT = U / 'review' / 'TRAIN_RESULTS_VERIFICATION.json'
OLD = U.parent / 'word100_stage_feedback_cycle_20261031'
CHECKS = Counter()
PINS = {}
LIMITS = [
    'Same Codex separate arithmetic review, not independent human KSL truth.',
    'Stored probabilities are recomputed into counts only; no model inference or re-scoring command is run.',
    'Logs and frozen code establish recorded exposure and final-checkpoint policy; they are not a separate replay of optimization.',
    'Checkpoint/probability hashes and shapes are matched to saved completion records; generation of probabilities from weights is not replayed.',
    'Agreement on selected exact annotation windows does not evaluate full-video segmentation, sign-word correctness, or educational correctness.',
    'New-feature expression evaluation clips have prior TCN TRAIN/native/reference history; they are not universally unseen public data.',
]


def require(ok, label):
    if not bool(ok):
        raise AssertionError(label)
    CHECKS[label.split(':')[0].split('/')[0]] += 1


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    PINS[str(path.resolve())] = value
    return value


def read(path):
    path = Path(path)
    sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def same(actual, expected, label, atol=2e-7, rtol=2e-6):
    """Compare the expected schema, retaining extra implementation metadata."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ':dict')
        for key, value in expected.items():
            require(key in actual, label + ':key ' + key)
            same(actual[key], value, label + '/' + key, atol, rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ':length')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label + '/' + str(i), atol, rtol)
    elif isinstance(expected, (float, np.floating)):
        require(actual is not None and math.isfinite(float(actual)) and
                math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), label + ':number')
    else:
        require(actual == expected, label + ':value')


def indexed(rows, keys, label):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(result) == len(rows), label + ':unique')
    return result


def arithmetic(prob, truth, known, names):
    """Independent per-cell confusion table, true row / predicted column."""
    require(prob.shape == truth.shape == known.shape, 'metric_shape')
    output = []
    for column, name in enumerate(names):
        table = [[0, 0], [0, 0]]
        for row in range(len(prob)):
            if known[row, column]:
                true_class = int(truth[row, column])
                pred_class = int(float(prob[row, column]) >= 0.5)
                table[true_class][pred_class] += 1
        tn, fp = table[0]
        fn, tp = table[1]
        total = tn + fp + fn + tp
        balanced = ((tp / (tp + fn) + tn / (tn + fp)) / 2
                    if tp + fn > 0 and tn + fp > 0 else None)
        output.append(dict(target=name, known=total, tp=tp, tn=tn, fp=fp, fn=fn,
                           accuracy=(tp + tn) / total if total else None,
                           balanced_accuracy=balanced))
    balanced = [r['balanced_accuracy'] for r in output if r['balanced_accuracy'] is not None]
    total = sum(r['known'] for r in output)
    return dict(targets=output,
                macro_balanced_accuracy=sum(balanced) / len(balanced) if balanced else None,
                balanced_supported_targets=len(balanced), known_cells=total,
                known_cell_accuracy=sum(r['tp'] + r['tn'] for r in output) / total if total else None)


def group_summary(rows):
    values = [r['macro_balanced_accuracy'] for r in rows if r['macro_balanced_accuracy'] is not None]
    require(len({r['known_cells'] for r in rows}) == 1, 'summary_fixed_known_denominator')
    require(len({r['balanced_supported_targets'] for r in rows}) == 1, 'summary_fixed_supported_targets')
    return dict(mean=sum(values) / len(values) if values else None,
                range=[min(values), max(values)] if values else None,
                known_cells=rows[0]['known_cells'], balanced_supported_targets=rows[0]['balanced_supported_targets'])


def load_array(path):
    sha(path)
    return np.load(path, allow_pickle=False)


def main():
    required = ['DATASET.json', 'TRAIN_FROZEN.json', 'NORMALIZATION.json', 'TRAINING.json',
                'SCORES.json', 'PREVIOUS_STORED_COMPARISON.json', 'DIAGNOSTICS.json']
    for name in required:
        require((U / name).is_file(), 'completed_input:' + name)
    require(not (U / 'TRAIN_RUNNING.json').exists(), 'no_active_training_marker')
    info, plan, norm, training, score, prior_compare, diagnostic = [read(U / n) for n in required]
    require(training['status'] == 'COMPLETE_SIX_RUNS', 'six_runs_complete_status')
    require(score['status'] == 'ONE_FROZEN_COMPARISON_COMPLETE', 'score_complete_status')
    require(prior_compare['status'] == 'NEW_RGB_LABEL_COMPARISON_ON_STORED_OUTPUTS', 'prior_compare_complete_status')
    require(training['backbone_updates'] == 0, 'frozen_backbone')
    same(plan, dict(conditions=['pose_only', 'pose_plus_signrep'], seeds=[42, 43, 44],
                    epochs=160, lr=.001, weight_decay=.01, threshold=.5), 'fixed_plan')
    require('final160' in plan['selection'] and 'no evaluation tuning' in plan['selection'], 'final_checkpoint_policy')
    for mapping in [info['pins'], plan['pins'], plan['source_pins'], plan['protected']]:
        for path, digest in mapping.items():
            require(sha(path) == digest, 'frozen_hash:' + path)
    same(plan['source_pins'], info['pins'], 'source_pins_exact')
    selection = read(U / 'SELECTION.json')
    same(plan['protected'], selection['protected'], 'protected_manifest')
    require(len(plan['protected']) == 32, 'protected32')

    rows, names, columns = info['rows'], info['selected_targets'], info['selected_indices']
    same(names, ['R_index_only','R_open_four','L_open_four','R_higher','L_higher',
                 'R_motion_down','R_motion_still','R_motion_mixed','L_motion_down','L_motion_still'],
         'pre_score_ten_supported_targets')
    require(len(rows) == 168 and len({r['sample_id'] for r in rows}) == 168, '168_unique_windows')
    same(names, [info['targets'][c] for c in columns], 'selected_column_names')
    same(plan['targets'], names, 'frozen_target_names')
    roles = sorted({r['role'] for r in rows})
    same(dict(Counter(r['role'] for r in rows)), {'fit':108, 'development':36,
         'new_feature_expression_evaluation':24}, 'split_windows')
    role_indices = {role:[i for i,r in enumerate(rows) if r['role'] == role] for role in roles}
    for role, people in [('fit', {1,2,3}), ('development', {4}), ('new_feature_expression_evaluation', {7,8})]:
        require({rows[i]['person'] for i in role_indices[role]} == people, 'person_split:' + role)
    fit_ids = [r['sample_id'] for r in rows if r['role'] == 'fit']
    same(plan['fit_windows'], fit_ids, '108_frozen_fit_ids')
    role_clips = {role:{rows[i]['clip_id'] for i in ix} for role,ix in role_indices.items()}
    for a,b in itertools.combinations(roles, 2):
        require(not role_clips[a] & role_clips[b], 'clip_split_disjoint:' + a + '/' + b)
    train_words = {r['word_id'] for r in rows if r['role'] == 'fit'}
    dev_words = {r['word_id'] for r in rows if r['role'] == 'development'}
    new_words = {r['word_id'] for r in rows if r['role'] == 'new_feature_expression_evaluation'}
    require(train_words == dev_words and len(train_words) == 12 and len(new_words) == 4 and not train_words & new_words, 'word_split')
    truth = np.asarray([r['y'] for r in rows], dtype=np.float32)[:, columns]
    known = np.asarray([r['target_mask'] for r in rows], dtype=bool)[:, columns]
    require(np.isin(truth[known], [0,1]).all(), 'binary_known_truth')
    fit = role_indices['fit']
    counts = np.asarray([[(known[fit,j] & (truth[fit,j] == value)).sum() for j in range(len(names))]
                         for value in (0,1)], dtype=np.float32)
    require((counts > 0).all(), 'fit_both_classes')
    weights = counts.sum(axis=0)[None,:] / (2 * counts)

    inputs = read(U / 'INPUTS.json')['rows']
    input_map = {r['alias']:r for r in inputs}
    require(len(input_map) == 56, '56_unique_input_aliases')
    require(len({r['source_sha256'] for r in inputs}) == 56, '56_source_sha_unique')
    require(sum(bool(r['reused_input']) for r in inputs) == 48, '48_reused_inputs')
    cache, window_data = {}, []
    for item in inputs:
        for pathkey, hashkey in [('input','input_sha256'), ('source','source_sha256'),
                                 ('record','record_sha256'), ('observation','observation_sha256')]:
            require(sha(item[pathkey]) == item[hashkey], 'input_source_hash:' + item['alias'] + '/' + pathkey)
    for row in rows:
        source = input_map[row['alias']]
        for key in ['clip_id','word_id','person','role','input','input_sha256']:
            same(row[key], source[key], 'dataset_input_identity/' + key)
        if row['alias'] not in cache:
            with load_array(row['input']) as a:
                cache[row['alias']] = {key:a[key] for key in ['frame_id','timestamp_us','x','mask','rgb']}
        a = cache[row['alias']]
        lookup = {int(f):i for i,f in enumerate(a['frame_id'])}
        positions = [lookup[int(f)] for f in row['frame_ids']]
        require(len(positions) == 13 and positions == list(range(positions[0], positions[0]+13)), 'native_contiguous13')
        same(row['center_frame'], row['frame_ids'][6], 'exact_center_frame')
        same(row['timestamps_us'], a['timestamp_us'][positions].tolist(), 'exact_original_pts')
        same(row['center_timestamp_us'], row['timestamps_us'][6], 'center_pts')
        window_data.append((a['x'][positions], a['mask'][positions] & np.isfinite(a['x'][positions]), a['rgb'][positions]))
    fit_x = np.concatenate([window_data[i][0] for i in fit])
    fit_m = np.concatenate([window_data[i][1] for i in fit])
    fit_z = np.concatenate([window_data[i][2] for i in fit])
    pose_mean, pose_std = [], []
    for j in range(69):
        values = fit_x[:,j][fit_m[:,j]]
        pose_mean.append(float(values.mean()) if len(values) else 0.)
        pose_std.append(max(float(values.std()), .001) if len(values) else 1.)
    same(norm, dict(pose_mean=pose_mean, pose_std=pose_std,
         rgb_mean=fit_z.mean(axis=0).tolist(), rgb_std=np.maximum(fit_z.std(axis=0), .001).tolist()),
         'fit_only_normalization', atol=4e-6)
    require(fit_x.shape[0] == 108*13, 'normalization_occurrence_denominator1404')
    require('fit13-frame windows only' in norm['scope'], 'normalization_scope')

    # Deserialize tensor-only checkpoint state; do not import or instantiate Head.
    import torch
    expected_shapes = {'pose.weight':(32,138),'pose.bias':(32,), 'aux.weight':(32,768),'aux.bias':(32,),
                       'c1.weight':(48,64,3),'c1.bias':(48,), 'c2.weight':(32,48,3),'c2.bias':(32,),
                       'out.weight':(len(names),128),'out.bias':(len(names),)}
    parameter_count = sum(math.prod(shape) for shape in expected_shapes.values())
    require(parameter_count == 44250, 'fixed44250_parameters')
    run_keys = list(itertools.product(plan['conditions'], plan['seeds']))
    registered = indexed(training['runs'], ['condition','seed'], 'training_run_registry')
    require(set(registered) == set(run_keys), 'exact_six_runs')
    probabilities, completions = {}, {}
    for condition, seed in run_keys:
        folder = U / 'runs' / f'{condition}_{seed}'
        complete = read(folder / 'COMPLETE.json')
        completions[condition,seed] = complete
        same(complete, dict(condition=condition,seed=seed,epochs=160,selected_epoch=160,
             parameters=parameter_count,class_counts=counts.tolist(),class_weights=weights.tolist()), 'run_contract')
        for name, field in [('head.pt','checkpoint_sha256'),('probabilities.npy','probabilities_sha256'),('EXPOSURES.jsonl','exposures_sha256')]:
            require(sha(folder/name) == complete[field], 'run_artifact_hash:' + name)
        state = torch.load(folder/'head.pt', map_location='cpu', weights_only=True)
        require(set(state) == set(expected_shapes), 'checkpoint_only_head_tensors')
        for name, shape in expected_shapes.items():
            require(tuple(state[name].shape) == shape and bool(torch.isfinite(state[name]).all()), 'checkpoint_tensor:' + name)
        require(sum(v.numel() for v in state.values()) == parameter_count, 'checkpoint_parameters')
        require(len(complete['losses']) == 160 and all(math.isfinite(v) and v >= 0 for v in complete['losses']), '160_finite_losses')
        exposures = [json.loads(line) for line in (folder/'EXPOSURES.jsonl').read_text(encoding='utf8').splitlines()]
        require(len(exposures) == 160, '160_exposure_rows')
        for epoch, exposure in enumerate(exposures, 1):
            same(exposure, dict(epoch=epoch,sample_ids=fit_ids,loss=complete['losses'][epoch-1]), 'epoch_fit_only_exposure')
        require(sum(len(e['sample_ids']) for e in exposures) == 17280, '17280_window_exposures_each_run')
        require(math.isfinite(complete['training_seconds']) and complete['training_seconds'] > 0, 'positive_training_seconds')
        same(registered[condition,seed], {k:complete[k] for k in ['condition','seed','parameters','training_seconds']}, 'training_registry_identity')
        pred = load_array(folder/'probabilities.npy')
        require(pred.shape == (168,len(names)) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(), 'probability_shape_range')
        probabilities[condition,seed] = pred

    stored_scores = indexed(score['rows'], ['condition','seed','role'], 'score_rows')
    computed_scores = {}
    for (condition,seed), pred in probabilities.items():
        for role, ix in role_indices.items():
            result = dict(condition=condition,seed=seed,role=role,windows=len(ix),
                          **arithmetic(pred[ix],truth[ix],known[ix],names))
            key = condition,seed,role
            require(key in stored_scores, 'score_key')
            same(stored_scores[key], result, 'independent_score_arithmetic')
            computed_scores[key] = result
    require(set(stored_scores) == set(computed_scores), 'all18_score_rows')
    summaries = indexed(score['summary'], ['condition','role'], 'score_summary')
    require(set(summaries) == set(itertools.product(plan['conditions'],roles)), 'all6_score_summaries')
    for key, stored in summaries.items():
        same(stored, group_summary([computed_scores[key[0],seed,key[1]] for seed in plan['seeds']]), 'independent_seed_mean_range')
    same(score['targets'], names, 'score_target_order')
    require(score['not_sign_word_or_educational_accuracy'] is True, 'nonlinguistic_score_scope')

    verify_previous(info, prior_compare, probabilities)
    verify_diagnostics(rows, names, truth, known, plan, probabilities, completions, diagnostic, inputs)
    return dict(status='PASS_STORED_ARITHMETIC_AND_CONTRACTS', reviewer='same Codex independent arithmetic implementation',
                at=datetime.datetime.now().isoformat(), script_sha256=sha(__file__),
                checks=dict(CHECKS), artifact_sha256=PINS, limitations=LIMITS,
                counts=dict(windows=168,clips=56,fit_windows=108,development_windows=36,new_feature_evaluation_windows=24,
                            conditions=2,seeds=3,runs=6,epochs_each=160,window_exposures_each_run=17280,
                            window_exposures_all_runs=103680,targets=len(names),parameters_each=parameter_count,
                            protected_files=32),
                verified_summaries=list(summaries.values()),
                no_inference=True,no_training=True,no_production_or_prior_result_write=True,
                optimization_not_replayed=True,linguistic_truth_verified=False)


def verify_previous(info, stored, probabilities):
    prior = read(OLD/'LEARNING_INPUTS.json')
    prior_by_clip = {r['clip_id']:(i,r) for i,r in enumerate(prior['rows'])}
    require(len(prior_by_clip) == len(prior['rows']), 'prior_unique_clip_ids')
    shared = [name for name in ['R_index_only','R_open_four','L_open_four'] if name in info['selected_targets']]
    same(stored['targets'], shared, 'prior_shared_targets')
    selected = [(i,r) for i,r in enumerate(info['rows']) if r['clip_id'] in prior_by_clip]
    require(len(selected) == 144 and len({r['clip_id'] for _,r in selected}) == 48, 'prior_exact48clips144centers')
    yy = np.asarray([[r['y'][info['targets'].index(t)] for t in shared] for _,r in selected])
    mm = np.asarray([[r['target_mask'][info['targets'].index(t)] for t in shared] for _,r in selected], bool)
    frame_maps, points = {}, []
    for _, r in selected:
        clip, oldrow = prior_by_clip[r['clip_id']]
        if r['clip_id'] not in frame_maps:
            with load_array(oldrow['input']) as data:
                require(len(set(data['frame_id'].tolist())) == len(data['frame_id']), 'prior_unique_frame_ids')
                frame_maps[r['clip_id']] = {int(f):j for j,f in enumerate(data['frame_id'])}
        points.append((clip, frame_maps[r['clip_id']][r['center_frame']]))
    expected = {}
    for model, old, condition in [('previous_pose',True,'pose_capacity_matched'),('previous_pose_signrep',True,'pose_plus_signrep'),
                                   ('new_pose',False,'pose_only'),('new_pose_signrep',False,'pose_plus_signrep')]:
        for seed in [42,43,44]:
            folder = (OLD if old else U)/'runs'/f'{condition}_{seed}'
            path = folder/'probabilities.npy'
            require(stored['pins'][str(path)] == sha(path), 'prior_comparison_probability_pin')
            manifest = read(folder/('RUN.json' if old else 'COMPLETE.json'))
            require(sha(path) == manifest['prediction_sha256' if old else 'probabilities_sha256'], 'prior_probability_original_manifest')
            if old:
                array = load_array(path)
                require(array.ndim == 3 and array.shape[0] == len(prior['rows']) and array.shape[2] == len(prior['targets']), 'prior_probability_axes')
                require(np.isfinite(array).all() and ((array>=0)&(array<=1)).all(), 'prior_probability_range')
                oldcols = [prior['targets'].index(t) for t in shared]
                pp = np.asarray([[array[c,t,j] for j in oldcols] for c,t in points])
            else:
                newcols = [info['selected_targets'].index(t) for t in shared]
                pp = probabilities[condition,seed][[i for i,_ in selected]][:,newcols]
            for role in ['fit','development']:
                ix = [j for j,(_,r) in enumerate(selected) if r['role'] == role]
                expected[model,seed,role] = arithmetic(pp[ix],yy[ix],mm[ix],shared)
    actual = indexed(stored['rows'], ['model','seed','role'], 'prior_comparison_rows')
    require(set(actual) == set(expected), '24_prior_comparison_rows')
    for key, result in expected.items():
        same(actual[key], result, 'prior_exact_frame_confusion_arithmetic')
    summaries = indexed(stored['summary'], ['model','role'], 'prior_comparison_summaries')
    require(len(summaries) == 8, '8_prior_comparison_summaries')
    for (model,role), row in summaries.items():
        same(row, group_summary([expected[model,seed,role] for seed in [42,43,44]]), 'prior_mean_range')
    require(stored['prior_probabilities_inference_count'] == 0 and stored['previous_result_files_overwritten'] is False, 'prior_readonly_no_inference_contract')


def verify_diagnostics(rows,names,truth,known,plan,probabilities,completions,stored,inputs):
    details = indexed(stored['details'], ['condition','sample_id'], 'diagnostic_detail_rows')
    require(len(details) == 336, 'all336_diagnostic_rows')
    for condition in plan['conditions']:
        for i,row in enumerate(rows):
            got = details[condition,row['sample_id']]
            same(got['role'], row['role'], 'diagnostic_role')
            require(set(got['targets']) == set(names), 'diagnostic_target_set')
            for j,target in enumerate(names):
                pp = [float(probabilities[condition,seed][i,j]) for seed in plan['seeds']]
                same(got['targets'][target], dict(known=bool(known[i,j]),truth=int(truth[i,j]) if known[i,j] else None,
                     probabilities=pp,all_three_wrong=bool(known[i,j] and all(int(p>=.5)!=truth[i,j] for p in pp))), 'diagnostic_detail_arithmetic')
    subgroups = indexed(stored['subgroups'], ['condition','seed','role','group_key','group_value'], 'diagnostic_subgroups')
    expected_keys = set()
    for (condition,seed), pred in probabilities.items():
        for key in ['person','word_id']:
            for value in {r[key] for r in rows}:
                for role in {r['role'] for r in rows}:
                    ix = [i for i,r in enumerate(rows) if r[key] == value and r['role'] == role]
                    if not ix:
                        continue
                    k = condition,seed,role,key,value
                    expected_keys.add(k)
                    require(k in subgroups, 'diagnostic_subgroup_key')
                    same(subgroups[k], dict(windows=len(ix),**arithmetic(pred[ix],truth[ix],known[ix],names)), 'diagnostic_subgroup_arithmetic')
    require(set(subgroups) == expected_keys, 'all_diagnostic_subgroup_rows')
    contradictions = indexed(stored['contradictions'], ['condition','seed','role'], 'diagnostic_contradictions')
    require(len(contradictions) == 18, '18_contradiction_rows')
    for (condition,seed), pred in probabilities.items():
        for role in {r['role'] for r in rows}:
            ix = [i for i,r in enumerate(rows) if r['role']==role]
            groups = []
            for side in ['R','L']:
                for kind in ['shape','motion']:
                    jj = [j for j,t in enumerate(names) if t.startswith(side+'_') and
                          (t[len(side)+1:] in ['index_only','open_four','closed_four'] if kind=='shape' else t.startswith(side+'_motion_'))]
                    if len(jj)>1:
                        groups.append(dict(side=side,kind=kind,targets=[names[j] for j in jj],windows=len(ix),
                                           multi_positive_windows=sum(sum(float(pred[i,j])>=.5 for j in jj)>1 for i in ix)))
            jj=[j for j,t in enumerate(names) if t in ['R_higher','L_higher']]
            if len(jj)==2:
                groups.append(dict(side='both',kind='bilateral_height',targets=[names[j] for j in jj],windows=len(ix),
                                   multi_positive_windows=sum(sum(float(pred[i,j])>=.5 for j in jj)>1 for i in ix)))
            unknown=[(i,j) for i in ix for j in range(len(names)) if not known[i,j]]
            same(contradictions[condition,seed,role],dict(exclusive_groups=groups,unknown_label_cells=len(unknown),
                 unknown_label_cells_predicted_positive=sum(float(pred[i,j])>=.5 for i,j in unknown),unknown_not_scored_as_true=True), 'diagnostic_exclusivity_unknown_arithmetic')
    latencies = indexed(stored['latencies'], ['condition','seed'], 'diagnostic_latencies')
    require(set(latencies) == {(c,42) for c in plan['conditions']}, 'two_final42_latency_records')
    for (condition,seed), row in latencies.items():
        same(row, dict(samples=100,threads=4,checkpoint_sha256=completions[condition,seed]['checkpoint_sha256'],
                       unit='milliseconds per3 prepared13-frame windows'), 'latency_contract')
        require(math.isfinite(row['median']) and math.isfinite(row['p95']) and 0<row['median']<=row['p95'], 'latency_summary_bounds')
        samples = next((row[k] for k in ['raw_milliseconds','raw_ms','samples_ms','timings_ms'] if k in row), None)
        if samples is None:
            LIMITS.append(condition + ': raw 100 latency samples unavailable; median/p95 arithmetic not independently reproducible, only summary bounds/scope/checkpoint verified.')
        else:
            require(len(samples)==100 and all(math.isfinite(x) and x>0 for x in samples), 'latency_raw100')
            same(row,dict(median=float(np.median(samples)),p95=float(np.percentile(samples,95))), 'latency_summary_arithmetic')
    new = [r for r in inputs if not r['reused_input']]
    same(stored['encoder_preparation'],dict(new_clips=8,total_rgb_seconds=sum(r['rgb_seconds'] for r in new),
         per_clip_seconds=[r['rgb_seconds'] for r in new]), 'encoder_saved_time_arithmetic')
    text=' '.join(stored['limitations']).lower()
    for token in ['excludes video decode','not recording-end-to-feedback','unknown','exact annotation windows','16-frame','noncausal']:
        require(token in text, 'diagnostic_scope:' + token)


if __name__ == '__main__':
    if OUT.exists():
        raise SystemExit('Verification output already exists; inspect it, do not duplicate execution.')
    try:
        result = main()
    except BaseException:
        result = dict(status='FAIL_FIRST_EXECUTION_PRESERVED', at=datetime.datetime.now().isoformat(),
                      error=traceback.format_exc(), checks=dict(CHECKS), artifact_sha256=PINS,
                      no_inference=True,no_training=True)
        with OUT.open('x',encoding='utf8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        raise
    with OUT.open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','counts','limitations']},ensure_ascii=False))
