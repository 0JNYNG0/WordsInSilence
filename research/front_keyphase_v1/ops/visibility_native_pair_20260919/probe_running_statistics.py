"""Evaluation-only counterfactual: replace running mean/variance, keep trained parameters.

This is an independent cause probe, not a trained or registered candidate.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
import numpy as np
import torch
torch.set_num_threads(2)
import tools_prep_union93_candidate as u
import tools_visibility_experiment as v
from wis_lab.service import neural
from wis_lab.core.observation import Observation

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

if (OPS / 'bn_counterfactual_LOCAL_ONLY.json').exists():
    raise FileExistsError('Counterfactual probe already recorded.')
plan = read(OPS / 'plan.json')
aud = read(OPS / 'native_pair_result_audit.json')
web_plan = read(OPS / 'web_plan_LOCAL_ONLY.json')
fixed = read(OPS / 'followup_validation_LOCAL_ONLY.json')
assert aud['status'] == 'PASS_OUTPUT_INTEGRITY' and fixed['status'] == 'PASS_FIXED_EVALUATIONS'
parent_path = Path(plan['checkpoint'])
assert sha(parent_path) == plan['checkpoint_sha256']
parent = torch.load(parent_path, weights_only=True, map_location='cpu')
keys = sorted(k for k in parent['state_dict'] if k.endswith(('.running_mean', '.running_var')))
assert len(keys) == 16
probe_plan = {
    'fixed_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
    'hypothesis': 'Drift of stored normalization statistics alone may explain loss of the previously learned missing-hand robustness after common native fine-tuning.',
    'operation': 'For BOTH new final models, replace ONLY the 16 running_mean/running_var tensors with the parent visibility tensors in a disposable in-memory model.',
    'keep': 'All trained parameters, class order, channel builder, observation span, data and labels.',
    'baselines': 'Reuse saved unmodified final-model results; do not re-infer them.',
    'buffers': keys, 'native_n': 699, 'web_development_n': 18, 'reference_intact_n': 18, 'reference_masked_n': 5,
    'no_training_checkpoint_save_registration_or_promotion': True,
    'causal_limit': 'A conditional inference-time intervention, not proof of the best training strategy. Old statistics and new parameters can be mismatched.',
}
write(OPS / 'bn_counterfactual_plan.json', probe_plan)
classes = parent['classes']
idx = {label:i for i,label in enumerate(classes)}
source_dev = u.load_source_split('validation', idx)
assert len(source_dev) == 696
candidate_dev = []
for r in read(Path(plan['candidate_report']))['rows']:
    if r['split_role'] != 'development_only':
        continue
    p = Path(r['npz'])
    assert sha(p) == r['npz_sha256']
    with np.load(p, allow_pickle=False) as z:
        x = np.array(z['x'], dtype=np.float32, copy=True)
    lab = plan['label_mapping'][r['raw_label']]
    candidate_dev.append({'sample_id': r['clip_id'], 'label': lab, 'class_index': idx[lab], 'x': x, 'frames': len(x)})
assert len(candidate_dev) == 3
public = u.load_nikl({word:idx[label] for word,label in neural.PROJECT_18.items()})
old_meta = read(Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/WordsInSilence_기능완성_20260913/visibility_experiment_20260914a/inputs.json'))
eligible = {r['id']:{'side':r['reviewed_non_sign_hand']} for r in old_meta['nikl'] if r.get('reviewed_non_sign_hand') in ('left','right')}
assert len(eligible) == 5
refs = v.reference_clips(public, eligible)
web_clips = []
for r in web_plan['rows']:
    p = Path(r['observation'])
    assert sha(p) == r['observation_sha256']
    clip, why, _ = neural._channels(Observation.load(str(p)))
    assert why is None
    web_clips.append({'sample_id':r['trial_id'], 'label':r['actual_label'], 'x':clip['x'], 'condition':'development'})
assert len(web_clips) == 18 and len(refs) == 23
arms = {}
for arm,spec in aud['checkpoints'].items():
    p = Path(spec['path'])
    assert sha(p) == spec['sha256']
    ck = torch.load(p, weights_only=True, map_location='cpu')
    model, got_classes = u.rebuild_from_payload(ck)
    assert got_classes == classes
    parameter_names = set(dict(model.named_parameters()))
    changed_parameters = [k for k in parameter_names if not torch.equal(ck['state_dict'][k], parent['state_dict'][k])]
    buffer_changes = {k: {'max_abs': float((ck['state_dict'][k] - parent['state_dict'][k]).abs().max()),
                          'l2': float(torch.linalg.vector_norm(ck['state_dict'][k] - parent['state_dict'][k]))} for k in keys}
    before = {k:t.detach().clone() for k,t in model.named_parameters()}
    with torch.no_grad():
        buffers = dict(model.named_buffers())
        for key in keys:
            buffers[key].copy_(parent['state_dict'][key])
    assert all(torch.equal(t, before[k]) for k,t in model.named_parameters())
    assert all(torch.equal(model.state_dict()[k], parent['state_dict'][k]) for k in keys)
    native = u.evaluate(model, source_dev + candidate_dev, classes)
    ref_rows = v.reference_rows(model, refs, classes, sorted(neural.PROJECT_18.values()))
    web_rows = v.reference_rows(model, web_clips, classes, sorted(neural.PROJECT_18.values()))
    summary = {}
    for scope in ('project18','full93'):
        summary[scope] = {
            'web_correct': sum(r[scope]['correct'] for r in web_rows),
            'web_errors': [{'true':r['true'],'pred':r[scope]['pred']} for r in web_rows if not r[scope]['correct']],
            'intact_correct': sum(r['condition']=='intact' and r[scope]['correct'] for r in ref_rows),
            'masked_correct': sum(r['condition']=='reviewed_masked' and r[scope]['correct'] for r in ref_rows)}
    addition_ids = {r['sample_id'] for r in candidate_dev}
    summary['native'] = {'full699_correct':native['correct'], 'macro_f1':native['macro_f1'],
                          'source696_correct':sum(r['id'] not in addition_ids and r['pred']==r['true'] for r in native['rows']),
                          'additional3_correct':sum(r['id'] in addition_ids and r['pred']==r['true'] for r in native['rows'])}
    arms[arm] = {'original_checkpoint_sha256':spec['sha256'], 'parent_statistics_sha256':plan['checkpoint_sha256'],
                 'learned_parameter_tensor_count':len(parameter_names), 'changed_parameter_tensors_since_parent':len(changed_parameters),
                 'buffer_changes_since_parent':buffer_changes, 'summary':summary,
                 'native_rows':native['rows'], 'reference_rows':ref_rows, 'web_rows':web_rows,
                 'parameters_identical_before_after_buffer_intervention':True,
                 'checkpoint_file_unchanged':sha(p)==spec['sha256']}
assert sha(parent_path) == plan['checkpoint_sha256']
write(OPS / 'bn_counterfactual_LOCAL_ONLY.json', {
    'status':'COMPLETED_EVALUATION_ONLY_COUNTERFACTUAL', 'plan_sha256':sha(OPS/'bn_counterfactual_plan.json'),
    'arms':arms, 'inference_calls':2*(699+23+18), 'training_steps':0, 'saved_models':0,
    'heldout_arrays_read':0, 'limitations':probe_plan['causal_limit']})
print(json.dumps({arm:r['summary'] for arm,r in arms.items()}, ensure_ascii=True))
