"""Independent arithmetic/synthetic checks; no Head.forward, training or score reads."""
import ast
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(U))
import numpy as np
import torch
import experiment as ex
from head import Head, masked_loss
from build_targets import labels, TARGETS

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

checks = {}
codefiles = ['head.py', 'build_targets.py', 'experiment.py', 'compare_saved.py', 'diagnostics.py', 'setup_prepare.py', 'make_report.py']
for name in codefiles:
    ast.parse((U/name).read_text(encoding='utf-8-sig'), filename=name)
checks['ast_files'] = len(codefiles)

# Synthetic logits, not network predictions. The network forward method is never called.
torch.manual_seed(52)
logits = torch.randn(7, 4, requires_grad=True)
y = torch.randint(0, 2, (7, 4)).float()
mask = torch.rand(7, 4) > .35
weights = torch.tensor([[.5, 1., 2., 1.5], [2., 1.5, .75, .5]])
event_weights = torch.rand(7, 4) + .1
base = masked_loss(logits, y, mask, weights, event_weights)
grad = torch.autograd.grad(base, logits, retain_graph=True)[0]
for replacement in (float('nan'), float('inf'), -1e8, 1e8):
    modified = y.clone()
    modified[~mask] = replacement
    loss = masked_loss(logits, modified, mask, weights, event_weights)
    g = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert torch.equal(base, loss) and torch.equal(grad, g)
assert torch.equal(grad[~mask], torch.zeros_like(grad[~mask]))
all_unknown = masked_loss(logits, y, torch.zeros_like(mask), weights, event_weights)
assert all_unknown.item() == 0 and torch.isfinite(all_unknown)
assert torch.equal(torch.autograd.grad(all_unknown, logits)[0], torch.zeros_like(logits))
checks['masked_loss'] = dict(unknown_values_loss_gradient_invariant=True, unknown_logits_gradient_zero=True, all_unknown_zero=True)

parameter_counts = {}
for condition in ('pose_only', 'pose_plus_signrep'):
    torch.manual_seed(42)
    h = Head(10)
    parameter_counts[condition] = sum(p.numel() for p in h.parameters() if p.requires_grad)
assert set(parameter_counts.values()) == {42960 + 129*10}
assert np.array_equal(ex.projection(), ex.projection())
checks['parameters'] = parameter_counts
checks['fixed_projection_is_repeatable'] = True

row = dict(center_shape={'R':[0,0,None], 'L':[None,None,None]}, center_relation=None,
           motion={'R':None, 'L':None}, roi_center={'R':True,'L':True}, roi_window={'R':True,'L':True})
vals, _ = labels(row)
assert vals['R_open_four'] == 0 and vals['R_closed_four'] is None
assert all(vals[f'{h}_motion_{d}'] is None for h in 'RL' for d in ('up','down','horizontal','still','mixed'))
row['motion']['R'] = 'mixed'
vals, _ = labels(row)
assert vals['R_motion_mixed'] == 1 and vals['R_motion_still'] == 0
row['roi_window']['R'] = False
vals, _ = labels(row)
assert vals['R_index_only'] == 0 and vals['R_motion_mixed'] is None
row['roi_window']['R'] = True
row['roi_center']['R'] = False
row['center_relation'] = 'R_higher'
vals, _ = labels(row)
assert vals['R_index_only'] is None and vals['R_motion_mixed'] == 1
assert vals['R_higher'] is vals['L_higher'] is None
checks['labels'] = dict(open_zero_not_closed=True, unknown_not_still=True, mixed_preserved=True,
                       center_window_roi_separate=True, relation_requires_both_roi=True)

# Mock input loading so normalization isolation uses no real clip data.
rng = np.random.default_rng(82)
arrays = {}
rows = []
for i, role in enumerate(('fit','development','new_feature_expression_evaluation')):
    key = f'synthetic_{i}'
    x = rng.normal(size=(13,69)).astype('float32')
    m = rng.random((13,69)) > .25
    x[~m] = np.nan
    arrays[key] = dict(x=x, mask=m, rgb=rng.normal(size=(13,768)).astype('float32'),
                       frame_id=np.arange(13), timestamp_us=np.arange(13)*33333)
    rows.append(dict(alias=key,input=key,input_sha256='synthetic',role=role,frame_ids=list(range(13)),
                     center_frame=6,timestamps_us=arrays[key]['timestamp_us'].tolist(),y=[1,0],target_mask=[True,False]))
info = dict(rows=rows,selected_indices=[0,1])
with patch.object(ex,'sha',return_value='synthetic'), patch.object(ex.np,'load',side_effect=lambda name,**kw: arrays[name]):
    batch, stats = ex.assemble(info)
original_fit = [a[0].clone() for a in batch[:2]]
for i in (1,2):
    arrays[f'synthetic_{i}']['x'] *= 1e6
    arrays[f'synthetic_{i}']['rgb'] *= 1e6
    info['rows'][i]['y'] = [-1e7,1e7]
for a in arrays.values():
    a['x'][~a['mask']] = 1e12
with patch.object(ex,'sha',return_value='synthetic'), patch.object(ex.np,'load',side_effect=lambda name,**kw: arrays[name]):
    changed_batch, changed_stats = ex.assemble(info)
assert stats == changed_stats
assert all(torch.equal(a,b[0]) for a,b in zip(original_fit,changed_batch[:2]))
assert torch.equal(batch[0],changed_batch[0]) is False  # Evaluation inputs did change.
checks['normalization'] = dict(fit_only=True, evaluation_labels_unused=True, masked_pose_values_ignored=True)

# Recount frozen label support, not model outcomes.
dataset = read(U/'DATASET.json')
audit = read(U/'LABEL_AUDIT.json')
rubric = read(U/'RUBRIC.json')
rows = dataset['rows']
assert len(rows) == 168 and len({r['sample_id'] for r in rows}) == 168
assert dict(Counter(r['role'] for r in rows)) == {'fit':108,'development':36,'new_feature_expression_evaluation':24}
assert len({r['clip_id'] for r in rows}) == 56
expected_people = {'fit':{1,2,3},'development':{4},'new_feature_expression_evaluation':{7,8}}
for role, people in expected_people.items():
    assert {r['person'] for r in rows if r['role']==role} == people
assert {r['word_id'] for r in rows if r['role']=='fit'}.isdisjoint(
    {r['word_id'] for r in rows if r['role']=='new_feature_expression_evaluation'})
for r in rows:
    assert r['frame_ids'] == list(range(r['center_frame']-6,r['center_frame']+7))
    assert len(r['timestamps_us']) == 13 and r['center_timestamp_us'] == r['timestamps_us'][6]
    assert all(b>a for a,b in zip(r['timestamps_us'],r['timestamps_us'][1:]))
    assert len(r['y']) == len(r['target_mask']) == 18
    assert all(v in (0.,1.) for v in r['y'])
independent_support = {}
for j,t in enumerate(dataset['targets']):
    ss = {}
    for sign, value in (('positive',1),('negative',0)):
        rr = [r for r in rows if r['role']=='fit' and r['target_mask'][j] and r['y'][j]==value]
        ss[sign] = dict(windows=len(rr), clips=len({r['clip_id'] for r in rr}),
                        words=len({r['word_id'] for r in rr}),people=len({r['person'] for r in rr}))
    passed = all(ss[sign][metric] >= rubric['class_gate'][f'fit_{sign}_{metric}_min']
                 for sign in ('positive','negative') for metric in ('windows','clips','words','people'))
    independent_support[t] = dict(passed=passed, **ss)
assert independent_support == audit['support']
selected = [t for t in dataset['targets'] if independent_support[t]['passed']]
assert selected == audit['selected'] == dataset['selected_targets']
assert dataset['selected_indices'] == [dataset['targets'].index(t) for t in selected]
checks['class_gate'] = dict(exact_support_match=True,selected=selected,selected_count=len(selected),
                          all_18_targets_retained_in_dataset=True,role_person_word_separation=True)

result = dict(status='PASS_ARITHMETIC_AND_SYNTHETIC_CODE_CHECKS',checks=checks,
              head_forward_calls=0,training_steps=0,checkpoint_loads=0,score_files_read=0,
              checked_code_sha256={n:sha(U/n) for n in codefiles},
              dataset_sha256=sha(U/'DATASET.json'),label_audit_sha256=sha(U/'LABEL_AUDIT.json'),
              limits='Same-Codex separate code/arithmetic checks; no model quality, visual-label truth, or linguistic validity claim.')
(U/'review/TRAIN_CODE_SELF_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
