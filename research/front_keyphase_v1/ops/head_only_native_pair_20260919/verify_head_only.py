"""Independent synthetic checks of newly authored head-only training behavior."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys

OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
import numpy as np
import torch
import tools_native_data_pair as base
import tools_prep_union93_candidate as u
import tools_head_only_native_pair as helper
from pilot.model_masked_v1 import MaskedSignTCN

torch.set_num_threads(2)
checks = []
def check(name, ok):
    checks.append({'name': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)

classes = ['class_A', 'class_B', 'class_C']
generator = np.random.default_rng(901)
items = [{'sample_id': f'synthetic_{i}', 'label': classes[i % 3], 'class_index': i % 3,
          'x': generator.normal(size=(7+i, 280)).astype(np.float32)} for i in range(6)]
setting = {'optimizer_steps': 5, 'batch': {'total': 6, 'source': 6, 'nikl': 0},
           'lr': {'encoder': 0.0, 'head': 0.001}, 'weight_decay': 0.001,
           'seed': 901, 'batch_size': 6, 'cpu_threads': 2, 'augmentation': False}
torch.manual_seed(902)
parent = MaskedSignTCN(width=8, dilations=(1, 2), dropout=0.7, n_classes=3)
for name, buf in parent.named_buffers():
    if name.endswith('num_batches_tracked'):
        buf.fill_(17)
    elif name.endswith('running_mean'):
        buf.fill_(0.31)
    elif name.endswith('running_var'):
        buf.fill_(1.7)
uniforms = generator.random(2*5*6).tolist()
outputs, trained, mode_observations = [], [], []
for _ in range(2):
    model = copy.deepcopy(parent)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = {k: v.clone() for k,v in model.state_dict().items()}
    def modes(m, inputs):
        mode_observations.append(all(not v.training for n,v in m.named_modules() if n != 'head') and m.head.training)
    handle = model.register_forward_pre_hook(modes)
    schedule = base.UniformSchedule(uniforms)
    cfg = copy.deepcopy(setting)
    result = helper.run_head_only_training(model, items, [], classes, cfg, schedule, log=lambda _:None)
    handle.remove()
    after = model.state_dict()
    check('all encoder parameters and buffers bitwise preserved', all(torch.equal(v,after[k]) and v.dtype == after[k].dtype for k,v in before.items() if not k.startswith('head.')))
    check('head actually updated', any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('head.')))
    check('stale encoder gradients cleared and never recreated', all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if not n.startswith('head.')))
    check('fixed schedule and no augmented branch', schedule.consumed == 60 and result['steps'] == 5 and result['augmentation_examples'] == [] and result['distinct_nikl_clips_seen'] == 0)
    check('effective metadata declares frozen encoder', cfg['lr']['encoder'] == 0 and cfg['training_scope'] == 'head_only_frozen_encoder_and_buffers' and cfg['encoder_buffers_frozen'] is True)
    check('all modules evaluation mode after training', all(not m.training for m in model.modules()))
    outputs.append(result)
    trained.append(copy.deepcopy(after))
check('every training forward keeps encoder eval and head train', len(mode_observations) == 10 and all(mode_observations))
check('identical schedule exactly reproducible', outputs[0] == outputs[1] and all(torch.equal(v,trained[1][k]) for k,v in trained[0].items()))

for case in ['nonzero_encoder_lr', 'augmentation_true', 'nikl_nonempty', 'wrong_label']:
    model, cfg, data, nikl = copy.deepcopy(parent), copy.deepcopy(setting), copy.deepcopy(items), []
    if case == 'nonzero_encoder_lr': cfg['lr']['encoder'] = 0.0001
    elif case == 'augmentation_true': cfg['augmentation'] = True
    elif case == 'nikl_nonempty': nikl = [{}]
    elif case == 'wrong_label': data[0]['label'] = 'class_B'
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rejected = False
    try:
        helper.run_head_only_training(model,data,nikl,classes,cfg,base.UniformSchedule(uniforms),log=lambda _:None)
    except (ValueError, TypeError, AssertionError):
        rejected = True
    check(case + ' rejected before weight changes', rejected and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

# A real illegal buffer write must be detected, not hidden by restoring snapshots.
model = copy.deepcopy(parent)
def corrupt_buffer(m, inputs):
    m.blocks[0].norm1.running_mean.add_(0.01)
h = model.register_forward_pre_hook(corrupt_buffer)
rejected = False
try:
    helper.run_head_only_training(model,items,[],classes,copy.deepcopy(setting),base.UniformSchedule(uniforms),log=lambda _:None)
except (RuntimeError, AssertionError):
    rejected = True
h.remove()
check('illegal encoder buffer mutation detected without restoration', rejected and not torch.equal(model.blocks[0].norm1.running_mean,parent.blocks[0].norm1.running_mean))

# Capture a stub original runner on a separately imported module, never run real data here.
original_run, original_training, original_main = base.run, u.run_training, base.main
captured = []
def fake_run(plan, prepared, out):
    captured.append((copy.deepcopy(prepared), u.run_training))
    raise LookupError('synthetic downstream failure')
try:
    base.run = fake_run
    spec = importlib.util.spec_from_file_location('head_only_wrapper_test', WEB / 'tools_head_only_native_pair.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
finally:
    base.run = original_run
plan = {'head_only': {'encoder_lr':0.0, 'freeze_encoder_buffers':True, 'training_scope':'head_only_frozen_encoder_and_buffers'}}
prepared = {'settings': {'encoder_lr':0.0001}}
before = copy.deepcopy(prepared)
try:
    wrapper.run(plan,prepared,'unused')
except LookupError:
    pass
check('wrapper supplies effective zero lr and installs correct helper', len(captured)==1 and captured[0][0]['settings']['encoder_lr']==0 and captured[0][1] is wrapper.run_head_only_training)
check('wrapper restores training hook after exception and preserves plan', u.run_training is original_training and prepared == before)
def fake_main(argv):
    check('main delegates through new wrapper', base.run is wrapper.run)
    raise LookupError('synthetic main failure')
try:
    base.main = fake_main
    try:
        wrapper.main([])
    except LookupError:
        pass
finally:
    base.main = original_main
check('main restores original runner after exception', base.run is original_run)

report = {'status':'PASS_SYNTHETIC_HEAD_ONLY', 'checks':checks, 'passed':sum(c['passed'] for c in checks),
          'total':len(checks), 'helper_sha256':hashlib.sha256((WEB/'tools_head_only_native_pair.py').read_bytes()).hexdigest(),
          'real_training_or_heldout_inputs_read':False, 'returned_freeze_proof':outputs[0]['freeze_proof']}
with (OPS/'synthetic_verification.json').open('x',encoding='utf-8') as f:
    json.dump(report,f,indent=2)
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total']}))
