"""Codex independent synthetic verification of a Claude-authored component.

No real sample scoring, optimizer steps, dataset changes, or production changes.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
import torch

U=Path(__file__).resolve().parent
OPS=U.parent
OLD=OPS/'stgcn_execution_ready_20261005'
RUN=OPS.parent/'runs/stgcn_full_path_v1'
OUT=U/'VALID_WINDOW_CHECK.json'
assert not OUT.exists(), 'Preserve existing independent check'
sys.path[:0]=[str(U),str(OLD),str(OPS/'stgcn_reference_alignment_20261005'),
             str(OPS/'stgcn_transfer_compare_20261005')]
from valid_window_forward import forward_valid_windows, self_test
import reference_adapter as ra
import generic_workflow as gw
from stgcn_core import read_encoder_tensors

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(20261006)
result={'status':'RUNNING','checks':[], 'training_updates':0,'real_clip_inferences':0}
def check(name, condition, **detail):
    result['checks'].append(dict(name=name,pass_check=bool(condition),**detail))
    if not condition:
        result['status']='FAIL'
        OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
        raise AssertionError(name)
def call(m,x,mask):
    return forward_valid_windows(m.conv_encoder,m.agg,m.final_bn,m.final_fc,x,mask)

try:
    self_test()
    check('claude_self_test',True)
    context=json.loads((OLD/'EXECUTION_CONTEXT.json').read_text(encoding='utf8'))
    pretrained=OPS/'openhands_artifact_20261005'
    tensors,_=read_encoder_tensors(pretrained/'raw_dpc.zip',pretrained/'TENSOR_METADATA.json',
                                  context['source_pins']['pretrained_zip'],encoder_prefix='model.')
    encoder={k[len('conv_encoder.'):]:v for k,v in tensors.items() if k.startswith('conv_encoder.')}
    gru={k[len('agg.'):]:v for k,v in tensors.items() if k.startswith('agg.')}
    dpc,_,_=ra.load_reference(OPS/'stgcn_reference_alignment_20261005/reference',context['source_pins']['source_pins'])
    model,_=ra.make_reference_model(dpc,encoder,gru,num_classes=93,seed=20261005)
    model=gw.load_final_state(model,str(RUN),context)
    model.eval()
    before=gw.state_digest(model.state_dict())
    x=torch.randn(3,6,2,10,27)
    allvalid=torch.ones((3,6),dtype=torch.bool)
    mixed=torch.tensor([[False,False,True,True,True,True],
                        [True,False,True,False,False,True],
                        [False,False,False,False,False,True]])
    with torch.no_grad():
        ref=model(x)
        got=call(model,x,allvalid)
        delta=float((ref-got).abs().max())
        check('actual_model_all_valid_reference',delta<=1e-5,max_difference=delta)
        got=call(model,x,mixed)
        manual=torch.cat([model(x[i:i+1,mixed[i]]) for i in range(3)])
        delta=float((got-manual).abs().max())
        check('mixed_lengths_chronological_manual_reference',delta<=1e-5,max_difference=delta)
        poison=x.clone();poison[~mixed]=10000
        changed=call(model,poison,mixed)
        check('invalid_content_excluded',torch.equal(got,changed))
        order=torch.tensor([2,0,1])
        perm=call(model,x[order],mixed[order])
        delta=float((perm-got[order]).abs().max())
        check('sample_order_preserved',delta<=1e-5,max_difference=delta)
        zero=x.clone();zero[0,2]=0
        zero_got=call(model,zero,mixed)
        zero_ref=torch.cat([model(zero[i:i+1,mixed[i]]) for i in range(3)])
        check('valid_all_zero_coordinates_kept',float((zero_got-zero_ref).abs().max())<=1e-5)
    inp=x.clone().requires_grad_(True)
    out=call(model,inp,mixed)
    out.square().sum().backward()
    check('invalid_input_grad_zero',bool((inp.grad[~mixed]==0).all()))
    check('valid_input_grad_present',float(inp.grad[mixed].abs().sum())>0)
    for name,block in [('encoder',model.conv_encoder),('gru',model.agg),('norm',model.final_bn),('classifier',model.final_fc)]:
        grads=[p.grad for p in block.parameters() if p.requires_grad]
        check(name+'_gradient_path',bool(grads) and all(g is not None and torch.isfinite(g).all() for g in grads)
              and sum(float(g.abs().sum()) for g in grads)>0)
    check('actual_model_eval_state_unchanged',gw.state_digest(model.state_dict())==before)
    check('eval_mode_not_changed',all(not m.training for m in model.modules()))

    # Training mode: use copies and identical seeds; compare actual encoder BN stats.
    a=copy.deepcopy(model).train();b=copy.deepcopy(model).train()
    with torch.no_grad():
        torch.manual_seed(4106);train_a=call(a,x,mixed)
        torch.manual_seed(4106);train_b=call(b,poison,mixed)
    check('training_poison_same_output',torch.equal(train_a,train_b))
    stats=[k for k in a.state_dict() if 'running_' in k or 'num_batches_tracked' in k]
    check('training_poison_same_batchnorm_stats',all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in stats),state_entries=len(stats))
    check('train_mode_not_changed',all(m.training for m in a.modules()))

    rejected=[]
    badmask=mixed.clone();badmask[0]=False
    malformed=[('all_invalid',x,badmask),('mask_dtype',x,mixed.float()),
               ('mask_shape',x,mixed[:,:5]),('input_shape',x[:,:,:,0],mixed)]
    nan=x.clone();nan[0,0,0,0,0]=float('nan')
    malformed.append(('nan_invalid',nan,mixed))
    for name,values,mask in malformed:
        try:call(model,values,mask)
        except (ValueError,TypeError):rejected.append(name)
    check('explicit_input_rejections',len(rejected)==len(malformed),rejected=rejected)
    expected='4c1554468e48d8a671b2a40be337f87f5bc4ff2935869e6257f3b38569d9625d'
    check('frozen_file_hash_preserved',hashlib.sha256((RUN/'final_model.npz').read_bytes()).hexdigest()==expected)
    result.update(status='PASS',component_sha256=hashlib.sha256((U/'valid_window_forward.py').read_bytes()).hexdigest(),model_sha256=expected)
    OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
except Exception as exc:
    if not OUT.exists():
        result.update(status='FAIL',error=repr(exc))
        OUT.write_text(json.dumps(result,indent=2),encoding='utf8')
    raise
