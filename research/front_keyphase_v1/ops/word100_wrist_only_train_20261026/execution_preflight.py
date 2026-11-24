"""Synthetic execution checks before long training. No dataset/model checkpoint access."""
from pathlib import Path
import sys, json, hashlib, datetime, copy, importlib.util
import numpy as np
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
U = Path(__file__).resolve().parent
B = U.parent/'word100_body_boundary_20261017'
sys.path.insert(0, str(B))
import training_driver as dr
import training_core as core
import torch

def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(a.tobytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))

def main():
    assert not (U/'CONTEXT.json').exists(), 'This check must precede real training.'
    assert not (U/'EXECUTION_PREFLIGHT.json').exists()
    plan=read(U/'PLAN.json')
    for p,h in plan['pins'].items(): assert sha(p)==h,p
    checker=module('wrist_batch_execution_check',U/'verify_results.py')
    dr.configure_torch()
    rows=[];arrays={};rng=np.random.default_rng(910246)
    for i in range(899):
        cid=f'SYNTHETIC_{i:04d}';arrays[cid]={}
        for v in core.VIEWS:
            arrays[cid][v]={}
            for t in ('x_whole','x_context'):
                a=rng.normal(size=(1+(i+(v=='sensor')+2*(t=='x_context'))%7,280)).astype(np.float32)
                if i%17==0:a[0]=0  # zero-valued observed frame is not padding
                arrays[cid][v][t]=a
        rows.append(dict(clip_id=cid,label_index=i%100,arrays=arrays[cid]))
    outputs={};last_records=None
    for seed in core.SEEDS:
        records=[]
        for epoch in range(1,5):
            order=np.random.default_rng(seed+epoch).permutation(899)
            for at in range(0,899,32):
                batch=[rows[i] for i in order[at:at+32]]
                x,p,y=dr.torch_batch(batch,epoch=epoch,arm='C')
                records.append(dict(seed=seed,epoch=epoch,clip_ids=[r['clip_id'] for r in batch],
                    choices=[list(core.schedule(r['clip_id'],epoch,'C')) for r in batch],
                    x_sha256=ah(x.numpy()),pad_sha256=ah(p.numpy()),labels_sha256=ah(y.numpy())))
        outputs[str(seed)]=checker.actual_batches(rows,records,4,seed,arrays)
        last_records=records
    rejected=[]
    for name in ('duplicate_id','wrong_view','damaged_tensor','damaged_padding','wrong_labels','missing_batch','extra_batch'):
        bad=copy.deepcopy(last_records)
        if name=='duplicate_id':bad[0]['clip_ids'][1]=bad[0]['clip_ids'][0]
        elif name=='wrong_view':bad[0]['choices'][0][0]='native'
        elif name=='damaged_tensor':bad[0]['x_sha256']='0'*64
        elif name=='damaged_padding':bad[0]['pad_sha256']='0'*64
        elif name=='wrong_labels':bad[0]['labels_sha256']='0'*64
        elif name=='missing_batch':bad.pop()
        else:bad.append(copy.deepcopy(bad[-1]))
        try:checker.actual_batches(rows,bad,4,core.SEEDS[-1],arrays)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Corruption was not rejected: '+name)
    # Cross-input evaluator runs fewer rows but retains the original group time length.
    # Test that removing batch mates in eval changes neither normalization state nor predictions materially.
    torch.manual_seed(910246)
    m=dr.MaskedSignTCN(**core.MODEL).eval()
    x=np.zeros((8,41,280),np.float32);p=np.zeros((8,41),np.float32)
    for i,n in enumerate([1,7,11,19,23,29,35,41]):
        x[i,:n]=rng.normal(size=(n,280));p[i,:n]=1
    before=dr.state_hash(m.state_dict())
    chosen=[0,2,5,7]
    with torch.inference_mode():
        full=m(torch.from_numpy(x),torch.from_numpy(p)).numpy()
        subset=m(torch.from_numpy(x[chosen]),torch.from_numpy(p[chosen])).numpy()
    gap=float(np.max(np.abs(full[chosen]-subset)))
    assert np.allclose(full[chosen],subset,rtol=1e-5,atol=1e-6)
    assert dr.state_hash(m.state_dict())==before
    for path,h in plan['pins'].items():assert sha(path)==h,path
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),
        status='PASS_SYNTHETIC_EXECUTION_CONTRACT',batch_reconstruction=outputs,
        corruption_rejections=rejected,cross_eval_max_absolute_logit_difference=gap,
        cross_eval_state_unchanged=True,synthetic_forward_passes=2,optimizer_steps=0,
        real_video_access=0,trained_checkpoint_access=0,original_source_changed=False,
        scope='Synthetic tensors verify batch/schedule/trace and eval batching assumptions. Not real input correctness, trained W verification, accuracy or speed evidence.',
        pins={str(q):sha(q) for q in (Path(__file__),U/'verify_results.py',U/'control.py',U/'cross_inputs.py',B/'training_driver.py',B/'training_core.py')})
    with (U/'EXECUTION_PREFLIGHT.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],actual_synthetic_batches=sum(r['actual_batches_checked'] for r in outputs.values()),corruptions_rejected=len(rejected),eval_difference=gap,real_training=0)))

if __name__=='__main__':main()
