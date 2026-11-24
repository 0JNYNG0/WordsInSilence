"""Single predeclared weak interval training, with source/query/class weights."""
import os,json,time,random,hashlib,math,traceback
from pathlib import Path
import numpy as np
import torch
from local_events import LocalEventNet,bag_logit
U=Path(__file__).resolve().parent;O=U.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def checkpins(pins):
    for p,h in pins.items():assert sha(p)==h, 'Frozen artifact changed: '+p

def training_groups(d):
    wanted=set(d['training_group_ids'])
    groups=[g for g in d['groups'] if g['group_id'] in wanted]
    assert len(groups)==len(wanted) and len(groups)<=45
    for g in groups:
        assert g['role']=='fit' and g['person'] in range(1,10) and g['train_eligible']
        assert g['target_index'] in d['supported_targets']
    return groups

def class_weights(groups,targets):
    counts={(t,y):sum(any(b['train_eligible'] and b['label']==y for b in g['bags']) for g in groups if g['target_index']==t) for t in targets for y in [0,1]}
    assert all(v>=3 for v in counts.values())
    # Equal target and class influence across the complete fixed training set.
    # Multiple dependent same-source bags of one class are averaged first.
    return {f'{t}:{y}':len(groups)/(2*len(targets)*n) for (t,y),n in counts.items()},counts

def group_loss(out,g,weights):
    t=g['target_index'];terms=[]
    for y in [0,1]:
        bags=[b for b in g['bags'] if b['train_eligible'] and b['label']==y]
        if not bags:continue
        losses=[]
        for b in bags:
            lo,hi=b['indices']
            assert b['evidence_complete'] and out['motion_anchor'][0,lo:hi+1].all()
            z=bag_logit(out,lo,hi,t)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(z,z.new_tensor(float(y))))
        terms.append(torch.stack(losses).mean()*weights[f'{t}:{y}'])
    assert terms
    return torch.stack(terms).sum()

def main():
    from data import load_inputs
    assert not (U/'TRAIN_STARTED.json').exists(),'Do not repeat a started training run'
    d=read(U/'DATASET_FROZEN.json');review=read(U/'REVIEW_CODE.json');dr=read(U/'DATA_REVIEW.json')
    assert review['approved_for_one_training'] and dr['approved_for_one_training']
    checkpins(review['code_pins']);checkpins(dr['pins']);checkpins(d['pins'])
    inputs=load_inputs(d);groups=training_groups(d)
    weights,counts=class_weights(groups,d['supported_targets'])
    init=O/'word100_learned_rgb_motion_20261102/runs/rgb_residual_42/model.pt'
    assert str(init) in d['pins']
    steps=80*math.ceil(len(groups)/12)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model=LocalEventNet()
    pretrained=torch.load(init,map_location='cpu',weights_only=True)
    model.initialize_from_frozen_motion(pretrained)
    assert torch.equal(model.geometry_valid,torch.tensor(d['geometry_valid'],dtype=torch.float32))
    assert torch.equal(model.geometry.weight,pretrained['direction.embed.0.weight'])
    assert torch.equal(model.rgb.weight,pretrained['direction.rgb.weight'])
    params={n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in model.parameters())==39013
    pins=dict(d['pins'])
    for name in ['DATASET_FROZEN.json','INPUT_CHECKS.json','DATA_REVIEW.json','REVIEW_CODE.json','data.py','prepare_data.py','local_events.py','train_local.py','evaluate_local.py']:
        pins[str(U/name)]=sha(U/name)
    contract=dict(pins=pins,seed=42,epochs=80,steps=steps,group_batch=12,
        optimizer='AdamW',learning_rate=.0005,weight_decay=.01,clipnorm=2,
        trainable_parameters=params,parameter_count=39013,device='cpu',threads=4,
        initialization=str(init),initialization_sha256=sha(init),supported_targets=d['supported_targets'],
        training_group_ids=[g['group_id'] for g in groups],class_weights=weights,
        class_group_counts={f'{t}:{y}':n for (t,y),n in counts.items()},
        loss='One mean per source-query-class, then frozen inverse group-class weights. Broad bag has zero training weight.',
        raw_threshold=.5,confirmed_threshold=.8,low_threshold=.2,
        evaluation='One full native forward per public clip; original timestamps, fixed reviewed bags and whole-context max. No old model rerun.',
        gate='Per supported target DEV original-positive high confirmations at least2/3; known local0 high false confirms0. Absent DEVnegative means unverified. Exact timing and education correctness not certified.',
        public_sources_only=True,dev_optimization=False,backbone_updated=False,old_checkpoints_changed=False)
    write(U/'EXECUTION_FROZEN.json',contract)
    write(U/'TRAIN_STARTED.json',dict(pid=os.getpid(),at=time.time(),steps=steps))
    arrays={cid:torch.from_numpy(np.asarray(p['x'],dtype='float32'))[None] for cid,p in inputs.items() if any(g['clip_id']==cid for g in groups)}
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    rng=np.random.default_rng(42);history=[];step=0;beg=time.perf_counter()
    model.train()
    for epoch in range(80):
        order=rng.permutation(len(groups));losses=[]
        for offset in range(0,len(groups),12):
            selected=[groups[i] for i in order[offset:offset+12]]
            opt.zero_grad(set_to_none=True)
            # Shared full-sequence calculation for two targets of the same clip.
            outputs={cid:model(arrays[cid],torch.tensor([arrays[cid].shape[1]])) for cid in sorted({g['clip_id'] for g in selected})}
            loss=torch.stack([group_loss(outputs[g['clip_id']],g,weights) for g in selected]).mean()
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
            opt.step();step+=1;losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),steps=step))
        if (epoch+1)%10==0:
            (U/'TRAIN_PROGRESS.json').write_text(json.dumps(history[-1]),encoding='utf-8')
            print(history[-1],flush=True)
    assert step==steps
    model.eval();checkpins(pins)
    out=U/'runs/local_42';out.mkdir(parents=True,exist_ok=True)
    with (out/'model.pt').open('xb') as f:torch.save(model.state_dict(),f)
    write(U/'TRAIN_RESULT.json',dict(status='ONE_FIXED_TRAINING_COMPLETE',seed=42,epochs=80,steps=step,
        seconds=time.perf_counter()-beg,history=history,model=str(out/'model.pt'),model_sha256=sha(out/'model.pt'),
        geometry_projection_changed=not torch.equal(model.geometry.weight,pretrained['direction.embed.0.weight']),
        rgb_projection_changed=not torch.equal(model.rgb.weight,pretrained['direction.rgb.weight']),
        original_initialization_unchanged=sha(init)==contract['initialization_sha256'],
        backbone_updated=False,dev_used_in_optimizer=False,user_or_mentor_data=False,whole_goal_complete=False))
    write(U/'TRAIN_FINISHED.json',dict(status='COMPLETE',pid=os.getpid(),at=time.time(),result_sha256=sha(U/'TRAIN_RESULT.json')))

if __name__=='__main__':
    try:main()
    except Exception:
        path=U/('TRAIN_FAILURE_'+str(time.time_ns())+'.txt');path.write_text(traceback.format_exc(),encoding='utf-8');raise
