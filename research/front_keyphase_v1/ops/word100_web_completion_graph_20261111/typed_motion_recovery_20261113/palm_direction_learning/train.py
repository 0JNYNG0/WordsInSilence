from core_palm import *
from collections import defaultdict
from torch.nn import functional as F
import os

def make_tree(rows, eligible, keys, predicate):
    tree={}
    for i,r in enumerate(rows):
        if not eligible[i] or not predicate(r):continue
        node=tree
        for key in keys[:-1]:node=node.setdefault(key(r),{})
        node.setdefault(keys[-1](r),[]).append(i)
    return tree

def sample(tree,rng):
    node=tree
    while isinstance(node,dict):
        keys=list(node);assert keys
        node=node[keys[int(rng.integers(len(keys)))]]
    assert len(node)
    return int(node[int(rng.integers(len(node)))])

def main():
    assert not(UNIT/'TRAINING_CONTRACT.json').exists()
    stop_if_requested();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    meta,z=load_inputs();cfg=read(UNIT/'PREPARATION_CONTRACT.json');hp=cfg['training_proposal']
    assert cfg['seeds']==[42,43,44] and cfg['steps']==800
    assert hp['primary_batch']==48 and hp['partial_batch']==hp['static_batch']==24
    tensors={k:torch.tensor(z[k],dtype=torch.float32) for k in ['primary_geometry','partial_geometry','static_geometry']}
    targets=torch.tensor(z['primary_y'],dtype=torch.long)
    prim=make_tree(meta['primary_rows'],z['primary_eligible'],[lambda r:r['target'],lambda r:r['origin']=='original_motion_replay',lambda r:r['clip_id']],lambda r:r['split']=='train')
    part=make_tree(meta['partial_rows'],z['partial_eligible'],[lambda r:r['clip_id'],lambda r:(r['view'],r['reverse'])],lambda r:r['split']=='train')
    static=make_tree(meta['static_rows'],z['static_eligible'],[lambda r:r['source_collection'],lambda r:r['clip_id'],lambda r:r['source_index']],lambda r:r['split']=='train')
    assert set(prim)=={0,1,2} and part and set(static)=={'primary','partial'}
    rcpath=OPS/'word100_guide_requirement_learning_20261111/requirement_core.py'
    modelpath=OPS/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
    normpath=OPS/'word100_broad_factorized_learning_20261110/NORMALIZATION.json'
    pins={str(p):sha(p) for p in [Path(__file__),UNIT/'core_palm.py',UNIT/'PREPARATION_CONTRACT.json',UNIT/'INPUTS_COMPLETE.json',Path(meta['path']),rcpath,modelpath,normpath]}
    contract=dict(at=stamp(),seeds=cfg['seeds'],steps=cfg['steps'],training=hp,
        pins=pins,checkpoint='final step only; all seeds retained',
        normalization='FIT primary only, as prepared; never update on partial/static/probe',
        partial_scale_gradient='detached direction mass; verified before each run',
        runtime_scope='axis2 only. Existing C/T3-F/common100/other axes/shape/extent untouched',
        evaluations='Save logits for all prepared primary/partial/static rows once per seed. Eligibility retained separately; no threshold or checkpoint selection.',
        private_access=False,final8=False,web_changed=False)
    write(UNIT/'TRAINING_CONTRACT.json',contract)
    rc=old.load('_palm_training_frozen_observer',rcpath);observer=rc.frozen_observer();runs=[]
    for seed in cfg['seeds']:
        stop_if_requested();torch.manual_seed(seed);rng=np.random.default_rng(seed)
        net=PalmDirection(observer,z['mean'],z['std'],float(z['delta_scale']))
        for name in ('motion_conv','motion_gru','motion_head'):
            for p,q in zip(getattr(net,name).parameters(),getattr(observer,name).parameters()):
                assert torch.equal(p,q) and p.data_ptr()!=q.data_ptr()
        # Verify algebra/gradient contract before optimization, not an accuracy score.
        ids=[sample(prim[k],rng) for k in (0,1,2)]
        gg=tensors['primary_geometry'][ids];p=net(gg);pr=net(reverse_geometry(gg))
        assert torch.allclose(p[:,[2,1,0]],pr,atol=2e-5,rtol=2e-5)
        net.zero_grad();partial_progression_loss(net(tensors['partial_geometry'][:4])).backward()
        assert net.raw_scale.grad is None or float(net.raw_scale.grad)==0
        net.zero_grad()
        optim=torch.optim.AdamW(net.parameters(),lr=hp['lr'],weight_decay=hp['weight_decay'])
        tic=time.perf_counter();log=[]
        for step in range(cfg['steps']):
            stop_if_requested()
            ip=[sample(prim[k],rng) for k in (0,1,2) for _ in range(16)]
            it=[sample(part,rng) for _ in range(24)]
            ist=[sample(static[k],rng) for k in ('primary','partial') for _ in range(12)]
            a=F.cross_entropy(net(tensors['primary_geometry'][ip]),targets[ip])
            b=partial_progression_loss(net(tensors['partial_geometry'][it]))
            c=F.cross_entropy(net(tensors['static_geometry'][ist]),torch.ones(24,dtype=torch.long))
            loss=a+b+c;assert torch.isfinite(loss),(seed,step)
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),hp['gradient_clip']);optim.step()
            if step in (0,199,399,599,cfg['steps']-1):
                q=dict(step=step+1,loss=float(loss.detach()),primary=float(a.detach()),partial=float(b.detach()),static=float(c.detach()))
                log.append(q);write(UNIT/'TRAINING_PROGRESS.json',dict(at=stamp(),pid=os.getpid(),seed=seed,step=step+1,completed=len(runs),log=q))
                print('palm',seed,step+1,round(q['loss'],5),flush=True)
        net.eval();dest=UNIT/'runs'/str(seed);dest.mkdir(parents=True);mp=dest/'model.pt';torch.save(net.state_dict(),mp)
        output={}
        with torch.inference_mode():
            for name in ('primary','partial','static'):
                g=tensors[name+'_geometry'];output[name+'_logits']=torch.cat([net(g[i:i+256]) for i in range(0,len(g),256)]).numpy()
        pp=dest/'predictions.npz';np.savez_compressed(pp,**output)
        run=dict(seed=seed,model=str(mp),model_sha256=sha(mp),predictions=str(pp),predictions_sha256=sha(pp),seconds=time.perf_counter()-tic,log=log,scale=float(F.softplus(net.raw_scale).detach()),partial_direction_labels=False)
        write(dest/'COMPLETE.json',run);runs.append(run)
    for p,h in pins.items():assert sha(p)==h,p
    write(UNIT/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=runs,pins=pins,old_models_changed=False,web_changed=False,private_access=False,final8=False))

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('TRAINING');raise
