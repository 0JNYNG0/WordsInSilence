from common import *
import time,traceback,os
def main():
    assert not(HERE/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=read(HERE/'FROZEN.json');meta=read(HERE/'INPUTS_COMPLETE.json');assert sha(meta['path'])==meta['sha256']
    for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(p)==h,p
    audit=read(HERE/'INPUT_CONTRACT_AUDIT.json');assert audit['normal_context_unchanged']and audit['fit_reverse_context_changed']>0
    z=np.load(meta['path']);ctx=torch.tensor(z['context']);yy=torch.tensor(z['y'],dtype=torch.long);rows=meta['rows'];train=[i for i,r in enumerate(rows)if r['split']=='train']
    assert all(rows[i]['person'] in range(1,8) for i in train)
    assert all(np.isfinite(z[k]).all()for k in z.files)
    pools={}
    for origin in ['original_motion_replay','new_reviewed']:
        ii=[i for i in train if(rows[i]['origin']=='original_motion_replay')==(origin=='original_motion_replay')]
        pools[origin]={k:[i for i in ii if(rows[i]['axis'],rows[i]['target'])==k]for k in sorted({(rows[i]['axis'],rows[i]['target'])for i in ii})}
    rc=old.load('_recovery_req',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();runs=[]
    write(HERE/'TRAINING_CONTRACT.json',dict(at=stamp(),pins={str(p):sha(p)for p in[Path(__file__),HERE/'common.py',HERE/'FROZEN.json',HERE/'INPUTS_COMPLETE.json',HERE/'INPUT_CONTRACT_AUDIT.json']},normalization='original geometry retains copied original observer gm/gs, projected geometry uses saved FIT-only projected normalization exactly as prior experiment',runtime_scope='Axis2 only for paired end-to-end comparison, exactly matching saved projected experiment; other component states/spans copied unchanged',steps=cfg['steps'],selected='final_step_only',conditions=cfg['conditions'],seeds=cfg['seeds']))
    for condition in cfg['conditions']:
        name='projected'if'projected'in condition else'original';g=torch.tensor(z[name+'_geometry'])
        for seed in cfg['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);head=old.DirectionHead(observer)
            if name=='projected':
                head.gm.copy_(torch.tensor(meta['norm'][name]['mean']));head.gs.copy_(torch.tensor(meta['norm'][name]['std']))
                om=read(PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json')['norm']
                assert np.allclose(head.gm.numpy(),om['mean'])and np.allclose(head.gs.numpy(),om['std'])
            opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[]
            for step in range(cfg['steps']):
                ix=[]
                for groups in pools.values():
                    keys=list(groups)
                    for _ in range(cfg['batch_per_origin']):
                        key=keys[int(rng.integers(len(keys)))];ix.append(int(rng.choice(groups[key])))
                ix=torch.tensor(ix);out=head(ctx[ix],g[ix]);loss=torch.nn.functional.cross_entropy(out,yy[ix]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),2);opt.step()
                if step%200==0 or step==cfg['steps']-1:
                    log.append(dict(step=step+1,loss=float(loss.detach())));print(condition,seed,step+1,float(loss.detach()),flush=True)
                    write(HERE/'TRAINING_PROGRESS.json',dict(at=stamp(),pid=os.getpid(),condition=condition,seed=seed,step=step+1,completed=runs))
            head.eval();dest=HERE/'runs'/condition/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(head.state_dict(),dest/'model.pt')
            with torch.inference_mode():pred=torch.cat([head(ctx[i:i+512],g[i:i+512])for i in range(0,len(rows),512)]).numpy()
            np.savez_compressed(dest/'predictions.npz',logits=pred);runs.append(dict(condition=condition,seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),seconds=time.perf_counter()-start,log=log))
    write(HERE/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,web_changed=False,private_access=False,final8=False))
if __name__=='__main__':
    try:main()
    except BaseException:(HERE/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
