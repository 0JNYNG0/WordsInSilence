from isolation_net import *
import time,traceback
def main():
    assert not(S/'FROZEN.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    meta=read(HERE/'INPUTS_COMPLETE.json');assert sha(meta['path'])==meta['sha256'];z=np.load(meta['path']);rows=meta['rows'];y=torch.tensor(z['y'],dtype=torch.long);g=torch.tensor(z['projected_geometry']);ctx=torch.zeros((len(rows),192))
    cfg=dict(at=stamp(),hypothesis='Separate observed body-wrist direction from hand/phase context; test reversal equivariance as an explicit physical-direction constraint.',motivation='Body-only axis2 geometry has no finger dependency. Paired FIT hand removal changed numerical context but no verdicts (0/21), so hand-context as the cause is NOT established. Existing development changes combine span/context; compare both structural hypotheses, no promised gain.',conditions=['body_only','body_only_symmetric'],seeds=[42,43,44],steps=800,lr=.0003,weight_decay=.001,selected='final step only; all seeds; no threshold change',runtime_scope='axis2 only; all spans, shapes and other axes frozen',labels='unchanged typed axes1..4, FIT/PROBE preserved; no geometry-derived labels',symmetry='three classes negative/hold/positive only; do not apply to action order, contact or completion',limits='synthetic reversals and upstream-exposed development fixtures; no new-user accuracy claim',pins={str(p):sha(p)for p in[Path(__file__),S/'isolation_net.py',HERE/'common.py',HERE/'INPUTS_COMPLETE.json',HERE/'context_isolation_audit/RESULTS.json',HERE/'VERIFICATION.json']},web_changed=False,final8=False,private_access=False)
    cfg['pins'].pop(str(S/'isolation_net.py'),None)
    cfg['pins'][str(S/'isolation_net.py')]=sha(S/'isolation_net.py')
    write(S/'FROZEN.json',cfg)
    tr=[i for i,r in enumerate(rows)if r['split']=='train'];assert not({rows[i]['source_sha256']for i in tr}&{r['source_sha256']for r in rows if r['split']=='probe'})
    pools={}
    for origin in ['original_motion_replay','new_reviewed']:
        ix=[i for i in tr if(rows[i]['origin']=='original_motion_replay')==(origin=='original_motion_replay')]
        pools[origin]={k:[i for i in ix if(rows[i]['axis'],rows[i]['target'])==k]for k in sorted({(rows[i]['axis'],rows[i]['target'])for i in ix})}
    rc=old.load('_di_req',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();runs=[]
    assert torch.allclose(reverse_geometry(reverse_geometry(g)),g,atol=1e-5)
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);head=BodyDirection(observer,condition.endswith('symmetric'));head.gm.copy_(torch.tensor(meta['norm']['projected']['mean']));head.gs.copy_(torch.tensor(meta['norm']['projected']['std']));opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[]
            for step in range(cfg['steps']):
                ii=[]
                for groups in pools.values():
                    keys=list(groups)
                    for _ in range(128):k=keys[int(rng.integers(len(keys)))];ii.append(int(rng.choice(groups[k])))
                ii=torch.tensor(ii);out=head(ctx[ii],g[ii]);loss=torch.nn.functional.cross_entropy(out,y[ii]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),2);opt.step()
                if step%200==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print(condition,seed,step+1,float(loss.detach()),flush=True);write(S/'PROGRESS.json',dict(at=stamp(),condition=condition,seed=seed,step=step+1,completed=runs))
            head.eval();dest=S/'runs'/condition/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(head.state_dict(),dest/'model.pt')
            with torch.inference_mode():
                pred=torch.cat([head(ctx[i:i+512],g[i:i+512])for i in range(0,len(rows),512)])
                test=head(torch.randn(32,192),g[:32]);assert torch.equal(test,head(ctx[:32],g[:32]))
                err=float((head(ctx[:32],reverse_geometry(g[:32]))[:,[2,1,0]]-pred[:32]).abs().max())
                if head.symmetric:assert err<1e-4
            np.savez_compressed(dest/'predictions.npz',logits=pred.numpy());runs.append(dict(condition=condition,seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),seconds=time.perf_counter()-start,log=log,context_independence_exact=True,reversal_max_logit_difference=err))
    write(S/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,web_changed=False))
if __name__=='__main__':
    try:main()
    except BaseException:(S/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise

