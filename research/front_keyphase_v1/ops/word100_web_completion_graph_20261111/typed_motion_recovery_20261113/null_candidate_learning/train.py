from null_net import *
import time,traceback
def main():
    assert not(N/'FROZEN.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    audit=read(N/'FIT_BASELINE_AUDIT.json');assert audit['training_justified']and audit['no_probe_scoring']
    pos=read(old.L/'INPUTS_COMPLETE.json');neg=read(N/'INPUTS_COMPLETE.json');rows=[dict(r,origin='positive_source_extent')for r in pos['bags']if r['family']in[3,4,5,7]]+neg['bags'];data=[]
    for r in rows:
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);x=z['local_full'];y=z['y'];assert np.isfinite(x).all()
        assert bool(y.any())==(r['origin']=='positive_source_extent');data.append(dict(x=x,y=y))
    tr=[i for i,r in enumerate(rows)if r['split']=='train'];assert {rows[i]['person']for i in tr}=={2,3,4,5,6,7}
    assert not({rows[i]['source_sha256']for i in tr}&{r['source_sha256']for r in rows if r['split']=='probe'})
    pools={(f,o):[i for i in tr if rows[i]['family']==f and rows[i]['origin']==o]for f in[3,4,5,7]for o in ['positive_source_extent','reviewed_local_hold']};assert all(pools.values())
    cfg=dict(at=stamp(),scope='Generic observed-movement families3/4/5/7; family2 keeps old selector, unsupported absence class. No word-specific branch.',hypothesis='Rank-only supervision lacks no-candidate cases. Add learned null choice using continuously reviewed non-progression, preserve positive extent supervision.',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,loss='same listwise logsumexp with appended learned family-null logit; positive bags sum existing positive candidates, local-hold bags target null',sampling='2 positive+2 hold bags per family; all positive spans and up to56 other spans per bag',selection='final_step_only; no probe checkpoint/threshold choice',runtime='same candidate generator/body visibility/presence gate; select null when its logit exceeds all admitted candidate logits. family2 uses original frozen output.',gate='allseeds preserve normal/speed/cross known anchors, reduce FIT and held-out-person local hold false selections without all-abstain. Then full error suite; G2 is not automatically passed.',limits='local non-progression != whole sign omission/completion; sparse normal action witnesses not dense human truth; upstream-exposed probe, not final accuracy',pins={str(p):sha(p)for p in[Path(__file__),N/'null_net.py',N/'INPUTS_COMPLETE.json',N/'FIT_BASELINE_AUDIT.json',old.L/'INPUTS_COMPLETE.json',old.L/'TRAINING_COMPLETE.json',HERE/'absence_review/REVIEW_COMPLETE.json',HERE/'absence_review/extension/REVIEW_COMPLETE.json']},common100_frozen=True,private_access=False,final8=False,web_changed=False)
    write(N/'FROZEN.json',cfg);write(N/'TRAINING_INPUT_INDEX.json',dict(rows=rows,train_indices=tr));runs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);path=old.L/'runs/local_full'/str(seed)/'model.pt';st=torch.load(path,map_location='cpu',weights_only=True);rank=old.LocalRankHead(st['mean'].numpy(),st['std'].numpy());rank.load_state_dict(st);net=NullRank(rank);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[]
        for step in range(cfg['steps']):
            chosen=[int(i)for pool in pools.values()for i in rng.choice(pool,2,replace=False)];xs=[];info=[];offset=0
            for i in chosen:
                d=data[i];pi=np.flatnonzero(d['y']);ni=np.flatnonzero(~d['y']);ni=rng.choice(ni,min(56,len(ni)),replace=False);ix=np.r_[pi,ni];xs.append(d['x'][ix]);info.append((offset,offset+len(ix),len(pi),old.F.index(rows[i]['family'])));offset+=len(ix)
            logits=net(torch.tensor(np.concatenate(xs)));losses=[]
            for a,b,np_,j in info:
                den=torch.logsumexp(torch.cat([logits[a:b,j],net.null_logits[j:j+1]]),0)
                num=torch.logsumexp(logits[a:a+np_,j],0)if np_ else net.null_logits[j]
                losses.append(den-num)
            loss=torch.stack(losses).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('null',seed,step+1,float(loss.detach()),flush=True);write(N/'PROGRESS.json',dict(at=stamp(),seed=seed,step=step+1,completed=runs))
        net.eval();dest=N/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(net.state_dict(),dest/'model.pt')
        with torch.inference_mode():pred={f'logits_{i}':net(torch.tensor(d['x'])).numpy()for i,d in enumerate(data)}
        np.savez_compressed(dest/'predictions.npz',**pred,null_logits=net.null_logits.detach().numpy());runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),seconds=time.perf_counter()-start,initial_model=str(path),initial_sha256=sha(path),log=log))
    write(N/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,web_changed=False,private_access=False,final8=False));print('Null training complete',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:(N/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
