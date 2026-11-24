from presence_core import *
import time,traceback

def main():
    assert not(P/'FROZEN.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows,data=labeled_data();tr=[i for i,r in enumerate(rows)if r['split']=='train']
    assert {rows[i]['person']for i in tr}=={2,3,4,5,6,7}
    assert not({rows[i]['source_sha256']for i in tr}&{r['source_sha256']for r in rows if r['split']=='probe'})
    pools={};audit=[];common={}
    for f in FAMILIES:
        for y in [0,1]:
            for b in range(5):
                pool=[]
                for i in tr:
                    if rows[i]['family']!=f:continue
                    ix=np.flatnonzero(data[i]['known']&(data[i]['y']==y)&(data[i]['buckets']==b))
                    if len(ix):pool.append((i,ix))
                pools[f,y,b]=pool
                audit.append(dict(family=f,label=y,duration_bucket=b,bags=len(pool),candidates=sum(len(ix)for _,ix in pool),people=sorted({rows[i]['person']for i,_ in pool})))
        common[f]=[b for b in range(5)if pools[f,0,b]and pools[f,1,b]]
        assert common[f],f
    write(P/'FIT_LABEL_AUDIT.json',dict(at=stamp(),rows=audit,matched_buckets=common,
        supervision='partial>=2 same-family direct progression witnesses OR old full-compatible extent = observed progression only; continuous RGB reviewed hold=no local progression; unreviewed excluded',
        limitations='No educational completion. Final-posture confound and unsupported long-hold duration remain. No family2 absence labels.',
        train_people=[2,3,4,5,6,7],probe_people=[8,9],probe_upstream_exposed=True))
    cfg=dict(at=stamp(),seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,
        hypothesis='Separate observed partial progression from complete-extent ranking. The old null rank suppressed real partial motion.',
        architecture='frozen common100 TCN+GRU features; copy old rank hidden layer, reset binary output layer; train separate1160->96->5 head; old extent ranker unchanged',
        loss='BCE with logits; family/label/matched-duration/bag balanced FIT sampling,16 examples per family/label per step',
        duration='Sample only FIT buckets having both supervised labels; frame bins4..6,7..10,11..15,16..21,22+. No duration feature or runtime duration gate added.',
        runtime='fixed logit>=0 binary boundary; inspect all old body-gate admitted candidates. Preserve old chosen extent separately. If chosen unsupported but another candidate has progression, report observed progression with uncertain extent, never omission.',
        family2='exact original runtime output preserved; unsupported new head output ignored',
        evaluation='local held-out-person progression/hold and full public regression; no threshold/checkpoint tuning; final_step_only',
        gate='retain normal/speed known spans and short observed progression while reducing false hold selections; no all-abstain; G2 remains rework until complete error suite passes',
        common100_frozen=True,private_access=False,final8=False,web_changed=False,
        pins={str(p):sha(p)for p in[Path(__file__),P/'presence_core.py',N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',old.L/'TRAINING_COMPLETE.json',P/'FIT_LABEL_AUDIT.json']})
    write(P/'FROZEN.json',cfg);runs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        src=old.L/'runs/local_full'/str(seed)/'model.pt'
        st=torch.load(src,weights_only=True,map_location='cpu');net=ProgressionHead(st['mean'].numpy(),st['std'].numpy());net.load_state_dict(st)
        net.head[2].reset_parameters();net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();log=[]
        for step in range(cfg['steps']):
            xx=[];yy=[];jj=[]
            for f in FAMILIES:
                for y in [0,1]:
                    for _ in range(16):
                        b=int(rng.choice(common[f]));pool=pools[f,y,b];i,ix=pool[int(rng.integers(len(pool)))];k=int(rng.choice(ix))
                        xx.append(data[i]['x'][k]);yy.append(y);jj.append(old.F.index(f))
            z=net(torch.tensor(np.asarray(xx)));v=z[torch.arange(len(xx)),torch.tensor(jj)]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(v,torch.tensor(yy,dtype=torch.float32));assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%300==0 or step==cfg['steps']-1:
                log.append(dict(step=step+1,loss=float(loss.detach())));print('progression',seed,step+1,float(loss.detach()),flush=True)
                write(P/'PROGRESS.json',dict(at=stamp(),seed=seed,step=step+1,completed=runs))
        net.eval();dest=P/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(net.state_dict(),dest/'model.pt')
        with torch.inference_mode():pred={f'logits_{i}':net(torch.tensor(d['x'])).numpy()for i,d in enumerate(data)}
        np.savez_compressed(dest/'predictions.npz',**pred)
        runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),seconds=time.perf_counter()-start,log=log,initial_model=str(src),initial_sha256=sha(src)))
    write(P/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,ranker_changed=False,web_changed=False));print('Progression training complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(P/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
