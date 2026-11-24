"""Separate body presence classification; root runs after code review."""
from body_core import *
import time

def main():
    assert not(B/'FROZEN.json').exists();assert not(B/'TRAINING_COMPLETE.json').exists();stop_if_requested()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows,data=labeled_data();meta=read(B/'INPUTS_COMPLETE.json');check_pins(meta['pins']);static={};paired={};anchor_audit=[]
    for m in meta['static_bags']:
        i=m['bag_index'];r=rows[i];assert all(m[k]==r[k] for k in ['clip_id','split','view','reverse','family','source_sha256'])
        assert sha(m['path'])==m['sha256'];z=np.load(m['path']);x=z['x'];mapping=z['mapping'];check_body_features(x);positive=np.flatnonzero(data[i]['known'])
        assert np.array_equal(z['positive_indices'],positive) and np.array_equal(mapping[:,0],np.repeat(positive,3)) and np.array_equal(mapping[:,1],np.tile(np.arange(3),len(positive)))
        axis=old.AX[r['family']]*2;coverage=x[:,64+axis*10+8];assert np.isin(coverage,[0,1]).all();ok=coverage>.5;assert np.array_equal(ok,z['eligible'])
        static[i]=dict(x=x,mapping=mapping);paired[i]={int(k):np.flatnonzero(ok&(mapping[:,0]==k)) for k in positive};assert all(len(v) for v in paired[i].values())
        anchor_audit.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],split=r['split'],view=r['view'],reverse=r['reverse'],generated=len(ok),eligible=int(ok.sum()),excluded_unobserved=int((~ok).sum())))
    tr=[i for i,r in enumerate(rows) if r['split']=='train'];assert {rows[i]['person'] for i in tr}=={2,3,4,5,6,7}
    assert not {rows[i]['source_sha256'] for i in tr}&{r['source_sha256'] for r in rows if r['split']=='probe'}
    pools={};matched={};posbags={}
    for f in FAMILIES:
        posbags[f]=[i for i in tr if rows[i]['family']==f and rows[i]['origin']=='positive_source_extent'];assert posbags[f]
        for y in [0,1]:
            for b in range(5):
                pool=[(i,np.flatnonzero(data[i]['known']&(data[i]['y']==y)&(data[i]['buckets']==b))) for i in tr if rows[i]['family']==f]
                pools[f,y,b]=[(i,ix) for i,ix in pool if len(ix)]
        matched[f]=[b for b in range(5) if pools[f,0,b] and pools[f,1,b]];assert matched[f]
    write(B/'FIT_LABEL_AUDIT.json',dict(at=stamp(),matched_buckets=matched,static_anchor_eligibility=anchor_audit,scope='Same known full/partial observed progression and reviewed real holds; paired static has zero input movement, not linguistic wrong-sign labels. Relevant-body invisible anchors excluded.'))
    initial_rows={r['seed']:r for r in read(old.L/'TRAINING_COMPLETE.json')['rows'] if r['condition']=='local_no_hands'};assert set(initial_rows)=={42,43,44}
    initpins={r['model']:r['model_sha256'] for r in initial_rows.values()};check_pins(initpins)
    pinned=[Path(__file__),B/'body_core.py',B/'INPUTS_COMPLETE.json',B/'FIT_LABEL_AUDIT.json',B/'PREPARATION_CONTRACT.json',P/'presence_core.py',N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',old.L/'TRAINING_COMPLETE.json',S/'hand_information_contract/RESULTS.json']
    cfg=dict(at=stamp(),seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,
        architecture='same1160->96->5. Initialize frozen-copy local_no_hands hidden+normalization; reset last linear for separate body progression classification. Original ranker/common100 untouched.',
        purpose='Body progression remains inspectable despite missing/wrong handshape. This is not no-hands action identity/extent ranking.',
        objective='BCE: per family8 same-source/same-span positive-static pairs plus8 FIT-duration-matched real-positive/real-hold pairs;128 examples per step. All observed start/middle/end anchors sampled, no failure-driven selection.',
        separation='FIT02..07 only for updates; prior-exposed probe08/09 only saved evaluation. Existing normalization retained; no DEV threshold/checkpoint choice.',
        runtime='Fixed>=0 progression support, original span preserved; no semantic completion/omission claim. family2, handshape heads, original extent and C/T3-F/common100 preserved.',
        gate='Retain normal/speed/cross, short progression and wrong-handshape attempted movement; reject stationary observation controls; full G2 remains independent.',
        pins={**{str(p):sha(p) for p in pinned},**initpins},private_access=False,final8=False,web_changed=False)
    write(B/'FROZEN.json',cfg);runs=[]
    for seed in cfg['seeds']:
        stop_if_requested();torch.manual_seed(seed);rng=np.random.default_rng(seed)
        initial=Path(initial_rows[seed]['model']);assert sha(initial)==initial_rows[seed]['model_sha256']
        st=torch.load(initial,map_location='cpu',weights_only=True);net=ProgressionHead(st['mean'].numpy(),st['std'].numpy());net.load_state_dict(st);net.head[2].reset_parameters();net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();logs=[]
        for step in range(cfg['steps']):
            if step%50==0:stop_if_requested()
            xx=[];yy=[];jj=[]
            for f in FAMILIES:
                j=old.F.index(f)
                for _ in range(8):
                    i=int(rng.choice(posbags[f]));d=data[i];ix=np.flatnonzero(d['known']);b=int(rng.choice(np.unique(d['buckets'][ix])));k=int(rng.choice(ix[d['buckets'][ix]==b]));si=int(rng.choice(paired[i][k]))
                    xx.extend([d['x'][k],static[i]['x'][si]]);yy.extend([1,0]);jj.extend([j,j])
                    b=int(rng.choice(matched[f]))
                    for y in [1,0]:
                        pool=pools[f,y,b];bi,ki=pool[int(rng.integers(len(pool)))];k=int(rng.choice(ki));xx.append(data[bi]['x'][k]);yy.append(y);jj.append(j)
            assert len(xx)==128
            logits=net(torch.tensor(np.asarray(xx)));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[torch.arange(len(xx)),torch.tensor(jj)],torch.tensor(yy,dtype=torch.float32));assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%300==0 or step==cfg['steps']-1:
                logs.append(dict(step=step+1,loss=float(loss.detach())));write(B/'PROGRESS.json',dict(at=stamp(),seed=seed,step=step+1,completed=runs));print('body progression',seed,step+1,float(loss.detach()),flush=True)
        net.eval();dest=B/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True)
        assert not any((dest/n).exists() for n in ['model.pt','predictions.npz','static_predictions.npz'])
        assert torch.equal(net.mean,st['mean']) and torch.equal(net.std,st['std']);torch.save(net.state_dict(),dest/'model.pt')
        with torch.inference_mode():
            predictions={f'logits_{i}':net(torch.tensor(d['x'])).numpy() for i,d in enumerate(data)}
            static_predictions={f'logits_{i}':net(torch.tensor(d['x'])).numpy() for i,d in static.items()}
        np.savez_compressed(dest/'predictions.npz',**predictions);np.savez_compressed(dest/'static_predictions.npz',**static_predictions)
        runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),static_predictions=str(dest/'static_predictions.npz'),static_predictions_sha256=sha(dest/'static_predictions.npz'),initial_model=str(initial),initial_sha256=sha(initial),seconds=time.perf_counter()-start,log=logs))
    check_pins(cfg['pins']);check_pins(meta['pins'])
    write(B/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,ranker_changed=False,handshape_head_changed=False,web_changed=False,private_access=False,final8=False))
    print('Body-only progression training complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('TRAIN');raise
