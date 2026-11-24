from static_core import *
import time,traceback

def main():
    assert not(S/'FROZEN.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows,data=labeled_data();meta=read(S/'INPUTS_COMPLETE.json');static={};valid={};paired={};audit=[]
    for r in meta['bags']:
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);i=r['bag_index'];x=z['x'];mapping=z['mapping'];axis=old.AX[r['family']]*2
        assert all(r[k]==rows[i][k]for k in ['clip_id','split','view','reverse','family'])
        assert np.array_equal(z['positive_indices'],np.flatnonzero(data[i]['known']))
        assert np.array_equal(mapping[:,0],np.repeat(z['positive_indices'],3))
        assert np.array_equal(mapping[:,1],np.tile(np.arange(3),len(z['positive_indices'])))
        # z is [local-encoder64, relation100, local-hand126], pooled by thirds.
        # Relation channel8 is the original binary body-visibility mask, never standardized.
        coverage=x[:,64+axis*10+8];assert np.isin(coverage,[0,1]).all()
        ok=coverage>.5;static[i]=dict(x=x,mapping=mapping);valid[i]=ok;paired[i]={int(k):np.flatnonzero(ok&(mapping[:,0]==k))for k in z['positive_indices']}
        assert all(len(v)>0 for v in paired[i].values()),'Visible candidate endpoints should give a supported anchor'
        audit.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],split=r['split'],view=r['view'],reverse=r['reverse'],generated=len(ok),eligible=int(ok.sum()),excluded_unobserved=int((~ok).sum())))
    tr=[i for i,r in enumerate(rows)if r['split']=='train'];pools={};common={};posbags={}
    for f in FAMILIES:
        posbags[f]=[i for i in tr if rows[i]['family']==f and rows[i]['origin']=='positive_source_extent']
        assert posbags[f]
        for y in [0,1]:
            for b in range(5):
                pools[f,y,b]=[(i,np.flatnonzero(d['known']&(d['y']==y)&(d['buckets']==b)))for i,d in enumerate(data)if i in tr and rows[i]['family']==f]
                pools[f,y,b]=[(i,ix)for i,ix in pools[f,y,b]if len(ix)]
        common[f]=[b for b in range(5)if pools[f,0,b]and pools[f,1,b]];assert common[f]
    write(S/'FIT_LABEL_AUDIT.json',dict(at=stamp(),matched_buckets=common,static_anchor_eligibility=audit,scope='Zero input movement with visible relevant body relationship; unobserved anchors excluded before learning/negative evaluation',feature_mask_index='pooled part0:64 + (axis*2)*10 + 8'))
    cfg=dict(at=stamp(),seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,
        architecture='copy separate progression-head P, same1160->96->5; original extent/common100 unchanged',
        objective='balanced BCE: per family8 original-progress/paired-static pairs plus8 duration-matched actual-progress/real-hold pairs. Each paired static uses same source candidate, length, dt; anchor uniformly chosen from all observed start/mid/end.',
        selection='all source candidates with verified full/partial progression, not failed cases; sample FIT bags equally and positive duration buckets equally. Probe never training/normalization/selection.',
        static_labels='mathematical no-change in observed coordinates, not linguistic incorrectness or sign omission. Invisible anchors excluded.',
        runtime='identical P progression-vs-extent separation and fixed>=0 boundary; family2 original output',
        selection_rule='final1200step only, no probe-driven threshold or checkpoint selection',
        gate='real normal/speed/cross/full/partial preservation plus same-pose static rejection; still needs full G2 order/omission correctness',
        pins={str(p):sha(p)for p in[Path(__file__),S/'static_core.py',S/'INPUTS_COMPLETE.json',S/'FIT_LABEL_AUDIT.json',P/'presence_core.py',P/'TRAINING_COMPLETE.json',P/'FROZEN.json']},private_access=False,final8=False,web_changed=False)
    write(S/'FROZEN.json',cfg);runs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);initial=P/'runs'/str(seed)/'model.pt';st=torch.load(initial,weights_only=True,map_location='cpu');net=ProgressionHead(st['mean'].numpy(),st['std'].numpy());net.load_state_dict(st);net.train();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();logs=[]
        for step in range(cfg['steps']):
            xx=[];yy=[];jj=[]
            for f in FAMILIES:
                j=old.F.index(f)
                for _ in range(8):
                    i=int(rng.choice(posbags[f]));d=data[i];ix=np.flatnonzero(d['known']);b=int(rng.choice(np.unique(d['buckets'][ix])));k=int(rng.choice(ix[d['buckets'][ix]==b]));s=int(rng.choice(paired[i][k]));xx.extend([d['x'][k],static[i]['x'][s]]);yy.extend([1,0]);jj.extend([j,j])
                    b=int(rng.choice(common[f]))
                    for y in [1,0]:
                        pool=pools[f,y,b];bi,ki=pool[int(rng.integers(len(pool)))];k=int(rng.choice(ki));xx.append(data[bi]['x'][k]);yy.append(y);jj.append(j)
            z=net(torch.tensor(np.asarray(xx)));loss=torch.nn.functional.binary_cross_entropy_with_logits(z[torch.arange(len(xx)),torch.tensor(jj)],torch.tensor(yy,dtype=torch.float32));assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            if step%300==0 or step==cfg['steps']-1:
                logs.append(dict(step=step+1,loss=float(loss.detach())));print('paired-static',seed,step+1,float(loss.detach()),flush=True);write(S/'PROGRESS.json',dict(at=stamp(),seed=seed,step=step+1,completed=runs))
        net.eval();dest=S/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);assert not(dest/'model.pt').exists();torch.save(net.state_dict(),dest/'model.pt')
        with torch.inference_mode():
            pred={f'logits_{i}':net(torch.tensor(d['x'])).numpy()for i,d in enumerate(data)}
            staticpred={f'logits_{i}':net(torch.tensor(d['x'])).numpy()for i,d in static.items()}
        np.savez_compressed(dest/'predictions.npz',**pred);np.savez_compressed(dest/'static_predictions.npz',**staticpred)
        runs.append(dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),static_predictions=str(dest/'static_predictions.npz'),static_predictions_sha256=sha(dest/'static_predictions.npz'),initial_model=str(initial),initial_sha256=sha(initial),seconds=time.perf_counter()-start,log=logs))
    write(S/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,common100_changed=False,ranker_changed=False,web_changed=False));print('Paired-static training complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(S/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
