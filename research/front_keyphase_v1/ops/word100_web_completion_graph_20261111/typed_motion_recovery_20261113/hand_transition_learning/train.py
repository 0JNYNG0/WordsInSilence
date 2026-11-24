from hand_core import *
from collections import defaultdict
import time
from torch.nn import functional as F

def materialize(rec,cache):
    z=cache[rec['clip_id'],rec['side']];a,b=rec['span'];x=z['x'][a:b+1].copy()
    y=z['y'][a:b+1].copy();mask=z['mask'][a:b+1].copy()*z['available'][a:b+1,None]
    if rec['kind']=='synthetic_repeated_pose':
        k=rec['anchor'];x[:]=z['x'][k];y[:]=z['y'][k];mask[:]=z['mask'][k]*z['available'][k]
    if rec['reverse']:x=x[::-1].copy();y=y[::-1].copy();mask=mask[::-1].copy()
    return x,y,mask

def grouped(ids,key):
    out=defaultdict(list)
    for i in ids:out[key(i)].append(i)
    return dict(out)

def main():
    assert not(U/'TRAINING_CONTRACT.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=read(U/'FROZEN.json');meta=read(U/'INPUTS_COMPLETE.json')
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    cache={};seqrows={}
    for r in meta['sequences']:
        assert sha(r['path'])==r['sha256'];cache[r['clip_id'],r['side']]=np.load(r['path']);seqrows[r['clip_id'],r['side']]=r
    labels=meta['labels'];win=meta['windows']+meta['controls'];frames=[]
    for cid,f,side in sorted({(r['clip_id'],r['index'],r['side']) for r in labels}):
        z=cache[cid,side];a=max(0,f-3);b=min(len(z['x'])-1,f+3)
        frames.append(dict(clip_id=cid,side=side,person=seqrows[cid,side]['person'],index=f,source_frame=int(z['frames'][f]),span=[a,b],center=f-a))
    write(U/'EVALUATION_ITEMS.json',dict(frames=frames,windows=win,labels=labels))
    contract=dict(at=stamp(),pins={str(p):sha(p) for p in[Path(__file__),U/'hand_core.py',U/'FROZEN.json',U/'INPUTS_COMPLETE.json',U/'EVALUATION_ITEMS.json']},
        training_only_people=list(range(1,8)),folds=cfg['folds'],seeds=cfg['seeds'],steps=cfg['steps'],
        normalization='All source rows of trainpeople, one copy per side/source. Fit mean/std for42coords with std floor.05; masks mean0/std1. No heldperson frame used in fitstats.',
        loss='0.5 sampled center attribute BCE +0.5 perattribute/target-balanced window attribute BCE + transition CE. All attribute loss entries explicitly labelled and available.',
        sequence_sampling='Uniform class; uniform person within class. For stable class choose real/synthetic with .5/.5, then person. No sample chosen by score.',
        output='Every heldperson labelled frame, realwindow and syntheticconstant once perseed. No checkpoint/threshold tuning. No allFIT production model yet.',
        previous_models_reinferred=False,web_changed=False,final8=False)
    write(U/'TRAINING_CONTRACT.json',contract);runs=[]
    for held in cfg['folds']:
        fitseq=[r for r in meta['sequences'] if r['person']!=held];trainpeople=sorted({r['person'] for r in fitseq});assert held not in trainpeople
        fitx=np.concatenate([cache[r['clip_id'],r['side']]['x'] for r in fitseq]);mean=fitx.mean(0);std=fitx.std(0).clip(.05);mean[42:]=0;std[42:]=1
        pool=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(labels):
            if r['person']!=held and r['available']:pool[r['attribute'],int(r['target'])][r['person']].append(i)
        assert len(pool)==8,'Every attribute/target must have training evidence'
        seqpool=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(win):
            if r['person']!=held:seqpool[r['target'],r['kind']=='synthetic_repeated_pose'][r['person']].append(i)
        assert all(k in seqpool for k in [(0,False),(1,False),(1,True),(2,False)])
        testframes=[i for i,r in enumerate(frames) if r['person']==held];testwin=[i for i,r in enumerate(win) if r['person']==held]
        split=dict(heldperson=held,trainpeople=trainpeople,train_source_hashes=sorted({r['source_sha256'] for r in fitseq}),
            held_source_hashes=sorted({r['source_sha256'] for r in meta['sequences'] if r['person']==held}),
            test_frame_ids=testframes,test_window_ids=testwin)
        assert not(set(split['train_source_hashes'])&set(split['held_source_hashes']))
        for seed in cfg['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=HandObserver(mean,std);optim=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);log=[];tic=time.perf_counter()
            for step in range(cfg['steps']):
                assert not(U/'STOP_REQUESTED').exists()
                attrkeys=list(pool);batch=[];target=[];attr=[];cent=[]
                for _ in range(cfg['batch_attributes']):
                    k=attrkeys[int(rng.integers(len(attrkeys)))];person=int(rng.choice(list(pool[k])));r=labels[int(rng.choice(pool[k][person]))]
                    z=cache[r['clip_id'],r['side']];i=r['index'];a=max(0,i-3);b=min(len(z['x'])-1,i+3)
                    batch.append(z['x'][a:b+1]);target.append(r['target']);attr.append(r['attribute']);cent.append(i-a)
                bx,bl=padded(batch);out=m(bx,bl);af=F.binary_cross_entropy_with_logits(out['attributes'][torch.arange(len(batch)),cent,attr],torch.tensor(target))
                batch=[];yy=[];mm=[];targets=[]
                for i in range(cfg['batch_transitions']):
                    cls=i%3;synthetic=cls==1 and bool(rng.integers(2));q=seqpool[cls,synthetic];person=int(rng.choice(list(q)));r=win[int(rng.choice(q[person]))]
                    x,y,mask=materialize(r,cache);batch.append(x);yy.append(y);mm.append(mask);targets.append(cls)
                bx,bl=padded(batch);out=m(bx,bl);ys=torch.zeros((len(batch),bx.shape[1],4));ms=torch.zeros_like(ys)
                for i,(y,mask) in enumerate(zip(yy,mm)):ys[i,:len(y)]=torch.tensor(y);ms[i,:len(y)]=torch.tensor(mask)
                raw=F.binary_cross_entropy_with_logits(out['attributes'],ys,reduction='none');parts=[]
                for j in range(4):
                    for value in (0,1):
                        selected=(ms[:,:,j]>0)&(ys[:,:,j]==value)
                        if selected.any():parts.append(raw[:,:,j][selected].mean())
                aw=torch.stack(parts).mean();tr=F.cross_entropy(out['transition'],torch.tensor(targets));loss=.5*af+.5*aw+tr
                assert torch.isfinite(loss), ('nonfinite_loss',held,seed,step)
                optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);optim.step()
                if step in [0,199,399,cfg['steps']-1]:log.append(dict(step=step+1,loss=float(loss.detach()),frame=float(af.detach()),window_attribute=float(aw.detach()),transition=float(tr.detach())))
            dest=U/'folds'/str(held)/str(seed);dest.mkdir(parents=True);m.eval();mp=dest/'model.pt';torch.save(m.state_dict(),mp)
            fl=[];wl=[]
            with torch.inference_mode():
                for off in range(0,len(testframes),128):
                    ids=testframes[off:off+128];rr=[frames[i] for i in ids];xx=[cache[r['clip_id'],r['side']]['x'][r['span'][0]:r['span'][1]+1] for r in rr];bx,bl=padded(xx);oo=m(bx,bl)
                    fl.extend(oo['attributes'][torch.arange(len(rr)),[r['center'] for r in rr]].numpy())
                for off in range(0,len(testwin),128):
                    ids=testwin[off:off+128];xx=[materialize(win[i],cache)[0] for i in ids];bx,bl=padded(xx);wl.extend(m(bx,bl)['transition'].numpy())
            pp=dest/'predictions.npz';np.savez_compressed(pp,frame_ids=testframes,frame_logits=np.asarray(fl),window_ids=testwin,window_logits=np.asarray(wl))
            rec=dict(heldperson=held,seed=seed,model=str(mp),model_sha256=sha(mp),predictions=str(pp),predictions_sha256=sha(pp),seconds=time.perf_counter()-tic,split=split,log=log)
            write(dest/'COMPLETE.json',rec);runs.append(rec);write(U/'TRAINING_PROGRESS.json',dict(at=stamp(),done=len(runs),total=18,last=dict(heldperson=held,seed=seed)));print('hand transition fold',held,'seed',seed,'complete',len(runs),18,flush=True)
    write(U/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=runs,folds=6,seeds=3,allfit_model_trained=False,old_models_changed=False,private_access=False,final8=False,web_changed=False));print('All fold diagnostics complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('TRAIN');raise
