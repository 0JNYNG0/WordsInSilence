"""One bounded physical wrist-window branch. Does NOT predict clause membership.

prepare reads saved public FIT geometry, never calls a pose/neural encoder.
run requires root-authored PRETRAIN_VERIFICATION.json tied to the frozen hash.
"""
from audit_saved import U,C6,read,sha
from pathlib import Path
import sys,json,datetime,time,collections
import numpy as np
sys.dont_write_bytecode=True
PEOPLE=list(range(2,8));SEEDS=[42,43,44]

def put(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

def features(q,valid,exists,centre):
    """q is native-order body-relative 2D wrist, not pixels or expression code."""
    q=np.asarray(q,float);valid=np.asarray(valid,bool)&np.asarray(exists,bool)&np.isfinite(q).all(1);exists=np.asarray(exists,bool)
    ii=np.flatnonzero(exists);gg=np.flatnonzero(valid);n=len(ii);ng=len(gg)
    assert q.shape==(9,2) and 0<=centre<9 and n>=1
    cur=q[centre]if valid[centre]else np.zeros(2)
    before=valid&(np.arange(9)<centre);after=valid&(np.arange(9)>centre)
    mb=bool(valid[centre]and before.any());ma=bool(valid[centre]and after.any())
    db=q[before].mean(0)-cur if mb else np.zeros(2)
    da=q[after].mean(0)-cur if ma else np.zeros(2)
    net=q[gg[-1]]-q[gg[0]]if ng>=2 else np.zeros(2)
    span=np.ptp(q[gg],axis=0)if ng else np.zeros(2)
    pair=valid[1:]&valid[:-1];path=float(np.linalg.norm(np.diff(q,axis=0)[pair],axis=1).sum())
    maxdist=float(np.linalg.norm(q[gg,None]-q[gg][None,:],axis=-1).max())if ng else 0.
    seen_span=float((gg[-1]-gg[0])/max(n-1,1))if ng>=2 else 0.
    out=np.r_[cur,db,da,net,span,path,maxdist,int(valid[centre]),int(mb),int(ma),ng/n,ng/9,n/9,pair.sum()/max(n-1,1),seen_span].astype(np.float32)
    gaps=[];length=0
    for i in ii:
        if valid[i]:gaps.append(length);length=0
        else:length+=1
    gaps.append(length)
    eligible=bool(ng>=4 and ng/n>=.75 and valid[ii[0]]and valid[ii[-1]]and max(gaps)<=2)
    assert out.shape==(20,)and np.isfinite(out).all()
    return out,eligible,dict(observed=ng,actual=n,coverage=ng/n,max_missing_run=max(gaps),first_seen=bool(valid[ii[0]]),last_seen=bool(valid[ii[-1]]))

def weights(rows,y,eligible,person):
    w=np.zeros(len(rows),float);training=(np.array([r['person']for r in rows])!=person)&eligible&(y>=0)
    # Every class/hand/record receives equal weight; views of a source window share
    # that window's weight, so a crop is not an independent additional annotation.
    for value in [0,1]:
        for side in ['R','L']:
            for pid in PEOPLE:
                if pid==person:continue
                take=training&(y==value)&np.array([r['side']==side and r['person']==pid for r in rows])
                keys=sorted(set(rows[i]['source_key']for i in np.flatnonzero(take)));assert keys,(value,side,pid)
                for key in keys:
                    ix=take&np.array([r['source_key']==key for r in rows]);w[ix]=1/(20*len(keys)*ix.sum())
    assert np.isclose(w.sum(),1.)and not w[~training].any()
    return w.astype(np.float32)

def calibrate_rule(x,y,w):
    score=x[:,11];unique=np.unique(score[w>0]).astype(float)
    thresholds=np.r_[np.nextafter(unique[0],-np.inf),(unique[:-1]+unique[1:])/2,np.nextafter(unique[-1],np.inf)]
    accuracies=np.array([float(w@((score>t)==y))for t in thresholds])
    best=np.flatnonzero(np.isclose(accuracies,accuracies.max(),rtol=0,atol=1e-10));selected=int(best[len(best)//2])
    return dict(threshold=float(thresholds[selected]),train_weighted_accuracy=float(accuracies[selected]),candidates=len(thresholds),tied_best=len(best),selection='middle indexed optimal threshold; never arithmetic midpoint across disjoint optimal regions',feature='max_pairwise_wrist_displacement',operator='>')

def prepare():
    assert not(U/'PHYSICAL_FROZEN.json').exists()
    labels=read(U/'PHYSICAL_SOURCE_LABELS.json');original=read(C6/'INDEX.json')['rows'];z=np.load(C6/'INPUTS.npz');raw=z['absolute_context'].reshape(-1,3,10,5)[:,1]
    data=[];meta=[]
    for label in labels['rows']:
        seqs=[s for s in original if s['clip_id']==label['clip_id']];assert len(seqs)==6
        for s in seqs:
            assert s['split']=='train'and s['person']in PEOPLE and s['source_sha256']==label['source_sha256']
            idx=np.arange(s['start'],s['stop']);frames=z['source_frame_id'][idx]
            assert len(frames)==len(set(frames.tolist()))
            centre_global=int(idx[np.flatnonzero(frames==label['frame'])[0]])
            # Index by original native time, then retain the saved forward/reverse
            # order. Clip boundaries have no fabricated repeated frames.
            chosen=idx[np.abs(frames-label['frame'])<=4]
            positions=chosen-centre_global+4
            assert ((positions>=0)&(positions<9)).all()
            for side,axis,name in [('R',0,'right'),('L',1,'left')]:
                q=np.zeros((9,2),np.float32);valid=np.zeros(9,bool);exists=np.zeros(9,bool)
                q[positions]=raw[chosen,axis,:2];valid[positions]=raw[chosen,axis,4]>.5;exists[positions]=True
                x,ok,obs=features(q,valid,exists,4)
                y={'M':1,'H':0,'U':-1}[label[name]]
                sid=f"{label['key']}:{side}";meta.append(dict(source_key=sid,clip_id=label['clip_id'],person=s['person'],source_sha256=label['source_sha256'],word_id=label['word_id'],kind=label['kind'],side=side,centre_frame=label['frame'],view=s['view'],reverse=s['reverse'],label=label[name],observation=obs,source_frames=z['source_frame_id'][chosen].tolist(),source_indices=chosen.tolist(),positions=positions.tolist()))
                data.append((x,ok,y,q,valid,exists))
    assert len(data)==1380
    arrays=dict(x=np.stack([d[0]for d in data]),eligible=np.array([d[1]for d in data]),y=np.array([d[2]for d in data],np.int8),q=np.stack([d[3]for d in data]),valid=np.stack([d[4]for d in data]),exists=np.stack([d[5]for d in data]),person=np.array([r['person']for r in meta],np.int8))
    np.savez_compressed(U/'PHYSICAL_INPUTS.npz',**arrays);put(U/'PHYSICAL_INDEX.json',dict(rows=meta,source_windows=115,source_hand_cells=230,view_time_repeats=6))
    folds=[]
    for person in PEOPLE:
        w=weights(meta,arrays['y'],arrays['eligible'],person);tr=w>0;te=arrays['person']==person
        assert set(r['source_sha256']for i,r in enumerate(meta)if tr[i]).isdisjoint(r['source_sha256']for i,r in enumerate(meta)if te[i])
        q=arrays['x'][tr].astype(float);norm=dict(mean=q.mean(0).tolist(),std=np.maximum(q.std(0),1e-4).tolist())
        rule=calibrate_rule(arrays['x'],arrays['y'],w)
        d=U/'physical_folds'/str(person);d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'WEIGHTS.npz',weights=w);put(d/'NORMALIZATION_RULE.json',dict(normalization=norm,rule=rule,training_record_ids=[p for p in PEOPLE if p!=person],held_record_id=person))
        folds.append(dict(person=person,weights=str(d/'WEIGHTS.npz'),normalization_rule=str(d/'NORMALIZATION_RULE.json'),train_eligible=int(tr.sum()),held_source_hand_cells=len(set(r['source_key']for i,r in enumerate(meta)if te[i]))))
    files=[Path(__file__),U/'PHYSICAL_SOURCE_LABELS.json',U/'PHYSICAL_CONTRACT_DRAFT.txt',U/'PHYSICAL_INPUTS.npz',U/'PHYSICAL_INDEX.json',C6/'INPUTS.npz',C6/'INDEX.json',C6/'FROZEN.json']
    for f in folds:files.extend([Path(f['weights']),Path(f['normalization_rule'])])
    pins={str(p):sha(p)for p in files}
    put(U/'PHYSICAL_FROZEN.json',dict(status='PREPARED_AWAIT_ROOT_PRETRAIN_REVIEW',created_at=stamp(),task='event.window_contains_wrist_translation',condition='shared_physical_wrist_mlp',folds=folds,seeds=SEEDS,steps=1200,lr=.001,weight_decay=.001,gradient_clip=2.,cpu_threads=2,architecture='20 ->16 GELU ->1 =353 trainable parameters',baseline='same-fold FIT-only calibrated maximum wrist displacement',readout='logit>=0 = M; else H; common eligibility failure =U; labelU never training H',target_source='Direct RGB labels, not pose thresholds or old clause category',comparison='New task only; no C6/C9 metric subtraction',input='Saved body-relative wrist2D and availability; no new detector/encoder/old-model forward',old_models_unchanged=True,operational_use=False,pins=pins))
    put(U/'PHYSICAL_INPUT_AUDIT.json',dict(status='PREPARED',rows=len(meta),source_hand_cells=230,source_known=228,source_unknown=2,eligible_by_view={view:int(sum(d[1]for r,d in zip(meta,data)if r['view']==view))for view in ['full','upper','actual_crop']},class_counts=dict(collections.Counter(r['label']for r in meta)),exposure='All6FIT record IDs have prior research exposure; new readout head held-record CV, not final untouched evaluation',no_training=True))
    print('Physical inputs prepared; root review required before18 trainings.',flush=True)

def run():
    import torch
    from torch import nn
    cfg=read(U/'PHYSICAL_FROZEN.json');verify=read(U/'PHYSICAL_PRETRAIN_VERIFICATION.json')
    assert verify['status']=='PASS'and verify['frozen_sha256']==sha(U/'PHYSICAL_FROZEN.json')
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    assert not(U/'PHYSICAL_TRAINING_COMPLETE.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    z=np.load(U/'PHYSICAL_INPUTS.npz');xx=torch.tensor(z['x']);yy=torch.tensor(z['y'],dtype=torch.float32);done=[]
    for fold in cfg['folds']:
        person=fold['person'];foldmeta=read(fold['normalization_rule']);norm=foldmeta['normalization'];w=np.load(fold['weights'])['weights'];tr=w>0;te=z['person']==person
        for seed in cfg['seeds']:
            if(U/'STOP').exists():raise RuntimeError('STOP requested at completed model boundary')
            d=U/'physical_runs'/f'{person}_{seed}';d.mkdir(parents=True,exist_ok=True)
            if(d/'COMPLETE.json').exists():done.append(read(d/'COMPLETE.json'));continue
            assert not(d/'STARTED.json').exists(),'Preserve and audit interrupted run before retry.'
            put(d/'STARTED.json',dict(at=stamp(),person=person,seed=seed,frozen_sha256=sha(U/'PHYSICAL_FROZEN.json')))
            start=time.perf_counter();torch.manual_seed(seed)
            net=nn.Sequential(nn.Linear(20,16),nn.GELU(),nn.Linear(16,1));assert sum(p.numel()for p in net.parameters())==353
            mean=torch.tensor(norm['mean'],dtype=torch.float32);std=torch.tensor(norm['std'],dtype=torch.float32);xt=(xx-mean)/std
            opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);wt=torch.tensor(w[tr]);lossfn=nn.BCEWithLogitsLoss(reduction='none');net.train()
            for step in range(cfg['steps']):
                opt.zero_grad(set_to_none=True);loss=(lossfn(net(xt[tr]).squeeze(1),yy[tr])*wt).sum();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
            net.eval()
            with torch.inference_mode():logits=net(xt).squeeze(1).numpy()
            model=d/'model.pt';pred=d/'predictions.npz';torch.save(dict(state_dict=net.state_dict(),mean=mean,std=std,task=cfg['task'],person=person,seed=seed),model)
            np.savez_compressed(pred,logits=logits,held=te,eligible=z['eligible'],rule_prediction=z['x'][:,11]>foldmeta['rule']['threshold'])
            result=dict(person=person,seed=seed,model=str(model),model_sha256=sha(model),prediction=str(pred),prediction_sha256=sha(pred),training_seconds=time.perf_counter()-start,final_training_loss=float(loss),frozen_sha256=sha(U/'PHYSICAL_FROZEN.json'))
            put(d/'COMPLETE.json',result);done.append(result);print(f'Physical run complete {len(done)}/18 person={person} seed={seed}',flush=True)
    assert len(done)==18
    put(U/'PHYSICAL_TRAINING_COMPLETE.json',dict(status='COMPLETE_AWAIT_INDEPENDENT_VERIFICATION',runs=done,frozen_sha256=sha(U/'PHYSICAL_FROZEN.json')))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1]in ['prepare','run']
    globals()[sys.argv[1]]()
