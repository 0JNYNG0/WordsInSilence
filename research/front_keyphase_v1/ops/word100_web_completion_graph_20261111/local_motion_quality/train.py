"""Train matched isolated quality copies after data/code admission.

Neither the original Conv/GRU, shared100 nor deployed checkpoints are written.
The REAL07 split is excluded only from this continuation. Older encoders,
localizers and the copied quality MLP already trained on these FIT people.
"""
from pathlib import Path
import sys,json,time,hashlib,traceback
import numpy as np,torch
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
from quality_model import LocalMotionQuality


def pack(items,indices):
    selected=[items[i] for i in indices];n=max(len(v['sequence']) for v in selected)
    sequence=np.zeros((len(selected),n,11),np.float32)
    for i,v in enumerate(selected):sequence[i,:len(v['sequence'])]=v['sequence']
    return (torch.tensor(np.stack([v['summary'] for v in selected])),torch.tensor(sequence),
      torch.tensor([len(v['sequence']) for v in selected]),
      torch.tensor([v['extent_target'] for v in selected],dtype=torch.long),
      torch.tensor([v['extent_mask'] for v in selected],dtype=torch.bool),
      torch.tensor([v['aux_target'] for v in selected],dtype=torch.float32),
      torch.tensor([v['aux_mask'] for v in selected],dtype=torch.bool))


def main():
    assert not (W/'TRAINING_COMPLETE.json').exists()
    admission=read(W/'TRAIN_ADMISSION.json');assert admission['approved'] is True
    for name in ['INPUTS_COMPLETE.json','FROZEN.json']:
        assert hashlib.sha256((W/name).read_bytes()).hexdigest()==admission['pins'][name]
    cfg=read(W/'FROZEN.json');manifest=read(W/'INPUTS_COMPLETE.json')
    assert cfg['conditions']==['B','C'] and cfg['seeds']==[42,43,44]
    for path,sha in cfg['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    train=[];probe=[]
    for row in manifest['rows']:
        assert row['split'] in ['train','head_probe']
        p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        with np.load(p,allow_pickle=False) as z:
            v={k:z[k].copy() for k in ['summary','sequence']}
        for k in ['extent_target','extent_mask','aux_target','aux_mask']:v[k]=row[k]
        v['clip_id']=row['clip_id'];v['path']=str(p)
        (train if row['split']=='train' else probe).append(v)
    assert train and probe
    assert not ({v['clip_id'] for v in train}&{v['clip_id'] for v in probe})
    counts=np.bincount([v['extent_target'] for v in train if v['extent_mask']],minlength=4)
    assert (counts>0).all();classweight=torch.tensor(1/np.sqrt(counts),dtype=torch.float32)
    aux_enabled=bool(manifest.get('aux_enabled',False))
    assert aux_enabled or not any(v['aux_mask'] for v in train+probe)
    results=[]
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            dst=W/'runs'/condition/str(seed);done=dst/'TRAINING.json'
            if done.exists():
                r=read(done);assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];results.append(r);continue
            torch.manual_seed(seed);net=LocalMotionQuality(condition=condition,seed=seed,ops_root=O)
            optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            start=time.perf_counter();history=[];steps=0
            for epoch in range(cfg['epochs']):
                net.train();indices=np.random.default_rng(seed*1000+epoch).permutation(len(train));losses=[]
                for offset in range(0,len(indices),cfg['batch']):
                    batch=pack(train,indices[offset:offset+cfg['batch']]);summary,seq,lens,y,ym,ay,am=batch
                    extent,aux=net(summary,seq,lens);loss=extent.sum()*0
                    if ym.any():loss+=torch.nn.functional.cross_entropy(extent[ym],y[ym],weight=classweight)
                    if aux_enabled and am.any():loss+=cfg['aux_loss_weight']*torch.nn.functional.binary_cross_entropy_with_logits(aux[am],ay[am])
                    assert ym.any() or am.any();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);optimizer.step();steps+=1;losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            net.eval();predictions=[]
            with torch.inference_mode():
                for offset in range(0,len(probe),cfg['batch']):
                    part=probe[offset:offset+cfg['batch']];summary,seq,lens,y,ym,ay,am=pack(part,range(len(part)))
                    extent,aux=net(summary,seq,lens);pp=extent.softmax(-1).numpy()
                    for v,p in zip(part,pp):predictions.append(dict(clip_id=v['clip_id'],input_path=v['path'],extent_target=v['extent_target'],extent_mask=v['extent_mask'],probability=p.tolist(),prediction=int(p.argmax()),auxiliary_used=False))
            dst.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dst/'model.pt');write(dst/'HEAD_PROBE.json',dict(rows=predictions,scope='REAL07 excluded only from this continuation. Existing encoders, localizer and copied quality head previously trained on these clips. Diagnostic only, not unseen-person generalization.'))
            r=dict(condition=condition,seed=seed,path=str(dst/'model.pt'),sha256=hashlib.sha256((dst/'model.pt').read_bytes()).hexdigest(),seconds=time.perf_counter()-start,steps=steps,epochs=cfg['epochs'],history=history,class_counts=counts.tolist(),train_items=len(train),probe_items=len(probe),aux_enabled=aux_enabled,checkpoint_selection='last epoch only; probe not used to select epochs/seed')
            write(done,r);results.append(r);print(condition,seed,round(r['seconds'],2),'seconds',steps,'steps',flush=True)
    write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),runs=results,web_changed=False,private_access=False,original_models_changed=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        W.mkdir(exist_ok=True);(W/f'FIRST_TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
