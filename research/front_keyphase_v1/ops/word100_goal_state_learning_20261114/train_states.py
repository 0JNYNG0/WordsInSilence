from pathlib import Path
import json,hashlib,time,collections
import numpy as np
import torch
from state_model import StateHeads,CLASSES
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(U/'TRAINING_FROZEN.json');meta=read(U/'LEARNING_INPUTS.json');torch.set_num_threads(2)
for p,h in cfg['pins'].items():assert sha(Path(p))==h
assert not(U/'TRAINING_COMPLETE.json').exists()
fit={d:[r for r in meta['rows']if r['domain']==d and r['split']=='fit'and r['target']>=0 and r['available']]for d in CLASSES}
allrows={d:[r for r in meta['rows']if r['domain']==d]for d in CLASSES}
train={}
for d,rr in fit.items():
    count=collections.Counter((r['target'],r['clip_id'])for r in rr);byclass=collections.Counter(r['target']for r in rr)
    unique=collections.Counter(t for t,c in count)
    wt=np.array([1/(len(CLASSES[d])*unique[r['target']]*count[r['target'],r['clip_id']])for r in rr]);wt/=wt.sum()
    train[d]=(torch.tensor([r['feature']for r in rr]),torch.tensor([r['target']for r in rr]),torch.tensor(wt,dtype=torch.float32))
out=[];(U/'models').mkdir(exist_ok=True)
for kind in cfg['models']:
 for seed in cfg['seeds']:
    dest=U/'models'/f'{kind}_{seed}.pt';assert not dest.exists()
    torch.manual_seed(seed);model=StateHeads(kind);initial={k:v.clone()for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    start=time.perf_counter();history=[]
    for step in range(cfg['steps']):
        loss=0
        for d,(x,y,w)in train.items():loss+=(torch.nn.functional.cross_entropy(model(x,d),y,reduction='none')*w).sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%200==0 or step==cfg['steps']-1:history.append(dict(step=step+1,loss=float(loss.detach())))
    model.eval();torch.save(model.state_dict(),dest)
    pred={}
    with torch.no_grad():
        for d,rr in allrows.items():
            pred[d]=torch.softmax(model(torch.tensor([r['feature']for r in rr]),d),-1).numpy()
            pred[d+'_indices']=np.array([r['index']for r in rr])
    pf=U/'models'/f'{kind}_{seed}_predictions.npz';np.savez_compressed(pf,**pred)
    out.append(dict(kind=kind,seed=seed,steps=cfg['steps'],fit_indices=[r['index']for rr in fit.values()for r in rr],
        model=str(dest),model_sha256=sha(dest),predictions=str(pf),predictions_sha256=sha(pf),
        seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in model.parameters()),log=history,
        weights_changed=any(not torch.equal(initial[k],v)for k,v in model.state_dict().items())))
    print(kind,seed,'trained',round(out[-1]['seconds'],2),'s',flush=True)
with (U/'TRAINING_COMPLETE.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=out,training_frozen_sha256=sha(U/'TRAINING_FROZEN.json')),f,ensure_ascii=False,indent=2)
