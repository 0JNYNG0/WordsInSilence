"""Shared relative-motion observer; never a word correctness classifier."""
from motion_review import *
import numpy as np, torch
from torch import nn

PALMS=([91,96,100,104,108],[112,117,121,125,129])
def feature(ch,a,b,edge=False):
    if a < 0 or b >= len(ch['frame_id']) or b <= a:return None
    ids=PALMS[0]+PALMS[1]
    if not ch['point_available'][a:b+1,ids].all():return None
    xy=ch['body_relative_xy'][a:b+1]
    p=xy[:,PALMS[1]].mean(1)-xy[:,PALMS[0]].mean(1)
    if not np.isfinite(p).all():return None
    radius=np.linalg.norm(p,axis=1);step=np.diff(radius);dv=np.diff(p,axis=0)
    out=np.r_[p[-1]-p[0],radius[-1]-radius[0],np.linalg.norm(dv,axis=1).sum(),
        np.ptp(p,axis=0),step.mean(),step.std(),step.min(),step.max()].astype(np.float32)
    if edge:
        # Distance between observed hand extents in the target palm's own axes.
        # This avoids treating two palm centres as if they were the contacting edges.
        ids2=list(range(91,133))
        if not ch['point_available'][a:b+1,ids2].all():return None
        left=xy[:,91:112];right=xy[:,112:133]
        axis=left[:,9]-left[:,0];length=np.linalg.norm(axis,axis=1)
        if np.any(length<1e-5):return None
        axis=axis/length[:,None];normal=np.stack([-axis[:,1],axis[:,0]],axis=1)
        gaps=[]
        for basis in [axis,normal]:
            l=(left*basis[:,None]).sum(-1);r=(right*basis[:,None]).sum(-1)
            gap=np.maximum(np.maximum(r.min(1)-l.max(1),l.min(1)-r.max(1)),0)
            dg=np.diff(gap);gaps.extend([gap[-1]-gap[0],np.ptp(gap),dg.mean(),dg.min(),dg.max()])
        out=np.r_[out,gaps].astype(np.float32)
    return out

class Net(nn.Module):
    def __init__(self,kind,n=10):
        super().__init__();self.net=nn.Linear(n,3) if kind=='linear' else nn.Sequential(nn.Linear(n,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)

class Observer:
    def __init__(self,kind,run_dir=None,edge=False):
        self.edge=edge
        self.models=[]
        for seed in [42,43,44]:
            ck=torch.load((run_dir or D/'runs')/f'{kind}_{seed}.pt',map_location='cpu',weights_only=False)
            m=Net(kind,len(ck['mean']));m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
    def predict(self,x):
        if x is None:return -1
        labels=[]
        with torch.inference_mode():
            for m,mu,sd in self.models:
                p=m(torch.tensor((x-mu)/sd)).softmax(-1).numpy()
                labels.append(int(p.argmax()) if p.max()>=.7 else -1)
        return labels[0] if len(set(labels))==1 else -1
    def sequence(self,ch,radius=3):
        return np.array([self.predict(feature(ch,i-radius,i+radius,self.edge)) for i in range(len(ch['frame_id']))])

def metrics(rows,key):
    return dict(total=len(rows),correct=sum(r[key]==r['label']for r in rows),
        wrong=sum(r[key]>=0 and r[key]!=r['label']for r in rows),unknown=sum(r[key]<0 for r in rows))

def main():
    import sys,time
    sys.path.insert(0,str(U));from core import Features
    from basic_runtime import RelationObserver
    torch.set_num_threads(2)
    assert not (D/'MOTION_COMPARISON.json').exists()
    plan=read(D/'MOTION_FROZEN.json'); assert sha(D/'MOTION_TRUTH.json')==plan['truth_sha256']
    sources={r['clip_id']:r for r in read(U/'INPUTS.json')['rows']}
    channels={k:dict(np.load(r['channels']))for k,r in sources.items()}
    truth=read(D/'MOTION_TRUTH.json')['rows'];fit=[r for r in truth if r['split']=='fit']
    xf=[feature(channels[r['clip_id']],r['start'],r['end'])for r in fit]
    assert all(x is not None for x in xf),'Do not silently discard missing fit observations'
    x=np.stack(xf);y=np.array([r['label']for r in fit]);mu=x.mean(0);sd=np.maximum(x.std(0),1e-4)
    xx=torch.tensor((x-mu)/sd);yy=torch.tensor(y);weights=torch.tensor(len(y)/(3*np.bincount(y)),dtype=torch.float32)
    (D/'runs').mkdir(exist_ok=True);runs=[]
    for kind in plan['models']:
        for seed in plan['seeds']:
            torch.manual_seed(seed);m=Net(kind);opt=torch.optim.AdamW(m.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);t=time.perf_counter()
            for step in range(plan['steps']):
                opt.zero_grad();loss=nn.functional.cross_entropy(m(xx),yy,weight=weights);loss.backward();opt.step()
            p=D/'runs'/f'{kind}_{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd,kind=kind,seed=seed,truth_sha=plan['truth_sha256']),p)
            runs.append(dict(kind=kind,seed=seed,seconds=time.perf_counter()-t,loss=float(loss.detach()),sha256=sha(p)))
    ob={k:Observer(k)for k in plan['models']};base=RelationObserver();factory=Features();states={}
    for k,ch in channels.items():
        f,q,_=factory(ch);states[k]=base(f,q,ch['timestamp_us'])[0]
    results=[]
    for r in truth:
        v=feature(channels[r['clip_id']],r['start'],r['end']);s=states[r['clip_id']];a,b=s[r['start']],s[r['end']]
        # A limited earlier-state proxy, explicitly not a learned motion classifier.
        old=1 if (a,b)==(1,2) else 2 if (a,b)==(2,1) else 0 if a==b and a>0 else -1
        results.append(dict(**r,static_state_proxy=old,**{k:o.predict(v)for k,o in ob.items()}))
    dev=[r for r in results if r['split']!='fit'];summary={k:metrics(dev,k)for k in ['static_state_proxy']+plan['models']}
    write(D/'MOTION_COMPARISON.json',dict(runs=runs,fit_intervals=len(fit),development_intervals=len(dev),summary=summary,rows=results,
        static_proxy_limit='Endpoint near/separated-state comparison, not an original full feedback score.',
        completion=False,web_admitted=False,independent_people=True,
        limitation='Small gratitude-only motion labels; common code and no word ID do not establish all-word generalization.'))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
