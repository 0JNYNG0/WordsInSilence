"""A separate process readout; no mutation of observed states/shape/motion."""
from pathlib import Path
import sys, numpy as np, torch
from torch import nn
from functools import lru_cache
A=Path(__file__).resolve().parent;O=A.parent;T=O/'word100_learned_state_context_balance_20261102'
sys.path.insert(0,str(T));import state_model as state
from state_model import read,sha,exclusive,write,clock,WORDS
S=state.S;P=state.old.P;G=state.old.G;U=state.old.U

@lru_cache(maxsize=64)
def labels(cid):
    r=next(r for r in read(A/'ROLE_TARGETS.json')['rows'] if r['clip_id']==cid)
    return {p['frame']:p['label'] for p in r['points'] if p['mask']}
def target(cid,mapping):return np.array([labels(cid).get(int(i),-1) for i in mapping],np.int64)

@lru_cache(maxsize=1)
def observer():
    m=state.StateNet(False);m.load_state_dict(torch.load(T/'runs/current_42/model.pt',weights_only=True,map_location='cpu'));m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m

def geometric(x,mask,available):
    # Static normalized body coordinates only; exclude old velocity channels.
    indices=list(range(8))+list(range(36,40))
    z=x[:,indices];valid=mask[:,indices].astype(bool)&np.isfinite(z)&available[:,None]
    elbow=x[:,4:6];wrist=x[:,2:4];palm=x[:,36:38];v=wrist-elbow;d=palm-elbow;norm=(v*v).sum(1)
    good=mask[:,[2,3,4,5,36,37]].all(1)&np.isfinite(norm)&(norm>.0025)&available
    along=(d*v).sum(1)/np.maximum(norm,1e-6);side=(d[:,0]*v[:,1]-d[:,1]*v[:,0])/np.maximum(norm,1e-6)
    z=np.column_stack([z,along,side]);valid=np.column_stack([valid,good,good]);valid&=np.isfinite(z)
    return np.where(valid,z,0).astype('float32'),valid

def attach_geometry(r,meta,short):
    a=dict(np.load(r['input']));path=Path(r['input']);extra=[]
    if 'x' not in a:
        if r['input'] in short:
            rr=short[r['input']];path=Path(rr['source_input']);aa=dict(np.load(path));gap=rr['gap_frames']
        elif r['kind']=='native':path=Path(meta[r['clip_id']]['input']);aa=dict(np.load(path));gap=[]
        elif r['kind'].startswith('pose_'):
            path=Path(r['axis_path']);aa=dict(np.load(path));gap=next(x['frames'] for x in read(G/'SENSOR_GAP_RESULTS.json')['rows'] if r['kind']==f"{x['kind']}_f{x['center']}" and x['word']==r['word_id'])
        else:raise ValueError(('Missing geometry',r['input']))
        a['x']=aa['x'].copy();a['mask']=aa['mask'].copy()
        a['x'][gap]=0;a['mask'][gap]=0;extra.append((str(path),sha(path)))
    available=r.get('available',a.get('image_available',np.ones(len(r['x']),bool))).astype(bool)
    assert len(a['x'])==len(r['x'])==len(available)
    z,m=geometric(a['x'],a['mask'],available)
    with torch.inference_mode():
        xx,ll,ww,_=state.batch([r]);pr=observer()(xx,ll,ww).softmax(-1)[0].numpy()
    # This new FIT observer use is necessary input construction, not a repeat
    # of completed T development evaluation. DEV uses saved predictions below.
    return z,m,pr,available,extra

def inputs(fit=True):
    if fit:rows=state.fit_rows()
    else:
        import evaluate_phase
        rows=evaluate_phase.cases()
    meta={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
    short={r['input']:r for r in read(S/'SHORT_GAP_FIT_MANIFEST.json')['rows']}
    for r in rows:
        if fit:
            z,m,pr,av,extra=attach_geometry(r,meta,short)
        else:
            # Avoid re-inference of the completed T comparison.
            saved=T/'predictions/current_42'/f"{r['key']}.npz"
            pr=np.load(saved)['probability'].copy();extra=[(str(saved),sha(saved))]
            a=dict(np.load(r['input']));av=r['available']
            if 'x' not in a:
                path=Path(meta[r['clip_id']]['input']) if r['kind']=='native' else Path(r['axis_path'])
                aa=dict(np.load(path));a['x']=aa['x'].copy();a['mask']=aa['mask'].copy();extra.append((str(path),sha(path)))
                if r['kind'].startswith('pose_'):
                    gap=next(x['frames'] for x in read(G/'SENSOR_GAP_RESULTS.json')['rows'] if r['kind']==f"{x['kind']}_f{x['center']}" and x['word']==r['word_id'])
                    a['x'][gap]=0;a['mask'][gap]=0
            z,m=geometric(a['x'],a['mask'],av)
        r.update(z=z,m=m,current=pr,available=av,extra_pins=extra,y=target(r['clip_id'],r['mapping']))
        r['y'][~av]=-1
        assert np.isfinite(z).all() and len(z)==len(r['x']) and np.isfinite(pr).all()
    return rows

def stats(rows):
    z=np.concatenate([r['z'] for r in rows]);m=np.concatenate([r['m'] for r in rows]);count=m.sum(0).clip(1)
    mean=(z*m).sum(0)/count;std=np.sqrt((((z-mean)*m)**2).sum(0)/count).clip(.05)
    return dict(mean=mean.tolist(),std=std.tolist())
def features(r,s):
    z=np.where(r['m'],(r['z']-s['mean'])/s['std'],0).astype('float32');m=r['m']
    delta=np.zeros_like(z);dm=np.zeros_like(m)
    # Adjacent displacement, not absolute clock or assumed fixed gesture speed.
    dm[1:]=m[1:]&m[:-1];delta[1:]=np.where(dm[1:],z[1:]-z[:-1],0)
    return np.concatenate([r['x'],r['current'],z,m,delta,dm],-1).astype('float32')
def batch(rows,s):
    lengths=torch.tensor([len(r['x']) for r in rows]);mx=int(lengths.max());x=torch.zeros(len(rows),mx,130);y=torch.full((len(rows),mx),-1,dtype=torch.long)
    for i,r in enumerate(rows):x[i,:lengths[i]]=torch.from_numpy(features(r,s));y[i,:lengths[i]]=torch.from_numpy(r['y'])
    return x,lengths,y
class RoleNet(nn.Module):
    def __init__(self,temporal=False):
        super().__init__();self.temporal=temporal;self.embed=nn.Sequential(nn.Linear(130,32),nn.GELU())
        self.context=nn.GRU(32,24,batch_first=True,bidirectional=True) if temporal else nn.Sequential(nn.Linear(32,48),nn.GELU())
        self.output=nn.Linear(48,3)
    def forward(self,x,lengths):
        h=self.embed(x)
        if self.temporal:
            h,_=self.context(nn.utils.rnn.pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False))
            h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        else:h=self.context(h)
        return self.output(h)

def measure(model,rows,s):
    model.eval();points=[]
    with torch.inference_mode():
        for r in rows:
            x,l,y=batch([r],s);pr=model(x,l).softmax(-1)[0].numpy()
            for i in np.flatnonzero(r['y']>=0):points.append(dict(clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
    return points
