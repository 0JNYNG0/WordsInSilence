"""Separate learned process branch. Existing observation arrays are immutable."""
from pathlib import Path
import sys, numpy as np, torch
from torch import nn
B=Path(__file__).resolve().parent; O=B.parent; A=O/'word100_learned_action_role_20261102'
sys.path.insert(0,str(A)); import role_model as old
from role_model import read,sha,exclusive,write,clock,WORDS,T,S,P,G,U,state

def add_rgb(rows):
    meta={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
    short={r['input']:r for r in read(S/'SHORT_GAP_FIT_MANIFEST.json')['rows']}
    for r in rows:
        p=Path(r['input']); a=dict(np.load(p))
        if 'rgb' not in a:
            if r['input'] in short:p=Path(short[r['input']]['source_input'])
            elif r['kind']=='native':p=Path(meta[r['clip_id']]['input'])
            else:p=Path(r['axis_path'])
            a=dict(np.load(p));r['extra_pins'].append((str(p),sha(p)))
        if 'rgb' not in a:
            # Native observer input may keep RGB separately. Resolve the exact
            # public prepared projection row; never search private directories.
            raise ValueError(('RGB not in exact public input',str(p),list(a)))
        rgb=a['rgb'].copy().astype('float32');assert rgb.shape==(len(r['x']),768)
        rgb[~r['available']]=0;r['rgb']=rgb
        r['physical']=state.labels_for(r['clip_id'],r['mapping']);r['physical'][~r['available']]=-1
        r['pairs']=pair_targets(r)
    return rows

def pair_targets(r):
    # Direct physical endpoints, not semantic correctness or a forced decoder.
    # Two anchors on opposite sides of an image gap do not become evidence.
    graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'][r['word_id']]
    edges={tuple(v) for v in graph.get('edges',[])}
    if not edges:
        edges={tuple(v) for v in graph.get('precedences',[])}
    ys=r['physical'];known=np.flatnonzero((ys>=0)&(ys<9));pairs=[]
    for i,j in zip(known[:-1],known[1:]):
        a,b=int(ys[i]),int(ys[j]);label=1 if (a,b) in edges else (0 if (b,a) in edges else -1)
        if label>=0 and r['available'][i:j+1].all():pairs.append([int(i),int(j),label])
    return pairs

def stats(rows):
    z=np.concatenate([r['rgb'][r['available']] for r in rows]);v=old.stats(rows)
    v.update(rgb_mean=z.mean(0).tolist(),rgb_std=z.std(0).clip(.05).tolist());return v

def batch(rows,s):
    x,l,y=old.batch(rows,s);mx=x.shape[1];rgb=torch.zeros(len(rows),mx,769)
    physical=torch.full_like(y,-1)
    for i,r in enumerate(rows):
        n=len(r['x']);v=np.where(r['available'][:,None],(r['rgb']-s['rgb_mean'])/s['rgb_std'],0)
        rgb[i,:n,:768]=torch.from_numpy(v.astype('float32'));rgb[i,:n,768]=torch.from_numpy(r['available'].astype('float32'))
        physical[i,:n]=torch.from_numpy(r['physical'])
    words=torch.tensor([WORDS.index(r['word_id']) for r in rows])
    return x,rgb,l,words,y,physical

class ProcessNet(nn.Module):
    def __init__(self,pilot=False):
        super().__init__();self.role=old.RoleNet(True)
        tag='pilot_temporal_42' if pilot else 'temporal_42'
        self.role.load_state_dict(torch.load(A/'runs'/tag/'model.pt',map_location='cpu',weights_only=True))
        self.rgb=nn.Sequential(nn.Linear(769,32),nn.GELU(),nn.Linear(32,130))
        nn.init.zeros_(self.rgb[-1].weight);nn.init.zeros_(self.rgb[-1].bias)
        self.physical=state.StateNet(False)
        self.physical.load_state_dict(torch.load(T/'runs/current_42/model.pt',map_location='cpu',weights_only=True))
        # This is a copy, not the protected observer used for output.
        self.query_delta=nn.Linear(48,64);nn.init.zeros_(self.query_delta.weight);nn.init.zeros_(self.query_delta.bias)
        self.pair=nn.Sequential(nn.Linear(48*3,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x,rgb,l,w):
        h=self.role.embed(x+self.rgb(rgb)*rgb[:,:,-1:])
        h,_=self.role.context(nn.utils.rnn.pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False))
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        return self.role.output(h),self.physical(x[:,:,:64]+self.query_delta(h),l,w),h
    def pairs(self,h,rows):
        xx=[]; yy=[]
        for k,r in enumerate(rows):
            for i,j,y in r['pairs']:
                xx.append(torch.cat([h[k,i],h[k,j],h[k,j]-h[k,i]])); yy.append(y)
        if not xx:return h.sum()*0,torch.empty(0,device=h.device)
        return self.pair(torch.stack(xx)).flatten(),torch.tensor(yy,dtype=h.dtype,device=h.device)

def point_records(rows,prs,key='y'):
    result=[]
    for r,p in zip(rows,prs):
        for i in np.flatnonzero(r[key]>=0):
            result.append(dict(key=r.get('key',r['clip_id']+'_'+r['kind']),word=r['word_id'],person=r['person'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r[key][i]),pred=int(p[i].argmax()),probability=p[i].tolist()))
    return result
