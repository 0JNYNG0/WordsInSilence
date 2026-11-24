"""Learn source-event content, preserving the independent physical observer."""
from audit import *
import copy,numpy as np,torch
from torch import nn
import process_model as process
ALIASES=[[0],[1],[2],[3],[4],[5,6,7,8,9],[10],[11,12]]

def target_map():
    return {r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'ACTION_TARGETS.json')['rows']}

def attach_targets(rows,targets):
    for r in rows:
        lookup=targets[r['clip_id']];mp=np.asarray(r['mapping'],int)
        y=np.asarray([lookup.get(int(f),-1) for f in mp],np.int64)
        y[~r['available']]=-1
        # An edit can alter the local event context; do not make automatic
        # source-frame labels on its artificial seam. Native time stays intact.
        seams=np.flatnonzero(np.abs(np.diff(mp))>3)+1
        for f in seams:y[max(0,f-2):min(len(y),f+3)]=-1
        r['episode']=y;r['seam_unlabeled']=seams.tolist()
    return rows

def fit_rows():
    rows=[]
    for meta in read(B/'PREPARED.json')['rows']:
        assert meta['person'] in range(1,10) and sha(meta['cache'])==meta['cache_sha256']
        r=dict(meta);r.update(dict(np.load(meta['cache'])));rows.append(r)
    return attach_targets(rows,target_map())

def batch(rows,s):
    x,rgb,l,w,y,p=process.batch(rows,s)
    # The old current-state posterior is target-word masked. Exclude it from
    # this observational branch so changing the requested word cannot change
    # what action the model claims to see.
    x[:,:,64:74]=0
    y=torch.full(x.shape[:2],-1,dtype=torch.long)
    for i,r in enumerate(rows):y[i,:len(r['x'])]=torch.from_numpy(r['episode'])
    return x,rgb,l,y

class EpisodeNet(nn.Module):
    def __init__(self,temporal):
        super().__init__();base=process.ProcessNet()
        base.load_state_dict(torch.load(B/'runs/joint_42/model.pt',map_location='cpu',weights_only=True))
        self.embed=copy.deepcopy(base.role.embed);self.rgb=copy.deepcopy(base.rgb)
        self.temporal=temporal
        self.context=copy.deepcopy(base.role.context) if temporal else nn.Sequential(nn.Linear(32,48),nn.GELU())
        self.delta=nn.Linear(48,64);nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        self.query=copy.deepcopy(base.physical.current.query);self.guide=copy.deepcopy(base.physical.current.guide)
        self.embedding=nn.Parameter(base.physical.current.embedding.detach().clone())
        self.register_buffer('guide_vectors',base.physical.current.guide_vectors.detach().clone())
        self.rest=copy.deepcopy(base.physical.rest)
    def forward(self,x,rgb,l):
        h=self.embed(x+self.rgb(rgb)*rgb[:,:,-1:])
        if self.temporal:
            h,_=self.context(nn.utils.rnn.pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False))
            h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        else:h=self.context(h)
        feature=x[:,:,:64]+self.delta(h)
        keys=self.guide(self.guide_vectors)+self.embedding
        logits=self.query(feature)@keys.T/(32**.5)
        families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in ALIASES],-1)
        return torch.cat([families,self.rest(feature)],-1)

def loss_fn(out,y,weights):
    loss=nn.functional.cross_entropy(out.transpose(1,2),y,weight=weights,ignore_index=-1,reduction='none')
    return (loss.sum(1)/(y>=0).sum(1).clamp_min(1)).mean()
