"""Independent process-role head over an immutable-or-copied action encoder."""
from pathlib import Path
import sys
F=Path(__file__).resolve().parent;O=F.parent;E=O/'word100_guide_conditioned_episode_20261102';D=O/'word100_action_episode_supervision_20261102'
sys.path.insert(0,str(D));import episode_model as episode
from episode_model import torch,nn,np,process,read,write,exclusive,sha,clock,B,A,T,P,S

def fit_rows():
    rows=episode.fit_rows()
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(E/'CONTEXT_TARGETS.json')['rows']}
    for r in rows:r['action']=r['episode'].copy()
    episode.attach_targets(rows,targets)
    for r in rows:
        z=r['episode'];r['role']=np.where(z<0,-1,np.where(z<8,1,np.where(z==8,2,0))).astype('int64');r['episode']=r['action']
    return rows

def batch(rows,stats):
    x,rgb,l,w,_,_=process.batch(rows,stats)
    action=torch.full(x.shape[:2],-1,dtype=torch.long);role=action.clone()
    for i,r in enumerate(rows):
        n=len(r['x']);action[i,:n]=torch.from_numpy(r['episode']);role[i,:n]=torch.from_numpy(r['role'])
    return x,rgb,l,action,role

class SplitNet(nn.Module):
    def __init__(self,frozen):
        super().__init__();self.frozen=frozen;self.actor=episode.EpisodeNet(True)
        self.actor.load_state_dict(torch.load(D/'runs/episode_42/model.pt',map_location='cpu',weights_only=True))
        # Freeze the learned feature encoder in the primary, not the new
        # interpretation readout. Original T physical observations stay outside
        # this copied model and are immutable in BOTH conditions.
        for module in [self.actor.embed,self.actor.rgb,self.actor.context,self.actor.delta]:
            for p in module.parameters():p.requires_grad_(not frozen)
        self.role_embed=nn.Sequential(nn.Linear(122,32),nn.GELU())
        self.role_context=nn.GRU(32,24,batch_first=True,bidirectional=True)
        self.role_output=nn.Linear(48,3)
    def encode(self,x,rgb,l):
        a=x.clone();a[:,:,64:74]=0
        h=self.actor.embed(a+self.actor.rgb(rgb)*rgb[:,:,-1:])
        h,_=self.actor.context(nn.utils.rnn.pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False))
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        feature=a[:,:,:64]+self.actor.delta(h)
        return feature,h
    def forward(self,x,rgb,l,action_logits=True):
        feature,h=self.encode(x,rgb,l)
        z=self.role_embed(torch.cat([feature,h,x[:,:,64:74]],-1))
        z,_=self.role_context(nn.utils.rnn.pack_padded_sequence(z,l.cpu(),batch_first=True,enforce_sorted=False))
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        role=self.role_output(z);actions=None
        if action_logits:
            logits=self.actor.query(feature)@(self.actor.guide(self.actor.guide_vectors)+self.actor.embedding).T/(32**.5)
            families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
            actions=torch.cat([families,self.actor.rest(feature)],-1)
        return actions,role

loss_fn=episode.loss_fn
