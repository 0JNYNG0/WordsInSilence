"""Matched interpretation factors; immutable existing observation arrays."""
from pathlib import Path
import sys,copy
E=Path(__file__).resolve().parent;D=E.parent/'word100_action_episode_supervision_20261102'
sys.path.insert(0,str(D));import episode_model as episode
from episode_model import np,torch,nn,process,read,write,exclusive,sha,clock,B,A,T,P
CONDITIONS=[(False,False),(True,False),(False,True),(True,True)]
def tag(condition,context):return f'guide{int(condition)}_context{int(context)}_42'

def fit_rows(context):
    rows=episode.fit_rows()
    if context:
        targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(E/'CONTEXT_TARGETS.json')['rows']}
        episode.attach_targets(rows,targets)
    return rows

def batch(rows,stats,condition):
    x,rgb,l,w,_,_=process.batch(rows,stats)
    if not condition:x[:,:,64:74]=0
    y=torch.full(x.shape[:2],-1,dtype=torch.long)
    for i,r in enumerate(rows):y[i,:len(r['x'])]=torch.from_numpy(r['episode'])
    return x,rgb,l,y

class ConditionNet(episode.EpisodeNet):
    def __init__(self):
        super().__init__(True)
        self.approach_return=nn.Linear(64,1)
        nn.init.zeros_(self.approach_return.weight);nn.init.constant_(self.approach_return.bias,-4.)
    def forward(self,x,rgb,l):
        h=self.embed(x+self.rgb(rgb)*rgb[:,:,-1:])
        h,_=self.context(nn.utils.rnn.pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False))
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        feature=x[:,:,:64]+self.delta(h)
        keys=self.guide(self.guide_vectors)+self.embedding
        logits=self.query(feature)@keys.T/(32**.5)
        families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
        return torch.cat([families,self.rest(feature),self.approach_return(feature)],-1)

loss_fn=episode.loss_fn
