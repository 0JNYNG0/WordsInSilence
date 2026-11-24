"""Learn guide-conditioned occurrence relevance without editing observation."""
from pathlib import Path
import sys
ORDER_UNIT=Path(__file__).resolve().parent
sys.path.insert(0,str(ORDER_UNIT.parent/'word100_learned_flow_role_20261102'))
from flow_model import *
ORDER_OUT=Path('W:/WordsInSilence_derived/word100_guided_event_order_20261102')
INITIAL=FLOW_OUT/'run/model.pt'

class OccurrenceOrderNet(FlowRoleNet):
    def __init__(self):
        super().__init__()
        assert sha(INITIAL)==read(FLOW_UNIT/'TRAINING_RESULT.json')['sha256']
        self.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True))
        with torch.no_grad():
            keys=self.actor.guide(self.actor.guide_vectors)+self.actor.embedding
            family_keys=torch.stack([keys[ids].mean(0) for ids in episode.ALIASES])
        assert family_keys.shape==(8,32)
        self.register_buffer('family_guide_keys',family_keys)
        self.occurrence_query=nn.Linear(48,32,bias=False);nn.init.zeros_(self.occurrence_query.weight)
    def forward(self,x,rgb,lengths,flow):
        with torch.no_grad():
            feature,h=self.encode(x,rgb,lengths)
            logits=self.actor.query(feature)@(self.actor.guide(self.actor.guide_vectors)+self.actor.embedding).T/(32**.5)
            families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
            action=torch.cat([families,self.actor.rest(feature)],-1)
        z=self.role_embed(torch.cat([feature,h,x[:,:,64:74]],-1)+self.flow_bridge(flow))
        z,_=self.role_context(nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False))
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        role=self.role_output(z)
        occurrence=(role[:,:,1]-role[:,:,0])[:,:,None]+self.occurrence_query(z)@self.family_guide_keys.T/(32**.5)
        return action,role,occurrence

def directed_probability(physical,occurrence,available,first,second):
    """Soft directed evidence, not a standalone completion certificate."""
    a=physical[:,first]*occurrence[:,first]*available
    b=physical[:,second]*occurrence[:,second]*available
    earlier_a=torch.cat([a.new_zeros(1),torch.cumsum(a,0)[:-1]])
    earlier_b=torch.cat([b.new_zeros(1),torch.cumsum(b,0)[:-1]])
    forward=(b*earlier_a).sum();reverse=(a*earlier_b).sum()
    probability=(forward+1e-8)/(forward+reverse+2e-8)
    return probability,forward+reverse

def attach_queries(rr):
    queries=read(ORDER_UNIT/'ORDER_SUPERVISION.json')['rows'];by={}
    for q in queries:by.setdefault(q['flow_key'],[]).append(q)
    for r in rr:r['order_queries']=by.get(r['flow_key'],[])
    assert sum(q['target']>=0 for r in rr for q in r['order_queries'])==576
    assert sum(q['target']<0 for r in rr for q in r['order_queries'])==54
    return rr

def occurrence_targets(rr,max_n):
    target=torch.full((len(rr),max_n,8),-1.,dtype=torch.float32)
    for i,r in enumerate(rr):
        nonmain=np.flatnonzero((r['role']>=0)&(r['role']!=1))
        target[i,nonmain,:]=0
        for f in np.flatnonzero((r['episode']>=0)&(r['episode']<8)):
            target[i,int(f),int(r['episode'][f])]=1
    return target

def occurrence_loss(logits,target,positive_weights):
    known=target>=0
    value=nn.functional.binary_cross_entropy_with_logits(logits,target.clamp_min(0),pos_weight=positive_weights,reduction='none')
    # Average within each video, then across videos; query/word edge counts
    # do not multiply the weight of a whole recording.
    return ((value*known).sum((1,2))/known.sum((1,2)).clamp_min(1)).mean()

def order_loss(physical,occurrence,rr):
    losses=[];used=0
    for i,r in enumerate(rr):
        n=len(r['x']);av=physical.new_tensor(r['available']);row=[]
        for q in r['order_queries']:
            if q['target']<0:continue
            p,_=directed_probability(physical[i,:n],occurrence[i,:n],av,q['asked_before'],q['asked_after'])
            row.append(nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),p.new_tensor(float(q['target']))));used+=1
        losses.append(torch.stack(row).mean() if row else occurrence[i].sum()*0)
    return torch.stack(losses).mean(),used
