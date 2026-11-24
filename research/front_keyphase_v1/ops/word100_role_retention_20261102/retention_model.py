"""Keep current action content fixed; retain prior process knowledge softly.

Unique module name avoids the legacy dependency's module named model.
"""
from pathlib import Path
import sys
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_event_temporal_finetune_20261102'))
import experiment as ex
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import flow_model as fl
torch=ex.torch;nn=ex.nn;np=ex.np
OUT=Path('W:/WordsInSilence_derived/word100_role_retention_20261102')
INITIAL=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102/runs/new_adapt/model.pt')
TEACHER=Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102/run/model.pt')
read=ex.read;sha=ex.sha;exclusive=ex.exclusive;clock=ex.clock

class RetainedRoleNet(ex.WitnessNet):
    def __init__(self):
        super().__init__();self.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True))
        for p in self.actor.parameters():p.requires_grad_(False)
        self.blank.requires_grad_(False)
        self.flow_bridge=nn.Linear(6,122,bias=False);nn.init.zeros_(self.flow_bridge.weight)
    def forward(self,x,rgb,lengths,flow):
        with torch.no_grad():
            feature,h=self.encode(x,rgb,lengths)
            logits=self.actor.query(feature)@(self.actor.guide(self.actor.guide_vectors)+self.actor.embedding).T/(32**.5)
            families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in ex.episode.ALIASES],-1)
            action=torch.cat([families,self.actor.rest(feature)],-1)
        z=self.role_embed(torch.cat([feature,h,x[:,:,64:74]],-1)+self.flow_bridge(flow))
        z,_=self.role_context(nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False))
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        return action,self.role_output(z)

def fit_rows():
    _,rows,_=ex.data();normal=read(fl.FLOW_UNIT/'FLOW_INPUTS.json')
    for i,r in enumerate(rows):
        r['flow_key']=f'fit_{i:03d}';item=normal['items'][r['flow_key']]
        assert r['person'] in range(1,10) and (item['clip_id'],item['kind'],item['person'])==(r['clip_id'],r['kind'],r['person'])
    return rows,normal

def retention_mask(r):
    mask=(r['role']<0)&r['available']
    for f in np.flatnonzero(np.abs(np.diff(r['mapping']))>3)+1:mask[max(0,f-2):min(len(mask),f+3)]=False
    return mask

def retention_loss(logits,teacher,mask):
    # KL is a soft consistency objective, not newly certified truth. Each
    # clip contributes equally, and labelled/absent/edit-seam frames never do.
    value=nn.functional.kl_div(logits.log_softmax(-1),teacher,reduction='none').sum(-1)
    return ((value*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()

def teacher_batch(rows):
    n=max(len(r['x']) for r in rows);p=np.full((len(rows),n,2),.5,np.float32);mask=np.zeros((len(rows),n),bool)
    for i,r in enumerate(rows):
        a=np.load(r['teacher_path']);assert sha(r['teacher_path'])==r['teacher_sha256']
        p[i,:len(r['x'])]=a['probability'];mask[i,:len(r['x'])]=a['mask']
    return torch.from_numpy(p),torch.from_numpy(mask)
