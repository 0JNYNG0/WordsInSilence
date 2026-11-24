"""Keep the physical readout immutable; let process interpretation read it."""
from pathlib import Path
import sys
V=Path(__file__).resolve().parent;O=V.parent
L=O/'word100_seated_fit_transfer_20261102'
sys.path.insert(0,str(L));from transfer_data import *
OUT=Path('W:/WordsInSilence_derived/word100_content_role_bridge_20261102')
START=DEST/'runs/seated_variation/model.pt'

class BridgeNet(WitnessNet):
    def __init__(self,connected):
        super().__init__();self.load_state_dict(torch.load(START,map_location='cpu',weights_only=True))
        self.connected=connected
        self.content_bridge=nn.Linear(9,122,bias=False);nn.init.zeros_(self.content_bridge.weight)
        for p in self.actor.parameters():p.requires_grad_(False)
        self.blank.requires_grad_(False)
    def forward(self,x,rgb,lengths):
        with torch.no_grad():
            feature,h=self.encode(x,rgb,lengths)
            logits=self.actor.query(feature)@(self.actor.guide(self.actor.guide_vectors)+self.actor.embedding).T/(32**.5)
            families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
            action=torch.cat([families,self.actor.rest(feature)],-1)
        observed=action.softmax(-1)
        if not self.connected:observed=torch.zeros_like(observed)
        z=self.role_embed(torch.cat([feature,h,x[:,:,64:74]],-1)+self.content_bridge(observed))
        z,_=self.role_context(nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False))
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        return action,self.role_output(z)

def rows():
    base,extra=load_pairs();return base+extra['seated_variation']
