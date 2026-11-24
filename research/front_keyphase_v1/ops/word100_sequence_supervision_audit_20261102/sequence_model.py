"""Retain rest observation; learn activity and sampled-witness sequence jointly."""
from pathlib import Path
import sys
J=Path(__file__).resolve().parent
sys.path.insert(0,str(J.parent/'word100_separate_action_context_20261102'))
from model import *

class WitnessNet(SplitNet):
    def __init__(self):
        super().__init__(True)
        self.role_output=nn.Linear(48,2)
        self.blank=nn.Parameter(torch.tensor(0.0))
    def forward(self,x,rgb,l):
        action,activity=super().forward(x,rgb,l)
        # CTC blank is a DIFFERENT parameter from action/rest output 8.
        sequence=torch.cat([action[:,:,:8],self.blank.expand(*action.shape[:2],1)],-1)
        return action,activity,sequence

def sequence_loss(logits,y):
    """CTC is restricted to the labelled witness subsequence, not the video.

    Every included point has a directly reviewed action label. Unlabelled
    frames still provide recurrent context, but are never CTC target times.
    Collapsing consecutive labels here does NOT assert full-video absence.
    """
    selected=[];targets=[]; lengths=[]; target_lengths=[]
    for i in range(len(y)):
        index=torch.nonzero((y[i]>=0)&(y[i]<8),as_tuple=True)[0]
        if not len(index):continue
        yy=y[i,index]; keep=torch.cat([torch.ones(1,dtype=torch.bool,device=yy.device),yy[1:]!=yy[:-1]])
        target=yy[keep]; selected.append(logits[i,index]);targets.append(target)
        lengths.append(len(index));target_lengths.append(len(target))
    if not selected:return logits.sum()*0
    scores=nn.utils.rnn.pad_sequence(selected,batch_first=False).log_softmax(-1)
    value=nn.functional.ctc_loss(scores,torch.cat(targets),torch.tensor(lengths),torch.tensor(target_lengths),blank=8,reduction='mean',zero_infinity=False)
    assert torch.isfinite(value)
    return value

def hierarchy(activity,current,available):
    """Independent rest evidence; disagreement held, not forced adoption."""
    rest=current[:,9].copy()
    role=np.column_stack([(1-rest)*activity[:,0],(1-rest)*activity[:,1],rest])
    conflict=(rest>=.8)&(activity[:,1]>=.8)&available
    return role,conflict

def batch_j(rows,stats):
    x,rgb,l,a,roles=batch(rows,stats)
    activity=torch.where(roles<0,-1,(roles==1).long())
    return x,rgb,l,a,activity
