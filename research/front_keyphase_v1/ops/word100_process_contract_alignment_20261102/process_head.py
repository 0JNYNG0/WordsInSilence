"""Additive typed physical-process evidence head, not whole-sign correctness.

No word/person/edit kind, elapsed-time percentage, source index, seam flag or
padding count is an answer feature. Existing models are not imported/changed.
Actual learning is permitted only by the separately frozen comparison contract.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

QUERIES = (
    'right_hand_toward_left_wrist_along_arm',
    'hands_move_apart',
    'right_hand_rises_relative_left',
    'right_hand_descends_face_to_torso',
    'right_hand_falls_relative_left',
)
GEOMETRY_DIM = 59  # body-relative shape/position/relation; omit diagnostics/velocity
INPUT_DIM = GEOMETRY_DIM*2+768+1

def make_input(a, rgb_stats):
    n=len(a['frame_id'])
    assert a['x'].shape==(n,69) and a['mask'].shape==(n,69) and a['rgb'].shape==(n,768)
    availability=np.asarray(a['image_available'],bool)
    assert availability.shape==(n,)
    observed=np.asarray(a['mask'][:,:GEOMETRY_DIM],bool)&np.isfinite(a['x'][:,:GEOMETRY_DIM])&availability[:,None]
    geometry=np.where(observed,a['x'][:,:GEOMETRY_DIM],0).astype(np.float32)
    video_good=availability&np.isfinite(a['rgb']).all(1)
    video=np.where(video_good[:,None],(a['rgb']-np.asarray(rgb_stats['mean']))/np.asarray(rgb_stats['std']),0)
    x=np.concatenate([geometry,observed.astype(np.float32),video,video_good[:,None]],axis=1).astype(np.float32)
    assert x.shape==(n,INPUT_DIM) and np.isfinite(x).all()
    return x

class ProcessEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry=nn.Linear(GEOMETRY_DIM*2,32)
        self.video=nn.Linear(768,32,bias=False)
        self.query=nn.Embedding(len(QUERIES),8)
        self.combine=nn.Sequential(nn.Linear(32+32+1+8,48),nn.GELU())
        self.temporal=nn.GRU(48,32,batch_first=True,bidirectional=True)
        self.point=nn.Linear(64,2) # separate setup/terminal relation, may overlap
        self.process=nn.Linear(64,1)
        self.motion=nn.Linear(64,1) # interval-level query motion, not dense truth

    def embed(self,x,query):
        assert x.ndim==3 and x.shape[-1]==INPUT_DIM
        available=x[:,:,-1:]
        # Defence against mistakenly passing copied/propagated geometry on blank
        # pixels. Missing image is neither normal execution nor a negative label.
        g=x[:,:,:GEOMETRY_DIM*2]*available
        v=x[:,:,GEOMETRY_DIM*2:-1]*available
        q=self.query(query)[:,None,:].expand(-1,x.shape[1],-1)
        return self.combine(torch.cat([torch.nn.functional.gelu(self.geometry(g)),self.video(v),available,q],dim=-1))

    def encode(self,h,lengths):
        assert torch.all(lengths>=2) and torch.all(lengths<=h.shape[1])
        packed=pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False)
        sequence,last=self.temporal(packed)
        sequence,_=pad_packed_sequence(sequence,batch_first=True,total_length=h.shape[1])
        return sequence,torch.cat([last[0],last[1]],dim=-1)

    def forward(self,x,lengths,query):
        sequence,last=self.encode(self.embed(x,query),lengths)
        return {'point_logits':self.point(sequence),'process_logits':self.process(last).squeeze(-1)}

    def interval_logits(self,x,lengths,query):
        # Called on only the explicitly labelled interval, with no external
        # recurrent context. Frozen RGB windows may still carry local context;
        # this is not an independent frame-by-frame motion annotation.
        _,last=self.encode(self.embed(x,query),lengths)
        return self.motion(last).squeeze(-1)

def batch(sequences):
    lengths=torch.tensor([len(x) for x in sequences],dtype=torch.long)
    assert len(sequences)>0 and int(lengths.min())>=2
    x=torch.zeros(len(sequences),int(lengths.max()),INPUT_DIM)
    for i,s in enumerate(sequences):x[i,:len(s)]=torch.as_tensor(s,dtype=torch.float32)
    return x,lengths
