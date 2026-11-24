"""Unique module name avoids collision with the frozen candidate core.py."""
from pathlib import Path
import sys
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S.parent))
from common import *

def reverse_geometry(g):
    """Reverse the SAME observed 9-point trajectory, without a target sign/word."""
    out=g.clone();v=g[:,:36].reshape(-1,9,4)
    out[:,:36]=(v.flip(1)-v[:,-1:]).reshape(-1,36)
    out[:,36:40]=g[:,36:40]+g[:,40:44]
    out[:,40:44]=-g[:,40:44]
    return out

class BodyDirection(old.DirectionHead):
    def __init__(self,observer,symmetric):
        super().__init__(observer);self.symmetric=symmetric
    def forward(self,context,g):
        # Finger appearance and target-word context do not determine physical
        # movement direction. Keep inherited dimensions/checkpoint compatible.
        zero=torch.zeros((len(g),192),device=g.device,dtype=g.dtype)
        f=super().forward(zero,g)
        if not self.symmetric:return f
        r=super().forward(zero,reverse_geometry(g))
        return (f+r[:,[2,1,0]])*.5
