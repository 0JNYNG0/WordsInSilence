from pathlib import Path
import sys
N=Path(__file__).resolve().parent
sys.path.insert(0,str(N.parent))
from common import *
class NullRank(torch.nn.Module):
    def __init__(self,rank):
        super().__init__();self.rank=rank;self.null_logits=torch.nn.Parameter(torch.zeros(5))
    def forward(self,x):return self.rank(x)

