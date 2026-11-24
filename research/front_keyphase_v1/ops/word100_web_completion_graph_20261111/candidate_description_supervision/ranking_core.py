from source_boundary_core import *
D=W/'additional_source_review';K=W/'source_rank_learning'
class RankHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(600,96),nn.GELU(),nn.Linear(96,5))
 def forward(self,x):return self.head((x-self.mean)/self.std)
