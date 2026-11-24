from ranking_core import *
L=W/'local_rank_learning'
class LocalRankHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(1160,96),nn.GELU(),nn.Linear(96,5))
 def forward(self,x):return self.head((x-self.mean)/self.std)
def drop_hand_information(pose,q):
 x=pose.copy();qq=q.reshape(-1,10,10).copy();x[:,18:186]=0;x[:,195:279]=0;qq[:,:,4:8]=0;qq[:,:,9]=0
 return x,qq.reshape(-1,100)
