import torch
from torch import nn
class StatePath(nn.Module):
    def __init__(self,dim=59):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(dim,24,3,padding=1),nn.Tanh(),nn.Conv1d(24,24,3,padding=1),nn.Tanh())
        self.state=nn.Sequential(nn.Linear(24+dim,24),nn.Tanh(),nn.Linear(24,1))
        self.scale_raw=nn.Parameter(torch.tensor(20.))
        self.margin_raw=nn.Parameter(torch.tensor(-1.5))
    def once(self,x):
        h=self.conv(x.transpose(1,2))[:,:,3]
        return self.state(torch.cat([h,x[:,3]],1))[:,0]
    def position(self,x):
        # Physical configuration does not change when time is reversed.
        return torch.sigmoid((self.once(x)+self.once(x.flip(1)))/2)
    def forward(self,x):
        n=len(x);p=self.position(x.reshape(-1,7,x.shape[-1])).reshape(n,2)
        d=p[:,1]-p[:,0];scale=torch.nn.functional.softplus(self.scale_raw)
        margin=torch.sigmoid(self.margin_raw)
        return torch.stack([-d,torch.ones_like(d)*margin,d],1)*scale,p
