import torch
class Rank(torch.nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.head=torch.nn.Sequential(torch.nn.Linear(1160,96),torch.nn.GELU(),torch.nn.Linear(96,5))
    def forward(self,x):return self.head((x-self.mean)/self.std)
class ExtentChoice(torch.nn.Module):
    def __init__(self,state):
        super().__init__();self.rank=Rank(state['rank.mean'],state['rank.std']);self.null_logits=torch.nn.Parameter(torch.zeros(5));self.load_state_dict(state)
    def forward(self,x):return self.rank(x)
