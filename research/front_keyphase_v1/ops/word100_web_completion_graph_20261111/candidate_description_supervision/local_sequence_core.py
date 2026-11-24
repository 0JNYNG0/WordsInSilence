from core import *
Q=W/'local_sequence_candidate';N=W/'dense_window_supervision'

def local_input(z,x):
 # No clip time, guide order, outside-window recurrent latent or word-ID input.
 return np.c_[z[:,292:392],x[:,102:186],x[:,237:279]].astype(np.float32)

class LocalCandidate(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(226,48,3,padding=1),nn.GELU(),nn.Conv1d(48,48,3,padding=1),nn.GELU());self.gru=nn.GRU(48,24,batch_first=True,bidirectional=True);self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,15))
 def forward(self,x,lengths):
  valid=torch.arange(x.shape[1],device=x.device)[None]<lengths[:,None];v=x.transpose(1,2)
  for layer in self.conv:v=layer(v)*valid[:,None]
  pack=nn.utils.rnn.pack_padded_sequence(v.transpose(1,2),lengths.cpu(),batch_first=True,enforce_sorted=False);h,_=self.gru(pack);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1]);last=h[torch.arange(len(x)),lengths-1];avg=(h*valid[:,:,None]).sum(1)/lengths[:,None];mx=h.masked_fill(~valid[:,:,None],-1e4).max(1).values
  return self.head(torch.cat([h[:,0],last,avg,mx],-1)).reshape(-1,5,3)

def batch(sequences):
 lens=np.array([len(s) for s in sequences]);out=np.zeros((len(lens),int(lens.max()),226),np.float32)
 for i,s in enumerate(sequences):out[i,:len(s)]=s
 return torch.tensor(out),torch.tensor(lens)
