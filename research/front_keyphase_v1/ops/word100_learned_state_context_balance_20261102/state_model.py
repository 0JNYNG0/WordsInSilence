"""Present readout and a confidence-limited extra context branch.

The retained64D latent already contains temporal context. 'Current' here means
no NEW recurrent layer, not an independent single-image measurement.
"""
from pathlib import Path
from functools import lru_cache
import sys,numpy as np,torch
from torch import nn
T=Path(__file__).resolve().parent;O=T.parent;S=O/'word100_learned_phase_sequence_20261102'
sys.path.insert(0,str(S));import phase_model as old
from phase_model import read,sha,exclusive,write,clock,WORDS,batch,metrics

class StateNet(nn.Module):
    def __init__(self,context=False):
        super().__init__();self.current=old.PhaseNet('phase_readout');self.rest=nn.Linear(64,1)
        nn.init.zeros_(self.rest.weight);nn.init.zeros_(self.rest.bias);self.with_context=context
        if context:
            self.context=nn.GRU(64,16,batch_first=True,bidirectional=True);self.delta=nn.Linear(32,10)
            nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def current_logits(self,x,lengths,word):return torch.cat([self.current(x,lengths,word),self.rest(x)],-1)
    def forward(self,x,lengths,word,details=False):
        base=self.current_logits(x,lengths,word)
        if not self.with_context:return (base,base,torch.zeros(base.shape[:2],dtype=torch.bool)) if details else base
        h,_=self.context(nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False))
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        # Fixed pre-existing0.8 evidence criterion. Confident current output is
        # exactly preserved; this cannot repair a confidently wrong current head.
        allow=base.detach().softmax(-1).amax(-1)<.8
        out=base+self.delta(h)*allow[:,:,None]
        return (out,base,allow) if details else out
    def freeze_current(self):
        for p in self.current.parameters():p.requires_grad_(False)
        for p in self.rest.parameters():p.requires_grad_(False)

@lru_cache(maxsize=64)
def source_labels(cid):
    r=next(r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows'] if r['clip_id']==cid)
    rr=next(r for r in read(T/'BACKGROUND_TARGETS.json')['rows'] if r['clip_id']==cid)
    d={p['frame']:p['group'] for p in r['points'] if p['mask']}
    for p in rr['points']:
        if p['mask']:assert p['frame'] not in d;d[p['frame']]=9
    return d

def labels_for(cid,mapping):
    d=source_labels(cid);return np.asarray([d.get(int(k),-1) for k in mapping],np.int64)

def fit_rows():
    rows=old.fit_rows()
    for r in rows:r['y']=labels_for(r['clip_id'],r['mapping'])
    for r in read(T/'LOCAL_FIT_MANIFEST.json')['rows']:
        assert r['role']=='fit' and 1<=r['person']<=9 and sha(r['input'])==r['analysis_sha256']
        with np.load(r['input']) as a:x=a['latent'].copy()
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],kind=r['kind'],input=r['input'],input_sha256=r['analysis_sha256'],x=x,
          y=labels_for(r['clip_id'],r['source_frame_indices']),mapping=r['source_frame_indices']))
    assert len(rows)==216 and all(np.isfinite(r['x']).all() and len(r['x'])==len(r['mapping']) for r in rows)
    return rows

def measure(model,rows):
    points=[];model.eval()
    with torch.inference_mode():
        for r in rows:
            x,l,w,y=batch([r]);pr=model(x,l,w).softmax(-1)[0].numpy()
            for i in np.flatnonzero(r['y']>=0):points.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
    return points
