"""Separate supervised phase readout; old observation weights never change."""
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
S=Path(__file__).resolve().parent;O=S.parent;P=O/'word100_learned_guide_feedback_20261102';N=O/'word100_learned_motion_invariance_20261102';M=O/'word100_learned_motion_order_20261102';G=O/'word100_learned_feedback_integration_20261102';U=O/'word100_learned_feature_retention_20261102';V=O/'word100_learned_body_relation_20261102'
sys.path.insert(0,str(N))
from prepare_real_fit import read,sha,exclusive,write,clock
WORDS=['WORD0739','WORD1133','WORD1567','WORD2523']

class PhaseNet(nn.Module):
    def __init__(self,condition):
        super().__init__();self.condition=condition
        base=torch.load(P/'runs/projection_finetune_42/model.pt',map_location='cpu',weights_only=True)
        self.query=nn.Linear(64,32);self.query.load_state_dict({k:base['stage_query.'+k] for k in ['weight','bias']})
        self.guide=nn.Sequential(nn.Linear(base['guide_vectors'].shape[1],32),nn.GELU(),nn.Linear(32,32))
        self.guide.load_state_dict({k[len('guide_encoder.'):]:v for k,v in base.items() if k.startswith('guide_encoder.')})
        self.embedding=nn.Parameter(base['stage_embedding'].clone())
        self.register_buffer('guide_vectors',base['guide_vectors']);self.register_buffer('guide_words',base['guide_words'])
        groups=read(S/'PHYSICAL_STATE_TARGETS.json')['groups'];self.aliases=[g['old_stage_aliases'] for g in groups]
        self.register_buffer('group_words',torch.tensor([WORDS.index(g['word_id']) for g in groups]))
        if condition=='temporal_readout':
            self.context=nn.GRU(64,16,batch_first=True,bidirectional=True)
            self.residual=nn.Linear(32,32);nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
    def forward(self,x,lengths,word):
        q=self.query(x)
        if self.condition=='temporal_readout':
            h,_=self.context(nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False))
            h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1]);q=q+self.residual(h)
        keys=self.guide(self.guide_vectors)+self.embedding
        stage=q@keys.T/(32**.5)
        groups=torch.stack([torch.logsumexp(stage[:,:,ids],-1) for ids in self.aliases],-1)
        return groups.masked_fill(self.group_words[None,None]!=word[:,None,None],-1e4)

def batch(rows):
    lengths=torch.tensor([len(r['x']) for r in rows]);T=int(lengths.max());x=torch.zeros(len(rows),T,64);y=torch.full((len(rows),T),-1,dtype=torch.long)
    for i,r in enumerate(rows):x[i,:len(r['x'])]=torch.from_numpy(r['x']);y[i,:len(r['y'])]=torch.from_numpy(r['y'])
    return x,lengths,torch.tensor([WORDS.index(r['word_id']) for r in rows]),y

def labels_for(clip_id,mapping):
    row=next(r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows'] if r['clip_id']==clip_id)
    labels={p['frame']:p['group'] for p in row['points'] if p['mask']}
    return np.asarray([labels.get(int(k),-1) for k in mapping],np.int64)

def fit_rows():
    result=[]
    for rec in read(N/'INPUT_MANIFEST.json')['rows']+read(S/'REVERSE_FIT_MANIFEST.json')['rows']+read(S/'SHORT_GAP_FIT_MANIFEST.json')['rows']:
        assert rec['role']=='fit' and rec['person'] in range(1,10) and sha(rec['input'])==rec['analysis_sha256']
        with np.load(rec['input']) as a:
            x=a['latent'].copy();assert len(x)==len(rec['source_frame_indices']) and np.isfinite(x).all()
        result.append(dict(clip_id=rec['clip_id'],person=rec['person'],word_id=rec['word_id'],kind=rec['kind'],input=rec['input'],input_sha256=rec['analysis_sha256'],
            x=x,y=labels_for(rec['clip_id'],rec['source_frame_indices']),mapping=rec['source_frame_indices']))
    assert len(result)==180
    return result

def measure(model,rows):
    points=[];model.eval()
    with torch.inference_mode():
        for r in rows:
            x,l,w,y=batch([r]);pr=model(x,l,w).softmax(-1)[0].numpy()
            for i in np.flatnonzero(r['y']>=0):points.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
    return points

def metrics(points):
    groups=sorted({p['y'] for p in points});per={str(g):dict(known=sum(p['y']==g for p in points),correct=sum(p['y']==g and p['pred']==g for p in points)) for g in groups}
    return dict(known=len(points),correct=sum(p['y']==p['pred'] for p in points),macro_recall=float(np.mean([v['correct']/v['known'] for v in per.values()])),groups=per)
