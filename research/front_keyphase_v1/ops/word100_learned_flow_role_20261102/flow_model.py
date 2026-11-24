"""Target flow may guide process roles; it never edits physical observations."""
from pathlib import Path
import sys,importlib.util
FLOW_UNIT=Path(__file__).resolve().parent
sys.path.insert(0,str(FLOW_UNIT.parent/'word100_content_role_bridge_20261102'))
from bridge_model import *
from evaluate_transfer import eval_rows
FLOW_OUT=Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102')
PROTO=P/'runs/projection_finetune_42/model.pt'
spec=importlib.util.spec_from_file_location('flow_alignment',P/'alignment.py')
alignment=importlib.util.module_from_spec(spec);spec.loader.exec_module(alignment)

def flow_features(r,prototype,guides):
    n=len(r['x']);ref=prototype[process.WORDS.index(r['word_id'])]
    m=len(ref);al=alignment.dtw(r['x'],ref);path=al['path'];cost=al['cost']
    guide_mask=np.zeros(m,bool)
    for g in guides:
        if g['word_id']==r['word_id']:guide_mask[g['frames'][0]:g['frames'][1]+1]=True
    reuse=np.bincount(path[:,1],minlength=m);features=[]
    for i in range(n):
        js=path[path[:,0]==i,1]
        features.append([float(np.mean(js)/max(m-1,1)),float(cost[i,js].mean()),float(guide_mask[js].mean()),
            float(np.log1p(len(js)/(m/n))),float(np.log1p(reuse[js].mean()/(n/m))),float(r['available'][i])])
    z=np.asarray(features,np.float32);assert np.isfinite(z).all()
    z[~r['available'],:5]=0
    return z,path

def get_rows(split):
    rr=rows() if split=='fit' else eval_rows()
    for i,r in enumerate(rr):r['flow_key']=f'{split}_{i:03d}'
    return rr

def flow_batch(rr,normal):
    max_n=max(len(r['x']) for r in rr);out=np.zeros((len(rr),max_n,6),np.float32)
    for i,r in enumerate(rr):
        if '_normalized_flow' in r:
            v=r['_normalized_flow'];out[i,:len(v)]=v;continue
        item=normal['items'][r['flow_key']];p=Path(item['path']);assert sha(p)==item['sha256']
        a=np.load(p);v=a['features'].copy();assert len(v)==len(r['x']) and np.array_equal(a['source_frame_id'],r['mapping'])
        av=r['available'];v[:,:5]=(v[:,:5]-np.asarray(normal['mean']))/np.asarray(normal['std']);v[~av,:5]=0
        assert np.array_equal(v[:,5]>0,av);r['_normalized_flow']=v;out[i,:len(v)]=v
    return torch.from_numpy(out)

class FlowRoleNet(BridgeNet):
    def __init__(self):
        super().__init__(False)
        self.content_bridge.weight.requires_grad_(False)
        self.flow_bridge=nn.Linear(6,122,bias=False);nn.init.zeros_(self.flow_bridge.weight)
    def forward(self,x,rgb,lengths,flow):
        with torch.no_grad():
            feature,h=self.encode(x,rgb,lengths)
            logits=self.actor.query(feature)@(self.actor.guide(self.actor.guide_vectors)+self.actor.embedding).T/(32**.5)
            families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
            action=torch.cat([families,self.actor.rest(feature)],-1)
        z=self.role_embed(torch.cat([feature,h,x[:,:,64:74]],-1)+self.flow_bridge(flow))
        z,_=self.role_context(nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False))
        z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
        return action,self.role_output(z)
