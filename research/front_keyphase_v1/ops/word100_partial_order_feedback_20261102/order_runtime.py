"""Isolated frozen partial-order inference, reusing the existing native input.

Run in a separate Python process: historical training modules have generic
module names and must not be imported into the already running web server.
No labels, participant IDs, source witnesses, or reference videos enter infer.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np

Z=Path(__file__).resolve().parent; O=Z.parent
Y=O/'word100_direct_occurrence_order_20261102'
sys.path.insert(0,str(Y))
import readout_model as model
from pair_contract import observe
import torch

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_frozen_runtime():
    frozen=read(Z/'RUNTIME_FROZEN.json')
    for p,h in frozen['pins'].items(): assert sha(p)==h,p
    return frozen

def validate_input(a,word,base):
    assert word in model.process.WORDS and base['word_id']==word
    ids=np.asarray(a['frame_id']); ts=np.asarray(a['timestamp_us']); n=len(ids)
    assert n>=16 and ids.shape==ts.shape==(n,)
    assert np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
    for key,shape in [('x',(n,69)),('mask',(n,69)),('latent',(n,64)),('rgb',(n,768)),('image_available',(n,))]:
        assert np.asarray(a[key]).shape==shape,(key,shape)
    assert np.isfinite(a['latent']).all() and np.isfinite(a['rgb']).all()
    assert np.isfinite(np.asarray(a['x'])[np.asarray(a['mask'],bool)]).all()
    assert base['frames']==n
    assert np.allclose(np.asarray(base['overlay']['times']),ts/1e6,rtol=0,atol=1e-9)

def normalized_input(a,current,stats):
    """Same frozen preprocessing, no training target construction."""
    av=np.asarray(a['image_available'],bool)
    z,mask=model.process.old.geometric(a['x'],a['mask'],av)
    r=dict(x=np.asarray(a['latent']),current=np.asarray(current),z=z,m=mask)
    x=model.process.old.features(r,stats)
    v=np.where(av[:,None],(a['rgb']-np.asarray(stats['rgb_mean']))/np.asarray(stats['rgb_std']),0)
    rgb=np.column_stack([v,av]).astype('float32')
    assert x.shape==(len(av),130) and rgb.shape==(len(av),769)
    return x,rgb

def flow_from_path(a,path,prototype,guides,word,normal):
    """Reuse the already computed learned-prototype path, no second DTW."""
    q=np.asarray(a['latent'],np.float64); ref=np.asarray(prototype,np.float64)
    n,m=len(q),len(ref); path=np.asarray(path,int); av=np.asarray(a['image_available'],bool)
    assert path.ndim==2 and path.shape[1]==2
    assert np.array_equal(np.unique(path[:,0]),np.arange(n))
    assert np.array_equal(np.unique(path[:,1]),np.arange(m))
    assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[n-1,m-1])
    d=np.diff(path,axis=0)
    assert ((d>=0)&(d<=1)).all() and (d.sum(1)>0).all()
    cost=((q[:,None,:]-ref[None,:,:])**2).mean(-1)
    mask=np.zeros(m,bool)
    for g in guides:
        if g['word_id']==word: mask[g['frames'][0]:g['frames'][1]+1]=True
    reuse=np.bincount(path[:,1],minlength=m); rows=[]
    for i in range(n):
        js=path[path[:,0]==i,1]
        rows.append([float(np.mean(js)/max(m-1,1)),float(cost[i,js].mean()),float(mask[js].mean()),
                     float(np.log1p(len(js)/(m/n))),float(np.log1p(reuse[js].mean()/(n/m))),float(av[i])])
    raw=np.asarray(rows,np.float32); raw[~av,:5]=0
    normalized=raw.copy()
    normalized[:,:5]=(normalized[:,:5]-np.asarray(normal['mean']))/np.asarray(normal['std'])
    normalized[~av,:5]=0
    return raw,normalized

class PartialOrderEngine:
    def __init__(self):
        verify_frozen_runtime()
        torch.set_num_threads(2)
        ck=model.OUT/'runs/tied_readout/model.pt'
        assert sha(ck)==read(ck.parent/'COMPLETE.json')['checkpoint_sha256']
        self.net=model.ReadoutNet('tied_readout')
        self.net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
        self.net.eval().requires_grad_(False)
        self.current=model.process.old.observer()
        self.stats=read(model.D/'NORMALIZATION.json')
        context=read(Z/'RUNTIME_CONTEXT.json')
        self.flow=context['flow_normalization']
        state=torch.load(model.PROTO,map_location='cpu',weights_only=True)
        self.prototypes=[state['prototype'][i,:n].numpy() for i,n in enumerate(state['prototype_lengths'].tolist())]
        self.guides=context['guides']
        self.graphs=context['graphs']
        self.families=context['families']
        self.model_sha256=sha(ck)

    @torch.inference_mode()
    def infer(self,a,word,base):
        validate_input(a,word,base); start=time.perf_counter(); n=len(a['frame_id'])
        lengths=torch.tensor([n]); wi=model.process.WORDS.index(word)
        current=self.current(torch.from_numpy(np.asarray(a['latent'],np.float32)[None]),lengths,torch.tensor([wi])).softmax(-1)[0].numpy()
        x,rgb=normalized_input(a,current,self.stats)
        rawflow,flow=flow_from_path(a,base['path'],self.prototypes[wi],self.guides,word,self.flow)
        ac,ro,oc=self.net(torch.from_numpy(x[None]),torch.from_numpy(rgb[None]),lengths,torch.from_numpy(flow[None]))
        physical=ac.softmax(-1)[0].numpy(); occurrence=oc.sigmoid()[0].numpy()
        pairs=[observe(physical,occurrence,a['image_available'],np.asarray(a['timestamp_us'])/1e6,i,j,'tied_readout')
               for i,j in self.graphs[word]['precedences']]
        # Pair indices are array indices. Keep native frame IDs as a separate
        # mapping so skipped decode frames never change their meaning.
        for p in pairs:
            p['first_native_frame_ids']=np.asarray(a['frame_id'])[p['first_evidence_frames']].tolist()
            p['second_native_frame_ids']=np.asarray(a['frame_id'])[p['second_evidence_frames']].tolist()
            p['unobserved_native_frame_ids']=np.asarray(a['frame_id'])[p['unobserved_frames']].tolist()
        return dict(pairs=pairs,seconds=time.perf_counter()-start,
            physical=physical,role=ro.softmax(-1)[0].numpy(),occurrence=occurrence,current=current,
            flow=rawflow,frame_id=np.asarray(a['frame_id']),timestamp_us=np.asarray(a['timestamp_us']),
            image_available=np.asarray(a['image_available']),model_sha256=self.model_sha256,
            individual_reference_video_comparisons=0,additional_DTW=0,training=False,
            scope='research partial observed order, not completion/omission/repetition correctness')

def main():
    import argparse
    from order_feedback import attach_partial_order
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); dest=Path(args.output).resolve(); assert dest.is_relative_to(Z.resolve())
    frozen=verify_frozen_runtime();pair=frozen['public_job']
    assert Path(args.input).resolve()==Path(pair['input']).resolve() and sha(args.input)==pair['input_sha256']
    assert Path(args.base).resolve()==Path(pair['base']).resolve() and sha(args.base)==pair['base_sha256']
    assert Path(args.base).parent.resolve()==Path(args.input).parent.resolve()
    assert not dest.exists(); dest.mkdir(parents=True)
    a=dict(np.load(args.input,allow_pickle=False));base=read(args.base)
    assert base['word_id']==pair['word_id'] and base['source_sha256']==pair['source_sha256']
    start=time.perf_counter();engine=PartialOrderEngine();load_seconds=time.perf_counter()-start
    result=engine.infer(a,base['word_id'],base)
    out=attach_partial_order(base,result['pairs'],engine.families,engine.graphs[base['word_id']])
    out['partial_order_runtime']=dict(load_seconds=load_seconds,head_seconds=result['seconds'],additional_DTW=0,
        individual_reference_video_comparisons=0,new_capture=False,new_backbone_inference=False,
        backbone_scope='No MediaPipe/SignRep/RGB extraction; fixed current-state head and tied neural head each run once.',
        model_sha256=result['model_sha256'],source_input_sha256=sha(args.input),base_result_sha256=sha(args.base),
        model_use='post-comparison research component; not independent generalization validation',
        full_feedback_latency_measured=False)
    np.savez_compressed(dest/'order_arrays.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    for name,value in [('PARTIAL_ORDER_FEEDBACK.json',out),('PAIRS.json',result['pairs'])]:
        (dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(out['partial_order_runtime'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    try: main()
    except BaseException:
        import traceback
        p=Z/'FIRST_RUNTIME_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
