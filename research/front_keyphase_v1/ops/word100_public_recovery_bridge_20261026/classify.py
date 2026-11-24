"""Frozen public TRAIN diagnostic. No refitting or user data, exact-input cache."""
from common import *
import time
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src'),str(B),str(T)]
from pilot.data import build_channels
from pilot.web_input import to_native_like,active_span
from training_core import MODEL,validate_channels,padded_arrays,prediction
KINDS=('native','retry','retry_yolo','retry_fixed_scale_span','retry_yolo_fixed_scale_span')
def array_sha(x):return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()
def nullable(q):return {k:(v if np.isfinite(v) else None) for k,v in q.items()}

def freeze():
    f=read(U/'FROZEN.json');assert read(U/'VERIFIED.json')['status']=='PASS';ctx=read(B/'training_CONTEXT.json')
    public={r['clip_id']:r for r in ctx['ordered_rows'] if r['role']=='train_candidate'}
    rows=[{k:public[r['clip_id']][k] for k in ('clip_id','word_code','signer_id','label_index')} for r in f['rows']]
    models={'C13':str(B/'training_C_20261013_best.pt'),**{f'T3F{s}':str(T/f'T3F_{s}_best.pt') for s in (20261013,20261014,20261015)}}
    paths=[U/'classify.py',U/'FROZEN.json',U/'VERIFIED.json',B/'training_CONTEXT.json',B/'training_core.py',T/'frozen_temporal.py',T/'temporal.py',
           ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']
    paths+=list((U/'inputs').glob('*.npz'))+[Path(p) for p in models.values()]
    write(U/'CLASSIFIER_FROZEN.json',dict(at=now(),rows=rows,models=models,kinds=list(KINDS),pins={str(p):sha(p) for p in paths},
        protocol='Same46 known TRAIN clips, diagnostic only. Frozen C13 and all3 T3F, plus previously frozen uniform E3. No learning or selection. First3 exact existing channel builder with its natural active span and per-clip scales. Additional2 keep native span/body/hand scales to isolate recovered positions from normalization changes; if original hand scale absent, local hand channels remain masked. Raw original observations preserved; uncertain evidence sidecar used for DTW, not new classifier channels. HaMeR inferred/region data excluded.',
        denominator='All46 for every condition/model, failures remain incorrect. No fresh/public holdout inference. One forward pass per unique array/model; identical arrays reuse logits.',
        limits='TRAIN scores cannot demonstrate unseen-person/general-webcam accuracy. No accuracy-based threshold/weight choices. Export gains and losses; coordinate/quality validation is separate. Operational C/T3F unchanged.'))
    print('CLASSIFIER FROZEN',flush=True)

def prepare():
    f=read(U/'CLASSIFIER_FROZEN.json');check(f['pins']);(U/'classifier_inputs').mkdir();idx=[];cache={}
    for r in f['rows']:
        p=load(U/'inputs'/(r['clip_id']+'.npz'));data=[];meta=[];base=None;span0=None
        for j,kind in enumerate(KINDS):
            k=j if j<3 else j-2
            xy=np.concatenate([p['pose_xy'],p['xy'][k].reshape(-1,42,2)],1)
            valid=np.concatenate([p['pose_valid'],p['valid'][k].reshape(-1,42)],1)
            nx,nv=to_native_like(xy,valid);span=active_span(nv) if j<3 else span0
            if span is None:
                data.append(None);meta.append(dict(error='No usable hand'));continue
            lo,hi=span;q=build_channels(nx[lo:hi+1],nv[lo:hi+1],p['timestamp_us'][lo:hi+1])
            if j==0:base=q;span0=span
            if j>=3:
                q['x'][:,:102]*=q['body_scale']/base['body_scale']
                for s,key in enumerate(('left','right')):
                    orig=base['hand_scales'][key];cur=q['hand_scales'][key];a=102+s*42;b=a+42
                    if np.isfinite(orig) and np.isfinite(cur):q['x'][:,a:b]*=cur/orig
                    else:q['x'][:,a:b]=0;q['x'][:,237+s*21:258+s*21]=0
            validate_channels(q['x']);h=array_sha(q['x']);cache.setdefault(h,q['x']);data.append(h)
            meta.append(dict(span=list(span),body_scale=q['body_scale'],hand_scales=nullable(q['hand_scales']),fixed_scale_span=j>=3))
        # Fixed-scale variants must preserve every already observed baseline channel.
        if data[0]:
            a=cache[data[0]]
            for j in (3,4):
                b=cache[data[j]];m=a[:,186:279].astype(bool);assert b.shape==a.shape
                assert np.all(b[:,186:279][m]==1)
                mask_body=np.repeat(m[:,:51],2,1);mask_hand=np.repeat(m[:,51:],2,1)
                np.testing.assert_allclose(b[:,:102][mask_body],a[:,:102][mask_body],rtol=1e-6,atol=2e-6)
                np.testing.assert_allclose(b[:,102:186][mask_hand],a[:,102:186][mask_hand],rtol=1e-6,atol=2e-6)
                assert same(a[:,279],b[:,279])
        idx.append(r|dict(arrays=dict(zip(KINDS,data)),meta=dict(zip(KINDS,meta))))
    for h,x in cache.items():np.savez_compressed(U/'classifier_inputs'/(h+'.npz'),x=x)
    write(U/'CLASSIFIER_INPUTS.json',dict(at=now(),rows=idx,unique_inputs=len(cache),pins={str(p):sha(p) for p in (U/'classifier_inputs').glob('*.npz')},fixed_existing_coordinates_preserved=True))
    print('UNIQUE INPUTS',len(cache),flush=True)

def run():
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    from frozen_temporal import FrozenCTemporal
    f=read(U/'CLASSIFIER_FROZEN.json');check(f['pins']);idx=read(U/'CLASSIFIER_INPUTS.json');check(idx['pins'])
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for name,path in f['models'].items():
        target=U/(name+'_SCORES.json')
        if target.exists():assert read(target)['context_sha256']==sha(U/'CLASSIFIER_FROZEN.json');continue
        write(U/(name+'_STARTED.json'),dict(at=now(),context_sha256=sha(U/'CLASSIFIER_FROZEN.json')))
        ck=torch.load(path,map_location='cpu',weights_only=True);m=MaskedSignTCN(**MODEL) if name=='C13' else FrozenCTemporal(**MODEL)
        m.load_state_dict(ck['state_dict'],strict=True);m.eval();state={k:v.clone() for k,v in m.state_dict().items()};scores={};t=time.perf_counter()
        with torch.inference_mode():
            for pp in sorted(idx['pins']):
                x=load(pp)['x'];h=Path(pp).stem;assert h==array_sha(x);xx,mask=padded_arrays([x]);out=m(torch.from_numpy(xx),torch.from_numpy(mask)).numpy()[0]
                assert np.isfinite(out).all();scores[h]=out
        elapsed=time.perf_counter()-t
        assert all(torch.equal(state[k],v) for k,v in m.state_dict().items());check(f['pins'])
        pred={kind:[prediction(r,scores[r['arrays'][kind]]) if r['arrays'][kind] else prediction(r,reason='Input absent') for r in idx['rows']] for kind in KINDS}
        write(target,dict(at=now(),context_sha256=sha(U/'CLASSIFIER_FROZEN.json'),model=name,checkpoint=path,predictions=pred,
            unique_forwards=len(scores),total_load_and_forward_seconds=elapsed,training=0,model_state_unchanged=True))
        print(name,'complete',flush=True)

def aggregate():
    f=read(U/'CLASSIFIER_FROZEN.json');saved={name:read(U/(name+'_SCORES.json'))['predictions'] for name in f['models']};saved['E3']={}
    for kind in KINDS:
        rows=[]
        for i,r in enumerate(f['rows']):
            three=[saved[f'T3F{s}'][kind][i] for s in (20261013,20261014,20261015)]
            assert all(x['clip_id']==r['clip_id'] for x in three)
            if any(x['logits'] is None for x in three):rows.append(prediction(r,reason='Input absent'));continue
            a=np.asarray([x['logits'] for x in three]);prob=np.exp(a-a.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            out=prediction(r,np.log(prob.mean(0)));rows.append(out)
        saved['E3'][kind]=rows
    write(U/'E3_SCORES.json',dict(at=now(),predictions=saved['E3'],weights=[1/3]*3,training=0))
    summary={}
    for name,ps in saved.items():
        base={r['clip_id']:r for r in ps['native']};out={}
        for kind,rows in ps.items():
            out[kind]=dict(total=len(rows),correct=sum(r['correct'] for r in rows),top5=sum(r['top5_correct'] for r in rows),
                gained=[r['clip_id'] for r in rows if r['correct'] and not base[r['clip_id']]['correct']],
                lost=[r['clip_id'] for r in rows if not r['correct'] and base[r['clip_id']]['correct']],
                label_changed=[r['clip_id'] for r in rows if r['prediction']!=base[r['clip_id']]['prediction']])
        summary[name]=out
    write(U/'CLASSIFIER_RESULT.json',dict(at=now(),summary=summary,scope='Known TRAIN diagnostic; not generalization accuracy',candidate_selected=False))
    print(json.dumps({k:{a:v['correct'] for a,v in b.items()} for k,b in summary.items()}),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
