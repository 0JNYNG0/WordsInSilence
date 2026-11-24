"""Public DEV input-repair diagnostic. Frozen weights, no training/selection."""
from common import *
import time
from collections import Counter
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src'),str(B),str(T)]
from pilot.data import build_channels
from pilot.web_input import to_native_like,active_span
from training_core import MODEL,validate_channels,padded_arrays,prediction
KINDS=('native','retry','retry_yolo','retry_fixed_scale_span','retry_yolo_fixed_scale_span')
SEEDS=(20261013,20261014,20261015)
def array_sha(x):return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()
def nullable(q):return {k:(v if np.isfinite(v) else None) for k,v in q.items()}

def freeze():
    f=read(U/'FROZEN.json');v=read(U/'INPUTS_VERIFIED.json');assert v['counts']['clips']==300
    assert v['status']=='NUMERIC_PASS_VISUAL_REVIEW_PENDING';check(f['pins']);check(v['pins'])
    models={'C13':str(B/'training_C_20261013_best.pt'),**{f'T3F{s}':str(T/f'T3F_{s}_best.pt') for s in SEEDS}}
    prior={'C13':str(B/'training_C_20261013_RESULT.json'),**{f'T3F{s}':str(T/f'T3F_{s}_RESULT.json') for s in SEEDS}}
    paths=[U/'classify.py',U/'verify_scores.py',U/'FROZEN.json',U/'INPUTS_VERIFIED.json',B/'training_core.py',T/'frozen_temporal.py',T/'temporal.py',
      ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']
    paths+=list(map(Path,models.values()))+list(map(Path,prior.values()))
    write(U/'CLASSIFIER_FROZEN.json',dict(at=now(),rows=f['rows'],models=models,prior=prior,kinds=KINDS,pins={str(p):sha(p) for p in paths},
      protocol='Same public DEV300 (IDs11/12/13). No learning or condition selection. Three natural channel-builder conditions and two original-span/body/hand-size-fixed diagnostics. Existing masked inputs must reproduce byte-exact, their saved logits are reused. Each distinct changed input/model inferred once; identical arrays share cached logits. E3 remains equal softmax mean of existing three T3F seeds.',
      interpretation='DEV development diagnosis, not unseen-user final accuracy. All300 including failures in denominator. Added hands are predicted evidence, not 21-joint ground truth; visual uncertainty prevents operational adoption. No automatic model/threshold selection. No user/mentor/extra199/200 data.',
      normalization='For fixed diagnostics original active span/body scale/each hand scale retained. If original hand scale unavailable, hand-local coordinates stay masked, body-relative evidence can still be used. Raw old valid coordinates preserved.'))
    print('CLASSIFIER FROZEN',flush=True)

def prepare():
    f=read(U/'CLASSIFIER_FROZEN.json');check(f['pins']);check(read(U/'INPUTS_VERIFIED.json')['pins'])
    (U/'classifier_inputs').mkdir();idx=[];cache={};original={};totals=Counter()
    for r in f['rows']:
        p=load(U/'outputs'/(r['clip_id']+'.npz'));data=[];meta=[];base=None;span0=None
        for j,kind in enumerate(KINDS):
            k=j if j<3 else j-2
            nx,nv=to_native_like(p['xy_px'][k],p['usable'][k]);span=active_span(nv) if j<3 else span0
            if span is None:data.append(None);meta.append(dict(error='No usable hand'));continue
            lo,hi=span;q=build_channels(nx[lo:hi+1],nv[lo:hi+1],p['timestamp_us'][lo:hi+1])
            if j==0:
                base=q;span0=span;old=load(read(r['record'])['views']['masked']['channel_file'])['x_whole'];assert same(old,q['x'])
                original.setdefault(array_sha(old),r['clip_id']);totals['exact_original_channel_cells']+=old.size
            if j>=3:
                q['x'][:,:102]*=q['body_scale']/base['body_scale']
                for s,key in enumerate(('left','right')):
                    orig=base['hand_scales'][key];cur=q['hand_scales'][key];a=102+s*42;b=a+42
                    if np.isfinite(orig) and np.isfinite(cur):q['x'][:,a:b]*=cur/orig
                    else:q['x'][:,a:b]=0;q['x'][:,237+s*21:258+s*21]=0
            validate_channels(q['x']);h=array_sha(q['x']);cache.setdefault(h,q['x']);data.append(h)
            meta.append(dict(span=list(span),body_scale=q['body_scale'],hand_scales=nullable(q['hand_scales']),fixed_scale_span=j>=3))
        if data[0]:
            a=cache[data[0]]
            for j in (3,4):
                b=cache[data[j]];m=a[:,186:279].astype(bool);assert b.shape==a.shape and np.all(b[:,186:279][m]==1)
                for aa,bb,mask in ((a[:,:102],b[:,:102],np.repeat(m[:,:51],2,1)),(a[:,102:186],b[:,102:186],np.repeat(m[:,51:],2,1))):
                    np.testing.assert_allclose(aa[mask],bb[mask],rtol=1e-6,atol=2e-6)
                assert same(a[:,279],b[:,279])
        idx.append({k:r[k] for k in ('clip_id','word_code','signer_id','label_index')}|dict(arrays=dict(zip(KINDS,data)),meta=dict(zip(KINDS,meta))))
    for h,x in cache.items():np.savez_compressed(U/'classifier_inputs'/(h+'.npz'),x=x)
    write(U/'CLASSIFIER_INPUTS.json',dict(at=now(),rows=idx,unique_inputs=len(cache),original_array_to_clip=original,counts=dict(totals),
       pins={str(p):sha(p) for p in (U/'classifier_inputs').glob('*.npz')},fixed_existing_coordinates_preserved=True))
    print('UNIQUE',len(cache),'REUSED',len(original),'NEW',len(cache)-len(original),flush=True)

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
        m.load_state_dict(ck['state_dict'],strict=True);m.eval();state={k:v.clone() for k,v in m.state_dict().items()}
        old={r['clip_id']:r for r in read(f['prior'][name])['predictions']['masked']}
        assert set(old)=={r['clip_id'] for r in idx['rows']}
        scores={};timing={};provenance={};t=time.perf_counter()
        with torch.inference_mode():
            for pp in sorted(idx['pins']):
                x=load(pp)['x'];h=Path(pp).stem;assert h==array_sha(x)
                if h in idx['original_array_to_clip']:
                    cid=idx['original_array_to_clip'][h];assert old[cid]['logits'] is not None
                    scores[h]=np.array(old[cid]['logits']);provenance[h]=dict(kind='original_masked_logits',clip_id=cid);continue
                xx,mask=padded_arrays([x]);tt=time.perf_counter();out=m(torch.from_numpy(xx),torch.from_numpy(mask)).numpy()[0];timing[h]=time.perf_counter()-tt
                assert np.isfinite(out).all();scores[h]=out;provenance[h]=dict(kind='new_forward')
        elapsed=time.perf_counter()-t
        assert all(torch.equal(state[k],v) for k,v in m.state_dict().items());check(f['pins'])
        pred={kind:[prediction(r,scores[r['arrays'][kind]]) if r['arrays'][kind] else prediction(r,reason='Input absent') for r in idx['rows']] for kind in KINDS}
        for r in pred['native']:assert r['logits']==old[r['clip_id']]['logits']
        write(target,dict(at=now(),context_sha256=sha(U/'CLASSIFIER_FROZEN.json'),model=name,checkpoint=path,predictions=pred,provenance=provenance,
          unique_forwards=len(timing),reused_original_arrays=len(scores)-len(timing),forward_seconds=timing,total_load_and_forward_seconds=elapsed,training=0,model_state_unchanged=True))
        print(name,'complete',len(timing),'forwards',flush=True)

def aggregate():
    f=read(U/'CLASSIFIER_FROZEN.json');saved={name:read(U/(name+'_SCORES.json'))['predictions'] for name in f['models']};saved['E3']={}
    for kind in KINDS:
        rows=[]
        for i,r in enumerate(f['rows']):
            three=[saved[f'T3F{s}'][kind][i] for s in SEEDS];assert all(x['clip_id']==r['clip_id'] for x in three)
            if any(x['logits'] is None for x in three):rows.append(prediction(r,reason='Input absent'));continue
            a=np.asarray([x['logits'] for x in three]);prob=np.exp(a-a.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            rows.append(prediction(r,np.log(prob.mean(0))))
        saved['E3'][kind]=rows
    write(U/'E3_SCORES.json',dict(at=now(),predictions=saved['E3'],weights=[1/3]*3,training=0))
    summary={}
    for name,ps in saved.items():
        base={r['clip_id']:r for r in ps['native']};out={}
        for kind,rows in ps.items():
            out[kind]=dict(total=len(rows),correct=sum(r['correct'] for r in rows),top5=sum(r['top5_correct'] for r in rows),
              gained=[r['clip_id'] for r in rows if r['correct'] and not base[r['clip_id']]['correct']],
              lost=[r['clip_id'] for r in rows if not r['correct'] and base[r['clip_id']]['correct']],
              label_changed=[r['clip_id'] for r in rows if r['prediction']!=base[r['clip_id']]['prediction']],
              by_signer={str(s):sum(r['correct'] for r in rows if r['signer_id']==s) for s in (11,12,13)})
        summary[name]=out
    write(U/'CLASSIFIER_RESULT.json',dict(at=now(),summary=summary,scope='Development300 input-repair diagnosis; not unseen-user final accuracy',candidate_selected=False))
    print(json.dumps({k:{a:v['correct'] for a,v in b.items()} for k,b in summary.items()}),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
