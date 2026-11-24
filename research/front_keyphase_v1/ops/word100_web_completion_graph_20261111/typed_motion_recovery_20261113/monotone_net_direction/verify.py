"""Saved two-parameter net-direction arithmetic audit; never call model.forward."""
from pathlib import Path
import importlib.util,json,datetime
from collections import Counter
import numpy as np
import torch
M=Path(__file__).resolve().parent;H=M.parent;G=H.parent
spec=importlib.util.spec_from_file_location('_saved_direction_audit',H/'verify.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
rd,sha,ck,pins,key=a.read,a.sha,a.ck,a.pins,a.key
def softmax(v):
    e=np.exp(np.asarray(v,dtype=float)-np.max(v,axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
def expected_logits(d,s,b):
    d=np.asarray(d,np.float32);s=np.float32(s);b=np.float32(b)
    return np.stack([-s*d,np.ones_like(d)*s*b,s*d],-1)
def main():
    assert not(M/'VERIFICATION.json').exists()
    cfg=rd(M/'FROZEN.json');tc=rd(M/'TRAINING_COMPLETE.json');ef=rd(M/'evaluation/FROZEN.json');res=rd(M/'evaluation/RESULTS.json');source=rd(H/'evaluation/RESULTS.json');meta=rd(H/'INPUTS_COMPLETE.json')
    pins(cfg['pins'],'training');pins(ef['pins'],'evaluation');pins({meta['path']:meta['sha256']},'input')
    original=np.load(meta['path']);ix=[i for i,r in enumerate(meta['rows']) if r['axis']==2];rows=[meta['rows'][i] for i in ix]
    ck('exact existing axis2 rows',rows==tc['rows'])
    ck('only train/probe',all(r['split'] in ['train','probe'] for r in rows))
    ck('source split',not{r['source_sha256'] for r in rows if r['split']=='train'}&{r['source_sha256'] for r in rows if r['split']=='probe'})
    ck('people split',all(r['person'] in range(1,8) for r in rows if r['split']=='train') and all(r['person'] in [8,9] for r in rows if r['split']=='probe'))
    for origin in [True,False]:
        for label in [0,1,2]:ck('FIT pool:'+str((origin,label)),any(r['split']=='train' and (r['origin']=='original_motion_replay')==origin and r['target']==label for r in rows))
    ck('fixed seeds and800step',cfg['seeds']==[42,43,44] and cfg['steps']==800 and {r['seed'] for r in tc['runs']}=={42,43,44})
    params={};symmetry=[]
    pairs={}
    for j,r in enumerate(rows):
        k=tuple(r.get(f) for f in ['clip_id','view','origin','a','b','axis','moving','anchor'])
        pairs.setdefault(k,{})[r['reverse']]=j
    for run in tc['runs']:
        pins({run[k]:run[k+'_sha256'] for k in ['model','predictions']},'run')
        state=torch.load(run['model'],weights_only=True,map_location='cpu')
        ck('exactly two scalar parameters:'+str(run['seed']),set(state)=={'raw_scale','raw_band'} and all(v.numel()==1 for v in state.values()))
        s=float(run['scale']);b=float(run['band']);params[run['seed']]=(s,b)
        ck('positive physical scale band:'+str(run['seed']),s>0 and b>0 and np.isclose(s,np.logaddexp(0,float(state['raw_scale'])),rtol=1e-6) and np.isclose(b,np.logaddexp(0,float(state['raw_band'])),rtol=1e-6))
        ar=np.load(run['predictions']);d=ar['d'];lo=ar['logits'];y=ar['y']
        ck('raw uncentered delta exact:'+str(run['seed']),np.array_equal(d,original['projected_geometry'][ix,40]) and np.array_equal(ar['original_indices'],ix))
        ck('original labels exact:'+str(run['seed']),np.array_equal(y,original['y'][ix]) and np.array_equal(y,[r['target'] for r in rows]))
        ck('finite logits:'+str(run['seed']),lo.shape==(len(rows),3) and np.isfinite(lo).all())
        err=float(np.max(np.abs(lo-expected_logits(d,s,b))));ck('two parameter saved arithmetic:'+str(run['seed']),err<1e-5,err)
        pos=d>0;neg=d<0;ck('direction never inverts physical sign:'+str(run['seed']),np.all(lo[pos,2]>lo[pos,0]) and np.all(lo[neg,0]>lo[neg,2]))
        order=np.argsort(d);p=softmax(lo);ck('probability monotonically increases with rawdelta:'+str(run['seed']),np.all(np.diff(p[order,2])>=-1e-7) and np.all(np.diff(p[order,0])<=1e-7))
        maxerr=0.;paired=0
        for k,q in pairs.items():
            ck('both reverse labels present:'+str((run['seed'],k)),set(q)=={False,True})
            if set(q)!={False,True}:continue
            n,r=q[False],q[True];paired+=1
            ck('raw reversal:'+str((run['seed'],k)),np.isclose(d[r],-d[n],atol=1e-6) and y[r]==2-y[n])
            e=float(np.max(np.abs(lo[n]-lo[r,[2,1,0]])));maxerr=max(maxerr,e)
        ck('exact reversed class exchange:'+str(run['seed']),maxerr<1e-5,maxerr)
        ck('final step:'+str(run['seed']),run['log'][-1]['step']==800)
        symmetry.append(dict(seed=run['seed'],pairs=paired,maximum_logit_flip_error=maxerr,scale=s,band=b))
    old={key(r):r for r in source['rows']};new={key(r):r for r in res['rows']};saved={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in source['direction_predictions'] if r['condition']=='reencoded_projected_geometry'}
    pred={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in res['predictions']}
    ck('246 original scenarios',len(new)==len(res['rows'])==246 and set(new)==set(old))
    ck('72 direction outputs',len(pred)==len(res['predictions'])==72 and set(pred)==set(saved))
    preserved=0;used=set();changes=[]
    for kk,row in new.items():
        cc=row['conditions'];before=old[kk]['conditions'];ck('four conditions:'+str(kk),set(cc)==set(ef['conditions']))
        for name in ['baseline','local_old','reencoded_projected_geometry']:ck('stored control exact:'+str((kk,name)),cc[name]==before[name])
        for c,oc in zip(cc['monotone_net'],before['local_old']):
            ck('card extent retained:'+str((kk,c['card'])),{k:v for k,v in c.items() if k not in ['evidence','status']}=={k:v for k,v in oc.items() if k not in ['evidence','status']} and len(c['evidence'])==len(oc['evidence']))
            for e,oe in zip(c['evidence'],oc['evidence']):
                if e['task']!='direction' or e.get('axis')!=2:ck('nonaxis event exact:'+str((kk,c['card'],preserved)),e==oe);preserved+=1;continue
                k=(*kk,c['card']);p=pred[k];sp=saved[k];used.add(k);s,b=params[kk[0]];d=sp['geometry'].get('signed_change',0.)
                ck('same span eligibility desired rawdelta:'+str(k),p['span']==e['span']==oe['span']==sp['span'] and p['eligible']==sp['eligible'] and p['desired']==e['desired']==oe['desired'] and p['delta']==d)
                v=expected_logits([d],s,b)[0];ck('runtime logits arithmetic:'+str(k),np.allclose(p['logits'],v,atol=1e-5,rtol=1e-6))
                pr=softmax(v);ck('runtime probability arithmetic:'+str(k),np.allclose(pr,p['probabilities'],atol=1e-6) and e['probabilities']==p['probabilities'])
                ck('existing verdict logic:'+str(k),a.verdict(p['probabilities'],p['desired'],p['eligible'])==p['state']==e['state'])
                ck('other direction fields retained:'+str(k),{f:v for f,v in e.items() if f not in ['probabilities','state','model_scope']}=={f:v for f,v in oe.items() if f not in ['probabilities','state','model_scope']})
                if e['state']!=oe['state']:changes.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],card=c['card'],old=oe['state'],new=e['state'],delta=d))
            ck('card status arithmetic:'+str((kk,c['card'])),c['status']==a.status(c))
    ck('all predictions consumed',used==set(pred));summary=[]
    for seed in [42,43,44]:
        for kind in sorted({r['kind'] for r in res['rows']}):
            rr=[r for r in res['predictions'] if r['seed']==seed and r['kind']==kind];summary.append(dict(seed=seed,kind=kind,n=len(rr),states=dict(Counter(r['state'] for r in rr))))
    ck('all result denominator arithmetic',summary==res['summary'])
    protections=rd(G/'PROTECTIONS.json');pins(protections,'protected');ck('protected477',len(protections)==477)
    ck('shared unchanged and no private',tc['shared_models_unchanged'] is True and all(cfg[k] is False for k in ['private_access','final8','web_changed']))
    output=dict(at=a.stamp(),status='PASS' if all(c['ok'] for c in a.checks) else 'FAIL',checks=len(a.checks),failed=[c for c in a.checks if not c['ok']],rows=len(new),predictions=len(pred),preserved_nonaxis_events=preserved,protected_files=len(protections),symmetry=symmetry,summary=summary,changes=changes,no_model_forward=True,no_training=True,adopted=False,full_G2_pass=False,script_sha256=sha(__file__),reviewer='Separate Codex saved arithmetic review; not independent human semantic truth',limitations=['Monotonic raw net displacement prevents reversed sign probability but does not certify action extent, trajectory, contact, repetition or completion.','Small true progression and local holds overlap. Removing opposite-direction warnings alone does not preserve normal support.'])
    (M/'VERIFICATION.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:output[k] for k in ['status','checks','failed','rows','predictions','preserved_nonaxis_events','protected_files','symmetry']},ensure_ascii=False))
    if output['failed']:raise SystemExit(1)
if __name__=='__main__':main()
