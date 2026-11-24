"""Separate saved-score arithmetic via sorted sweeps, no model import/forward."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import json,hashlib,datetime,time,traceback
C=Path(__file__).resolve().parent
J=C.parent/'splice_rank_learning'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distribution(z,index,prefix,t):
    spans=z[f'spans_{index}'];logits=z[f'{prefix}_logits_{index}'].astype(float);admit=z[f'presence_{index}']>=0
    assert len(np.unique(spans,axis=0))==len(spans)
    if not admit.any():return None
    v=logits[admit];weight=np.exp(v-v.max());weight/=weight.sum()
    return t[spans[admit]],weight

def sweep(a,b):
    if a is None or b is None:return None
    aa,wa=a;bb,wb=b
    # Sum Bstart>=Aend using an explicit cumulative scan, no searchsorted/CDF helper.
    bs=np.argsort(bb[:,0]);ae=np.argsort(aa[:,1]);at=0;excluded=0.;forward=0.
    for i in ae:
        while at<len(bs) and bb[bs[at],0]<aa[i,1]:
            excluded+=wb[bs[at]];at+=1
        forward+=wa[i]*(1-excluded)
    be=np.argsort(bb[:,1]);ast=np.argsort(aa[:,0]);at=0;included=0.;reverse=0.
    for i in ast:
        while at<len(be) and bb[be[at],1]<=aa[i,0]:
            included+=wb[be[at]];at+=1
        reverse+=wa[i]*included
    return np.asarray([forward,reverse,1-forward-reverse])

def main():
    dest=C/'INDEPENDENT_VERIFICATION.json';assert not dest.exists()
    freeze=read(C/'FROZEN.json');result=read(C/'RESULTS.json');comp=read(J/'evaluation/COMPARISON.json')
    for path,h in freeze['pins'].items():assert sha(path)==h,path
    assert freeze['temperature']==1 and freeze['threshold']==.65
    assert result['inference']==result['training']==result['web_changed']==result['g2_pass_claim']==False
    meta={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
    prior={(r['seed'],r['clip_id'],r['kind']):r for r in comp['rows']}
    sources={};npzs={};checks=[];out=[]
    for row in result['rows']:
        key=row['seed'],row['clip_id'],row['kind'];pr=prior[key];mr=meta[key];mode=row['condition']
        cards=pr['conditions'][mode]['cards'];indices=[next(i for i,c in enumerate(cards)if c['card']==name) for name in row['cards']]
        assert row['raw_chosen']==[cards[i]['chosen'] for i in indices]
        assert all(c is None or c['semantic_completion'] is None for c in row['raw_chosen'])
        path=mr['path'] if mode=='splice_rank' else mr['original_prediction']
        if path not in npzs:npzs[path]=dict(np.load(path))
        if mr['source'] not in sources:sources[mr['source']]=np.load(mr['source'])['ts']
        t=sources[mr['source']];assert t.ndim==1 and np.issubdtype(t.dtype,np.integer) and (np.diff(t)>0).all()
        ds=[distribution(npzs[path],i,mode,t)for i in indices]
        mass=sweep(*ds);relation=row['relation'];assert relation['action_presence_established'] is False
        if mass is None:
            assert relation['state']=='no_admitted_candidates'
            assert all(relation[k] is None for k in ['forward','reverse','overlap'])
            out.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],condition=mode,unknown=True))
            continue
        stored=np.array([relation[k]for k in ['forward','reverse','overlap']])
        assert np.isfinite(stored).all() and (stored>=0).all() and abs(stored.sum()-1)<1e-12
        assert np.allclose(stored,mass,atol=1e-10,rtol=0),(key,mode,stored,mass)
        swapped=sweep(ds[1],ds[0]);assert np.allclose(swapped,mass[[1,0,2]],atol=1e-10,rtol=0)
        reverse=lambda d:(-d[0][:,::-1],d[1])
        reversed_mass=sweep(reverse(ds[0]),reverse(ds[1]));assert np.allclose(reversed_mass,mass[[1,0,2]],atol=1e-10,rtol=0)
        i=int(stored.argmax());state=['forward','reverse','positive_duration_overlap'][i] if stored[i]>=.65 else 'mixed_candidate_relations'
        assert relation['state']==state
        checks.append(dict(key=list(key),condition=mode,max_saved_error=float(np.max(np.abs(stored-mass))),
            swap_error=float(np.max(np.abs(swapped-mass[[1,0,2]]))),time_reverse_error=float(np.max(np.abs(reversed_mass-mass[[1,0,2]])))))
        out.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],condition=mode,unknown=False,state=state,
            forward=float(stored[0]),reverse=float(stored[1]),overlap=float(stored[2])))
    assert len(out)==144
    for summary in result['summary']:
        rows=[r for r in result['rows'] if all(r[k]==summary[k]for k in ['seed','kind','condition'])]
        assert summary['count']==len(rows) and summary['states']==dict(Counter(r['relation']['state']for r in rows))
    states=defaultdict(Counter)
    for row in result['rows']:states[row['condition'],row['kind']][row['relation']['state']]+=1
    conclusion=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SAVED144_MASS_ARITHMETIC_SYMMETRY_PINS_PASS_NOT_ADOPTED',
        rows=144,observed_rows=len(checks),no_candidate_rows=144-len(checks),pins_verified=len(freeze['pins']),
        maximum_error=max(c['max_saved_error']for c in checks),
        summaries=[dict(condition=k[0],kind=k[1],states=dict(v))for k,v in sorted(states.items())],
        fast_unresolved=[r for r in out if r['condition']=='splice_rank'and r['kind']=='speed125'and r.get('state')!='forward'],
        interpretation=['The three masses partition conditional candidate pairs correctly; swappedfamilies/reversedtime swapforwardandreverse.',
            'Massreadout doesnotrepair the remaining fastsourceambiguity. Keepmixedstate; no thresholdchange or G2/websuccessclaim.',
            'Dynamicreversedblock relation isreverse9/9, omittedfinaleventforward0. Theseare controlledrelativeorderdiagnostics, not proofactionabsence/completion.',
            'A maskedfinalevent can retain aconditionalforwardmass. Actionexistence, correctshape and completion need their ownobservability/evidence guards.',
            'Root may investigate FIT speedcoverage separately; these saved results alone cannot establish speedtraining omission as the cause.'],
        comparisons_preserved=True,chosen_spans_preserved=True,no_model_execution=True,no_training=True,no_threshold_selection=True,
        reviewer='Separate Codex arithmetic implementation, not independent human KSL truth',
        pins={str(p):sha(p)for p in [Path(__file__),C/'FROZEN.json',C/'RESULTS.json',C/'INDEPENDENT_DESIGN_REVIEW.json']})
    dest.write_text(json.dumps(conclusion,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified144 rows; observed',len(checks),'unknown',144-len(checks),'maxerror',conclusion['maximum_error'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (C/f'INDEPENDENT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
