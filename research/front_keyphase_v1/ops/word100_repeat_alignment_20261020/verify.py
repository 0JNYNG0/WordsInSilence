"""Separate result verification, does not import alignment.py or audit.py."""
from pathlib import Path
from collections import defaultdict
import json,gzip,hashlib,traceback,time
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def wts(ts):
    t=(ts-ts[0])/1e6;left=np.r_[t[0],(t[:-1]+t[1:])/2];right=np.r_[(t[:-1]+t[1:])/2,t[-1]];return right-left
def source(r):
    assert sha(r['path'])==r['sha256']
    with np.load(r['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def independent_optimal(cost):
    n,m=cost.shape;b=max(1,round(.2*max(n,m)));last=np.full(m+1,np.inf);last[0]=0
    for i in range(n):
        curr=np.full(m+1,np.inf);center=i*(m-1)/(n-1)
        for j in range(max(0,int(np.floor(center-b))),min(m-1,int(np.ceil(center+b)))+1):curr[j+1]=cost[i,j]+min(last[j],last[j+1],curr[j])
        last=curr
    return last[-1]
def check_pair(q,r,entry,required,optimal=False):
    groups=[0,1] if required==['L'] else [2,3] if required==['R'] else list(range(4));p=np.asarray(entry['path'],int);i,j=p.T;k=entry['cycles'];m=len(r['ts'])
    assert p[0].tolist()==[0,0] and p[-1].tolist()==[len(q['ts'])-1,m*k-1]
    delta=np.diff(p,axis=0);assert np.all((delta>=0)&(delta<=1)) and np.all(delta.sum(1)>0)
    raw=j%m;valid=q['valid'][i][:,groups]&r['valid'][raw][:,groups];support=valid.mean(1);good=support>=.5
    diff=q['xy'][i][:,groups]-r['xy'][raw][:,groups];distance=np.sqrt(np.sum(np.nan_to_num(diff)**2,2));cost=np.where(valid,distance,1.5).mean(1)
    distinct=[]
    for cycle in range(k):
        z=good&(j//m==cycle)&r['reference_core'][raw];distinct.append(min(len(set(i[z])),len(set(raw[z]))))
    ok=support.mean()>=.35 and min(distinct)>=8
    balanced=np.mean(np.bincount(i,weights=cost,minlength=len(q['ts']))/np.bincount(i,minlength=len(q['ts'])))
    assert bool(ok)==entry['available'] and distinct==entry['supported_core_distinct']
    assert np.array_equal(good,entry['good']) and abs(float(support.mean())-entry['support'])<1e-8
    assert abs(balanced-entry['score'])<1e-8 and abs(float(cost.sum())-entry['accumulated_cost'])<1e-6
    if optimal:
        qa=q['xy'][:,groups];ra=r['xy'][:,groups];v=q['valid'][:,groups][:,None]&r['valid'][:,groups][None]
        c=np.where(v,np.linalg.norm(np.nan_to_num(qa[:,None]-ra[None]),axis=3),1.5).mean(2)
        assert abs(independent_optimal(np.tile(c,(1,k)))-entry['accumulated_cost'])<1e-6
    return p,good

def independent_votes(q,refs,bank,cycles,required):
    n=len(q['ts']);core=np.zeros(n,int);poss=np.zeros(n,int);outside=np.zeros(n,int);rest={s:np.zeros(n,int) for s in ('L','R')}
    strong={s:{a:np.zeros(n,int) for a in ('handshape','location','movement')} for s in rest};may={s:{a:np.zeros(n,int) for a in strong[s]} for s in rest};available=0
    def reduce(indices,values):
        mn=np.ones(n,bool);mx=np.zeros(n,bool);seen=np.bincount(indices,minlength=n)>0
        np.logical_and.at(mn,indices,values);np.logical_or.at(mx,indices,values)
        return mn&seen,mx
    for r,entries in zip(refs,bank):
        e=entries[cycles-1]
        if not e['available']:continue
        available+=1;p=np.asarray(e['path'],int)[np.asarray(e['good'],bool)];qi=p[:,0];ri=p[:,1]%len(r['ts'])
        allcore,anycore=reduce(qi,r['reference_core'][ri]);seen=np.bincount(qi,minlength=n)>0
        core+=allcore;poss+=anycore;outside+=seen&~anycore
        for s in rest:
            rr,_=reduce(qi,r[s+'_source_state'][ri]==3);rest[s]+=rr&allcore
            for a in strong[s]:
                expected=r['reference_core'][ri]&(r[s+'_source_state'][ri]==2)&r[s+'_'+a][ri]
                lo,hi=reduce(qi,expected);strong[s][a]+=lo;may[s][a]+=hi
    enough=available>=5;c=(core>=5)&enough;p=(poss>0)&enough;o=(outside>=5)&~p&enough
    answer=dict(core=c,possible_core=p,outside=o,unknown=~(c|o),available=available,core_votes=core,possible_core_votes=poss,outside_votes=outside,expected={},possible_expected={},rest={},semantic_verdict=None,whole_action_pass=None)
    for s in rest:
        rr=(rest[s]>=5)&c;answer['rest'][s]=rr
        answer['expected'][s]={a:(strong[s][a]>=5)&c&~rr if s in required else np.zeros(n,bool) for a in strong[s]}
        answer['possible_expected'][s]={a:(may[s][a]>0)&p&~rr if s in required else np.zeros(n,bool) for a in strong[s]}
    return answer
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b

def run():
    beg=time.time();index=read(U/'INDEX.json');by={(r['clip_id'],r['view']):r for r in index};cache={};byword=defaultdict(list)
    for r in index:
        if r['signer_id']<=9:byword[r['word'],r['view']].append(r)
    old=read(OPS/'word100_multiref_observation_compare_20261019/FROZEN.json');raw={r['clip_id']:r for r in old['inputs']}
    def cached(rec):
        key=(rec['clip_id'],rec['view'])
        if key not in cache:cache[key]=source(rec)
        return cache[key]
    allrows=read(U/'ROWS.json')+read(U/'DELETION.json');paths=0;optimal=0;orig={}
    for ix,row in enumerate(allrows):
        cid,view=row['clip_id'],row['view'];rec=by[cid,view];assert rec['signer_id'] in (11,12,13)
        d=cached(rec);q={k:d[k] for k in ('ts','xy','valid')};ids=np.arange(len(d['ts']));scenario=row['scenario']
        truth=np.zeros(len(d['ts']),bool)
        for a,b in raw[cid]['intervals']:truth|=(d['ts']>=a*1e6)&(d['ts']<=b*1e6)
        if scenario.startswith('repeat'):
            k=int(scenario[-1]);step=d['ts'][-1]-d['ts'][0]+int(np.median(np.diff(d['ts'])))
            q=dict(ts=np.concatenate([d['ts']-d['ts'][0]+j*step for j in range(k)]),xy=np.tile(d['xy'],(k,1,1)),valid=np.tile(d['valid'],(k,1)));ids=np.tile(ids,k);truth=np.tile(truth,k)
        elif scenario.startswith('delete'):
            ids=np.array(row['meta']['keep_rows']);q=dict(ts=np.arange(len(ids))*np.median(np.diff(d['ts'])),xy=d['xy'][ids],valid=d['valid'][ids]);truth=truth[ids]
        rr=byword[rec['word'],view];assert len({r['signer_id'] for r in rr})==len(rr) and len(rr) in (8,9)
        refs=[cached(r) for r in rr]
        with gzip.open(U/row['evidence_path'],'rt',encoding='utf-8') as f:saved=json.load(f)
        result=saved['result'];bank=result['bank'];required=saved['required'];scores=[]
        for refi,(r,entries) in enumerate(zip(refs,bank)):
            for e in entries:
                checkopt=(ix%200==0 and refi==0)
                check_pair(q,r,e,required,checkopt);paths+=1;optimal+=int(checkopt)
        for k in (1,2,3):
            count=sum(b[k-1]['available'] for b in bank);score=float(np.median([b[k-1]['score'] for b in bank])) if count>=5 else None
            scores.append(dict(cycles=k,score=score,valid_refs=count))
        assert scores==result['cycle_scores']==row['scores'];opts=[(r['score'],r['cycles']) for r in scores if r['score'] is not None];chosen=min(opts)[1] if opts else 1
        assert chosen==result['chosen_cycles']==row['chosen_cycles'];w=wts(q['ts'])
        for mode,k in [('single',1),('multi',chosen)]:
            p=independent_votes(q,refs,bank,k,required);equal(p,result[mode])
            for area in ('core','possible_core'):
                loc=row[mode+'_loc'][area];assert abs(float(w@(p[area]&truth))-loc['intersection_s'])<1e-7
                assert abs(float(w@p[area])-loc['predicted_s'])<1e-7 and abs(float(w@truth)-loc['annotation_s'])<1e-7
            for m in row[mode+'_metrics']:
                s,a=m['side'],m['key'];ev=d[s+'_'+a][ids];strong=p['expected'][s][a];possible=p['possible_expected'][s][a]
                assert abs(float(w@strong)-m['expected_s'])<1e-7 and abs(float(w@(strong&~ev))-m['missing_s'])<1e-7
                assert abs(float(w@(possible&~strong&~ev))-m['uncertain_missing_s'])<1e-7
                assert m['wrong_action'] is None and m['whole_action_pass'] is None
            if mode=='multi' and scenario=='original':orig[cid,view]=(p,q,d)
        if (ix+1)%200==0:print('verified rows',ix+1,flush=True)
    stress=read(U/'STRESS.json')
    for r in stress:
        a,b=r['before'],r['after'];assert a['expected_s']==b['expected_s'] and a['possible_s']==b['possible_s'] and r['extent_unchanged']
        assert abs(b['missing_s']-a['missing_s']-r['extra_strong_s'])<1e-7
        assert abs(b['uncertain_missing_s']-a['uncertain_missing_s']-(r['extra_possible_s']-r['extra_strong_s']))<1e-7
    f=read(U/'FROZEN.json');pins=0
    for k in ('protected','previous','code'):
        for p,h in f[k].items():assert sha(p)==h,p;pins+=1
    for name in ('INPUT_PINS.json','EVALUATION_PINS.json'):
        for p,h in read(U/name).items():assert sha(p)==h,p;pins+=1
    assert len(allrows)==2391 and len(stress)==7092
    write('VERIFICATION.json',dict(status='PASS',rows=len(allrows),independent_paths=paths,independent_dp_optimality_samples=optimal,
        votes_reconstructed=True,split_reference_people_verified=True,fixed_extent_masks=len(stress),hash_pins=pins,seconds=time.time()-beg,
        operator='Same Codex separate implementation, not another agent or sign expert',production_changed=False,training=0,private_data_used=0))
    print('VERIFIED',paths,pins,flush=True)
if __name__=='__main__':
    try:run()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
