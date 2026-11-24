"""Separate reconstruction; does not import this unit's run.py or consensus.py."""
from pathlib import Path
from collections import defaultdict, Counter
import json, gzip, hashlib, importlib.util, itertools, time, traceback
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;P=OPS/'word100_repeat_alignment_20261020'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('prior_independent_verifier',P/'verify.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
SIDES=('L','R');AXES=('handshape','location','movement')
def eq(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:eq(a[k],b[k])
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b,(a,b)
def evidence(q,r,entries,required,person):
    winners=[]
    available=[e for e in entries if e['available']]
    if available:
        cheapest=min(e['score'] for e in available);winners=[e['cycles'] for e in available if e['score']==cheapest]
    k=winners[0] if len(winners)==1 else None
    ans=dict(person=person,preferred=k,boundaries={})
    if k not in (2,3):return ans
    core=np.flatnonzero(r['reference_core'])
    if not len(core):return ans
    n=len(q['ts']);anchor=np.zeros(n,bool)
    ids=[0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]
    if core[0]>0 and core[-1]+1<len(r['ts']):
        ranges=[np.arange(core[0]),np.arange(core[-1]+1,len(r['ts'])),core]
        for i in range(n):
            joint=r['valid'][:,ids]&q['valid'][i,ids]
            delta=np.nan_to_num(r['xy'][:,ids]-q['xy'][i,ids])
            costs=np.where(joint,np.sqrt((delta*delta).sum(2)),1.5).mean(1)
            nearest=[rg[np.argmin(costs[rg])] for rg in ranges]
            anchor[i]=(joint[nearest[0]].mean()>=.5 and joint[nearest[1]].mean()>=.5
                       and costs[nearest[0]]<costs[nearest[2]] and costs[nearest[1]]<costs[nearest[2]])
    e=entries[k-1];path=np.array(e['path']);good=np.array(e['good']);m=len(r['ts'])
    core_pairs=[(i,j//m) for (i,j),g in zip(path,good) if g and r['reference_core'][j%m]]
    for c in range(k-1):
        left=[i for i,v in core_pairs if v==c];right=[i for i,v in core_pairs if v==c+1]
        ans['boundaries'][str(c+1)]=[int(i) for i in np.flatnonzero(anchor) if left and right and max(left)<i<min(right)]
    return ans
def decision(ev,n):
    counts=Counter(e['preferred'] for e in ev);known=[k for k in (1,2,3) if counts[k]]
    out=dict(count=None,status='COUNT_UNCERTAIN',reason='insufficient_people',preferences={str(k):counts[k] for k in (1,2,3)},
             abstaining_people=counts[None],supporting_people=[],boundaries=[])
    assert len(set(e['person'] for e in ev))==len(ev)
    if len(known)>1:out['reason']='reference_count_disagreement';return out
    if not known or counts[known[0]]<5:return out
    k=known[0];group=[e for e in ev if e['preferred']==k]
    if k==1:
        out.update(count=k,status='REFERENCE_SUPPORTED',reason='unopposed_single_traversal',supporting_people=[e['person'] for e in group]);return out
    for indices in itertools.combinations(range(len(group)),5):
        shared=[]
        for c in range(1,k):
            common=[j for j in range(n) if all(j in group[i]['boundaries'][str(c)] for i in indices)]
            shared.append(common)
        if all(shared) and all(a[-1]<b[0] for a,b in zip(shared,shared[1:])):
            out.update(count=k,status='REFERENCE_SUPPORTED',reason='same_people_observed_return_boundaries',supporting_people=[group[i]['person'] for i in indices],boundaries=shared);return out
    out['reason']='return_boundary_not_jointly_supported';return out
def merge(pp,dec,n):
    ks=[k for k in (1,2,3) if pp[k]['available']>=5];ss=[dec['count']] if dec['count'] else ks
    def intersect(arrays):return np.array([bool(arrays) and all(a[i] for a in arrays) for i in range(n)])
    def union(arrays):return np.array([any(a[i] for a in arrays) for i in range(n)])
    p=dict(core=intersect([pp[k]['core'] for k in ss]),possible_core=union([pp[k]['possible_core'] for k in ks]),
           expected={},possible_expected={},rest={},semantic_verdict=None,whole_action_pass=None)
    p['outside']=intersect([pp[k]['outside'] for k in ks])&~p['possible_core'];p['unknown']=~(p['core']|p['outside'])
    for s in SIDES:
        p['rest'][s]=intersect([pp[k]['rest'][s] for k in ks])
        p['expected'][s]={a:intersect([pp[k]['expected'][s][a] for k in ss]) for a in AXES}
        p['possible_expected'][s]={a:union([pp[k]['possible_expected'][s][a] for k in ks]) for a in AXES}
    return p
def main():
    beg=time.time();index=read(P/'INDEX.json');lookup={(r['clip_id'],r['view']):r for r in index};groups=defaultdict(list)
    for r in index:
        if r['signer_id']<=9:groups[r['word'],r['view']].append(r)
    cache={}
    def load(rec):
        if rec['path'] not in cache:
            assert sha(rec['path'])==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as z:cache[rec['path']]={k:z[k] for k in z.files}
        return cache[rec['path']]
    rows=read(U/'ROWS.json');original={};evidence_count=0;boundary_frames=0
    raw={r['clip_id']:r for r in read(OPS/'word100_multiref_observation_compare_20261019/FROZEN.json')['inputs']}
    for ix,row in enumerate(rows):
        cid,view=row['clip_id'],row['view'];rec=lookup[cid,view];assert rec['signer_id'] in (11,12,13)
        d=load(rec);ts=d['ts'];ids=np.arange(len(ts));scenario=row['scenario']
        if scenario.startswith('repeat'):
            k=int(scenario[-1]);period=ts[-1]-ts[0]+int(np.median(np.diff(ts)));ts=np.concatenate([ts-ts[0]+j*period for j in range(k)]);ids=np.tile(ids,k)
        elif scenario.startswith('delete'):
            ids=np.array(row['meta']['keep_rows']);ts=np.arange(len(ids))*np.median(np.diff(ts))
        q=dict(ts=ts,xy=d['xy'][ids],valid=d['valid'][ids]);w=V.wts(ts)
        rr=groups[rec['word'],view];people=[r['signer_id'] for r in rr];assert len(set(people))==len(people) and len(people)>=8
        refs=[load(r) for r in rr]
        with gzip.open(P/row['source_path'],'rt',encoding='utf-8') as h:old=json.load(h)
        with gzip.open(U/row['evidence_path'],'rt',encoding='utf-8') as h:saved=json.load(h)
        ev=[evidence(q,r,b,old['required'],person) for r,b,person in zip(refs,old['result']['bank'],people)]
        eq(ev,saved['reference_evidence']);evidence_count+=len(ev)
        dec=decision(ev,len(ts));eq(dec,saved['decision']);eq(dec,row['decision'])
        boundary_frames+=sum(len(v) for e in ev for v in e['boundaries'].values())
        pp={k:V.independent_votes(q,refs,old['result']['bank'],k,old['required']) for k in (1,2,3)}
        p=merge(pp,dec,len(ts));eq(p,saved['projection'])
        truth=np.zeros(len(d['ts']),bool)
        for a,b in raw[cid]['intervals']:truth|=(d['ts']>=a*1e6)&(d['ts']<=b*1e6)
        truth=truth[ids]
        for area in ('core','possible_core'):
            val=row['new_loc'][area]
            for key,mask in [('intersection_s',p[area]&truth),('annotation_s',truth),('predicted_s',p[area])]:assert abs(float(w@mask)-val[key])<1e-7
            if val['annotation_s']:assert abs(val['recall']-val['intersection_s']/val['annotation_s'])<1e-9
            if val['predicted_s']:assert abs(val['precision']-val['intersection_s']/val['predicted_s'])<1e-9
        for met in row['new_metrics']:
            s,a=met['side'],met['key'];present=d[s+'_'+a][ids]
            for key,mask in [('expected_s',p['expected'][s][a]),('possible_s',p['possible_expected'][s][a]),
                             ('missing_s',p['expected'][s][a]&~present),('uncertain_missing_s',p['possible_expected'][s][a]&~p['expected'][s][a]&~present)]:
                assert abs(float(w@mask)-met[key])<1e-7
            assert met['whole_action_pass'] is None and met['wrong_action'] is None
        for s in SIDES:
            for a in AXES:assert np.all(~pp[row['old_count']]['possible_expected'][s][a]|p['possible_expected'][s][a])
        if view=='sensor' and scenario=='original':original[cid]=(d,p,w)
        if (ix+1)%200==0:print('verified',ix+1,'seconds',round(time.time()-beg,1),flush=True)
    windows={(r['clip_id'],r['side'],r['condition']):r for r in read(OPS/'word100_missing_phase_validation_20261020/ROWS.json')}
    for row in read(U/'STRESS.json'):
        d,p,w=original[row['clip_id']];s,a=row['side'],row['key'];wr=windows[row['clip_id'],s,row['condition']]
        mask=np.ones(len(w),bool) if row['condition']=='whole' else (d['ts']/1e6>=wr['window']['start_s'])&(d['ts']/1e6<=wr['window']['end_s'])
        removed=mask&d[s+'_'+a];assert int(removed.sum())==row['removed_frames']
        assert abs(float(w@(removed&p['expected'][s][a]))-row['new_strong_s'])<1e-7
        assert abs(float(w@(removed&p['possible_expected'][s][a]))-row['new_possible_s'])<1e-7
        assert row['new_possible_s']+1e-8>=row['old_possible_s']
    full={}
    for row in read(U/'LOWERING.json'):
        cid=row['clip_id'];d,p,w=original[cid];s,a=row['side'],row['key']
        if cid not in full:
            with np.load(OPS/'word100_metric_evidence_20261020/features/sensor'/(cid+'.npz'),allow_pickle=False) as z:
                full[cid]={ss:z['full_'+ss+'_offline_full_view_state']==3 for ss in SIDES}
        rest_missing=full[cid][s]&~d[s+'_'+a]
        assert abs(float(w@(rest_missing&p['expected'][s][a]))-row['new_strong_s'])<1e-7
        assert abs(float(w@(rest_missing&p['possible_expected'][s][a]))-row['new_possible_s'])<1e-7
    summary=read(U/'SUMMARY.json')
    for key,g in summary['groups'].items():
        view,scenario=key.split('_');k=1 if scenario=='original' else int(scenario[-1]);rr=[r for r in rows if r['view']==view and r['scenario']==scenario]
        assert len(rr)==g['n']==300
        assert sum(r['decision']['count']==k for r in rr)==g['confirmed_matching']
        assert sum(r['decision']['count'] is None for r in rr)==g['abstained']
        assert g['confirmed_matching']+g['confirmed_different']+g['abstained']==300
        for mode in ('old','new'):
            for area in ('core','possible_core'):
                for metric in ('recall','precision'):
                    values=[r[mode+'_loc'][area][metric] for r in rr if r[mode+'_loc'][area][metric] is not None]
                    assert abs(float(np.median(values))-g['localization'][mode][area][metric])<1e-9
    pins=0;f=read(U/'FROZEN.json')
    for section in ('protected','previous','code'):
        for path,h in f[section].items():assert sha(path)==h,path;pins+=1
    for base,name in [(P,'PROJECTION_PINS.json'),(U,'EVIDENCE_PINS.json'),(U,'EVALUATION_PINS.json')]:
        for path,h in read(base/name).items():assert sha(base/path)==h,path;pins+=1
    for r in index:assert sha(r['path'])==r['sha256'];pins+=1
    result=dict(status='PASS',rows=len(rows),reference_evidence=evidence_count,boundary_witness_frames=boundary_frames,
                projections_reconstructed=3*len(rows),fixed_extent_masks=7092,lowered_metric_rows=1800,hash_pins=pins,
                seconds=time.time()-beg,operator='Same Codex separate implementation, not another agent or sign expert',
                private_data_read=0,training=0,production_changes=0)
    (U/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
