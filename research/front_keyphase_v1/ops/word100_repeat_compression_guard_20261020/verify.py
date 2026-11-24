"""Independent new guard reconstruction. Prior verified vote/path helpers reused explicitly."""
from pathlib import Path
from collections import defaultdict,Counter
import json,gzip,hashlib,time,traceback,importlib.util,itertools
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS/'word100_repeat_boundary_consensus_20261020';P=OPS/'word100_repeat_alignment_20261020'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
V=module('old_path_verifier',P/'verify.py');BV=module('old_consensus_verifier',B/'verify.py')
SIDES=('L','R');AXES=('handshape','location','movement')
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,np.floating)):assert b is not None and abs(a-b)<1e-9,(a,b)
    else:assert a==b,(a,b)
def point_ids(required):return [0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]
def reference_parts(r,required):
    points=point_ids(required);edge=np.zeros(len(r['ts'])-1)
    for j in range(len(edge)):
        joint=r['valid'][j,points]&r['valid'][j+1,points]
        if r['reference_core'][j] and r['reference_core'][j+1] and joint.mean()>=.5:
            delta=np.nan_to_num(r['xy'][j+1,points]-r['xy'][j,points])
            distance=np.sqrt((delta*delta).sum(1))
            edge[j]=np.where(joint,distance,0).sum()/max(joint.sum(),1)
    cum=np.concatenate([[0],np.cumsum(edge)])
    if cum[-1]<=1e-12:return []
    cuts=[int(np.searchsorted(cum,v,side='left')) for v in np.linspace(0,cum[-1],11)]
    cuts[0]=int(np.flatnonzero(edge>0)[0]);out=[]
    for part in range(10):
        a,b=cuts[part:part+2]
        if a<b and cum[b]>cum[a]:
            out.append(dict(part=part,start=a,end=b,travel=float(cum[b]-cum[a]),
                moving_edges=np.flatnonzero(edge[a:b]>0).astype(int).tolist(),
                source_fraction=(b-a)/(len(r['ts'])-1)))
            out[-1]['moving_edges']=[x+a for x in out[-1]['moving_edges']]
    return out
def returns(q,r,bank,required):
    n=len(q['ts']);m=len(r['ts']);points=point_ids(required);core=np.flatnonzero(r['reference_core']);ok=np.zeros(n,bool)
    if len(core) and core[0]>0 and core[-1]<m-1:
        joint=q['valid'][:,None,points]&r['valid'][None,:,points]
        delta=np.nan_to_num(q['xy'][:,None,points]-r['xy'][None,:,points])
        costs=np.where(joint,np.sqrt((delta*delta).sum(3)),1.5).mean(2)
        regions=[np.arange(core[0]),np.arange(core[-1]+1,m),core]
        for i in range(n):
            nearest=[rg[np.argmin(costs[i,rg])] for rg in regions]
            ok[i]=(joint[i,nearest[0]].mean()>=.5 and joint[i,nearest[1]].mean()>=.5
                   and costs[i,nearest[0]]<costs[i,nearest[2]] and costs[i,nearest[1]]<costs[i,nearest[2]])
    out={}
    for k in (2,3):
        e=bank[k-1];path=np.array(e['path'],int);good=np.array(e['good'],bool);bands=[]
        for cycle in range(k-1):
            before=[];after=[]
            for (qi,rj),supported in zip(path,good):
                if supported and r['reference_core'][rj%m]:
                    if rj//m==cycle:before.append(qi)
                    elif rj//m==cycle+1:after.append(qi)
            bands.append([int(i) for i in np.flatnonzero(ok) if e['available'] and before and after and max(before)<i<min(after)])
        out[str(k)]=dict(available=e['available'],boundaries=bands)
    return out
def jointly(witnesses,people,k,n):
    eligible=[i for i in range(len(people)) if witnesses[i][str(k)]['available']]
    for subset in itertools.combinations(eligible,5):
        bands=[]
        for boundary in range(k-1):
            bands.append([j for j in range(n) if all(j in witnesses[i][str(k)]['boundaries'][boundary] for i in subset)])
        if all(bands) and all(a[-1]<b[0] for a,b in zip(bands,bands[1:])):
            return dict(supported=True,people=[people[i] for i in subset],boundaries=bands)
    return dict(supported=False,people=[],boundaries=[])
def trace(r,e,parts,n):
    mask=np.zeros(n,bool);events=[];ratios=[];unobserved=0;eligible=0
    if not e['available']:return dict(available=False,mask=mask,events=[],eligible=0,unobserved=0,min_positive_ratio=None)
    m=len(r['ts']);p=np.array(e['path'],int);good=np.array(e['good'],bool)
    grouped=[defaultdict(set) for _ in range(e['cycles'])]
    for (i,j),g in zip(p,good):
        if g and r['reference_core'][j%m]:grouped[j//m][j%m].add(int(i))
    for cycle in range(e['cycles']):
        for part in parts:
            rs={j for j in grouped[cycle] if part['start']<=j<=part['end']}
            qs=sorted(set().union(*(grouped[cycle][j] for j in rs))) if rs else []
            eligible+=1
            if not qs or len(rs)<2 or not any(x in rs and x+1 in rs for x in part['moving_edges']):
                unobserved+=1;continue
            ratio=((qs[-1]-qs[0])/max(1,n-1))/part['source_fraction']
            if ratio>0:ratios.append(ratio)
            if len(qs)==1:
                mask[qs[0]]=True;events.append(dict(cycle=cycle,part=part['part'],reference_start=part['start'],reference_end=part['end'],
                    supported_reference_samples=len(rs),query_frame=qs[0],query_samples=1,source_fraction=part['source_fraction'],relative_duration=0.0))
    return dict(available=True,mask=mask,events=events,eligible=eligible,unobserved=unobserved,min_positive_ratio=min(ratios) if ratios else None)
def main():
    beg=time.time();rows=read(U/'ROWS.json');index=read(P/'INDEX.json');lookup={(r['clip_id'],r['view']):r for r in index};groups=defaultdict(list)
    for r in index:
        if r['signer_id']<=9:groups[r['word'],r['view']].append(r)
    cache={};parts_cache={}
    def load(rec):
        if rec['path'] not in cache:
            assert sha(rec['path'])==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as f:cache[rec['path']]={k:f[k] for k in f.files}
        return cache[rec['path']]
    raw={r['clip_id']:r for r in read(OPS/'word100_multiref_observation_compare_20261019/FROZEN.json')['inputs']}
    original={};ref_count=0;newpaths=0;optimal=0;fresh_index=0
    for ix,row in enumerate(rows):
        cid,view=row['clip_id'],row['view'];rec=lookup[cid,view];assert rec['signer_id'] in (11,12,13);d=load(rec);n0=len(d['ts'])
        rr=groups[rec['word'],view];people=[r['signer_id'] for r in rr];assert len(set(people))==len(people) and len(people)>=8
        refs=[load(r) for r in rr];fresh=row['fresh_control'];scenario=row['scenario']
        with gzip.open((U if fresh else P)/row['source_path'],'rt',encoding='utf-8') as h:source=json.load(h)
        if fresh:
            bank=source['bank'];baseline=source['baseline'];required=source['required'];mapping=source['mapping']
            if scenario=='pause':
                expected=np.r_[np.arange(n0//2),np.full(15,n0//2),np.arange(n0//2,n0)]
                assert mapping['left']==mapping['right']==expected.tolist();assert not any(mapping['fraction'])
            else:
                factor=.75 if scenario=='speed075' else 1.5;grid=np.linspace(0,n0-1,max(2,round((n0-1)*factor)+1))
                assert np.array_equal(np.floor(grid),mapping['left']) and np.array_equal(np.ceil(grid),mapping['right'])
                assert np.allclose(grid-np.floor(grid),mapping['fraction'])
            left=np.array(mapping['left']);right=np.array(mapping['right']);frac=np.array(mapping['fraction'])
            valid=d['valid'][left]&d['valid'][right];xy=d['xy'][left]*(1-frac[:,None,None])+d['xy'][right]*frac[:,None,None];xy[~valid]=np.nan
            q=dict(ts=np.arange(len(left))*np.median(np.diff(d['ts'])),xy=xy,valid=valid)
            flags={s:{a:d[s+'_'+a][left]&d[s+'_'+a][right] for a in AXES} for s in SIDES}
            for ri,(ref,entries) in enumerate(zip(refs,bank)):
                for e in entries:
                    check=fresh_index%100==0 and ri==0
                    V.check_pair(q,ref,e,required,check);newpaths+=1;optimal+=int(check)
            fresh_index+=1
        else:
            bank=source['result']['bank'];required=source['required']
            with gzip.open(B/row['baseline_path'],'rt',encoding='utf-8') as h:baseline=json.load(h)
            ids=np.arange(n0);ts=d['ts']
            if scenario.startswith('repeat'):
                k=int(scenario[-1]);ids=np.tile(ids,k);period=ts[-1]-ts[0]+int(np.median(np.diff(ts)))
                ts=np.concatenate([ts-ts[0]+j*period for j in range(k)])
            elif scenario.startswith('delete'):
                ids=np.array(row['meta']['keep_rows']);ts=np.arange(len(ids))*np.median(np.diff(d['ts']))
            q=dict(ts=ts,xy=d['xy'][ids],valid=d['valid'][ids])
            flags={s:{a:d[s+'_'+a][ids] for a in AXES} for s in SIDES}
        with gzip.open(U/row['evidence_path'],'rt',encoding='utf-8') as h:z=json.load(h)
        n=len(q['ts']);w=V.wts(q['ts'])
        witnesses=[returns(q,r,b,required) for r,b in zip(refs,bank)];equal(witnesses,z['return_witnesses']);ref_count+=len(refs)
        alternatives={str(k):jointly(witnesses,people,k,n) for k in (2,3)};equal(alternatives,z['alternative_evidence'])
        pp={k:V.independent_votes(q,refs,bank,k,required) for k in (1,2,3)}
        if fresh:
            ev=[]
            for person,entries,wit in zip(people,bank,witnesses):
                available=[e for e in entries if e['available']];best=min([e['score'] for e in available],default=None)
                winners=[e['cycles'] for e in available if e['score']==best];pref=winners[0] if len(winners)==1 else None
                ev.append(dict(person=person,preferred=pref,boundaries={str(i+1):v for i,v in enumerate(wit[str(pref)]['boundaries'])} if pref in (2,3) else {}))
            equal(ev,baseline['reference_evidence'])
            dec=BV.decision(ev,n);equal(dec,baseline['decision']);equal(BV.merge(pp,dec,n),baseline['projection'])
        before=baseline['decision']['count'];viable=[k for k in (1,2,3) if pp[k]['available']>=5];inspected=[before] if before is not None else viable
        traces=[];per=[]
        for rec_ref,r,entries,person in zip(rr,refs,bank,people):
            key=(rec_ref['path'],tuple(required))
            if key not in parts_cache:parts_cache[key]=reference_parts(r,required)
            parts=parts_cache[key];tt={str(k):trace(r,entries[k-1],parts,n) for k in inspected}
            mask=np.array([bool(inspected) and all(tt[str(k)]['mask'][j] for k in inspected) for j in range(n)])
            traces.append(dict(person=person,parts=parts,counts=tt,common_mask=mask));per.append(mask)
        equal(traces,z['compression_traces'])
        votes=np.array([sum(m[j] for m in per) for j in range(n)]);comp=votes>=5;equal(votes,z['compression_votes']);equal(comp,z['compression_mask'])
        reasons=[]
        if before==1 and any(a['supported'] for a in alternatives.values()):reasons.append('ALTERNATIVE_REPEAT_WITNESS')
        if comp.any():reasons.append('MOTION_COLLAPSED_TO_ONE_SAMPLE')
        count=None if reasons else before;assert count==row['new_count']==z['count'];assert reasons==row['guard_reasons']==z['guard_reasons']
        assert int(comp.sum())==row['compression_frames'] and abs(float(w@comp)-row['compression_s'])<1e-7
        p=BV.merge(pp,dict(count=count),n);p['core']&=~comp;p['unknown']=~(p['core']|p['outside'])
        for s in SIDES:
            p['rest'][s]&=~comp
            for a in AXES:p['expected'][s][a]&=~comp
        equal(p,z['projection']);assert z['semantic_error'] is None and z['complete_performance'] is None
        assert np.array_equal(p['possible_core'],baseline['projection']['possible_core'])
        for s in SIDES:
            for a in AXES:assert np.array_equal(p['possible_expected'][s][a],baseline['projection']['possible_expected'][s][a])
        truth=np.zeros(n0,bool)
        for a,b in raw[cid]['intervals']:truth|=(d['ts']>=a*1e6)&(d['ts']<=b*1e6)
        truth=(truth[left]&truth[right]) if fresh else truth[ids]
        for area in ('core','possible_core'):
            for name,mask in [('intersection_s',p[area]&truth),('predicted_s',p[area]),('annotation_s',truth)]:
                assert abs(float(w@mask)-row['new_loc'][area][name])<1e-7
        for met in row['new_metrics']:
            s,a=met['side'],met['key'];observed=flags[s][a]
            for name,mask in [('expected_s',p['expected'][s][a]),('possible_s',p['possible_expected'][s][a]),
                             ('missing_s',p['expected'][s][a]&~observed),('uncertain_missing_s',p['possible_expected'][s][a]&~p['expected'][s][a]&~observed)]:
                assert abs(float(w@mask)-met[name])<1e-7
            assert met['wrong_action'] is None and met['whole_action_pass'] is None
        if view=='sensor' and scenario=='original':original[cid]=(d,p,w)
        if (ix+1)%150==0:print('verified',ix+1,'seconds',round(time.time()-beg,1),flush=True)
    windows={(r['clip_id'],r['side'],r['condition']):r for r in read(OPS/'word100_missing_phase_validation_20261020/ROWS.json')}
    for row in read(U/'STRESS.json'):
        d,p,w=original[row['clip_id']];s,a=row['side'],row['key'];wr=windows[row['clip_id'],s,row['condition']]
        mask=np.ones(len(w),bool) if row['condition']=='whole' else (d['ts']/1e6>=wr['window']['start_s'])&(d['ts']/1e6<=wr['window']['end_s'])
        removed=mask&d[s+'_'+a];assert int(removed.sum())==row['removed_frames']
        assert abs(float(w@(removed&p['expected'][s][a]))-row['new_strong_s'])<1e-7
        assert abs(float(w@(removed&p['possible_expected'][s][a]))-row['new_possible_s'])<1e-7
        assert abs(row['new_possible_s']-row['old_possible_s'])<1e-7
    full={}
    for row in read(U/'LOWERING.json'):
        cid=row['clip_id'];d,p,w=original[cid];s,a=row['side'],row['key']
        if cid not in full:
            with np.load(OPS/'word100_metric_evidence_20261020/features/sensor'/(cid+'.npz'),allow_pickle=False) as z:
                full[cid]={ss:z['full_'+ss+'_offline_full_view_state']==3 for ss in SIDES}
        rest=full[cid][s]&~d[s+'_'+a]
        assert abs(float(w@(rest&p['expected'][s][a]))-row['new_strong_s'])<1e-7
        assert abs(float(w@(rest&p['possible_expected'][s][a]))-row['new_possible_s'])<1e-7
    summary=read(U/'SUMMARY.json')
    for name,g in summary['groups'].items():
        view,scenario=name.split('_');target=int(scenario[-1]) if scenario.startswith('repeat') else 1
        rr=[r for r in rows if r['view']==view and r['scenario']==scenario];assert len(rr)==g['n']==300
        for prefix in ('old','new'):
            assert sum(r[prefix+'_count']==target for r in rr)==g[prefix+'_correct']
            assert sum(r[prefix+'_count'] is None for r in rr)==g[prefix+'_abstain']
            assert g[prefix+'_correct']+g[prefix+'_wrong']+g[prefix+'_abstain']==300
        assert sum(r['old_count']==target and r['new_count'] is None for r in rr)==g['lost_correct']
    f=read(U/'FROZEN.json');pins=0
    for typ in ('protected','previous','code'):
        for path,h in f[typ].items():assert sha(path)==h,path;pins+=1
    for base,name in [(P,'PROJECTION_PINS.json'),(B,'EVIDENCE_PINS.json'),(U,'OUTPUT_PINS.json'),(U,'EVALUATION_PINS.json')]:
        for path,h in read(base/name).items():assert sha(base/path)==h,path;pins+=1
    for r in index:assert sha(r['path'])==r['sha256'];pins+=1
    result=dict(status='PASS',rows=len(rows),reference_evidence=ref_count,new_control_paths=newpaths,
        new_path_optimality_samples=optimal,fixed_extent_masks=7092,lowering_rows=1800,hash_pins=pins,seconds=time.time()-beg,
        operator='Same Codex separate new-guard implementation. Prior already-crosschecked vote/path helper reused; not an independent person or sign expert.',
        training=0,private_data_read=0,production_changes=0)
    (U/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
