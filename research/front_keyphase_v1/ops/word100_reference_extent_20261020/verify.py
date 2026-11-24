"""Independent vote/support/time checks. No imports from candidate or audit."""
from pathlib import Path
import json,hashlib,bisect,math,traceback
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def load(rec):
    assert sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def weights(ts):
    t=[(int(x)-int(ts[0]))/1e6 for x in ts]
    return np.array([(t[-1] if i==len(t)-1 else (t[i]+t[i+1])/2)-(0 if i==0 else (t[i-1]+t[i])/2) for i in range(len(t))])
def scalar_lift(ts,raw):
    result=[];a=list(ts)
    for t in raw:
        j=min(len(a)-1,bisect.bisect_left(a,t));i=max(0,j-1);result.append(i if abs(a[i]-t)<=abs(a[j]-t) else j)
    return np.array(result,int)
def verify_projection(p,q,refs,required):
    n=len(q['pose_timestamp_us']);core=np.zeros(n,int);out=np.zeros(n,int);rest={s:np.zeros(n,int) for s in ('L','R')}
    expected={s:{k:np.zeros(n,int) for k in ('handshape','location','movement')} for s in ('L','R')}
    groups=[0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]
    for pair in p['pairs']:
        r=refs[pair['reference_clip']];path=pair['path'];good=[];supports=[];costs=[]
        assert path[0]==[0,0] and path[-1]==[len(q['pose_timestamp_us'])-1,len(r['pose_timestamp_us'])-1]
        assert all((b[0]-a[0],b[1]-a[1]) in ((1,0),(0,1),(1,1)) for a,b in zip(path,path[1:]))
        for qi,ri in path:
            both=[j for j in groups if q['pose_valid'][qi,j] and r['pose_valid'][ri,j]];support=len(both)/len(groups);supports.append(support);good.append(support>=.5)
            terms=[]
            for j in groups:
                if j in both:
                    aa=q['pose_xy'][qi,j];bb=r['pose_xy'][ri,j];terms.append(math.hypot(float(aa[0]-bb[0]),float(aa[1]-bb[1])))
                else:terms.append(1.5)
            costs.append(sum(terms)/len(groups))
        support=sum(supports)/len(supports)
        nq=len({path[i][0] for i,g in enumerate(good) if g});nr=len({path[i][1] for i,g in enumerate(good) if g})
        status='AVAILABLE' if support>=.35 and min(nq,nr)>=8 else 'UNAVAILABLE'
        assert pair['status']==status and good==pair['good']
        assert abs(support-pair['support'])<1e-7 and abs(sum(costs)/len(costs)-pair['cost'])<1e-6
        if status!='AVAILABLE':continue
        picks=r['pose_picked_rows']
        for qi in range(n):
            rawref=[int(picks[ri]) for (i,ri),g in zip(path,good) if i==qi and g]
            if not rawref:continue
            cc=[bool(r['reference_core'][j]) for j in rawref]
            if all(cc):core[qi]+=1
            elif not any(cc):out[qi]+=1;continue
            else:continue
            for s in ('L','R'):
                if all(r[s+'_source_state'][j]==3 for j in rawref):rest[s][qi]+=1
                for k in expected[s]:
                    if all(r[s+'_source_state'][j]==2 and r[s+'_'+k][j] for j in rawref):expected[s][k][qi]+=1
    lift=scalar_lift(q['pose_timestamp_us'],q['pose_full_timestamp_us'])
    assert np.array_equal((core>=5)[lift],p['core']) and np.array_equal((out>=5)[lift],p['outside'])
    assert np.array_equal(core[lift],p['core_votes']) and np.array_equal(out[lift],p['outside_votes'])
    for s in ('L','R'):
        assert np.array_equal((rest[s]>=5)[lift]&(core>=5)[lift],p['rest'][s])
        for k in expected[s]:
            answer=(expected[s][k]>=5)[lift]&(core>=5)[lift]
            if s not in required:answer[:]=False
            assert np.array_equal(answer,p['expected'][s][k])
    assert np.array_equal(~(np.array(p['core'])|np.array(p['outside'])),p['unknown'])
    return len(p['pairs'])
def run():
    frozen=read(U/'FROZEN.json');index=read(U/'INDEX.json');idx={(r['clip_id'],r['view']):r for r in index}
    old=read(OPS/'word100_multiref_observation_compare_20261019/FROZEN.json');oldidx={e['clip_id']:e for e in old['inputs']}
    assert len(index)==2398 and sum(r['signer_id']<=9 for r in index)==1798
    base=read(U/'BASELINE.json');stress=read(U/'STRESS.json');summary=read(U/'SUMMARY.json');count=0;cache={};projcache={}
    for b in base:
        cid,view=b['clip_id'],b['view'];q=load(idx[cid,view]);assert idx[cid,view]['signer_id'] in (11,12,13)
        refs={r['clip_id']:cache.setdefault((r['clip_id'],view),load(r)) for r in index if r['word']==b['word'] and r['view']==view and r['signer_id']<=9}
        assert len({idx[r,view]['signer_id'] for r in refs})==len(refs)
        p=read(U/'projections'/view/(cid+'.json'));count+=verify_projection(p,q,refs,b['required_sides']);projcache[cid,view]=(p,q)
        ts=q['pose_full_timestamp_us'];w=weights(ts);truth=np.zeros(len(ts),bool)
        for a,z in oldidx[cid]['intervals']:truth|=(ts>=a*1e6)&(ts<=z*1e6)
        inter=float(w@(np.array(p['core'])&truth));loc=b['localization'];assert abs(inter-loc['intersection_s'])<1e-7
        assert abs(float(w@truth)-loc['annotation_s'])<1e-7
        for m in b['metrics']:
            s,k=m['side'],m['key'];e=np.array(p['expected'][s][k]);missing=e&~q[s+'_'+k]
            assert abs(float(w@e)-m['expected_s'])<1e-7 and abs(float(w@missing)-m['missing_expected_s'])<1e-7
            assert m['wrong_action'] is None and m['whole_action_pass'] is None
    for r in stress:
        assert r['extent_unchanged'] and r['before']['expected_s']==r['after']['expected_s']
        assert r['after']['missing_expected_s']+1e-8>=r['before']['missing_expected_s']
        assert abs(r['after']['missing_expected_s']-r['before']['missing_expected_s']-r['removed_expected_s'])<1e-7
        assert r['after']['wrong_action'] is None
    assert len(base)==600 and len(stress)==591*4*3 and count==5394
    pins=0
    for n in ('INPUT_PINS.json','EVALUATION_PINS.json'):
        for p,h in read(U/n).items():assert sha(p)==h,p;pins+=1
    for k in ('protected','previous','code'):
        for p,h in frozen[k].items():assert sha(p)==h,p;pins+=1
    assert summary['baseline_status']==dict(Counter(m['status'] for b in base for m in b['metrics']))
    write('VERIFICATION.json',dict(status='PASS',query_views=600,independent_arm_paths_and_votes=count,monotone_fixed_extent_masks=len(stress),
        reference_people_separate=True,dev_annotations_evaluation_only=True,input_protection_hashes=pins,
        operator='Codex separate scalar/path/vote implementation, not independent agent',production_changes=0,new_training=0,private_data_used=0))
    print('VERIFIED',count,pins)
if __name__=='__main__':
    try:run()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
