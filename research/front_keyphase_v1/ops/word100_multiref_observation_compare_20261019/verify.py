"""Separate saved-output arithmetic, source feature and data-split verification.

No execution of the comparison/scoring implementation, no model or private data.
Same Codex author; not a second-agent or language-expert review.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,math,statistics,traceback
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
GROUPS=[('arm_L',4),('arm_R',4),('shape_L',40),('shape_R',40),('orient_L',2),('orient_R',2),('handpos_L',2),('handpos_R',2),('facerel_L',2),('facerel_R',2),('bimanual',2),('motion_L',3),('motion_R',3),('bend_L',19),('bend_R',19)]
OFF={};i=0
for g,w in GROUPS:OFF[g]=slice(i,i+w);i+=w
GN=[g for g,w in GROUPS]
readsample=lambda p:dict(np.load(p,allow_pickle=False))
maxerr=0.;checks=0
def equal(a,b,tol=2e-4):
    global maxerr,checks
    if a is None or b is None:assert a is b,(a,b)
    else:
        e=abs(float(a)-float(b));assert e<=tol,(a,b,e);maxerr=max(maxerr,e)
    checks+=1
def valid(seq,name):
    v=seq['features'][:,OFF[name]]
    return v,seq['present'][:,GN.index(name)]&np.isfinite(v).all(axis=1)
def median(v):return statistics.median(v) if v else None
def verify():
    assert not (U/'INDEPENDENT.json').exists()
    f=read(U/'FROZEN.json');inputs=0;feature_values=0;max_feature_error=0.;max_cosine_error=0.
    assert read(U/'COMPARE_RESULT.json')['status']=='COMPLETE'
    for p,h in f['protected'].items():assert sha(p)==h
    byid={x['clip_id']:x for x in f['inputs']}
    cache={};stats=Counter()
    for v in f['views']:
        for row in f['inputs']:
            info=read(U/'inputs'/v/(row['clip_id']+'.json'));inputs+=1
            assert row['inputs'][v]==info['source_path'] and sha(info['source_path'])==info['source_sha256']==row['input_hashes'][v]
            stats[v+'_'+info['status']]+=1
            if info['status']!='OK':continue
            assert sha(info['sample_path'])==info['sample_sha256']
            q=readsample(info['sample_path']);cache[(v,row['clip_id'])]=q
            with np.load(info['source_path'],allow_pickle=False) as z:
                xy=z['xy_px'];ok=z['usable'];ts=z['timestamp_us'];fid=z['frame_id']
            lookup={int(n):j for j,n in enumerate(fid)};pick=np.array([lookup[int(n)] for n in q['frame_id']]);assert np.array_equal(ts[pick],q['timestamp_us'])
            anyhand=ok[:,33:].any(axis=1);ix=np.flatnonzero(anyhand)
            assert info['span']==[int(ix[0]),int(ix[-1])]
            goodbody=ok[:,11]&ok[:,12];scale=float(np.median(np.linalg.norm(xy[goodbody,11]-xy[goodbody,12],axis=1)))
            equal(scale,info['shoulder_width_px'])
            origin=(xy[pick,11]+xy[pick,12])/2
            for side,offset in [('L',33),('R',54)]:
                p=xy[pick,offset:offset+21];k=ok[pick,offset:offset+21];body=goodbody[pick]
                hv,hm=valid(q,'handpos_'+side)
                assert np.array_equal(hm,body&k[:,0])
                expected=(p[:,0]-origin)/scale
                if hm.any():
                    e=float(np.max(np.abs(hv[hm]-expected[hm])));assert e<1e-5;max_feature_error=max(max_feature_error,e);feature_values+=int(hm.sum())*2
                # atan2 independently checks all15 inner bone angles (scorer uses acos).
                bv,bm=valid(q,'bend_'+side)
                if bm.any():
                    expected=[]
                    for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
                        for j in range(1,4):
                            a=p[bm,chain[j-1]].astype(float)-p[bm,chain[j]].astype(float);b=p[bm,chain[j+1]].astype(float)-p[bm,chain[j]].astype(float)
                            expected.append(np.arctan2(np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]),(a*b).sum(axis=1))/np.pi)
                    expected=np.array(expected).T
                    e=float(np.max(np.abs(bv[bm,:15]-expected)))
                    # acos is ill-conditioned at straight/folded endpoints. Compare
                    # cosine first with a float32 operation-roundoff allowance;
                    # bound angle discrepancy using that same cosine allowance.
                    cos_tol=16*np.finfo(np.float32).eps
                    ce=float(np.max(np.abs(np.cos(bv[bm,:15].astype(float)*np.pi)-np.cos(expected*np.pi))))
                    assert ce<=cos_tol and e<=np.arccos(1-float(cos_tol))/np.pi,(row['clip_id'],v,ce,e)
                    max_cosine_error=max(max_cosine_error,ce)
                    max_feature_error=max(max_feature_error,e);feature_values+=int(bm.sum())*15
    pair_count=Counter();numeric_pairs=0;paths=0;reused=0
    oldrole=read(B/'ops/word100_role_dtw_20261017/ROLE_BANK.json')['words']
    olddev={x['clip_id']:x for x in read(B/'ops/word100_role_dtw_20261017/DEVELOPMENT.json')}
    for v in f['views']:
        files=sorted((U/'words'/v).glob('*.json'));assert len(files)==100
        for wp in files:
            d=read(wp);required=d['alignment_sides'];pairs=d['pairs'];seen=set();perquery={}
            for p in pairs:
                k=(p['query_clip'],p['reference_clip']);assert k not in seen;seen.add(k)
                qr,rr=byid[k[0]],byid[k[1]]
                assert qr['word_code']==rr['word_code']==d['word_code'] and rr['role']=='train_candidate' and qr['signer_id']!=rr['signer_id']
                pair_count[v+'_'+qr['role']]+=1;perquery.setdefault(k[0],[]).append(p)
                reused+=int(p['result_source']=='reused_frozen')
                path=p['diagnostics'].get('path',[])
                if path:
                    a=np.array(path);assert (a[0]==[0,0]).all();q=cache[(v,k[0])];r=cache[(v,k[1])]
                    assert (a[-1]==[len(q['features'])-1,len(r['features'])-1]).all()
                    assert all(tuple(x) in ((0,1),(1,0),(1,1)) for x in np.diff(a,axis=0));paths+=1
            for q in d['queries']:
                pp=perquery[q['clip_id']]
                for m in q['metrics']:
                    both=[]
                    for s in m['sides']:
                        values=[]
                        if s['side'] in required:
                            for p in pp:
                                if p['result']['status']!='OK':continue
                                mm=next(x for x in p['result']['metrics'] if x['key']==m['key']);ss=next(x for x in mm['sides'] if x['side']==s['side'])
                                if ss['value'] is not None and ss['support_fraction']>=.5 and ss.get('query_distinct_frames',0)>=8 and ss.get('reference_distinct_frames',0)>=8:values.append(ss['value'])
                        assert len(values)==s['valid_references']
                        value=median(values) if len(values)>=5 else None;equal(s['value'],value,1e-8)
                        if s['side'] in required:both.append(value)
                    equal(m['value'],sum(both)/len(both) if all(x is not None for x in both) else None,1e-8)
                    if v=='legacy' and q['clip_id'] in olddev:
                        om=next(x for x in olddev[q['clip_id']]['metrics'] if x['key']==m['key']);equal(m['value'],om['value'],1e-7)
            for key in ('handshape','location','movement'):
                base=d['baseline'][key]
                for side in ('L','R'):
                    vals=[]
                    for q in d['queries']:
                        if q['role']!='train_candidate':continue
                        m=next(x for x in q['metrics'] if x['key']==key);val=next(x for x in m['sides'] if x['side']==side)['value']
                        if val is not None:vals.append(val)
                    bb=base['sides'][side];assert bb['n']==len(vals);equal(bb['median'],median(vals),1e-8)
                    if vals:equal(bb['min'],min(vals),1e-8);equal(bb['max'],max(vals),1e-8)
                    if v=='legacy' and d['word_code'] in oldrole:
                        ob=oldrole[d['word_code']]['baseline'][key]['sides'][side];assert bb==ob
            # Fixed first available pair in every word/view: separate scalar arithmetic.
            p=next((p for p in pairs if p['result']['status']=='OK'),None)
            if p:
                q=cache[(v,p['query_clip'])];r=cache[(v,p['reference_clip'])];path=p['diagnostics']['path'];numeric_pairs+=1
                for key,group in [('handshape','bend'),('location','handpos'),('movement','handpos')]:
                    for side in required:
                        a,av=valid(q,group+'_'+side);b,bv=valid(r,group+'_'+side);indices=[(i,j) for i,j in path if av[i] and bv[j]]
                        expected=None
                        if indices:
                            if key=='handshape':values=[sum(abs(float(a[i,n])-float(b[j,n])) for n in range(15))/15*180 for i,j in indices]
                            else:
                                qa=a[np.flatnonzero(av)[0]] if key=='movement' else np.zeros(2);rb=b[np.flatnonzero(bv)[0]] if key=='movement' else np.zeros(2)
                                values=[math.hypot((float(a[i,0])-float(qa[0]))-(float(b[j,0])-float(rb[0])),(float(a[i,1])-float(qa[1]))-(float(b[j,1])-float(rb[1])))*100 for i,j in indices]
                            expected=sum(values)/len(values)
                        m=next(m for m in p['result']['metrics'] if m['key']==key);s=next(s for s in m['sides'] if s['side']==side);equal(s['value'],expected)
        assert pair_count[v+'_train_candidate']==7184 and pair_count[v+'_development_only']==2697
    out={'status':'PASS','method':'Codex separate source atan2/coordinate and stdlib median/scalar distance checks; no second agent or expert','inputs':inputs,'feature_values_checked':feature_values,'max_feature_error_normalized':max_feature_error,'max_angle_cosine_error':max_cosine_error,'cosine_roundoff_limit':float(16*np.finfo(np.float32).eps),'saved_scalar_checks':checks,'max_metric_error':maxerr,'numeric_pairs':numeric_pairs,'monotone_paths':paths,'pair_counts':dict(pair_count),'reused_metric_pairs':reused,'input_status':dict(stats),'legacy12_bank_and_DEV36_parity':True,'protected_unchanged':True,'training':0}
    (U/'INDEPENDENT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
    try:verify()
    except BaseException:
        p=U/'FIRST_VERIFY_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
