"""Separate source/cosine, scalar distance and saved-aggregation verification."""
from pathlib import Path
import json,hashlib,math,statistics,time
import numpy as np
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_multiref_observation_compare_20261019';B=U.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    assert not (U/'VERIFICATION.json').exists();f=read(U/'FROZEN.json');assert read(U/'COMPARISON.json')['status']=='PASS'
    triples=[(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
    inputs=anglevalues=aggregates=pairvalues=stateframes=0;maxcos=0.;maxmean=0.;beg=time.time()
    for row in f['inputs']:
        rec=read(B/'ops/word100_body_boundary_20261017/data/records'/(row['clip_id']+'.json'));native=load(rec['views']['native']['observation']);bottom=rec['calibration']['box_source_pixels'][3]
        for view in ('legacy','masked','sensor'):
            assert sha(row['inputs'][view])==row['input_hashes'][view]
            src=load(row['inputs'][view]);feat=load(U/'features'/view/(row['clip_id']+'.npz'))
            assert sha(U/'features'/view/(row['clip_id']+'.npz'))==read(U/'inputs'/view/(row['clip_id']+'.json'))['sha256']
            ix=np.searchsorted(src['frame_id'],feat['frame_id']);assert np.array_equal(src['frame_id'][ix],feat['frame_id'])
            oldmeta=read(OLD/'inputs'/view/(row['clip_id']+'.json'));old=load(oldmeta['sample_path']);offset=0;oldpos={}
            # Fixed original feature layout: arms8, shapes80, orient4, handposL2/R2.
            for side,start in [('L',92),('R',94)]:assert np.array_equal(feat['position_'+side],old['features'][:,start:start+2],equal_nan=True)
            for side,base in [('L',33),('R',54)]:
                actual=feat['angles_'+side]
                for j,ids in enumerate(triples):
                    p=src['xy_px'][ix][:,np.array(ids)+base].astype(float);ok=src['usable'][ix][:,np.array(ids)+base].all(1)&np.isfinite(p).all((1,2))
                    a=p[:,0]-p[:,1];b=p[:,2]-p[:,1];na=np.sqrt((a*a).sum(1));nb=np.sqrt((b*b).sum(1));ok&=(na>=1e-6)&(nb>=1e-6)
                    assert np.array_equal(ok,np.isfinite(actual[:,j]))
                    if ok.any():
                        cosine=(a[ok]*b[ok]).sum(1)/(na[ok]*nb[ok]);err=np.abs(np.cos(np.radians(actual[ok,j]))-np.clip(cosine,-1,1));maxcos=max(maxcos,float(err.max()));assert err.max()<1e-12
                    anglevalues+=int(ok.sum())
                # Offline lowered flags require visible complete source hand below boundary.
                st=feat['full_'+side+'_offline_full_view_state'];ids=np.flatnonzero(st==3)
                for t in ids:
                    lo=np.searchsorted(src['timestamp_us'],src['timestamp_us'][t]-300000);pp=native['xy_px'][lo:t+1,base:base+21];uu=native['usable'][lo:t+1,base:base+21]
                    assert uu.all() and np.isfinite(pp).all() and (pp[:,:,1]>=bottom).all() and t-lo+1>=8
                local=feat['full_'+side+'_any_hand_points'];want=src['usable'][:,base:base+21].any(1);assert np.array_equal(local,want)
                gap=feat['full_'+side+'_internal_gap'];assert not (gap&local).any()
                valididx=np.flatnonzero(local)
                if len(valididx):assert not gap[:valididx[0]].any() and not gap[valididx[-1]+1:].any()
                else:assert not gap.any()
                stateframes+=len(st)
            inputs+=1
    for view in ('legacy','masked','sensor'):
        for path in sorted((U/'words'/view).glob('*.json')):
            w=read(path);old=read(OLD/'words'/view/path.name);omap={(p['query_clip'],p['reference_clip']):p for p in old['pairs']}
            for q in w['queries']:
                pairs=[p for p in w['pairs'] if p['query_clip']==q['clip_id']]
                # TRAIN self-person exclusion inherited unchanged; no DEV used as reference.
                assert all(int(p['reference_clip'].split('REAL')[1][:2]) in range(1,10) and int(p['reference_clip'].split('REAL')[1][:2])!=q['signer_id'] for p in pairs)
                for side,result in q['sides'].items():
                    vals=[]
                    for j in range(15):
                        vv=[p['sides'][side]['angles']['values'][j] for p in pairs if p['sides'][side]['angles']['values'][j] is not None]
                        expected=statistics.median(vv) if len(vv)>=5 else None;actual=result['angles'][j]
                        assert (expected is None)==(actual is None)
                        if expected is not None:assert abs(expected-actual)<1e-10
                        assert result['valid_reference_counts'][j]==len(vv);vals.append(expected);aggregates+=1
                    if all(x is not None for x in vals):assert abs(statistics.mean(vals)-result['whole_hand'])<1e-10
                    else:assert result['whole_hand'] is None
                    for p in pairs:
                        a=p['sides'][side]['angles'];aligned=omap[p['query_clip'],p['reference_clip']]['result']['status']=='OK'
                        for j,v in enumerate(a['values']):
                            assert (v is not None)==(aligned and a['support'][j]>=.5 and min(a['query_frames'][j],a['reference_frames'][j])>=8)
            # Fixed first pair per word/view numeric verifier independent of implementation.
            p=w['pairs'][0];op=omap[p['query_clip'],p['reference_clip']];q=load(U/'features'/view/(p['query_clip']+'.npz'));r=load(U/'features'/view/(p['reference_clip']+'.npz'));dtw=op['diagnostics']['path']
            for side,a in p['sides'].items():
                for j,val in enumerate(a['angles']['values']):
                    good=[(i,k) for i,k in dtw if math.isfinite(q['angles_'+side][i,j]) and math.isfinite(r['angles_'+side][k,j])]
                    assert abs(len(good)/len(dtw)-a['angles']['support'][j])<1e-12
                    if val is not None:
                        avg=statistics.mean(abs(float(q['angles_'+side][i,j])-float(r['angles_'+side][k,j])) for i,k in good);maxmean=max(maxmean,abs(avg-val));assert abs(avg-val)<1e-10;pairvalues+=1
                valid=[(i,k) for i,k in dtw if q['position_valid_'+side][i] and r['position_valid_'+side][k]]
                if valid:
                    i0,k0=valid[0];cq=q['position_'+side][i0].astype(float);cr=r['position_'+side][k0].astype(float)
                    avg=statistics.mean(math.hypot(*(q['position_'+side][i].astype(float)-cq-r['position_'+side][k].astype(float)+cr))*100 for i,k in valid)
                    assert abs(avg-a['trajectory']['paired_start'])<1e-8
    assert all(sha(p)==h for p,h in f['protected'].items()) and sha(U/'candidate.py')==f['candidate_sha256'] and sha(U/'run.py')==f['runner_sha256']
    out={'status':'PASS','inputs':inputs,'source_angle_values':anglevalues,'source_angle_max_cosine_error':maxcos,'state_hand_frames':stateframes,'aggregate_angles':aggregates,'fixed_pairs_checked':300,'fixed_pair_angle_values':pairvalues,'max_pair_angle_error_degrees':maxmean,'position_preserved':True,'protected_unchanged':True,'same_operator_separate_verifier':True,'seconds':time.time()-beg}
    (U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(out))
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        p=U/'FIRST_VERIFY_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
