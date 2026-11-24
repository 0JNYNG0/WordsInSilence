"""Separate saved-result audit. No import of candidate or replay of tracking."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2
U=Path(__file__).resolve().parent;P=U.parent.parent/'word100_image_tracking_pair_20261020'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def stats(a):
    return dict(n=len(a),median=float(np.median(a)),mean=float(np.mean(a)),p95=float(np.quantile(a,.95)),
        max=float(max(a)),above_0_10=sum(float(v)>.1 for v in a)) if a else dict(n=0)
def main():
    c=read(U/'FROZEN.json');p=c['params'];counts=Counter();states=Counter();reasons=Counter()
    natural=[];synthetic=[];rechecks=[];lookup={};latency=[];flow_n=0;fit_n=0;obsproxy=[]
    for path,h in {**c['pins'],**c['protected']}.items():assert sha(path)==h,path
    for ix in read(U/'INDEX.json'):
        assert sha(ix['path'])==ix['sha256']
    for row in c['rows']:
        cid=row['clip_id'];x=read(U/'records'/f'{cid}.json');old=read(P/'records'/f'{cid}.json')
        lookup[cid]=x;counts['clips']+=1;counts['frames']+=len(x['frames']);latency+=x['per_frame_ms']
        with np.load(row['views']['sensor']['raw'],allow_pickle=False) as z:raw_hands=z['hand_xy_px'].copy()
        assert x['blocks']==old['blocks'] and len(x['frames'])==len(old['frames'])
        seen=[False,False]
        for e,b in zip(x['frames'],old['frames']):
            assert e['direct']==b['direct'] and e['baseline_sides']==b['baseline_sides']
            assert e['index']==b['index'] and e['time']==b['time'] and e['frame_id']==b['frame_id']
            fac=e['factor'];hidden={b['baseline_sides'][i] for i in b['suppressed']}
            for mode in ['natural','test']:
                o=e[mode];assert o['proposal']==b['baseline_sides'];assert o['observed_finger_points_added']==0
                for s in (0,1):
                    measured=None if b['direct'][s] is None or (mode=='test' and s in hidden) else (np.array(b['direct'][s])*fac).tolist()
                    if measured is None:assert o['observed_center'][s] is None
                    else:
                        assert np.allclose(o['observed_center'][s],measured,atol=1e-10)
                        assert np.allclose(o['center_used'][s],measured,atol=1e-10) and o['source'][s]=='observed'
                        assert o['recheck'][s]['current_observation_priority']
                    tr=o['traces'].get(str(s))
                    if tr and 'flow' in tr:
                        assert 0<tr['age']<=p['max_age_s']+1e-8 and tr['anchor_time']<e['time']
                        fl=tr['flow'];flow_n+=1
                        if 'good' in fl:
                            a=np.array(fl['previous'],float);q=np.array(fl['forward'],float);back=np.array(fl['backward'],float);err=np.array(fl['errors'],float)
                            w=p['image_width'];h=round(row['box'][3]*fac)
                            good=(np.array(fl['forward_status'])==1)&(np.array(fl['backward_status'])==1)&np.isfinite(err)&(err<=p['lk_error'])
                            good &= np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
                            good &= np.linalg.norm(back-a,axis=1)<=p['fb_px']
                            assert good.tolist()==fl['good']
                            passed=good.sum()>=p['min_points'] and good.mean()>=p['keep_fraction']
                            assert (fl['reason']=='region_points_tracked')==passed
                    fit=tr.get('center_fit') if tr else None
                    if fit and 'matrix' in fit:
                        fit_n+=1
                        a=np.array(fit['previous']);q=np.array(fit['current']);m=np.array(fit['matrix']);keep=np.array(fit['inliers'],bool)
                        if 'predicted_center' in fit:
                            predicted=np.array(fit['previous_center'])@m[:,:2].T+m[:,2]
                            assert np.allclose(predicted,fit['predicted_center'],atol=1e-8)
                        if 'scale' in fit:
                            assert abs(np.linalg.norm(m[:,0])-fit['scale'])<1e-9
                            assert abs(np.degrees(np.arctan2(m[1,0],m[0,0]))-fit['rotation_deg'])<1e-8
                            residual=np.linalg.norm(a[keep]@m[:,:2].T+m[:,2]-q[keep],axis=1)
                            assert abs(np.quantile(residual,.9)-fit['residual_p90'])<1e-8
                    retained=o.get('retained_region_points',{}).get(str(s),[])
                    if retained:
                        assert o['region'][s] is not None
                        idx=[i for i,a in enumerate(o['anchors']) if a==s and b['baseline_sides'][i]==s]
                        assert len(idx)==1
                        hull=cv2.convexHull((raw_hands[e['index'],idx[0],[0,5,9,13,17]]*fac).astype(np.float32))
                        for pt in retained:
                            assert cv2.pointPolygonTest(hull,tuple(map(float,pt)),False)>=0
                            assert any(np.allclose(pt,q,atol=1e-7) for q in o['region'][s]['points'])
                        if o['seed_reason'][s]=='current_palm_revalidated_existing_pixels':assert len(retained)>=p['min_points']
                    est=o['center_estimate'][s]
                    if est is not None:
                        assert tr['reason']=='region_and_center_estimate' and fit['reason']=='transform_candidate'
                        assert fit['residual_p90']<=p['residual_shoulder']*tr['shoulder']+1e-9
                        assert np.allclose(est,fit['predicted_center'])
                    if o['source'][s]=='estimated':
                        assert measured is None and est is not None and o['center_used'][s]==est
                    if o['source'][s]=='region_only':
                        assert measured is None and est is None and o['region'][s] is not None and o['center_used'][s] is None
                    if o['source'][s]=='missing':assert measured is None and est is None and o['center_used'][s] is None
            for s in (0,1):
                o=e['natural'];states.update([o['source'][s]]);reasons.update([o['seed_reason'][s]])
                direct=e['direct'][s];est=None if o['center_estimate'][s] is None else np.array(o['center_estimate'][s])/fac
                if direct is None:
                    counts['missing_core' if e['core'] else 'missing_outside']+=1
                    if o['region'][s] is not None or est is not None:
                        natural.append(dict(clip_id=cid,word=row['label'],person=row['signer_id'],side=s,index=e['index'],
                            frame_id=e['frame_id'],time=e['time'],core=e['core'],state=o['source'][s],
                            estimate=None if est is None else est.tolist(),old=e['old_estimate'][s],
                            region_center=None if o['region'][s] is None else (np.array(o['region'][s]['center'])/fac).tolist()))
                else:
                    counts['observed_preserved']+=1
                    if est is not None:
                        obsproxy.append(dict(clip_id=cid,side=s,index=e['index'],
                            error=float(np.linalg.norm(np.array(direct)-est)/e['scale_px'])))
                    rc=o['recheck'][s]
                    event='reappearance' if seen[s] and rc['gap_frames']>0 else 'first_appearance' if not seen[s] else 'continuous'
                    rotation=rc['rotation_deg'] is not None and abs(rc['rotation_deg'])>=p['rotation_event_deg']
                    if event!='continuous' or rotation:
                        rechecks.append(dict(clip_id=cid,word=row['label'],side=s,index=e['index'],frame_id=e['frame_id'],
                            time=e['time'],event=event,rotation=rotation,**rc))
                    seen[s]=True
                block=x['blocks'].get(str(s))
                if block and block[0]<=e['index']<block[1]:
                    assert e['test']['observed_center'][s] is None
                    new=e['test']['center_estimate'][s];oldest=e['old_test_estimate'][s]
                    err=None if new is None else float(np.linalg.norm(np.array(new)/fac-direct)/e['scale_px'])
                    olderr=None if oldest is None else float(np.linalg.norm(np.array(oldest)-direct)/e['scale_px'])
                    synthetic.append(dict(clip_id=cid,word=row['label'],side=s,index=e['index'],frame_id=e['frame_id'],time=e['time'],
                        old_error=olderr,new_error=err,new_state=e['test']['source'][s],
                        region_available=e['test']['region'][s] is not None))
    ann=read(P/'ANNOTATED.json');norm=[a for a in ann if a['label']=='normal_identity'];errors=[a for a in ann if a['label']=='error_identity']
    for a in ann:
        assert lookup[a['clip_id']]['frames'][a['index']]['natural']['proposal'][a['raw_index']]==a['before']
    for a in read(P/'CROSSINGS.json'):
        assert lookup[a['clip_id']]['frames'][a['index']]['natural']['proposal']==a['before']
    oe=[r['old_error'] for r in synthetic if r['old_error'] is not None];ne=[r['new_error'] for r in synthetic if r['new_error'] is not None]
    both=[r for r in synthetic if r['old_error'] is not None and r['new_error'] is not None]
    transitions=Counter(('old_supported' if r['old_error'] is not None else 'old_abstain')+'__'+
        ('new_supported' if r['new_error'] is not None else 'new_abstain') for r in synthetic)
    result=dict(status='SAVED_ARITHMETIC_CONTRACT_PASS_PENDING_VISUAL',counts=dict(counts),
        natural_states=dict(states),seed_reasons=dict(reasons),natural_additions=Counter(r['state'] for r in natural),
        natural_core_additions=Counter(r['state'] for r in natural if r['core']),
        normal_preserved=len(norm),known_errors_unchanged=len(errors),crossings_unchanged=len(read(P/'CROSSINGS.json')),
        side_changes=0,observed_fingers_added=0,
        recheck_events=dict(Counter(r['event'] for r in rechecks)),rotation_events=sum(r['rotation'] for r in rechecks),
        reappear_arm_confirmed=sum(r['event']=='reappearance' and r['arm_confirmed'] for r in rechecks),
        synthetic=dict(targets=len(synthetic),old=stats(oe),new=stats(ne),transitions=dict(transitions),
            common_old=stats([r['old_error'] for r in both]),common_new=stats([r['new_error'] for r in both]),
            common_improved=sum(r['new_error']<r['old_error']-1e-6 for r in both),
            common_worsened=sum(r['new_error']>r['old_error']+1e-6 for r in both)),
        observed_prediction_proxy=stats([r['error'] for r in obsproxy]),
        flow_traces_checked=flow_n,center_fits_checked=fit_n,
        latency=dict(p95_ms=float(np.quantile(latency,.95)),median_ms=float(np.median(latency))),
        protected_files=len(c['protected']),training=0,user_data=0,
        limitations='Same Codex separate arithmetic, not independent human or LK reimplementation. MP centers are proxy. Unchanged labels by construction, not demonstrated identity repair.')
    for n,x in [('VERIFY',result),('NATURAL',natural),('SYNTHETIC',synthetic),('RECHECKS',rechecks),('OBSERVED_PROXY',obsproxy)]:write(U/f'{n}.json',x)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
