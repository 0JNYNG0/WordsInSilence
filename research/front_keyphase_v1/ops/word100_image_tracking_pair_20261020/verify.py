"""Separate saved-trace arithmetic/chronology/proxy audit. No tracker import."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def pct(vals):
    return dict(n=len(vals),median=float(np.median(vals)),p95=float(np.quantile(vals,.95)),max=float(max(vals)),
                within_0_10=int(sum(v<=.1 for v in vals))) if vals else dict(n=0)
def main():
    c=read(U/'FROZEN.json');p=c['params'];counts=Counter();changes=[];episodes=[];synth=[];proxy=[];lat=[];look={}
    for path,h in {**c['pins'],**c['protected']}.items():assert sha(path)==h,path
    for row in c['rows']:
        cid=row['clip_id'];x=read(U/'records'/f'{cid}.json');look[cid]=x
        counts['clips']+=1;counts['frames']+=len(x['frames']);lat+=x['per_frame_ms']
        last=[[-float('inf')]*2 for _ in range(2)]
        with np.load(row['views']['sensor']['assignment'],allow_pickle=False) as a:base=a['sides'].copy()
        with np.load(row['views']['sensor']['raw'],allow_pickle=False) as a:
            width=a['width'].copy();height=a['height'].copy();ts=a['timestamp_us'].copy()
        assert len(x['frames'])==len(ts)
        for e in x['frames']:
            f=e['index'];assert abs(e['time']-ts[f]/1e6)<1e-9
            assert e['baseline_sides']==base[f,:len(e['baseline_sides'])].tolist()
            factor=p['image_width']/width[f];w=p['image_width'];h=round(height[f]*factor)
            for vi,(mode,estkey) in enumerate([('natural','estimate'),('test','test_estimate')]):
                out=e[mode]
                assert out['observed_finger_points_added']==0
                for side in (0,1):
                    tr=out['traces'].get(str(side))
                    if tr:
                        assert tr['anchor_time']<e['time'] and abs(tr['anchor_time']-last[vi][side])<1e-7
                        assert 0<tr['age']<=p['max_age_s']+1e-8
                        if 'forward' in tr:
                            a=np.array(tr['previous_points'],float);b=np.array(tr['forward'],float);back=np.array(tr['backward'],float)
                            err=np.array(tr['errors'],float)
                            good=(np.array(tr['forward_status'])==1)&(np.array(tr['backward_status'])==1)
                            good &= np.isfinite(b).all(1)&np.isfinite(err)&(err<=p['lk_error'])
                            good &= np.linalg.norm(back-a,axis=1)<=p['fb_px']
                            good &= (b[:,0]>=0)&(b[:,0]<w)&(b[:,1]>=0)&(b[:,1]<h)
                            assert int(good.sum())==tr['points_after']
                            enough=good.sum()>=p['min_points'] and good.sum()/len(good)>=p['keep_fraction']
                            if enough:
                                d=b[good]-a[good];med=np.median(d,axis=0)
                                proposed=np.array(tr['previous_center'])+med
                                spread=float(np.quantile(np.linalg.norm(d-med,axis=1),.9))
                                assert np.allclose(proposed,tr['proposed_center'],atol=3e-5)
                                assert abs(spread-tr['spread'])<3e-5
                                ok=spread<=p['residual_shoulder']*tr['shoulder'] and np.isfinite(proposed).all()
                                ok=ok and 0<=proposed[0]<w and 0<=proposed[1]<h
                                assert (tr['reason']=='flow_pass')==ok
                            else:assert tr['reason']=='flow_quality'
                            counts['flow_traces_checked']+=1
                    if e[estkey][side] is not None:
                        assert tr and tr['reason']=='flow_pass' and tr['decision']=='image_position_estimate'
                        assert np.allclose(np.array(e[estkey][side])*factor,tr['proposed_center'],atol=3e-5)
                    if out['seed'][side]:last[vi][side]=e['time']
            for i,(a,b) in enumerate(zip(e['baseline_sides'],e['natural']['proposal'])):
                if a!=b:changes.append(dict(clip_id=cid,word=row['label'],index=f,frame_id=e['frame_id'],raw_index=i,before=a,after=b,time=e['time']))
            for s in (0,1):
                direct,estimate=e['direct'][s],e['estimate'][s]
                counts['direct_palm_frames']+=int(direct is not None)
                if direct is None:
                    counts['missing_palm_frames']+=1
                    counts['missing_with_image_estimate']+=int(estimate is not None)
                elif estimate is not None and e['scale_px'] and e['scale_px']>0:
                    proxy.append(float(np.linalg.norm(np.array(direct)-estimate)/e['scale_px']))
                block=x['blocks'].get(str(s))
                if block and block[0]<=f<block[1]:
                    assert direct is not None
                    q=e['test_estimate'][s]
                    err=None if q is None else float(np.linalg.norm(np.array(q)-direct)/e['scale_px'])
                    synth.append(dict(clip_id=cid,side=s,index=f,frame_id=e['frame_id'],error_shoulder=err))
        for s in (0,1):
            fs=[e['index'] for e in x['frames'] if e['direct'][s] is None and e['estimate'][s] is not None]
            groups=[]
            for f in fs:
                if groups and groups[-1][-1]==f-1:groups[-1].append(f)
                else:groups.append([f])
            for g in groups:
                episodes.append(dict(id=len(episodes),clip_id=cid,word=row['label'],side=s,start=g[0],end=g[-1],
                    frames=len(g),time_start=x['frames'][g[0]]['time'],time_end=x['frames'][g[-1]]['time'],
                    core_frames=sum(x['frames'][f]['core'] for f in g),
                    baseline_candidate_present=sum(s in x['frames'][f]['baseline_sides'] for f in g)))
    ann=read(O/'word100_image_arm_connectivity_20261018/ANNOTATED.json')
    matched=[];normal=Counter();errors=Counter()
    for a in ann:
        if a['clip_id'] not in look or a['view']!='sensor':continue
        e=look[a['clip_id']]['frames'][a['index']];i=a['raw_index']
        out=dict(clip_id=a['clip_id'],word=a['word'],index=a['index'],raw_index=i,truth=a['actual_side'],
                 label=a['verdict'],before=e['baseline_sides'][i],after=e['natural']['proposal'][i])
        matched.append(out)
        if a['verdict']=='normal_identity':
            normal['total']+=1;normal['before_correct']+=out['before']==out['truth'];normal['after_correct']+=out['after']==out['truth']
            normal['new_wrong']+=out['before']==out['truth'] and out['after']!=out['truth']
        elif a['verdict']=='error_identity':
            errors['total']+=1;errors['before_correct']+=out['before']==out['truth'];errors['after_correct']+=out['after']==out['truth']
    crossings=[]
    for a in read(O/'word100_image_arm_connectivity_20261018/CROSSING.json'):
        if a['clip_id'] not in look or a['view']!='sensor':continue
        e=look[a['clip_id']]['frames'][a['index']]
        crossings.append(dict(clip_id=a['clip_id'],index=a['index'],before=e['baseline_sides'],after=e['natural']['proposal']))
    syerr=[a['error_shoulder'] for a in synth if a['error_shoulder'] is not None]
    latency=dict(median_ms=float(np.median(lat)),p95_ms=float(np.quantile(lat,.95)),total_s=sum(lat)/1000)
    for path,h in c['protected'].items():assert sha(path)==h
    if (U/'STORAGE_RECOVERY.json').exists():
        rec=read(U/'STORAGE_RECOVERY.json')
        for path,h in rec['completed_preserved'].items():assert sha(path)==h
    result=dict(status='PASS_SAVED_ARITHMETIC_AND_PROVENANCE_NOT_VISUAL_ADOPTION',counts=dict(counts),
        side_changes=len(changes),normal_labels=dict(normal),error_labels=dict(errors),
        crossing_frames=len(crossings),crossing_changed=sum(a['before']!=a['after'] for a in crossings),
        natural_estimate_episodes=len(episodes),synthetic_dropout=dict(target_frames=len(synth),estimated_frames=len(syerr),
            error_as_shoulder_width=pct(syerr)),one_step_observed_proxy=pct(proxy),latency=latency,
        protected_files=len(c['protected']),training=0,user_data=0,
        limitation='Position error uses saved MediaPipe palm centers as proxy, not manual joint ground truth. Same author separate arithmetic; image LK algorithm itself not independently reimplemented.')
    for name,val in [('VERIFY',result),('CHANGES',changes),('EPISODES',episodes),('ANNOTATED',matched),('CROSSINGS',crossings),('SYNTHETIC',synth)]:
        write(U/f'{name}.json',val)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
