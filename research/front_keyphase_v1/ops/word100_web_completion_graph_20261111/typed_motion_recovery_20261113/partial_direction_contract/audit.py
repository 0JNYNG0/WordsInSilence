"""FIT-only coordinate/annotation contract audit; no models or held-out scoring."""
from pathlib import Path
import sys, traceback
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent))
from common import *
from collections import defaultdict,Counter

N=HERE/'null_candidate_learning'

def numbers(v):
    a=np.asarray(v,float)
    if not len(a):return {'count':0}
    return dict(count=len(a),minimum=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),maximum=float(a.max()))

def components(x,ts,a,b):
    g,ok,info=old.geometry(x,ts,a,b,2,'R','L')
    out=dict(eligible=bool(ok),coverage=info['coverage'],observed=info['observed'],duration_seconds=float(ts[b]-ts[a])/1e6)
    if not ok:return out
    p,v=old.prior.motion.body_points(x)
    p[:,:,0]*=np.sign(p[:,2,0]-p[:,1,0])[:,None]
    rel=p[:,6]-p[:,3];arm=p[:,5]-p[:,3];length=np.linalg.norm(arm,axis=1)
    unit=arm/np.maximum(length[:,None],1e-5)
    good=v[:,[6,5,3]].all(1)&(length>.05)
    ii=np.flatnonzero(good[a:b+1])+a
    r,u=rel[ii],unit[ii]
    z=(r*u).sum(1); dz=np.diff(z)
    # Product difference identity: exact, no fitted threshold.
    translation=float(((r[-1]-r[0])*(u[-1]+u[0])/2).sum())
    rotation=float(((r[-1]+r[0])/2*(u[-1]-u[0])).sum())
    assert abs((translation+rotation)-float(g[40]))<1e-6
    dr=np.diff(r,axis=0);du=np.diff(u,axis=0)
    moving_steps=(dr*(u[1:]+u[:-1])/2).sum(1)
    axis_steps=((r[1:]+r[:-1])/2*du).sum(1)
    assert np.allclose(dz,moving_steps+axis_steps,atol=1e-10)
    path=float(np.abs(dz).sum());angular=np.arccos(np.clip((u[1:]*u[:-1]).sum(1),-1,1))
    out.update(delta=float(g[40]),direct_raw_delta=float(z[-1]-z[0]),path_variation=path,
        net_fraction=float(abs(z[-1]-z[0])/path) if path else 0.,
        endpoint_relative_motion=translation,endpoint_anchor_rotation=rotation,
        step_relative_motion_net=float(moving_steps.sum()),step_anchor_rotation_net=float(axis_steps.sum()),
        relative_motion_path=float(np.abs(moving_steps).sum()),anchor_rotation_path=float(np.abs(axis_steps).sum()),
        anchor_total_rotation_degrees=float(np.degrees(angular.sum())),
        anchor_max_step_rotation_degrees=float(np.degrees(angular.max())) if len(angular) else 0.,
        anchor_min_length=float(length[ii].min()),anchor_max_length=float(length[ii].max()),
        wrist_distance_delta=float(g[43]),raw_steps=len(dz),positive_steps=int((dz>0).sum()),negative_steps=int((dz<0).sum()),
        max_abs_projection_step=float(abs(dz).max()) if len(dz) else 0.)
    # Delete only hand coordinates/masks; all body coordinates/masks/PTS retained.
    xm=x.copy();xm[:,18:186]=0;xm[:,195:279]=0
    gm,om,_=old.geometry(xm,ts,a,b,2,'R','L')
    assert om==ok and np.array_equal(gm,g)
    gr,orr,_=old.geometry(x,ts,a,b,2,'R','L',True)
    assert orr and abs(float(gr[40])+float(g[40]))<1e-6
    return out

def main():
    assert not(U/'CONTRACT.json').exists() and not(U/'RESULTS.json').exists()
    source_paths=[GRAPH/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',PRIOR/'additional_source_review/SOURCE_EPISODES_ADDED.json']
    paths=[Path(__file__),N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',HERE/'INPUTS_COMPLETE.json',PRIOR/'projected_direction_core.py',OPS/'word100_motion_components_20261109/motion.py']+source_paths
    pins={str(p):sha(p) for p in paths}
    write(U/'CONTRACT.json',dict(at=stamp(),scope='FIT02..07 only, axis2/family4, existing positive full/partial and reviewed hold candidates; no probe arrays, models, training, evaluation or threshold search',
        labels='Existing progression labels retained. Partial means >=2 direct witnesses within one occurrence, not automatically a directional endpoint or complete/correct sign.',
        measurement='Shoulder-width projection onto current opposite-forearm unit vector; exact relative-motion/anchor-rotation product-difference decomposition. Mathematical zero only for sign audit.',
        checks='All fixed FIT candidates/views including original and whole-source reversed; same-frame hand-information removal and interval-time-reversal checks. Results never relabel originals.',
        provenance='Source RGB annotations are prior Codex review, not independent human gold. Read coordinate caches only.',pins=pins,private_access=False,final8=False,web_changed=False))
    rows=read(N/'TRAINING_INPUT_INDEX.json')['rows'];masks=read(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
    sources={r['clip_id']:r for p in source_paths for r in read(p)['rows'] if r['person'] in range(2,8)}
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['person'] in range(2,8)}
    crop={r['clip_id']:r for p in [old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json'] for r in read(p)['rows'] if r['clip_id'] in ready}
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_partial_direction_guard',OPS/'word100_observation_guard_20261108/guard.py')
    ad=old.load('_partial_direction_adapter',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cache={};input_pins={}
    def raw(r):
        key=(r['clip_id'],r['view'])
        if key not in cache:
            sr=ready[r['clip_id']];assert sr['person']==r['person']
            for path,h in [(sr['path'],sr['sha256']),(sr['channel'],sr['channel_sha256'])]:assert sha(path)==h;input_pins[path]=h
            z=np.load(sr['path']);ch=np.load(sr['channel']);ts=z['timestamp_us']
            if r['view']=='actual_crop':
                cr=crop[r['clip_id']];assert sha(cr['path'])==cr['sha256'];input_pins[cr['path']]=cr['sha256'];cz=np.load(cr['path']);assert np.array_equal(ts,cz['timestamp_us'])
                cl=guard.quarantine(cz['xy_px'],cz['score'],cz['timestamp_us'],640,480)
                x=ad.convert(cl['xy_px'],cz['score'],cz['timestamp_us'],cz['frame_id'],640,480)['tcn280']
            else:x=ch['full'] if r['view']=='full' else ch['feedback'] if 'feedback' in ch.files else ch['upper_mask']
            cache[key]=(x.copy(),ts.copy())
        x,ts=cache[key];x=x.copy();ts=ts.copy()
        if r['reverse']:
            x=x[::-1].copy();ts=ts[-1]-ts[::-1];x[:,279]=np.r_[0,np.diff(ts)/1e6]
        if r['origin']=='reviewed_local_hold':
            a,b=map(int,r['source_span']);x=x[a:b+1].copy();ts=ts[a:b+1];x[0,279]=0
        return x,ts
    out=[];direction_support=[]
    dr=read(HERE/'INPUTS_COMPLETE.json')['rows']
    for i,(r,mask) in enumerate(zip(rows,masks)):
        if r['split']!='train' or r['family']!=4:continue
        assert r['person'] in range(2,8) and mask['bag_index']==i and mask['clip_id']==r['clip_id']
        assert sha(r['path'])==r['sha256'];input_pins[r['path']]=r['sha256'];bag=np.load(r['path']);spans=bag['spans'];full=bag['y'].astype(bool)
        partial={v['candidate_index']:v for v in mask['candidates']}
        x,ts=raw(r);src=sources[r['clip_id']];assert src['source_sha256']==r['source_sha256']
        directions=[v for v in dr if v['split']=='train' and v['axis']==2 and v['clip_id']==r['clip_id'] and v['view']==r['view'] and not v['reverse'] and v['target']!=1]
        positive_directions=set(v['target'] for v in directions)
        direction_support.append(dict(bag_index=i,clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],existing_direction_targets=sorted(positive_directions),source_observation=src.get('observation_ko')))
        for k,(aa,bb) in enumerate(spans):
            kind='hold' if r['origin']=='reviewed_local_hold' else 'full' if full[k] else 'partial' if k in partial else None
            if kind is None:continue
            a,b=int(aa),int(bb);c=components(x,ts,a,b)
            entry=dict(bag_index=i,candidate_index=k,clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],label_kind=kind,span=[a,b],frames=b-a+1,source_span=r.get('source_span'),measurement=c,
                expected_direction_by_existing_occurrence=(0 if r['reverse'] else 2) if kind!='hold' and positive_directions=={2} else None,
                semantic_completion=None,direction_target_created=False)
            if kind=='partial':
                pm=partial[k];entry['original_span']=pm['original_span'];entry['groups']=pm['groups'];w=sorted(set(f for gg in pm['groups'] for f in gg['witnessed_frames']))
                if r['reverse']:w=sorted(len(x)-1-f for f in w)
                assert len(w)>=2 and a<=w[0]<w[-1]<=b
                entry['witness_core_span']=[w[0],w[-1]];entry['witness_core_measurement']=components(x,ts,w[0],w[-1])
            out.append(entry)
    summary=[]
    for key,rr in __import__('itertools').groupby(sorted(out,key=lambda r:(r['label_kind'],r['view'],r['reverse'])),lambda r:(r['label_kind'],r['view'],r['reverse'])):
        rr=list(rr);good=[r for r in rr if r['measurement']['eligible']];exp=-1 if key[2] else 1
        summary.append(dict(label_kind=key[0],view=key[1],reverse=key[2],rows=len(rr),eligible=len(good),delta=numbers([r['measurement']['delta'] for r in good]),
            projection_sign_opposes_occurrence=sum(r['measurement']['delta']*exp<0 for r in good) if key[0]!='hold' else None,
            relative_motion_sign_opposes_occurrence=sum(r['measurement']['endpoint_relative_motion']*exp<0 for r in good) if key[0]!='hold' else None,
            endpoint_projection_motion_sign_disagreement=sum(r['measurement']['delta']*r['measurement']['endpoint_relative_motion']<0 for r in good),
            anchor_rotation_larger_than_relative_motion=sum(abs(r['measurement']['endpoint_anchor_rotation'])>abs(r['measurement']['endpoint_relative_motion']) for r in good),
            net_fraction=numbers([r['measurement']['net_fraction'] for r in good]),anchor_total_rotation_degrees=numbers([r['measurement']['anchor_total_rotation_degrees'] for r in good])))
    full_forward=[r for r in out if r['view']=='full' and not r['reverse'] and r['measurement']['eligible']]
    holds=[r['measurement']['delta'] for r in full_forward if r['label_kind']=='hold'];max_hold=max(abs(v) for v in holds)
    short=[r for r in full_forward if r['label_kind']=='partial'];overlap=[r for r in short if abs(r['measurement']['delta'])<=max_hold]
    counter=[r for r in out if r['label_kind']!='hold' and r['measurement']['eligible'] and r['measurement']['delta']*(-1 if r['reverse'] else 1)<=0]
    witness_counter=[r for r in short if r['witness_core_measurement']['eligible'] and r['witness_core_measurement']['delta']<=0]
    report=dict(at=stamp(),status='COMPLETE_FIT_COORDINATE_AUDIT_NOT_TRAINING_PERMISSION',rows=out,summary=summary,source_direction_annotation_support=direction_support,
        full_forward=dict(clips=len(set(r['clip_id'] for r in full_forward)),hold_delta=numbers(holds),partial_delta=numbers([r['measurement']['delta'] for r in short]),
            maximum_absolute_reviewed_hold_delta=max_hold,partial_within_hold_amplitude=len(overlap),partial_total=len(short),partial_within_hold_amplitude_clips=sorted(set(r['clip_id'] for r in overlap)),
            eligible_witness_core=sum(r['witness_core_measurement']['eligible'] for r in short),witness_core_sign_conflicts=len(witness_counter)),
        sign_conflicts=counter,witness_core_sign_conflicts=witness_counter,input_pins=input_pins,contract_sha256=sha(U/'CONTRACT.json'),code_pins=pins,
        new_direction_labels=False,new_model_inference=False,new_training=False,probe_arrays_opened=False,web_changed=False)
    for p,h in {**pins,**input_pins}.items():assert sha(p)==h
    write(U/'RESULTS.json',report)
    print(json.dumps(dict(rows=len(out),summary=summary,full_forward=report['full_forward'],sign_conflicts=len(counter)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
