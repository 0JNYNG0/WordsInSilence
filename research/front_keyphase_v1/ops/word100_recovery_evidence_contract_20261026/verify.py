"""Independent arithmetic/data-consumer audit; no build/adapter import for replay."""
from common import *
from collections import Counter
TRIPLES=[c[k:k+3] for c in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)) for k in range(3)]
GROUPS={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}
def inside(p,b):return np.isfinite(p).all(-1)&(p[...,0]>=b[0])&(p[...,0]<b[2])&(p[...,1]>=b[1])&(p[...,1]<b[3])
def check_intervals(rows,mask,time):
    actual=np.zeros(len(mask),bool);prev=-2
    for r in rows:
        a,b=r['start_index'],r['end_index'];assert a>prev+1 and b>=a and b<len(mask)
        assert r['samples']==b-a+1 and r['start_seconds']==float(time[a]/1e6) and r['end_seconds']==float(time[b]/1e6)
        actual[a:b+1]=True;prev=b
    assert np.array_equal(actual,mask)

def main():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);result=read(U/'RESULT.json');check(result['output_pins'])
    independent=module('independent_complex_geometry',O/'word100_finger_evidence_20261020/verify.py')
    count=Counter();states_total=np.zeros((3,5),np.int64);groups={g:dict(computable=[0,0,0],eligible=[0,0,0]) for g in GROUPS}
    ci={r['clip_id']:r for r in read(V/'CLASSIFIER_INPUTS.json')['rows']}
    scores={p.stem.removesuffix('_SCORES'):{r['clip_id']:r for r in read(p)['predictions']['native']} for p in V.glob('*_SCORES.json')}
    for row in f['rows']:
        cid=row['clip_id'];p=load(V/'outputs'/(cid+'.npz'));z=load(U/'inputs'/(cid+'.npz'));rec=read(V/'records'/(cid+'.json'))['frames'];rr={r['i']:r for r in rec}
        receipt=read(U/'receipts'/(cid+'.json'));n=len(p['frame_id']);box=p['roi_box']
        assert same(z['xy'],p['xy_px'][:,:,33:].reshape(3,n,2,21,2)) and same(z['valid'],p['usable'][:,:,33:].reshape(3,n,2,21))
        assert same(z['point_origin'],p['point_origin'][:,:,33:].reshape(3,n,2,21))
        for key in ('timestamp_us','frame_id','position_only','roi_box'):assert same(z[key],p[key])
        assert same(z['pose_xy'],p['xy_px'][0,:,:33]) and same(z['pose_valid'],p['usable'][0,:,:33])
        original=load(read(row['record'])['views']['masked']['channel_file'])['x_whole'];kept=load(receipt['classifier_input']['path'])['x']
        assert kept.tobytes()==original.tobytes() and kept.shape==original.shape and kept.dtype==original.dtype
        assert sha(receipt['classifier_input']['path'])==receipt['classifier_input']['sha256']
        for model in scores:
            assert receipt['recognition']['saved_original_predictions'][model]=={k:scores[model][cid][k] for k in ('prediction','top5','correct','top5_correct','status')}
        assert receipt['recognition']['new_inference'] is False and receipt['feedback_observation']['whole_action_correct'] is None
        assert receipt['feedback_observation']['expected_motion_scope'] is None
        identities=np.zeros((n,2),np.uint8)
        for fr in rec:
            i=fr['i']
            for d in fr['mp_decisions']+fr['yolo_decisions']:
                if d['side'] in (0,1):
                    if d['reason']=='confirm_existing_hand':identities[i,d['side']]=1
                    if d['accepted']:identities[i,d['side']]=2
            for d in fr['mp_decisions']:
                if d['reason'] in ('existing_identity_conflict','unresolved_existing_overlap'):
                    for s in set([fr['old_sides'][j] for j in d.get('matched_old',[])]+[d['side']]):
                        if s in (0,1) and z['valid'][2,i,s].any():identities[i,s]=3
        assert same(identities,z['identity_state']) and same(identities==3,z['identity_conflict'])
        sh=z['pose_valid'][:,11]&z['pose_valid'][:,12];width=np.linalg.norm(z['pose_xy'][:,11]-z['pose_xy'][:,12],axis=-1)
        scale=float(np.median(width[sh&np.isfinite(width)&(width>1e-6)]));origin=(z['pose_xy'][:,11]+z['pose_xy'][:,12])/2
        assert np.all(z['body_scale']==scale)
        for k in range(3):
            x=z['xy'][k].astype(float);valid=z['valid'][k];st=np.zeros((n,2,15),np.uint8);deg=np.full(st.shape,np.nan)
            for a,(i,j,l) in enumerate(TRIPLES):
                v=x[:,:,i]-x[:,:,j];w=x[:,:,l]-x[:,:,j]
                present=valid[:,:,i]&valid[:,:,j]&valid[:,:,l]&np.isfinite(v).all(-1)&np.isfinite(w).all(-1)
                good=present&(np.linalg.norm(v,axis=-1)>1e-6)&(np.linalg.norm(w,axis=-1)>1e-6)
                deg[:,:,a]=np.where(good,np.degrees(np.arctan2(abs(v[:,:,0]*w[:,:,1]-v[:,:,1]*w[:,:,0]),np.sum(v*w,-1))),np.nan)
                st[:,:,a]=np.where(good,2,np.where(present,1,0))
            for i in range(n):
                for s in (0,1):
                    bx=None;bv=None
                    if i in rr and valid[i,s].any():
                        fr=rr[i];sl=slice(33+21*s,54+21*s)
                        if p['full_usable'][0,i,sl].any():
                            ds=[d for d in fr['mp_decisions'] if d['reason']=='confirm_existing_hand' and d['side']==s]
                            if len(ds)==1:bx=np.asarray(fr['mp_full'][ds[0]['index']]['xy'],float);bv=inside(bx,box)
                        elif k:
                            prefix='yolo' if k==2 and (valid[i,s]&~z['valid'][1,i,s]).any() else 'mp'
                            ds=[d for d in fr[prefix+'_decisions'] if d['accepted'] and d['side']==s]
                            if ds:assert len(ds)==1;bx=np.asarray(fr[prefix+'_crop'][ds[0]['crop_index']]['xy'],float);bv=valid[i,s]&inside(bx,box)
                    if bx is None:
                        assert not z['pair_valid'][k,i,s].any() and np.isnan(z['pair_xy'][k,i,s]).all();continue
                    assert same(bx,z['pair_xy'][k,i,s]) and same(bv,z['pair_valid'][k,i,s])
                    q=independent.reference(x[i,s],valid[i,s],bx,bv)
                    if q is not None:
                        av=q['angle_available']&np.isfinite(deg[i,s]);st[i,s,av]=np.where(q['angle_unresolved'][av],3,4);count['independent_geometry_pairs']+=1
            assert same(st,z['angle_state'][k]);np.testing.assert_allclose(deg,z['angle_degree'][k],atol=1e-9,rtol=1e-10,equal_nan=True)
            values=np.full((n,2,23),np.nan);available=np.zeros(values.shape,bool)
            values[:,:,:15]=deg;available[:,:,:15]=np.isfinite(deg)
            for s in (0,1):
                for a,pts in ((15,[0,5,9,13,17]),(21,[0])):
                    # Match stored float32 source precision then normalize; all required landmarks must exist.
                    vv=(np.mean(z['xy'][k,:,s,pts,:],axis=0)-origin)/scale
                    values[:,s,a:a+2]=vv;available[:,s,a:a+2]=(valid[:,s,pts].all(-1)&sh)[:,None]
                for a,j in ((17,13+s),(19,15+s)):
                    values[:,s,a:a+2]=(z['pose_xy'][:,j]-origin)/scale;available[:,s,a:a+2]=(z['pose_valid'][:,j]&sh)[:,None]
            eligible=available.copy();eligible[:,:,:15]&=st==4;eligible[:,:,:17]&=~(identities==3)[:,:,None];eligible[:,:,21:]&=~(identities==3)[:,:,None]
            assert same(available,z['available'][k]) and same(eligible,z['eligible'][k])
            np.testing.assert_allclose(np.where(available,values,np.nan),z['values'][k],atol=1e-6,rtol=1e-6,equal_nan=True)
            # Original raw hands are retained even if quality has not been corroborated.
            assert z['valid'][k][z['valid'][0]].all() and same(z['xy'][k][z['valid'][0]],z['xy'][0][z['valid'][0]])
            states_total[k]+=np.bincount(st.ravel(),minlength=5)
            for g,(a,b) in GROUPS.items():
                groups[g]['computable'][k]+=int(available[:,:,a:b].sum());groups[g]['eligible'][k]+=int(eligible[:,:,a:b].sum())
            obs=receipt['feedback_observation']['conditions'][k]
            for s in (0,1):
                h=obs['hands'][s];assert h['angle_state_counts']==np.bincount(st[:,s].ravel(),minlength=5).tolist()
                check_intervals(h['wrist_not_observed'],~valid[:,s,0],z['timestamp_us'])
                check_intervals(h['finger_quality_incomplete'],~eligible[:,s,:15].all(-1),z['timestamp_us'])
                check_intervals(h['identity_disagreement'],identities[:,s]==3,z['timestamp_us'])
        po=z['position_only'];assert not z['available'][2,:,:,:17][po].any();assert not z['eligible'][2,:,:,:17][po].any()
        cross=z['valid'][0,:,0,0]&z['valid'][0,:,1,0]&(z['xy'][0,:,0,0,0]<z['xy'][0,:,1,0,0])
        count.update(clips=1,frames=n,original_classifier_cells=original.size,raw_hand_points_preserved=int(z['valid'][0].sum()),
            angle_states=z['angle_state'].size,crossing_order_frames=int(cross.sum()),wrist_only=int(po.sum()))
    assert states_total.tolist()==result['angle_state_counts'] and groups==result['groups']
    for row in f['comparison_rows']:
        p=load(R/'comparisons'/(row['name']+'.npz'));receipt=read(U/'comparison_receipts'/(row['name']+'.json'))
        assert receipt['whole_action_correct'] is None and receipt['alignment_verified'] is False
        for k in range(3):
            for s in (0,1):
                hand=receipt['conditions'][k]['hands'][s]
                for g,(a,b) in GROUPS.items():
                    ex=p['expected'][:,s,a:b];ok=p['compared'][k,:,s,a:b];d=p['difference'][k,:,s,a:b];q=hand['groups'][g]
                    assert q['expected_cells']==int(ex.sum()) and q['compared_cells']==int(ok.sum()) and q['missing_cells']==int((ex&~ok).sum())
                    assert q['coverage']==(int(ok.sum())/int(ex.sum()) if ex.any() else None)
                    if ok.any():assert abs(q['mean_difference']-float(d[ok].mean()))<1e-12
                    else:assert q['mean_difference'] is None
                    check_intervals(q['missing_intervals'],(ex&~ok).any(-1),p['timestamp_us'])
                for a,q in enumerate(hand['angles']):
                    ex=p['expected'][:,s,a];ok=p['compared'][k,:,s,a];assert q['expected']==int(ex.sum()) and q['compared']==int(ok.sum())
                    check_intervals(q['missing_intervals'],ex&~ok,p['timestamp_us'])
        count['comparison_receipts']+=1;count['comparison_cells']+=p['compared'].size
    # Meaningful consumer fault-injection: absence leaves denominator and other angle channels unchanged.
    consumer=module('contract_consumer_for_perturbation',U/'adapter.py')
    fake=dict(expected=np.ones((5,2,23),bool),compared=np.ones((3,5,2,23),bool),difference=np.zeros((3,5,2,23)),timestamp_us=np.arange(5)*33333)
    original=consumer.comparison_receipt(fake);fake['compared'][2,2:4,0,2]=False;fake['difference'][2,2:4,0,2]=np.nan
    changed=consumer.comparison_receipt(fake);a=changed['conditions'][2]['hands'][0]
    assert a['angles'][2]['expected']==5 and a['angles'][2]['compared']==3
    assert a['angles'][1]==original['conditions'][2]['hands'][0]['angles'][1]
    assert a['groups']['angle']['evidence_status']=='partial_evidence' and changed['whole_action_correct'] is None
    fake['compared'][2,:,:,:]=False;fake['difference'][2,:,:,:]=np.nan;empty=consumer.comparison_receipt(fake)
    assert all(g['mean_difference'] is None and g['evidence_status']=='no_comparable_evidence' for h in empty['conditions'][2]['hands'] for g in h['groups'].values())
    count['consumer_fault_injection_checks']=5
    check(f['pins']);check(f['protected']);check(result['output_pins'])
    write(U/'VERIFIED.json',dict(at=now(),status='PASS',counts=dict(count),method='Separate Codex complex-similarity and angle/channel arithmetic, independent interval reconstruction and original-byte checks. Not independent human ground truth.',anatomical_accuracy_verified=False))
    print(json.dumps(dict(count)),flush=True)
if __name__=='__main__':main()
