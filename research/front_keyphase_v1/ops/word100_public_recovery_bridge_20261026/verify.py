"""Separate arithmetic verifier; no adapter or runner import, no detector inference."""
from common import *
from collections import Counter

def main():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    original_geometry=module('independent_finger_geometry',O/'word100_finger_evidence_20261020/verify.py')
    result=read(U/'INPUT_RESULT.json');counts=Counter();inputs={}
    cross=read(O/'word100_yolo_hand_fourway_20261025/CROSSING_PRESERVATION.json')
    triples=[ch[k:k+3] for ch in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)) for k in range(3)]
    hids={r['clip_id'] for r in f['hamer_manifest']}
    def inside(p,b):return np.isfinite(p).all(-1)&(p[...,0]>=b[0])&(p[...,0]<b[2])&(p[...,1]>=b[1])&(p[...,1]<b[3])
    for row in f['rows']:
        cid=row['clip_id'];p=load(U/'inputs'/(cid+'.npz'));inputs[cid]=p;z=load(row['arrays']);n=len(z['frame_id']);box=z['roi_box']
        assert same(p['frame_id'],z['frame_id']) and same(p['timestamp_us'],z['timestamp_us'])
        assert same(p['pose_xy'],z['native_xy_px'][:,:33])
        assert same(p['pose_valid'],z['native_usable'][:,:33]&inside(p['pose_xy'],box))
        expected=[z['native_usable'][:,33:].reshape(n,2,21)&inside(z['native_xy_px'][:,33:].reshape(n,2,21,2),box),z['redetected_roi_usable'][:,33:].reshape(n,2,21)]
        for k in range(2):assert same(p['valid'][k],expected[k])
        for k in range(1,3):
            v=p['valid'][k-1];assert p['valid'][k][v].all() and same(p['xy'][k][v],p['xy'][k-1][v])
        counts['old_valid_points_preserved']+=int(expected[1].sum())
        assert same(p['full_final_valid']&inside(p['full_final_xy'],box),p['valid'][2])
        assert np.allclose(p['full_final_xy'][p['valid'][2]],p['xy'][2][p['valid'][2]],atol=.0002,rtol=0)
        mr=read(row['source_assets']['detections'])['frames'];yr={r['i']:r for r in read(Y/'records'/(cid+'.json'))['frames']}
        native=z['native_usable'][:,33:].reshape(n,2,21)
        for k in range(3):
            x=p['xy'][k].astype(float);valid=p['valid'][k];states=np.zeros((n,2,15),np.uint8)
            computed=np.full(states.shape,np.nan)
            for a,(i,j,l) in enumerate(triples):
                u=x[:,:,i]-x[:,:,j];v=x[:,:,l]-x[:,:,j]
                present=valid[:,:,i]&valid[:,:,j]&valid[:,:,l]&np.isfinite(u).all(-1)&np.isfinite(v).all(-1)
                good=present&(np.linalg.norm(u,axis=-1)>1e-6)&(np.linalg.norm(v,axis=-1)>1e-6)
                value=np.degrees(np.arctan2(abs(u[:,:,0]*v[:,:,1]-u[:,:,1]*v[:,:,0]),np.sum(u*v,-1)))
                computed[:,:,a]=np.where(good,value,np.nan);states[:,:,a]=np.where(good,2,np.where(present,1,0))
            for i in range(n):
                for s in range(2):
                    b=None;vb=None
                    if not valid[i,s].any():continue
                    if k==2 and np.any(valid[i,s]&~p['valid'][1,i,s]):
                        ds=[d for d in yr[i]['decisions'] if d['accepted'] and d['side']==s];assert len(ds)==1
                        b=np.asarray(yr[i]['crop'][ds[0]['crop_index']]['xy']);vb=valid[i,s]&inside(b,box)
                    elif native[i,s].any():
                        ds=[d for d in mr[i]['decisions'] if d['reason']=='confirm_existing_hand' and d['side']==s]
                        if len(ds)==1:b=np.asarray(mr[i]['full'][ds[0]['index']]['xy']);vb=inside(b,box)
                    elif k>0:
                        ds=[d for d in mr[i]['decisions'] if d['accepted'] and d['side']==s]
                        if len(ds)==1:b=np.asarray(mr[i]['crop'][ds[0]['crop_index']]['xy']);vb=valid[i,s]&inside(b,box)
                    if b is None:continue
                    independent=original_geometry.reference(x[i,s],valid[i,s],b,vb)
                    if independent is None:continue
                    av=independent['angle_available']&np.isfinite(computed[i,s])
                    states[i,s,av]=np.where(independent['angle_unresolved'][av],3,4)
                    counts['independent_shape_pairs']+=1
            assert same(states,p['angle_state'][k]),cid
            np.testing.assert_allclose(computed,p['angle_degree'][k],rtol=1e-10,atol=1e-9,equal_nan=True)
            # No inferred/region channel can bypass either angle evidence or identity gate.
            expected_angles=(states==4)&~p['identity_conflict'][:,:,None]
            assert same(expected_angles,p['eligible'][k,:,:,:15])
            assert np.isnan(p['values'][k][~p['available'][k]]).all()
            counts['angle_states_checked']+=states.size
        if 'angle_evidence' in row['source_assets']:
            old=load(row['source_assets']['angle_evidence']);assert same(old['angle_state'],p['angle_state'][1]);counts['old_angle_sidecars_preserved']+=1
        for cr in cross:
            if cr['clip_id']==cid and cr['condition']=='M1':
                i=cr['frame_id'];v=p['valid'][1,i];assert same(p['xy'][1,i][v],p['xy'][2,i][v]);counts['crossing_frames_preserved']+=1
        if cid in hids:
            hz=load(H/'outputs'/(cid+'.npz'));assert same(~hz['identity_eligible'],p['identity_conflict'])
            assert same(hz['R_inferred'],p['inferred_valid'])
            assert same(hz['R_xy'][p['inferred_valid']],p['inferred_xy'][p['inferred_valid']])
        counts['frames']+=n;counts['clips']+=1
    compare=read(U/'COMPARE_RESULT.json');totals={};group={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}
    for row in f['pairs']:
        q=inputs[row['query']];r=inputs[row['reference']];p=load(U/'comparisons'/(row['name']+'.npz'));old=load(D/'cross'/(row['name']+'.npz'))
        assert same(p['path'],old['path']);domain=np.concatenate([old['expected'],old['expected'][:,:,15:17]],-1);assert same(p['expected'],domain)
        for k in range(3):
            for j in range(len(domain)):
                a,b=p['query_span'][j];assert (a,b)==(old['path'][old['path'][:,1]==j,0].min(),old['path'][old['path'][:,1]==j,0].max())
                ok=q['eligible'][k,a:b+1].all(0)&r['eligible'][1,j]&domain[j]
                delta=abs(np.mean(q['values'][k,a:b+1],axis=0)-r['values'][1,j])
                assert same(ok,p['compared'][k,j]);np.testing.assert_allclose(p['difference'][k,j],np.where(ok,delta,np.nan),equal_nan=True,rtol=1e-12,atol=1e-12)
        for g,(a,b) in group.items():
            mask=p['compared'][:,:,:,a:b];t=totals.setdefault(g,Counter());t['expected']+=int(domain[:,:,a:b].sum())
            for k in range(3):t['compared_'+str(k)]+=int(mask[k].sum())
            for name,x,y in [('retry',0,1),('yolo',1,2)]:
                t['added_'+name]+=int((mask[y]&~mask[x]).sum());t['lost_'+name]+=int((mask[x]&~mask[y]).sum())
        counts['compared_channel_cells']+=p['compared'].size
    assert {k:dict(v) for k,v in totals.items()}==compare['groups']
    check(f['protected']);check(f['pins'])
    write(U/'VERIFIED.json',dict(at=now(),status='PASS',method='Separate Codex arithmetic via complex similarity and independent per-angle/path loops; NOT independent human ground truth.',checks=dict(counts),protected=len(f['protected']),pins=len(f['pins'])))
    print(json.dumps(dict(counts)),flush=True)
if __name__=='__main__':main()
