"""Body-only observed motion and paired static features. Root runs explicitly."""
from body_core import *
from transfer_local_timefix_core import encode_windows,normalized_pose

def main():
    B.mkdir(parents=True,exist_ok=True)
    assert not(B/'PREPARATION_CONTRACT.json').exists()
    assert not(B/'INPUTS_COMPLETE.json').exists()
    stop_if_requested();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows,data=source_labeled_data();assert len(rows)==286
    assert sum(r['origin']=='positive_source_extent' for r in rows)==190
    assert sum(r['origin']=='reviewed_local_hold' for r in rows)==96
    assert all(r['family'] in FAMILIES for r in rows)
    fit={r['source_sha256'] for r in rows if r['split']=='train'};probe={r['source_sha256'] for r in rows if r['split']=='probe'}
    assert not fit&probe
    pinned=[Path(__file__),B/'body_core.py',P/'presence_core.py',N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',old.L/'INPUTS_COMPLETE.json',old.L/'FROZEN.json',old.L/'TRAINING_COMPLETE.json',S/'hand_information_contract/RESULTS.json',PRIOR/'local_rank_core.py',PRIOR/'transfer_local_timefix_core.py',HERE/'absence_review/REVIEW_COMPLETE.json',HERE/'absence_review/extension/REVIEW_COMPLETE.json']
    cfg=dict(at=stamp(),purpose='Separate visible body progression from handshape input; preserve old extent and handshape judgement.',
        scope='families3/4/5/7 observed body progression only; family2 unchanged',
        positive='Reuse all190 same-candidate local_no_hands caches; first candidate in each bag reencoded to verify raw-view contract.',
        hold='Reencode96 reviewed local hold bags inside their original source_span using the same candidate spans; remove hands before frozen encoder.',
        static='All known full/partial positive candidates times start/mid/end constant raw pose and mask; actual dt retained; recompute q/normalization then remove hands.',
        observation='Invisible relevant-body anchors excluded before learning/evaluation; not taught as negative.',
        preservation='Body18, body masks186:195, dt279 and explicit q-body retained; hand coordinates/masks/q-hand removed before shared encoder. Old rank/no-hands rank failure concerns extent identity, not this separate presence task.',
        no_completion_label=True,private_access=False,final8=False,web_changed=False,
        pins={str(p):sha(p) for p in pinned})
    write(B/'PREPARATION_CONTRACT.json',cfg)
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit'}
    crop={r['clip_id']:r for p in [old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json'] for r in read(p)['rows']}
    rc=old.load('_body_progression_role',OPS/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=old.encoder()
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_body_progression_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_body_progression_adapter',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cache={};inputpins={};body_bags=[];static_bags=[];bodyids=body_relation_indices()
    def raw_view(r):
        key=r['clip_id'],r['view']
        if key not in cache:
            sr=ready[r['clip_id']];assert sr['person']==r['person'] and r['person'] in [2,3,4,5,6,7,8,9]
            for path,h in [(sr['path'],sr['sha256']),(sr['channel'],sr['channel_sha256'])]:assert sha(path)==h;inputpins[path]=h
            z=np.load(sr['path']);ch=np.load(sr['channel']);ts=z['timestamp_us']
            if r['view']=='actual_crop':
                cr=crop[r['clip_id']];assert sha(cr['path'])==cr['sha256'];inputpins[cr['path']]=cr['sha256'];cz=np.load(cr['path'])
                assert np.array_equal(ts,cz['timestamp_us'])
                cl=guard.quarantine(cz['xy_px'],cz['score'],cz['timestamp_us'],640,480)
                xx=ad.convert(cl['xy_px'],cz['score'],cz['timestamp_us'],cz['frame_id'],640,480)['tcn280']
            elif r['view']=='full':xx=ch['full']
            else:
                assert r['view']=='upper';xx=ch['feedback'] if 'feedback' in ch.files else ch['upper_mask']
            assert len(xx)==len(ts) and np.isfinite(xx).all()
            cache[key]=xx
        xx=cache[key].copy()
        if r['reverse']:
            dt=xx[1:,279][::-1].copy();xx=xx[::-1].copy();xx[:,279]=np.r_[0,dt]
        return xx
    def normalize(raw):
        q=rc.old.relation_features(raw);qn=(q-jn.qm.numpy())/jn.qs.numpy()
        qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(-1,100)
        return normalized_pose(raw),qn
    def encode(px,qq,spans):
        out=[]
        for off in range(0,len(spans),128):
            stop_if_requested();out.extend(encode_windows(base,px,qq,spans[off:off+128]))
        return np.asarray(out,np.float32)
    for i,(r,d) in enumerate(zip(rows,data)):
        stop_if_requested();assert sha(r['path'])==r['sha256'];inputpins[r['path']]=r['sha256'];source=np.load(r['path']);raw=raw_view(r)
        ispos=r['origin']=='positive_source_extent'
        if not ispos:
            assert not r['reverse'];a,b=map(int,r['source_span']);assert 0<=a<b<len(raw)
            assert np.array_equal(source['source_frames'],np.arange(a,b+1))
            raw=raw[a:b+1].copy();raw[0,279]=0
        spans=d['spans'];assert len(spans)>0 and (spans[:,0]>=0).all() and (spans[:,1]<len(raw)).all()
        pose,q=normalize(raw);px,qq=old.drop_hand_information(pose,q)
        assert np.array_equal(px[:,:18],pose[:,:18]) and np.array_equal(px[:,186:195],pose[:,186:195]) and np.array_equal(px[:,279],pose[:,279])
        qr=q.reshape(-1,10,10);br=qq.reshape(-1,10,10);assert np.array_equal(qr[:,:,:4],br[:,:,:4]) and np.array_equal(qr[:,:,8],br[:,:,8])
        if ispos:
            real=source['local_no_hands'].copy();err=float(np.max(np.abs(encode(px,qq,spans[:1])[0]-real[0])));assert err<1e-5,(i,err)
            full_error=None;feature_origin='reused_frozen_local_no_hands_cache'
        else:
            full_error=float(np.max(np.abs(encode(pose,q,spans[:1])[0]-d['x'][0])));assert full_error<1e-5,(i,full_error)
            real=encode(px,qq,spans);err=None;feature_origin='new_body_only_encoding_of_existing_reviewed_hold'
        check_body_features(real);assert real.shape==d['x'].shape
        assert np.array_equal(real[:,bodyids],d['x'][:,bodyids]),('explicit body relation changed',i)
        dest=B/'body_bags'/f'{i}.npz';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
        np.savez_compressed(dest,local_body=real,spans=spans,y=d['full'])
        fields={k:r[k] for k in ['clip_id','person','split','family','view','reverse','origin','source_sha256']}
        body_bags.append(dict(fields,bag_index=i,path=str(dest),sha256=sha(dest),prior_path=r['path'],prior_sha256=r['sha256'],source_span=r.get('source_span'),candidate_count=len(real),feature_origin=feature_origin,first_cached_body_max_error=err,first_cached_full_max_error=full_error))
        if ispos:
            known=np.flatnonzero(d['known']);assert len(known) and (d['y'][known]==1).all()
            blocks=[];relations=[];ss=[];mapping=[];offset=0
            for k in known:
                a,b=map(int,spans[k]);n=b-a+1
                for anchor,t in enumerate([a,(a+b)//2,b]):
                    xx=np.repeat(raw[t:t+1],n,axis=0);xx[:,279]=raw[a:b+1,279]
                    pp,qt=normalize(xx);bp,bq=old.drop_hand_information(pp,qt)
                    assert np.ptp(bp[:,:279],axis=0).max()==0 and np.ptp(bq,axis=0).max()==0
                    blocks.append(bp);relations.append(bq);ss.append([offset,offset+n-1]);offset+=n;mapping.append([int(k),anchor,t])
            sx=encode(np.concatenate(blocks),np.concatenate(relations),np.asarray(ss,np.int32));check_body_features(sx)
            assert sx.shape==(len(known)*3,1160)
            axis=old.AX[r['family']]*2;coverage=sx[:,64+axis*10+8];assert np.isin(coverage,[0,1]).all()
            mp=np.asarray(mapping,np.int32);eligible=coverage>.5;assert all(eligible[mp[:,0]==k].any() for k in known)
            sp=B/'static_bags'/f'{i}.npz';sp.parent.mkdir(exist_ok=True);assert not sp.exists()
            np.savez_compressed(sp,x=sx,mapping=mp,positive_indices=known,eligible=eligible)
            static_bags.append(dict(fields,bag_index=i,path=str(sp),sha256=sha(sp),positive_candidates=len(known),static_candidates=len(sx),eligible=int(eligible.sum()),excluded_unobserved=int((~eligible).sum())))
        write(B/'PREP_PROGRESS.json',dict(at=stamp(),completed=i+1,total=len(rows),body_bags=len(body_bags),static_bags=len(static_bags)))
        print('body progression inputs',i+1,len(rows),flush=True)
    check_pins(cfg['pins']);check_pins(inputpins)
    assert len(body_bags)==286 and len(static_bags)==190
    write(B/'INPUTS_COMPLETE.json',dict(at=stamp(),body_bags=body_bags,static_bags=static_bags,pins=inputpins,source_clips=len({r['clip_id'] for r in rows}),train_people=[2,3,4,5,6,7],probe_people=[8,9],upstream_probe_exposed=True,labels_unchanged=True,no_new_educational_labels=True,common100_changed=False,old_rank_changed=False,private_access=False,final8=False,web_changed=False))
    print('Body-only progression inputs complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('PREP');raise
