from static_core import *
from transfer_local_timefix_core import encode_windows,normalized_pose
import time,traceback

def main():
    assert not(S/'PREPARATION_CONTRACT.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows,data=labeled_data();selected=[i for i,r in enumerate(rows)if r['origin']=='positive_source_extent']
    cfg=dict(at=stamp(),hypothesis='Presence trained on final holds also fires on constant observed coordinates. Add matched pose/duration static negatives for all known progression windows.',
        selection='Every existing full-compatible and >=2-witness partial candidate, all normal/reverse/full/upper/actual_crop conditions. No model-score selection.',
        transform='Repeat raw start/middle/end pose and masks of each candidate independently across its own length, preserve dt, recompute relations and frozen encoder in isolated window. Zero keypoint progression target only.',
        synthetic_scope='not wrong sign, missing action, educational completeness, or human static pose gold. Meaningful holds may be correct; classifier only reports change evidence.',
        original_positive_features='reuse existing saved1160 vectors; no old model rerun',private_access=False,final8=False,web_changed=False,
        pins={str(p):sha(p)for p in[Path(__file__),S/'static_core.py',P/'presence_core.py',P/'FROZEN.json',P/'TRAINING_COMPLETE.json',N/'TRAINING_INPUT_INDEX.json',N/'local_bag_evaluation/RULES_FROZEN.json',P/'static_contract_audit/RESULTS.json']})
    write(S/'PREPARATION_CONTRACT.json',cfg)
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'}
    crop={r['clip_id']:r for p in[old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']}
    rc=old.load('_static_aug_role',OPS/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=old.encoder()
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'));guard=old.load('_static_aug_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_static_aug_adapter',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cache={};pins={};out=[]
    for ii,i in enumerate(selected):
        r=rows[i];d=data[i];key=(r['clip_id'],r['view'])
        if key not in cache:
            sr=ready[r['clip_id']];assert sha(sr['path'])==sr['sha256']and sha(sr['channel'])==sr['channel_sha256'];pins[sr['path']]=sr['sha256'];pins[sr['channel']]=sr['channel_sha256'];ch=np.load(sr['channel']);ts=np.load(sr['path'])['timestamp_us']
            if r['view']=='actual_crop':
                cr=crop[r['clip_id']];assert sha(cr['path'])==cr['sha256'];pins[cr['path']]=cr['sha256'];z=np.load(cr['path']);clean=guard.quarantine(z['xy_px'],z['score'],z['timestamp_us'],640,480);x=ad.convert(clean['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],640,480)['tcn280']
            else:x=ch['full']if r['view']=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
            cache[key]=x
        x=cache[key].copy()
        if r['reverse']:
            dt=x[1:,279][::-1].copy();x=x[::-1].copy();x[:,279]=np.r_[0,dt]
        def normq(raw):
            q=rc.old.relation_features(raw);qn=(q-jn.qm.numpy())/jn.qs.numpy();return np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(-1,100)
        known=np.flatnonzero(d['known']);assert (d['y'][known]==1).all();spans=d['spans'][known];first=encode_windows(base,normalized_pose(x),normq(x),spans[:1])[0];err=float(np.max(np.abs(first-d['x'][known[0]])));assert err<1e-5,(r['clip_id'],r['view'],r['reverse'],err)
        blocks=[];qs=[];offset=0;synthetic_spans=[];mapping=[]
        for k,(a,b)in zip(known,spans):
            a,b=int(a),int(b);n=b-a+1
            for anchor,t in enumerate([a,(a+b)//2,b]):
                raw=np.repeat(x[t:t+1],n,axis=0);raw[:,279]=x[a:b+1,279];px=normalized_pose(raw);q=normq(raw)
                assert np.ptp(px[:,:279],axis=0).max()==0 and np.ptp(q,axis=0).max()==0
                blocks.append(px);qs.append(q);synthetic_spans.append([offset,offset+n-1]);offset+=n;mapping.append([int(k),anchor,t])
        xx=np.concatenate(blocks);qq=np.concatenate(qs);ss=np.array(synthetic_spans,np.int32);ff=[]
        for off in range(0,len(ss),128):ff.extend(encode_windows(base,xx,qq,ss[off:off+128]))
        ff=np.asarray(ff,np.float32);assert ff.shape==(len(known)*3,1160)and np.isfinite(ff).all()
        dest=S/'bags'/f'{i}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,x=ff,mapping=np.array(mapping,np.int32),positive_indices=known)
        out.append(dict(bag_index=i,path=str(dest),sha256=sha(dest),clip_id=r['clip_id'],person=r['person'],split=r['split'],family=r['family'],view=r['view'],reverse=r['reverse'],positive_candidates=len(known),static_candidates=len(ff),original_feature_max_error=err,source_sha256=r['source_sha256']))
        write(S/'PREP_PROGRESS.json',dict(at=stamp(),completed=ii+1,total=len(selected),static_candidates=sum(a['static_candidates']for a in out)));print('paired static features',ii+1,len(selected),len(ff),flush=True)
    write(S/'INPUTS_COMPLETE.json',dict(at=stamp(),bags=out,pins=pins,source_clips=len({r['clip_id']for r in out}),train_people=[2,3,4,5,6,7],probe_people=[8,9],upstream_probe_exposed=True,static_is_observation_counterfactual=True));print('Paired static inputs complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(S/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
