from presence_core import *
from transfer_local_timefix_core import encode_windows,normalized_pose
import time,traceback

def main():
    D=P/'static_contract_audit';D.mkdir(exist_ok=True);assert not(D/'FROZEN.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows=read(N/'TRAINING_INPUT_INDEX.json')['rows']; selected=[(i,r)for i,r in enumerate(rows)if r['split']=='train'and r['view']=='full'and not r['reverse']and r['origin']=='positive_source_extent']
    assert len(selected)==24 and {r['person']for _,r in selected}=={2,3,4,5,6,7}
    plan=[]
    for i,r in selected:
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);k=int(np.flatnonzero(z['y'])[0]);a,b=map(int,z['spans'][k]);plan.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],person=r['person'],span=[a,b],candidate_index=k,anchors=[a,(a+b)//2,b],input_path=r['path'],input_sha256=r['sha256']))
    cfg=dict(at=stamp(),scope='FIT-only synthetic input-contract diagnostic, no normal-sign/educational error claim',selection='All24 FIT full-view forward source bags; first pre-existing positive candidate, repeat its start/middle/end observation separately. No prediction-dependent selection.',
        transform='Repeat one raw observed pose frame throughout same isolated candidate duration; preserve real elapsed time channel279. Recompute normalized pose and relation features from this raw constant sequence, then reencode. No outside frames or retained velocity features.',
        expected='No evidence of observed progression in constant-keypoint inputs. Original positive logits reused from saved training predictions; only new constant-input model calls.',
        limitation='Synthetic held detections; not real camera jitter/occlusion or proof of physical static hands. May expose posture/encoder-position shortcut, does not alone isolate its cause.',
        plan=plan,pins={str(p):sha(p)for p in[Path(__file__),P/'TRAINING_COMPLETE.json',P/'FROZEN.json',N/'TRAINING_INPUT_INDEX.json',PRIOR/'transfer_local_timefix_core.py']},private_access=False,final8=False,probe_access=False,web_changed=False)
    write(D/'FROZEN.json',cfg)
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'}
    rc=old.load('_static_role',OPS/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=old.encoder();features=[];items=[];inputpins={}
    for p in plan:
        sr=ready[p['clip_id']];assert sha(sr['channel'])==sr['channel_sha256'];inputpins[sr['channel']]=sr['channel_sha256'];x=np.load(sr['channel'])['full'];q=rc.old.relation_features(x);qn=(q-jn.qm.numpy())/jn.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(-1,100);pose=normalized_pose(x);a,b=p['span'];n=b-a+1
        original=encode_windows(base,pose,qn,np.array([[a,b]]))[0];saved=np.load(p['input_path'])['local_full'][p['candidate_index']];delta=float(np.max(np.abs(original-saved)));assert delta<1e-5,(p['clip_id'],delta)
        for label,t in zip(['start','middle','end'],p['anchors']):
            raw=np.repeat(x[t:t+1],n,axis=0);raw[:,279]=x[a:b+1,279];xx=normalized_pose(raw)
            cq=rc.old.relation_features(raw);qq=(cq-jn.qm.numpy())/jn.qs.numpy();qq=np.concatenate([qq[...,:4]*cq[...,8,None],qq[...,4:8]*cq[...,9,None],cq[...,8:]],-1).reshape(-1,100)
            assert np.ptp(xx[:,:279],axis=0).max()==0 and np.ptp(qq,axis=0).max()==0
            f=encode_windows(base,xx,qq,np.array([[0,n-1]]))[0];features.append(f);items.append(dict(p,anchor=label,anchor_frame=t,original_feature_max_error=delta,coordinates_and_masks_constant=True))
    features=np.asarray(features);np.savez_compressed(D/'features.npz',x=features);out=[]
    for run in read(P/'TRAINING_COMPLETE.json')['rows']:
        assert sha(run['model'])==run['model_sha256']and sha(run['predictions'])==run['predictions_sha256'];st=torch.load(run['model'],weights_only=True,map_location='cpu');m=ProgressionHead(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();before=np.load(run['predictions'])
        with torch.inference_mode():pred=m(torch.tensor(features)).numpy()
        for i,r in enumerate(items):
            j=old.F.index(r['family']);value=float(pred[i,j]);oldvalue=float(before[f"logits_{r['bag_index']}"][r['candidate_index'],j]);out.append(dict(r,seed=run['seed'],original_logit=oldvalue,static_logit=value,original_progression=oldvalue>=0,static_false_progression=value>=0))
    summary=[]
    for seed in [42,43,44]:
        rr=[r for r in out if r['seed']==seed];summary.append(dict(seed=seed,source_clips=24,static_controls=len(rr),original_positive_controls=sum(r['original_progression']for r in rr),static_false_progression=sum(r['static_false_progression']for r in rr)))
    write(D/'RESULTS.json',dict(at=stamp(),rows=out,summary=summary,input_pins=inputpins,feature_sha256=sha(D/'features.npz'),probe_access=False,new_educational_labels=False))
    for r in summary:print(r,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(P/f'STATIC_AUDIT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
