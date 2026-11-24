"""Local, continuously reviewed non-progression windows, not whole-sign omissions."""
from common import *
from transfer_local_timefix_core import encode_windows
from collections import Counter
import time,traceback
N=HERE/'null_candidate_learning'
def main():
    N.mkdir(exist_ok=True);assert not(N/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    review_paths=[HERE/'absence_review/REVIEW_COMPLETE.json',HERE/'absence_review/extension/REVIEW_COMPLETE.json']
    assert all(p.exists()for p in review_paths)
    rows=[r for p in review_paths for r in read(p)['rows']]
    cfg=dict(at=stamp(),purpose='Detect actual non-progression in locally reviewed hold intervals. Not linguistic omission/completion and not whole-video absence.',reviewed_families=[3,4,5,7],unsupported_null_family=2,fit_people=[2,3,4,5,6,7],probe_people=[8,9],candidate_generator='All same body-visible intervals at least 4 frames,80% observation and visible endpoints, restricted to continuously RGB-reviewed hold region. No unobserved=negative.',normal_bags=str(old.L/'INPUTS_COMPLETE.json'),gate='Do not retrain a null selector without observed false selections on FIT local hold bags. Probe outputs withheld until training contract fixed.',pins={str(p):sha(p)for p in review_paths+[Path(__file__),HERE/'common.py',old.L/'INPUTS_COMPLETE.json',old.B/'TRAINING_COMPLETE.json',old.L/'TRAINING_COMPLETE.json']},web_changed=False,private_access=False,final8=False)
    write(N/'PREPARATION_CONTRACT.json',cfg)
    rc=old.load('_null_role',OPS/'word100_candidate_role_learning_20261111/role_core.py');jn=rc.frozen_model();base=old.encoder()
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_null_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_null_adapter',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'}
    crop={r['clip_id']:r for q in [old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(q)['rows']}
    heads={};ranks={}
    for seed in [42,43,44]:
        for cls,folder,out in [(old.BodyWindow,old.B/'runs/body_window',heads),(old.LocalRankHead,old.L/'runs/local_full',ranks)]:
            p=folder/str(seed)/'model.pt';st=torch.load(p,weights_only=True,map_location='cpu');m=cls(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();out[seed]=m
    result=[];diagnostic=[];pins={};skips=[]
    for r in rows:
        review=r['review'];span=review.get('hold_span')
        if not span or len(span)!=2:skips.append(dict(clip_id=r['clip_id'],reason='no_verified_continuous_hold'));continue
        a,b=span;family=review['target_family'];assert family in cfg['reviewed_families'];assert review['all_frames_inspected'];assert set(range(a,b+1))<=set(review['inspected_frames'])
        assert not(set(range(a,b+1))&set(review['uncertain_frames']))
        for key in ['source','contact_sheet']:
            assert sha(r[key])==r[key+'_sha256'];pins[r[key]]=r[key+'_sha256']
        cid=r['clip_id'];rr=ready[cid];assert rr['person']==r['person']
        for p,h in [(rr['path'],rr['sha256']),(rr['channel'],rr['channel_sha256'])]:assert sha(p)==h;pins[p]=h
        z=np.load(rr['path']);ch=np.load(rr['channel']);ts=z['timestamp_us'];assert np.array_equal(z['frame_id'][a:b+1],np.arange(a,b+1))
        views={'full':ch['full'],'upper':ch['feedback']if'feedback'in ch.files else ch['upper_mask']}
        if cid in crop:
            cr=crop[cid];assert sha(cr['path'])==cr['sha256'];cz=np.load(cr['path']);cl=guard.quarantine(cz['xy_px'],cz['score'],cz['timestamp_us'],640,480)
            views['actual_crop']=ad.convert(cl['xy_px'],cz['score'],cz['timestamp_us'],cz['frame_id'],640,480)['tcn280'];pins[cr['path']]=cr['sha256'];assert np.array_equal(ts,cz['timestamp_us'])
        for view,full in views.items():
            x=full[a:b+1].copy();x[0,279]=0;t=ts[a:b+1];q=rc.old.relation_features(x);qn=(q-jn.qm.numpy())/jn.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1).reshape(-1,100)
            axis=old.AX[family]*2;valid=q[:,axis,8]>.5;n=len(x)
            spans=np.array([(aa,bb)for aa in range(n-3)for bb in range(aa+3,n)if valid[aa]and valid[bb]and valid[aa:bb+1].mean()>=.8],np.int32).reshape(-1,2)
            if not len(spans):skips.append(dict(clip_id=cid,view=view,reason='insufficient_body_observation_not_negative'));continue
            bf=old.pool_windows(old.body_context(qn,t),spans);lf=[]
            for off in range(0,len(spans),128):lf.extend(encode_windows(base,old.normalized_pose(x),qn,spans[off:off+128]))
            lf=np.asarray(lf,np.float32);assert np.isfinite(lf).all()and np.isfinite(bf).all()
            p=N/'bags'/f'{cid}_{view}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,local_full=lf,body_features=bf,spans=spans,y=np.zeros(len(spans),bool),source_frames=np.arange(a,b+1))
            row=dict(clip_id=cid,person=r['person'],split='probe'if r['person']in[8,9]else'train',view=view,reverse=False,family=family,path=str(p),sha256=sha(p),source_sha256=r['source_sha256'],source_span=[a,b],origin='reviewed_local_hold',candidate_count=len(spans),semantic_completion=None)
            result.append(row)
            if row['split']=='train':
                j=old.F.index(family)
                for seed in [42,43,44]:
                    with torch.inference_mode():wl=heads[seed](torch.tensor(bf)).numpy()[:,j,0];rl=ranks[seed](torch.tensor(lf)).numpy()[:,j]
                    score=np.where(wl>=0,rl,-np.inf);i=int(score.argmax())if np.isfinite(score).any()else None
                    diagnostic.append(dict(clip_id=cid,view=view,seed=seed,family=family,eligible_candidates=int((wl>=0).sum()),chosen=None if i is None else spans[i].tolist(),false_progression_selection=i is not None))
        print('hold features',cid,flush=True)
    assert not({r['source_sha256']for r in result if r['split']=='train'}&{r['source_sha256']for r in result if r['split']=='probe'})
    write(N/'INPUTS_COMPLETE.json',dict(at=stamp(),bags=result,pins=pins,skipped=skips,counts=dict(Counter(r['split']for r in result)),original_reviews=[dict(path=str(p),sha256=sha(p))for p in review_paths],probe_predictions_computed=False,labels_not_human_gold=True))
    write(N/'FIT_BASELINE_AUDIT.json',dict(at=stamp(),rows=diagnostic,false_selection_count=sum(r['false_progression_selection']for r in diagnostic),total=len(diagnostic),training_justified=any(r['false_progression_selection']for r in diagnostic),scope='local hold bags only, not whole sign omitted',no_probe_scoring=True))
    print('FIT hold false selections',sum(r['false_progression_selection']for r in diagnostic),'/',len(diagnostic),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:N.mkdir(exist_ok=True);(N/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
