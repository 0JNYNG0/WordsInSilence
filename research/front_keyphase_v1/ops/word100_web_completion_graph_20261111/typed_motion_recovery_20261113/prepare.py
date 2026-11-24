"""Audit and repair the actual input, never relabel movement from predictions."""
from common import *
from collections import Counter
import time, traceback, os

def main():
    assert not (HERE/'INPUTS_COMPLETE.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=read(PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json')
    scoped=read(PRIOR/'boundary_direction_learning/SCOPED_INPUTS.json')
    for mm in [m,scoped]: assert sha(mm['path'])==mm['sha256']
    z=np.load(m['path']); orig=np.load(scoped['path']);rows=m['rows']
    assert np.array_equal(z['context'],orig['context']) and np.array_equal(z['y'],orig['y'])
    cfg=dict(at=stamp(),status='FROZEN_BEFORE_NEW_CANDIDATE_SCORES',hypothesis='Reversed geometry was paired with swapped endpoints of NORMAL temporal context; execution encodes the actually changed video. Correct this mismatch while preserving targets.',
      conditions=['reencoded_original_geometry','reencoded_projected_geometry'],
      saved_controls=['boundary_direction_learning','projected_direction_learning'],
      seeds=[42,43,44],steps=800,lr=.0003,weight_decay=.001,batch_per_origin=128,
      selection='final step only, all 3 seeds, no DEV checkpoint or threshold selection',
      transform='reflect x[a:b+1] and its PTS about endpoints, then re-encode full sequence; read NEW h[a],h[b],delta',
      labels='Preserve existing axes1..4 typed targets, holds and splits; axis0 remains excluded; no coordinate-derived labels',
      gate='At least preserve normal/speed and reverse/compound direction versus baseline and saved local_old in every seed; all other axes/shapes/spans frozen. This subtest cannot pass overall G2 alone.',
      caveats=['Synthetic reversal introduces splice boundaries, not natural novice errors.','Probe IDs08/09 and development fixtures exposed to upstream encoders; not independent final accuracy.','Hand position/shape annotation is Codex-reviewed, not expert gold.'],
      fit_people=list(range(1,8)),probe_people=[8,9],private_access=False,web_changed=False,final8=False,
      pins={str(p):sha(p)for p in [Path(__file__),HERE/'common.py',PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json',PRIOR/'boundary_direction_learning/SCOPED_INPUTS.json',PRIOR/'boundary_direction_core.py',PRIOR/'projected_direction_core.py',PRIOR/'transfer_local_timefix_core.py',OPS/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt']})
    assert not(HERE/'FROZEN.json').exists();write(HERE/'FROZEN.json',cfg)
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'}
    crop={r['clip_id']:r for q in [old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(q)['rows']}
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_tm_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_tm_adapter',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    base=old.encoder();contexts=[];cache={};revs={};pins={};audit=[];start=time.perf_counter()
    for i,r in enumerate(rows):
        cid=r['clip_id'];view=r['view'];key=(cid,view)
        if key not in cache:
            rr=ready[cid]
            for p,h in [(rr['path'],rr['sha256']),(rr['channel'],rr['channel_sha256'])]:assert sha(p)==h;pins[p]=h
            zz=np.load(rr['path']);ch=np.load(rr['channel']);ts=zz['timestamp_us']
            if view=='actual_crop':
                cr=crop[cid];assert sha(cr['path'])==cr['sha256'];aa=np.load(cr['path']);cl=guard.quarantine(aa['xy_px'],aa['score'],aa['timestamp_us'],640,480)
                x=ad.convert(cl['xy_px'],aa['score'],aa['timestamp_us'],aa['frame_id'],640,480)['tcn280'];pins[cr['path']]=cr['sha256'];assert np.array_equal(ts,aa['timestamp_us'])
            else:x=ch['full']if view=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
            cache[key]=(x,ts,old.hidden(base,old.normalized_pose(x)))
        x,ts,h=cache[key];a,b=r['a'],r['b'];normal=np.r_[h[a],h[b],h[b]-h[a]]
        if not r['reverse']:
            assert np.allclose(normal,z['context'][i],atol=1e-6),('normal-context mismatch',i)
            ctx=normal;diff=0.;geo_error=0.
        else:
            kk=(*key,a,b)
            if kk not in revs:
                xr,tr=reverse_interval(x,ts,a,b);hr=old.hidden(base,old.normalized_pose(xr));revs[kk]=(xr,tr,hr)
            xr,tr,hr=revs[kk];ctx=np.r_[hr[a],hr[b],hr[b]-hr[a]]
            # The geometry must describe EXACTLY the sequence encoded above.
            gp,ok,_=old.geometry(xr,tr,a,b,r['axis'],r['moving'],r['anchor'])
            go,oko,_=old.prior.geometry(xr,tr,a,b,r['axis'],r['moving'],r['anchor'])
            assert ok and oko
            geo_error=float(max(np.max(np.abs(gp-z['g'][i])),np.max(np.abs(go-orig['g'][i]))))
            assert geo_error<1e-5,('geometry reversal mismatch',i,geo_error)
            diff=float(np.max(np.abs(ctx-z['context'][i])))
        contexts.append(ctx);audit.append(dict(index=i,split=r['split'],reverse=r['reverse'],context_max_abs_difference=diff,geometry_max_abs_difference=geo_error))
        if i%100==0:
            write(HERE/'PREPARATION_PROGRESS.json',dict(at=stamp(),pid=os.getpid(),rows=i+1,total=len(rows),seconds=time.perf_counter()-start));print('input-contract',i+1,len(rows),flush=True)
    train=np.array([r['split']=='train'for r in rows]);assert not({r['source_sha256']for r in rows if r['split']=='train'}&{r['source_sha256']for r in rows if r['split']=='probe'})
    assert all(r['person']in(cfg['fit_people']if r['split']=='train'else cfg['probe_people'])for r in rows)
    path=HERE/'inputs.npz';np.savez_compressed(path,context=np.asarray(contexts),original_context=z['context'],original_geometry=orig['g'],projected_geometry=z['g'],y=z['y'])
    norms={name:dict(mean=arr[train].mean(0).tolist(),std=np.maximum(arr[train].std(0),.01).tolist())for name,arr in [('original',orig['g']),('projected',z['g'])]}
    aa=[a for a in audit if a['split']=='train'and a['reverse']]
    write(HERE/'INPUT_CONTRACT_AUDIT.json',dict(at=stamp(),rows=audit,normal_context_unchanged=True,geometry_and_labels_unchanged=True,fit_reverse_context_changed=sum(a['context_max_abs_difference']>1e-6 for a in aa),fit_reverse_count=len(aa),max_fit_context_difference=max(a['context_max_abs_difference']for a in aa),reviewer='root Codex with independent code review by motion_contract_review_1113; not independent human labels'))
    write(HERE/'INPUTS_COMPLETE.json',dict(at=stamp(),path=str(path),sha256=sha(path),rows=rows,norm=norms,pins=pins,source_meta=str(PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json'),source_sha256=sha(PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json'),seconds=time.perf_counter()-start))
    print('Prepared',len(rows),'rows; FIT reverse contexts changed',sum(a['context_max_abs_difference']>1e-6 for a in aa),'of',len(aa),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:(HERE/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
