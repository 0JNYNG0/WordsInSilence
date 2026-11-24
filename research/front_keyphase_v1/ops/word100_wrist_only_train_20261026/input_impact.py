"""Aggregate verified sparse-wrist inputs and recorded detector time; no inference."""
from pathlib import Path
import sys,json,hashlib,datetime,re
from collections import Counter,defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf8')
J=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def timing(xs):
    return dict(n=len(xs),sum_ms=float(sum(xs)),median_ms=float(np.median(xs)) if xs else None,p95_ms=float(np.percentile(xs,95)) if xs else None)
def main(unit):
    u=Path(unit).resolve();f=read(u/'FROZEN.json');v=read(u/'ALL_VERIFIED.json')
    assert v['status']=='PASS' and v['frozen_sha256']==sha(u/'FROZEN.json') and not (u/'RUNNING.json').exists()
    dest=J/('TRAIN_INPUT_IMPACT.json' if 'dev_inputs' not in u.name else 'DEV_INPUT_IMPACT.json');assert not dest.exists()
    counts=defaultdict(Counter);groups=defaultdict(Counter);calls=defaultdict(list);frame_ms=defaultdict(list);rows=[];pins={str(p):sha(p) for p in (u/'FROZEN.json',u/'ALL_VERIFIED.json',Path(__file__))}
    for r in f['rows']:
        uid=r['uid'];view=r['view'];rec=read(u/'records'/(uid+'.json'));outp=u/'outputs'/(uid+'.npz');fp=u/'features'/(uid+'.npz')
        assert rec['outputs'][str(outp)]==sha(outp) and rec['outputs'][str(fp)]==sha(fp)
        with np.load(outp,allow_pickle=False) as z:
            base=z['usable'][0,:,33:].reshape(-1,2,21);root=z['wrist_usable'];orig=z['point_origin'][2,:,33:].reshape(-1,2,21)[:,:,0]
            assert not np.any(root & base.any(axis=2))
            n=int(root.sum());mp=int((root & (orig==2)).sum());yo=int((root & (orig==3)).sum());assert n==mp+yo==rec['new_wrists']
            a=dict(views=1,frames=len(base),observed_hand_frames_before=int(base.any(axis=2).sum()),absent_hand_frames_before=int((~base.any(axis=2)).sum()),added_wrist_frames=n,mp_wrist_frames=mp,yolo_wrist_frames=yo,eligible_frame_attempts=len(rec['frames']),left_added=int(root[:,0].sum()),right_added=int(root[:,1].sum()))
        temporal={};changed_rows=0
        with np.load(fp,allow_pickle=False) as new,np.load(r['info']['source_arrays']['channels'],allow_pickle=False) as old:
            for name in ('x_whole','x_context'):
                if name not in old:continue
                oldvalid=old[name][:,[195,216]];newvalid=new[name][:,[195,216]]
                assert np.all(newvalid>=oldvalid)
                inserted=int((newvalid>oldvalid).sum());assert inserted==rec['changed_feature_wrists'][name[2:]]
                temporal[name]=dict(frames=len(oldvalid),insertions=inserted,changed_frame_count=int(np.any(newvalid!=oldvalid,axis=1).sum()))
                a[name+'_insertions']=inserted
                if inserted:changed_rows+=1
        a['views_with_any_insertions']=int(changed_rows>0)
        a['views_with_whole_insertions']=int(temporal['x_whole']['insertions']>0)
        for frame in rec['frames']:
            uncached=[];cached=False
            for call in frame['calls']:
                if call['cached']:a['cached_calls']=a.get('cached_calls',0)+1;cached=True;continue
                t=float(call['elapsed_ms']);assert np.isfinite(t) and t>=0
                calls[view+'_'+call['detector']].append(t);uncached.append(t);a['new_calls']=a.get('new_calls',0)+1
            if uncached and not cached:frame_ms[view].append(sum(uncached))
        counts[view].update(a);groups[str(r['signer_id'])+'_'+view].update(a)
        rows.append(dict(uid=uid,recording_id=r['signer_id'],view=view,**a,temporal=temporal))
    c=Counter()
    for q in counts.values():c.update(q)
    assert c['views']==v['counts']['views']
    assert c['x_whole_insertions']+c.get('x_context_insertions',0)==v['counts']['wrist_feature_insertions']
    assert c['eligible_frame_attempts']==v['counts']['geometry_frames']
    assert c['new_calls']==v['counts']['new_calls_recorded'] and c['cached_calls']==v['counts']['cache_calls_verified']
    assert c['added_wrist_frames']==len(v['review'])
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),status='VERIFIED_INPUT_IMPACT_AGGREGATE',unit=str(u),total=dict(c),by_view={k:dict(x) for k,x in counts.items()},by_recording={k:dict(x) for k,x in groups.items()},detector_calls={k:timing(x) for k,x in calls.items()},fully_uncached_trigger_frame_extra_time={k:timing(x) for k,x in frame_ms.items()},rows=rows,pins=pins,limitations=['Counts are observations and input availability, not correct recovery or recognition accuracy.','An absent hand may be outside camera/ROI, intentionally resting, occluded, or a detector miss. These counts do not distinguish ground truth.','Added wrist bodyXY does not restore fingers or certify precise anatomical position.','Timing is recorded additional detector calls only; excludes original MediaPipe capture, video decode, feature construction, TCN, DTW and browser. Fully cached/mixed frames excluded from trigger-frame timing. Not end-to-end FPS.','Same Codex independent aggregation of frozen saved evidence; no new inference, training, threshold changes or user data.'])
    with dest.open('x',encoding='utf8') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],by_view=result['by_view'],timing=result['fully_uncached_trigger_frame_extra_time']),ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
