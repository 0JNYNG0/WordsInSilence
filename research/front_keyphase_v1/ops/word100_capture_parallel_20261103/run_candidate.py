"""Two frozen public candidate captures, compared to saved serial observations."""
from pathlib import Path
import json,hashlib,time,traceback,os
import numpy as np
U=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(pins):
    for p,h in pins.items():assert sha(p)==h,p

def compare(baseline,candidate):
    rows=[]
    for name in ['raw.npz','baseline.npz','observations.npz','assignment.npz','face.npz']:
        with np.load(baseline/name,allow_pickle=False) as old,np.load(candidate/name,allow_pickle=False) as new:
            rows.append(dict(artifact=name,check='key_set',equal=set(old.files)==set(new.files)))
            for key in sorted(set(old.files)|set(new.files)):
                if key not in old.files or key not in new.files:
                    rows.append(dict(artifact=name,key=key,equal=False,reason='missing key'));continue
                a,b=old[key],new[key]
                same_shape=a.shape==b.shape;same_dtype=a.dtype==b.dtype
                rec=dict(artifact=name,key=key,shape_equal=same_shape,dtype_equal=same_dtype,equal=False)
                if same_shape and same_dtype:
                    if a.dtype.kind in 'fc':
                        equal=(a==b)|(np.isnan(a)&np.isnan(b))
                        finite=np.isfinite(a)&np.isfinite(b)
                        rec['maximum_finite_absolute_difference']=float(np.max(np.abs(a[finite]-b[finite]),initial=0))
                        rec['nan_locations_equal']=bool(np.array_equal(np.isnan(a),np.isnan(b)))
                    else:equal=a==b
                    rec['equal']=bool(equal.all())
                    rec['unequal_elements']=int((~equal).sum())
                    rec['first_unequal_indices']=np.argwhere(~equal)[:8].tolist()
                rows.append(rec)
    trace_equal=read(baseline/'assignment_trace.json')==read(candidate/'assignment_trace.json')
    rows.append(dict(artifact='assignment_trace.json',check='full_value_equal',equal=trace_equal))
    # Only the generated NPZ container checksum may differ in sidecars. It is
    # independently validated against the corresponding artifact in each run.
    for name,array in [('baseline.sidecar.json','baseline.npz'),('observations.sidecar.json','observations.npz')]:
        a,b=read(baseline/name),read(candidate/name)
        assert a.pop('observation_sha256')==sha(baseline/array)
        assert b.pop('observation_sha256')==sha(candidate/array)
        rows.append(dict(artifact=name,check='semantic_fields',equal=a==b,
            excluded_fields=['observation_sha256'],container_hashes_individually_verified=True))
    return dict(rows=rows,all_equal=all(r['equal'] for r in rows),unequal_checks=sum(not r['equal'] for r in rows))

def main():
    assert not (U/'RUN_STARTED.json').exists(),'Inspect existing attempt; do not repeat captured frames'
    frozen=read(U/'EXECUTION_FROZEN.json');verify(frozen['pins'])
    assert read(U/'CODE_REVIEW.json')['approved_for_two_candidate_captures'] is True
    items=read(U/'INPUT_PLAN.json')['rows'];assert len(items)==2
    from parallel_capture import capture_parallel
    write(U/'RUN_STARTED.json',dict(pid=os.getpid(),at=time.time(),planned_candidate_captures=2,
        baseline_captures=0,signrep_forwards=0,feedback_head_forwards=0,new_training=0))
    rows=[]
    for item in items:
        cid=item['case_id'];assert cid in ['AK_1133_ID10','AA_0739_ID12']
        destination=U/'captures'/cid;destination.mkdir(parents=True,exist_ok=False)
        assert sha(item['source'])==item['source_sha256']
        write(destination/'RUNNER_CAPTURE_REQUEST.json',dict(case_id=cid,source_sha256=item['source_sha256'],at=time.time()))
        print('Candidate capture starting: '+cid,flush=True)
        started=time.perf_counter()
        try:
            ob,face=capture_parallel(item['source'],destination)
            elapsed=time.perf_counter()-started
            assert ob.frame_id.tolist()==item['frame_id'] and ob.timestamp_us.tolist()==item['timestamp_us']
            comparison=compare(Path(item['baseline_directory']),destination)
            write(destination/'ARRAY_COMPARISON.json',comparison)
            profile=read(destination/'CAPTURE_TIMING.json')
            row=dict(case_id=cid,output_directory=str(destination),source_sha256=item['source_sha256'],
                status='COMPLETE',capture_wall_seconds=elapsed,profile=profile,
                exact_observation_preserved=comparison['all_equal'],unequal_checks=comparison['unequal_checks'],
                native_frames=ob.n_frames,baseline_capture_seconds=item.get('baseline_capture_seconds'),
                timing_comparison_is_controlled=False,
                timing_note='Historical serial capture overlapped RGB and image-quality work; this sensor-only candidate has different contention.')
            write(destination/'CAPTURE_FINISHED.json',dict(status='complete',at=time.time(),exact_observation_preserved=comparison['all_equal']))
        except Exception as exc:
            (destination/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
            row=dict(case_id=cid,output_directory=str(destination),source_sha256=item['source_sha256'],
                status='FAILED',error=str(exc),exact_observation_preserved=False)
        rows.append(row)
        (U/'PROGRESS.json').write_text(json.dumps(dict(done=len(rows),total=2,last=cid,status=row['status'])),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='profile'},ensure_ascii=False),flush=True)
    verify(frozen['pins'])
    result=dict(status='TWO_FIXED_CANDIDATE_ATTEMPTS_COMPLETE',rows=rows,
        all_exact=all(r['status']=='COMPLETE' and r['exact_observation_preserved'] for r in rows),
        baseline_reruns=0,semantic_or_SignRep_forwards=0,training=0,new_DTW=0,
        speed_improvement_proven=False,live_changed=False,whole_goal_complete=False)
    write(U/'RESULT.json',result)
    write(U/'RUN_FINISHED.json',dict(status='complete',at=time.time(),result_sha256=sha(U/'RESULT.json')))
    print('Two fixed candidate attempts finished; all_exact='+str(result['all_exact']),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        with (U/('RUN_FAILURE_'+str(time.time_ns())+'.txt')).open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
        raise
