"""Independent saved-output audit: never imports RTMW/stream runtime or runs inference."""
from pathlib import Path
import json,hashlib,math
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    plan=read(U/'FROZEN.json');complete=read(U/'COMPLETE.json');audits=[]
    points=[5,6,7,8,9,10]+list(range(91,133));maxerr=0.;count=0
    for row in plan['rows']:
        assert sha(row['source'])==row['source_sha256'];assert sha(row['saved_baseline'])==row['saved_baseline_sha256']
        with np.load(row['saved_baseline'],allow_pickle=False) as z:old={k:z[k] for k in z.files}
        for rep in range(plan['repetitions']):
            d=U/'runs'/row['clip_id']/str(rep);s=read(d/'SESSION.json');r=read(d/'RESULT.json')
            with np.load(d/'observations.npz',allow_pickle=False) as z:new={k:z[k] for k in z.files}
            with np.load(d/'analysis_input.npz',allow_pickle=False) as z:features={k:z[k] for k in z.files}
            assert s['completed']==s['detector_calls']==len(new['frame_id'])
            assert s['received']==len(old['frame_id'])
            assert s['accepted']==s['completed']+len(s['errors'])
            assert s['accepted']+s['dropped']==s['received']
            ids=new['frame_id'];assert np.array_equal(ids,[a['frame_id'] for a in s['receipts'] if a['accepted']])
            assert np.array_equal(new['timestamp_us'],old['timestamp_us'][ids])
            assert np.array_equal(new['frame_sha256'],old['frame_sha256'][ids])
            assert np.array_equal(features['frame_id'],ids) and np.array_equal(features['timestamp_us'],new['timestamp_us'])
            assert r['reuse_detector_calls']==0 and s['whole_video_redetection'] is False
            assert np.allclose(new['xy_px'],old['xy_px'][ids],equal_nan=True,atol=1e-4,rtol=1e-5)==r['coordinate_preserved']
            lag=np.array([1000*(a['completed_s']-a['received_s']) for a in s['records']])
            assert abs(float(np.percentile(lag,95))-r['lag_p95_ms'])<1e-6
            assert (s['status']=='COMPLETE' and r['lag_p95_ms']<=plan['lag_p95_budget_ms'] and r['post_capture_to_reused_input_ms']<=plan['post_capture_budget_ms'])==r['engineering_targets_pass']
            w,h=int(new['source_width']),int(new['source_height'])
            for t,xy in enumerate(new['xy_px']):
                def valid(p):return bool(np.isfinite(p).all() and 0<=p[0]<w and 0<=p[1]<h)
                center=(xy[5].astype(float)+xy[6].astype(float))/2
                scale=math.hypot(float(xy[5,0]-xy[6,0]),float(xy[5,1]-xy[6,1]))
                ok=valid(xy[5]) and valid(xy[6]) and math.isfinite(scale) and scale>1
                for j,idx in enumerate(points):
                    good=ok and valid(xy[idx]);assert bool(features['body_mask'][t,j])==good
                    expected=(xy[idx]-center)/scale if good else np.zeros(2)
                    error=float(np.abs(features['body_xy'][t,j]-expected).max());maxerr=max(maxerr,error);assert error<5e-5;count+=1
                for hi,start in enumerate([91,112]):
                    palm=math.hypot(float(xy[start+5,0]-xy[start+17,0]),float(xy[start+5,1]-xy[start+17,1]))
                    ok=valid(xy[start]) and valid(xy[start+5]) and valid(xy[start+17]) and math.isfinite(palm) and palm>1
                    for j in range(21):
                        good=ok and valid(xy[start+j]);assert bool(features['hand_local_mask'][t,hi,j])==good
                        expected=(xy[start+j].astype(float)-xy[start].astype(float))/palm if good else np.zeros(2)
                        error=float(np.abs(features['hand_local_xy'][t,hi,j]-expected).max());maxerr=max(maxerr,error);assert error<5e-5;count+=1
            audits.append(r)
    for p,h in plan['protected'].items():assert sha(p)==h,p
    assert len(audits)==len(complete['runs'])==12
    totals=dict(runs=len(audits),unique_clips=len(plan['rows']),submitted_frames=sum(r['source_frames'] for r in audits),
        completed_frames=sum(r['completed'] for r in audits),dropped_frames=sum(r['dropped'] for r in audits),
        all_coordinate_preserved=all(r['coordinate_preserved'] for r in audits),
        all_engineering_targets_pass=all(r['engineering_targets_pass'] for r in audits),
        lag_p95_range_ms=[min(r['lag_p95_ms'] for r in audits),max(r['lag_p95_ms'] for r in audits)],
        post_capture_to_reused_input_range_ms=[min(r['post_capture_to_reused_input_ms'] for r in audits),max(r['post_capture_to_reused_input_ms'] for r in audits)],
        post_capture_to_reused_input_median_ms=float(np.median([r['post_capture_to_reused_input_ms'] for r in audits])),
        reuse_detector_calls=sum(r['reuse_detector_calls'] for r in audits),relative_scalar_checks=count,max_relative_delta=maxerr)
    report=dict(status='PASS',totals=totals,protected_files=len(plan['protected']),
        method='same Codex separate code arithmetic audit; no new inference; not independent human labels',
        limits=['public paced predecoded frame test; not actual browser webcam','coordinate preservation is not finger accuracy','feedback network not trained or evaluated'])
    with (U/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
