from pathlib import Path
import json
import numpy as np
from freeze import sha,U
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    plan=read(U/'FROZEN.json');http=read(U/'HTTP_COMPLETE.json');n=0
    for r in http['runs']:
        row=next(a for a in plan['rows'] if a['clip_id']==r['clip_id']);d=Path(r['artifact']);s=read(d/'SESSION.json')
        assert d.resolve().parent==(U/'http_runs').resolve()
        with np.load(row['saved_baseline']) as b,np.load(d/'observations.npz') as a,np.load(d/'analysis_input.npz') as f:
            for k in ['xy_px','score','frame_id','timestamp_us','frame_sha256','bbox','n_people']:
                if a[k].dtype.kind in 'f':assert np.array_equal(a[k],b[k],equal_nan=True),k
                else:assert np.array_equal(a[k],b[k]),k
            for k in ['frame_id','timestamp_us']:assert np.array_equal(f[k],a[k])
            n+=len(a['frame_id'])
        assert r['completed']==r['received']==r['expected_frames']==s['detector_calls']
        assert r['dropped']==r['reuse_detector_calls']==0 and r['feedback_model_ready'] is False
        assert r['engineering_targets_pass'] and not s['missing_input_id_ranges'] and not s['errors']
    for p,h in plan['protected'].items():assert sha(p)==h,p
    obj=dict(status='PASS',http_runs=len(http['runs']),frames=n,coordinates_scores_timestamps_pixel_hashes_unchanged=True,
             post_capture_ms=[min(r['last_capture_to_input_ready_ms'] for r in http['runs']),max(r['last_capture_to_input_ready_ms'] for r in http['runs'])],
             control_cases=http['controls'],verifier='same Codex saved-output comparison, not independent human reviewer',
             limitation='client RTT/raw send timestamps not separately stored; timing values reported by runner, not independently reconstructed; no browser/camera/game or feedback head')
    with (U/'HTTP_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    print(json.dumps(obj,ensure_ascii=False))
if __name__=='__main__':main()
