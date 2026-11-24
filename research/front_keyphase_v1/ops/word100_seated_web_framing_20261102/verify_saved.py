"""Separate saved-array/arithmetic verification. No detector/model/DP execution."""
from pathlib import Path
from collections import Counter
import json,sys,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_multipeople_live_feedback_20261101'
sys.dont_write_bytecode=True;sys.path.insert(0,str(O.parent/'src'))
import adapter
from wis_lab.core.observation import Observation
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
obsmod=adapter._module('_framing_verify_observer',O/'word100_guide_video_feedback_20261028/motion_observer.py')
old=adapter._module('_framing_verify_old',O/'word100_body_boundary_20261017/boundary.py')
coef=read(O/'word100_body_boundary_20261017/design_data_COEFFICIENT.json')
common=adapter._module('_framing_verify_common',O/'word100_signrep_feasibility_20261029/common.py')
checks=[];records=[];stagecounts=Counter();featurecounts=Counter()
def check(name,v):assert v,name;checks.append(name)
ui=read(U/'LIVE_UI_VERIFICATION.json');check('browser PASS',ui['status']=='PASS')
for row in read(U/'INPUT_VALIDATION.json')['public_crops']:
    jobid=read(U/(row['clip_id']+'_UPLOAD.json'))['job_id']
    jobs=list((U/'sessions').glob('*/'+jobid));check(row['clip_id']+' unique job',len(jobs)==1)
    j=jobs[0];d=j/'adapter';request=read(j/'request.json');complete=read(d/'COMPLETE.json');result=read(j/'server_result.json')
    check(row['clip_id']+' submitted public pixels',sha(j/'submission.mp4')==row['video_sha256']==complete['source_sha256'])
    check(row['clip_id']+' target retained',request['word']==result['word']==row['word_id'])
    check(row['clip_id']+' not global rejected',complete['status']=='COMPLETE_RESEARCH_ESTIMATES' and len(result['references'])==9)
    check(row['clip_id']+' no learning/reference ingestion',not request['training_allowed'] and not request['reference_bank_addition_allowed'])
    check(row['clip_id']+' guard retained',result['evidence_guard']['applied'])
    ob=Observation.load(str(d/'observations.npz'));w,h=ob.sidecar['width'],ob.sidecar['height']
    full={k:getattr(ob,k).copy() for k in ('xy_px','usable','observed','timestamp_us','frame_id')}
    expected=adapter.roi_arrays(full,[0,0,w,h]);f=obsmod.extract_features(expected)
    with np.load(d/'analysis.npz',allow_pickle=False) as z:
        check(row['clip_id']+' all native frames/PTS',np.array_equal(z['frame_id'],ob.frame_id) and np.array_equal(z['timestamp_us'],ob.timestamp_us))
        check(row['clip_id']+' exact visible feature mask',np.array_equal(z['mask'],f['mask']))
        check(row['clip_id']+' exact visible feature values',np.array_equal(z['x'],np.nan_to_num(f['x']).astype(np.float32)))
        check(row['clip_id']+' exact temporal centers',np.array_equal(z['new_four_center_indices'],adapter.window_centers(ob.frame_id)))
        check(row['clip_id']+' old three predicted at every frame',np.isfinite(z['probabilities'][:,:3]).all())
    with np.load(d/'rgb_windows.npz',allow_pickle=False) as z:
        check(row['clip_id']+' all fixed 16frame RGB windows',np.array_equal(z['starts'],common.starts(ob.n_frames)))
    oldresult=old.calibrate(ob,w,h,coef)
    # Arithmetic of previous gate only: no old model inference is repeated.
    check(row['clip_id']+' no inferred cutoff in new path',complete['calibration']['box_source_pixels']==[0,0,w,h])
    first=ob.timestamp_us-ob.timestamp_us[0]<=1_000_000
    vis=expected['usable']&expected['observed'];hands=np.column_stack([vis[:,s:s+21].any(1) for s in (33,54)])
    for s in result['stages']:
        stagecounts[s['status']]+=1
        for x in s['features']:featurecounts[x['status']]+=1
    rec=dict(clip=row['clip_id'],label=row['label'],job_id=jobid,frames=ob.n_frames,
             seconds=read(j/'REQUEST_TIMING.json')['analysis_seconds'],old_gate_status=oldresult['status'],old_reason=oldresult['reason'],
             actual_sensor=[w,h],old_proposed_bottom=oldresult.get('proposed_bottom'),
             first_second_frames=int(first.sum()),first_second_any_visible_hand=int(hands[first].any(1).sum()),
             later_any_visible_hand=int(hands[~first].any(1).sum()),stage_statuses=[s['status'] for s in result['stages']])
    records.append(rec)
check('only four fresh public analyses',len(list((U/'sessions').glob('*/*/request.json')))==4)
for p,h in read(U/'CODE_MANIFEST.json')['base_pins'].items():check('old preserved '+p,sha(Path(p))==h)
report=dict(status='PASS',checks=len(checks),check_names=checks,records=records,stage_counts=dict(stagecounts),
            feature_counts=dict(featurecounts),new_inference=0,implementation='Codex',review='same Codex saved-array/code verification; no independent human labels',
            scope='input availability and execution correctness, not sign/educational accuracy')
with (U/'SAVED_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({k:report[k] for k in ['status','checks','records','stage_counts','feature_counts']},ensure_ascii=True))
