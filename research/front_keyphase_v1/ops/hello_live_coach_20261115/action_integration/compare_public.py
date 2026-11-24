"""Same public observations before/after; controlled challenges, not learner accuracy."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
V=Path(__file__).resolve().parent;U=V.parent;sys.path.insert(0,str(U))
from coach import Coach

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    suffix='_FT'if '--candidate'in sys.argv else '_V2'if '--v2'in sys.argv else''
    out=V/f'PUBLIC_COMPARISON{suffix}.json'
    assert not out.exists(),'Preserve previous runs; archive explicitly before a revised comparison.'
    spec=importlib.util.spec_from_file_location('coach_baseline_frozen',V/'before/coach.py')
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    with np.load(U/'source/reference.npz')as z:raw={k:z[k].copy()for k in z.files}
    before=baseline.Coach()
    if '--candidate'in sys.argv:
        from action_runtime import ActionObserver
        after=Coach(model=before.model,action=ActionObserver(candidate_path=V/'finetune/CANDIDATE.pt'))
    else:after=Coach(model=before.model)
    rows=[]
    def copied():return {k:v.copy()for k,v in raw.items()}
    cases=[]
    cases.append(('native_reference',copied(),[True,True]))
    a=copied()
    for k in ['xy_px','score','timestamp_us','frame_id']:a[k]=a[k][:49]
    cases.append(('first_stage_only',a,[True,False]))
    a=copied();a['xy_px'][34:46]=a['xy_px'][34:46][::-1];a['score'][34:46]=a['score'][34:46][::-1]
    cases.append(('reverse_sweep_frames',a,[False,False]))
    for name,frame in [('hold_two_fists',51),('one_downstroke_then_hold',56)]:
        a=copied();a['xy_px'][frame+1:]=a['xy_px'][frame];a['score'][frame+1:]=a['score'][frame]
        cases.append((name,a,[True,False]))
    a=copied();a['xy_px'][37:44,91:]=np.nan;cases.append(('missing_sweep_hands',a,[False,False]))
    a=copied();a['xy_px'][:]=np.nan;cases.append(('all_missing',a,[False,False]))
    for name,factor in [('fast_timestamps',.65),('slow_timestamps',1.5)]:
        a=copied();a['timestamp_us']=(a['timestamp_us']*factor).astype(np.int64);cases.append((name,a,[True,True]))
    a=copied();a['xy_px']=a['xy_px']*.75+np.array([90,40],np.float32);cases.append(('scale_translation',a,[True,True]))
    a=copied()
    for i in range(49,80):
        for start in [91,112]:
            a['xy_px'][i,start:start+21]=a['xy_px'][i,start]+raw['xy_px'][38,start:start+21]-raw['xy_px'][38,start]
    cases.append(('open_hands_in_fists_motion',a,[True,False]))
    a=copied()
    for i in range(34,46):a['xy_px'][i,112:]=a['xy_px'][i,112]+raw['xy_px'][55,112:]-raw['xy_px'][55,112]
    cases.append(('closed_sweep_hand',a,[False,False]))
    for side,pose in [(91,9),(112,10)]:
        a=copied();a['xy_px'][49:80,side:side+21]=raw['xy_px'][51,side:side+21];a['xy_px'][49:80,pose]=raw['xy_px'][51,pose]
        cases.append((f'held_hand_{side}',a,[True,False]))
    for step in [2,3]:
        a=copied()
        for k in ['xy_px','score','timestamp_us','frame_id']:a[k]=a[k][::step]
        cases.append((f'subsample_{30//step}fps',a,[True,True]if step==2 else [False,False]))
    pins={f:sha(U/f)for f in ['coach.py','action_runtime.py']}
    for name,r,expected in cases:
        row=dict(case=name,expected=expected,input_frames=len(r['timestamp_us']))
        for key,model in [('before',before),('after',after)]:
            model.reset();times=[];outputs=[]
            for i,ts in enumerate(r['timestamp_us']):
                tick=time.perf_counter();o=model.update(r['xy_px'][i],r['score'][i],int(ts),int(r['source_width']),int(r['source_height']))
                times.append((time.perf_counter()-tick)*1000)
                o.pop('assessed_xy',None);outputs.append(o)
            row[key]=dict(completed=model.completed.copy(),events=model.events,
                first_completion_received_frame=[next((i for i,o in enumerate(outputs)if o['completed'][s]),None)for s in [0,1]],
                update_ms_median=float(np.median(times[6:])),update_ms_p95=float(np.quantile(times[6:],.95)),
                uncertain_frames=sum(o['phase']in ['unknown','action_uncertain']for o in outputs),
                model_calls=(outputs[-1].get('action_model')or{}).get('calls',0))
            if name=='native_reference'and key=='after':
                (V/f'PUBLIC_TRACE{suffix}.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
                row[key]['action_update_ms_median']=float(np.median([(o.get('action_model')or{}).get('compute_ms',0)for o in outputs[6:]]))
        row['pass']=row['after']['completed']==expected
        rows.append(row);print(json.dumps(dict(case=name,before=row['before']['completed'],after=row['after']['completed'],passed=row['pass'],frames=row['after']['first_completion_received_frame']),ensure_ascii=False),flush=True)
    report=dict(rows=rows,passes=sum(r['pass']for r in rows),total=len(rows),
        code_hashes=pins,code_unchanged=all(sha(U/f)==h for f,h in pins.items()),raw_sha256=sha(U/'source/reference.npz'),
        model_hash=sha(U.parent/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'),
        scope=__doc__,timing='Cached public RTMW input -> handshape+action+controller CPU. Excludes RTMW extraction, webcam, network, image encoding and rendering.',
        training=False,personal_data=False,physical_camera_opened=False,independent_person_accuracy=None,
        timestamp_cases='Changes timestamp spacing, not real camera performance.',whole_word_accuracy=None)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passes=report['passes'],total=report['total']),ensure_ascii=False))
if __name__=='__main__':main()
