"""Public single-reference feasibility; derived negatives are NOT real learner accuracy."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from coach import Coach
U=Path(__file__).resolve().parent

def run():
    raw=dict(np.load(U/'source/reference.npz'));model=Coach();cases=[]
    def copied():return {k:v.copy()for k,v in raw.items()}
    a=copied();cases.append(('native_reference',a,[True,True]))
    a=copied()
    for k in ['xy_px','score','timestamp_us','frame_id']:a[k]=a[k][:49]
    cases.append(('first_stage_only',a,[True,False]))
    a=copied();a['xy_px'][34:46]=a['xy_px'][34:46][::-1];a['score'][34:46]=a['score'][34:46][::-1]
    cases.append(('reverse_sweep_frames',a,[False,False]))
    a=copied();a['xy_px'][51:]=a['xy_px'][51];a['score'][51:]=a['score'][51]
    cases.append(('hold_two_fists_without_moving',a,[True,False]))
    a=copied();a['xy_px'][57:]=a['xy_px'][56];a['score'][57:]=a['score'][56]
    cases.append(('one_downstroke_then_hold',a,[True,False]))
    a=copied();a['xy_px'][37:44,91:]=np.nan
    cases.append(('missing_hands_during_sweep',a,[False,False]))
    a=copied();a['xy_px'][:]=np.nan
    cases.append(('all_points_missing',a,[False,False]))
    for mode,factor in [('fast_timestamps',.65),('slow_timestamps',1.5)]:
        a=copied();a['timestamp_us']=(a['timestamp_us']*factor).astype(np.int64);cases.append((mode,a,[True,True]))
    a=copied();a['xy_px']=a['xy_px']*.75+np.array([90,40],np.float32);cases.append(('scale_and_translation',a,[True,True]))
    a=copied()
    for i in range(49,80):
        for start in [91,112]:
            origin=a['xy_px'][i,start].copy();template=raw['xy_px'][38,start:start+21]-raw['xy_px'][38,start]
            a['xy_px'][i,start:start+21]=origin+template
    cases.append(('open_hands_during_required_fists',a,[True,False]))
    a=copied()
    for i in range(34,46):
        origin=a['xy_px'][i,112].copy();a['xy_px'][i,112:]=origin+raw['xy_px'][55,112:]-raw['xy_px'][55,112]
    cases.append(('closed_hand_during_required_sweep',a,[False,False]))
    rows=[]
    for name,z,expected in cases:
        model.reset();outputs=[];t=time.perf_counter()
        for i,ts in enumerate(z['timestamp_us']):
            out=model.update(z['xy_px'][i],z['score'][i],int(ts),int(z['source_width']),int(z['source_height']))
            out.pop('assessed_xy',None);outputs.append(out)
        elapsed=time.perf_counter()-t
        first=[next((i for i,o in enumerate(outputs)if o['completed'][s]),None)for s in [0,1]]
        row=dict(case=name,expected=expected,actual=model.completed.copy(),passed=model.completed==expected,
            frames=len(outputs),seconds=elapsed,first_completion_received_frame=first,
            events=model.events,correction_frames=sum(o['phase']=='correction'for o in outputs),
            unknown_frames=sum(o['phase']=='unknown'for o in outputs))
        rows.append(row)
        if name=='native_reference':(U/'source/causal_reference.json').write_text(json.dumps(dict(rows=outputs,seconds=elapsed,completion=model.completed.copy()),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    result=dict(code_sha256=hashlib.sha256((U/'coach.py').read_bytes()).hexdigest(),rows=rows,
        passes=sum(r['passed']for r in rows),total=len(rows),training=False,private_data=False,
        accuracy=None,independent_signer_test=False,
        scope='One approved public reference plus controlled derived observations. Timestamp transforms are not real camera speed/fps/drop tests. Synthetic hand substitutions are safety challenges, not real learner videos.',
        runtime_changed=False)
    (U/'source/PUBLIC_COACH_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':run()
