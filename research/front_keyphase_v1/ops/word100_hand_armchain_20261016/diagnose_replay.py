"""Describe already-computed trace prerequisites; never execute/adjust assigner."""
from pathlib import Path
from collections import Counter
import json,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
counts=Counter();held=Counter();rows=[]
for rec in read(U/'REPLAY_RESULT.json')['rows']:
    cid,mode=rec['clip_id'],rec['mode'];d=U/'replay'/cid/mode
    trace=read(d/'trace.json')
    with np.load(Path(rec['raw_dir'])/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    rc=Counter()
    for f,t in enumerate(trace):
        arm=t['arm_chain'];prediction_sides={p['side'] for p in arm['predictions']}
        rc['frames']+=1;rc['with_any_arm_prediction']+=bool(prediction_sides)
        rc['with_both_arm_predictions']+=len(prediction_sides)==2
        rc['arm_anchor_seed_events']+=len(arm['seeded_sides'])
        rc['arm_anchor_expiry_events']+=len(arm['expired_sides'])
        rc['arm_recoveries']+=len(arm['recoveries'])
        rc['raw_detections']+=int(raw['raw_count'][f])
        rc['zero_raw_frames']+=int(raw['raw_count'][f])==0
        for r in arm['rejected']:
            for why in r['reasons']:held['explicit_reject_'+why]+=1
        for i,s in enumerate(t['baseline_assigned_sides']):
            if s is not None:continue
            lab=int(raw['raw_label'][f,i]);held['baseline_unassigned_raw']+=1
            held['baseline_reason_'+t['baseline_reasons'][i]]+=1
            held['label_side_has_arm_prediction' if lab in prediction_sides else 'label_side_has_no_arm_prediction']+=1
    counts.update(rc);rows.append(dict(clip_id=cid,label=rec['label'],mode=mode,counts=dict(rc)))
targets=[('NIA_SL_WORD0651_REAL03_F','crop',[56,57,58]),
 ('NIA_SL_WORD0766_REAL03_F','crop',[45,46,47]),('NIA_SL_WORD2721_REAL02_F','crop',[42,43,44]),
 ('NIA_SL_WORD1139_REAL01_F','native',[71,72]),('NIA_SL_WORD1139_REAL01_F','crop',[46,47,55,56,57]),
 ('NIA_SL_WORD0106_REAL01_F','crop',[50,51,52,53]),('NIA_SL_WORD1675_REAL03_F','crop',[41,42,43]),
 ('NIA_SL_WORD1140_REAL01_F','crop',[94]),('NIA_SL_WORD2572_REAL01_F','crop',[34])]
bykey={(r['clip_id'],r['mode']):r for r in read(U/'REPLAY_RESULT.json')['rows']};details=[]
for cid,mode,frames in targets:
    rec=bykey[cid,mode];trace=read(U/'replay'/cid/mode/'trace.json')
    with np.load(Path(rec['raw_dir'])/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    for f in frames:
        n=int(raw['raw_count'][f]);t=trace[f]
        details.append(dict(clip_id=cid,label=rec['label'],mode=mode,frame=f,raw_count=n,
            raw_label=raw['raw_label'][f,:n].tolist(),raw_score=raw['raw_score'][f,:n].tolist(),
            pose_chain_usable={'L':raw['pose_usable'][f,[11,13,15]].tolist(),'R':raw['pose_usable'][f,[12,14,16]].tolist()},
            baseline_sides=t['baseline_assigned_sides'],new_sides=t['assigned_sides'],reasons=t['reasons'],arm_chain=t['arm_chain']))
result=dict(status='POST_REPLAY_DESCRIPTIVE_ONLY_NO_TUNING',counts=dict(counts),unassigned_prerequisites=dict(held),
    rows=rows,known_target_frames=details,interpretation='Arm prerequisites and counts are engineering diagnostics, not recognition accuracy. Missing raw hands cannot be restored by assignment.',
    candidate_executions=0,training=0)
with (U/'REPLAY_DIAGNOSTIC.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'counts':dict(counts),'held':dict(held)},ensure_ascii=False))
