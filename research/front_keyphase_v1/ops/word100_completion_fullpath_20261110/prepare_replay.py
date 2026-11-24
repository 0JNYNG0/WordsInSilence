from common import *
import numpy as np
assert not (F/'REPLAY_INPUTS.json').exists()
public={r['clip_id']:r for r in read(J/'SOURCES.json') if r['feedback'] and r['role']=='development'}
U=O/'word100_attempt_event_supervision_20261109';meta=[r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development' and r['kind'] in ['full_normal','speed075','speed125','omit_final_event','mask_final_event','all_hand_missing']];rows=[]
for r in meta:
 cid=r['clip_id'];record=read(J/'records'/f'{cid}.json');assert record['source_sha256']==r['source_sha256'];assert sha(record['raw_path'])==record['raw_sha256'];assert sha(r['path'])==r['sha256']
 z=np.load(record['raw_path']);source=np.load(r['path']);ids=source['source_indices'];xy=z['xy_px'][ids].copy();score=z['score'][ids].copy()
 if r['kind'] in ['all_hand_missing','mask_final_event']:
  lo=0 if r['kind']=='all_hand_missing' else r['cut']+1;xy[lo:,91:133]=np.nan;score[lo:,91:133]=0
 raw=dict(xy_px=xy,score=score,timestamp_us=source['timestamp_us'].copy(),frame_id=np.arange(len(ids),dtype=np.int64),source_width=z['source_width'],source_height=z['source_height'])
 path=F/'replay_inputs'/f"{cid}_{r['kind']}.npz";path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**raw,source_frame_id=z['frame_id'][ids]);rows.append(dict(clip_id=cid,word_id=r['word_id'],person=r['person'],kind=r['kind'],path=str(path),sha256=sha(path),source_sha256=r['source_sha256'],raw_source=record['raw_path'],raw_source_sha256=record['raw_sha256'],source_indices=ids.tolist(),frames=len(ids),origin='historical DEV; new complete raw-path integration, not fresh accuracy'))
assert len(rows)==60
write(F/'REPLAY_INPUTS.json',dict(rows=rows,frame_ids='Strictly increasing stream IDs; source_frame_id retains resampling/cut provenance',mask='Only RTMW21 hand landmarks each side hidden; body wrists retained; no observation invented'))
pins=read(E/'FROZEN.json')['protected'];pins.update({str(p):sha(p) for p in [E/'runtime_adapter.py',E/'completion_guard_v2.py',E/'completion_guard.py',E/'core.py',E/'runs/final_42/model.pt',D/'consumer.py',O/'word100_observation_guard_20261108/guard.py',J/'runtime_candidate.py',J/'tcn_runs/finetune_C_20261013/model.pt',J/'feedback_viewfix/feedback_runs/pose_transfer_42/model.pt']})
write(F/'REPLAY_FROZEN.json',dict(status='FROZEN_BEFORE_FULLPATH_OUTPUT',conditions=60,primary=42,model_selection=False,fullpath='quarantine -> RTMW channel conversion -> current TCN & feedback -> frozen completion guard -> DTW results -> final quarantine-overlap display',baseline='same neural output before completion guard, followed by identical final quarantine; no second TCN inference',no_detector=True,no_training=True,no_private=True,inputs_sha256=sha(F/'REPLAY_INPUTS.json'),pins=pins,gates=['same TCN top3, DTW path and frame accounting','all normal/speed/crossing original display states preserved','no new false success for omitted/masked events','all old shape/direction differences preserved unless existing final quarantine withholds them','all 60 conditions retained in denominator; errors visible']))
print('PREPARED_RAW_REPLAY',len(rows))
