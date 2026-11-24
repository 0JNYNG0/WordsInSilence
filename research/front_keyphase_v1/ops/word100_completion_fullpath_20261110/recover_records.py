"""Recover already saved inference outputs after a duplicate dictionary-key reporting error."""
from common import *
import numpy as np
def compact(d):return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],top1=d['tcn_top3'][0]['word'],all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)))
assert not (F/'REPLAY_RECOVERED.json').exists();first=read(F/'REPLAY_COMPLETE.json');rows=[]
for r in first['rows']:
 key=r['clip_id']+'_'+r['kind'];p=F/'replay_results'/key
 assert 'multiple values for keyword argument' in r['error'];b=read(p/'before.json');d=read(p/'after.json');pre=read(p/'before_final_quarantine.json');assert b['tcn_top3']==d['tcn_top3'] and b['path']==d['path'];assert len(d['frame_id'])==r['frames']
 rr={k:v for k,v in r.items() if k not in ['status','error']};rows.append(dict(**rr,before=compact(b),after=compact(d),before_final_quarantine=compact(pre),result=str(p/'after.json'),result_sha256=sha(p/'after.json'),status='OK',seconds=d['compute_seconds'],timing_scope='consumer.raw internal elapsed, outer wrapper elapsed unavailable after reporting error',added_seconds=d['timing']['completion_identity_seconds'],quarantined_hand_frames=int(np.asarray(d['observation_guard']['hand_quarantined']).any(1).sum()),tcndtw_identical=True))
write(F/'RECOVERY_AUDIT.json',dict(first_error='dict(**input_row, sha256=result_hash) duplicated the input sha256 key after all outputs were saved',first_manifest_sha256=sha(F/'REPLAY_COMPLETE.json'),saved_outputs_preserved=True,neural_reruns=0,code_fix='Renamed output field result_sha256',corrected_code_sha256=sha(F/'replay.py'),recovered=len(rows)))
write(F/'REPLAY_RECOVERED.json',dict(rows=rows,neural_calls=first['neural_calls'],conditions=len(rows),model_load_seconds=first['model_load_seconds'],actual_camera=False,detector_calls=0,reporting_recovered_without_inference=True));print('RECOVERED_SAVED_OUTPUTS',len(rows))
