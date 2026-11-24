"""RAM diagnostic isolation and public fake-device wiring, no physical camera."""
from pathlib import Path
import sys,json,hashlib
U=Path(__file__).resolve().parents[1];V=Path(__file__).resolve().parent
sys.path.insert(0,str(U))
from diagnostic_ledger import DiagnosticLedger
checks=[]
def ck(name,v):checks.append(dict(name=name,passed=bool(v)));assert v,name
l=DiagnosticLedger();l.begin('camera',1)
ob=dict(assessed_timestamp_us=1000000,stage_index=0,diagnostic=dict(category='missing_keypoints',code='moving_hand_missing',message='손 관절 확인',held_progress=True),assessed_xy=[[1,2]],image='private-not-retained')
l.observe(ob,dict(observation_ms=40));l.observe(ob,dict(observation_ms=40))
ck('same_assessment_not_counted_twice',l.snapshot()['session']['observed_updates']==1)
s=l.snapshot();s['session']['counts']['missing_keypoints']=999
ck('returned_snapshot_cannot_mutate_ledger',l.snapshot()['session']['counts']['missing_keypoints']==1)
ck('no_coordinates_images_or_arbitrary_fields',not any(x in json.dumps(l.snapshot())for x in ['private-not-retained','assessed_xy']))
for i in range(1,60):
 ob['assessed_timestamp_us']+=33333;ob['diagnostic']['category']='shape_uncertain'if i%2 else'missing_keypoints';l.observe(ob,{})
ck('bounded_recent_reason_transitions',len(l.snapshot()['session']['recent_events'])==24)
l.end('done');before=l.snapshot();ob['assessed_timestamp_us']+=33333;l.observe(ob,{})
ck('ended_session_not_mutated',before==l.snapshot())
l.begin('public_replay',2)
ck('new_session_empty_previous_summary_preserved',l.snapshot()['session']['observed_updates']==0 and l.snapshot()['recent_sessions'][-1]['source']=='camera')
for g in range(3,10):l.begin('public_replay',g)
ck('history_bounded_to_five',len(l.snapshot()['recent_sessions'])==5)
l.observe(dict(assessed_timestamp_us=None,diagnostic=dict(category='progress')),{});ck('countdown_does_not_enter_diagnostic_counts',l.snapshot()['session']['observed_updates']==0)
ob['diagnostic']=dict(category='shape_mismatch',code='finger_changed',reset_reason='known_shape_mismatch')
for i in range(3):ob['assessed_timestamp_us']+=33333;l.observe(ob,{})
ck('same_reset_reason_counts_one_transition',l.snapshot()['session']['resets']==1)
before=l.snapshot()['session']['resets']
lost=dict(assessed_timestamp_us=None,diagnostic=dict(category='observation_delay',code='warming_up',reset_reason='capture_gap_exceeded'))
l.observe(lost,dict(elapsed_seconds=9.2))
ck('buffer_reset_without_assessment_timestamp_retains_reason',l.snapshot()['session']['resets']==before+1 and l.snapshot()['session']['last_reason']['time_basis']=='received_metric')
l.observe(lost,dict(elapsed_seconds=9.2))
ck('same_received_reset_not_counted_twice',l.snapshot()['session']['resets']==before+1)
import server,numpy as np
s=server.LiveService();s.generation=4;s.diagnostics.begin('public_replay',4)
image=np.zeros((20,20,3),np.uint8);xy=np.full((133,2),np.nan)
s.publish(image,xy,ob,{},3);ck('stale_generation_cannot_write_diagnostics',s.status()['diagnostics']['session']['observed_updates']==0)
s.publish(image,xy,ob,{},4);ck('accepted_frame_updates_ledger',s.status()['diagnostics']['session']['observed_updates']==1)
s.stop_event.set();ob['assessed_timestamp_us']+=33333;s.publish(image,xy,ob,{},4)
ck('stop_cannot_write_diagnostics',s.status()['diagnostics']['session']['observed_updates']==1)
report=dict(passed=all(c['passed']for c in checks),checks=checks,physical_camera=False,private_data=False,source='synthetic scalar status only',code_hashes={f:hashlib.sha256((U/f).read_bytes()).hexdigest()for f in ['diagnostic_ledger.py','server.py']})
(V/'LEDGER_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=report['passed'],checks=len(checks))))
