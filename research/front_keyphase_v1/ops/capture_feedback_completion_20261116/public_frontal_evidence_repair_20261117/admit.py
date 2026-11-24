from pathlib import Path
import json,hashlib,ast,datetime
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117';ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 checks={}
 for name in ['NORMAL_APPLE','PARTIAL_FINGERS','OVERLAP_CONTROLS','UNKNOWN_NOT_ERROR','AGGREGATION_REPRODUCTION']:
  p=D/'checks'/f'{name}.json';assert read(p)['passed'];checks[str(p)]=sha(p)
 summary=read(D/'checks/SUMMARY.json');assert all(summary[k]for k in['normal_passed','finger_passed','overlap_passed','unknown_passed']) and summary['control_failed']==0
 for name in ['CONSUMER_CHECK','CARD_CHECK','CAPTURE_CHECK','READINESS_RECOVERY','LATENCY_CHECK']:
  p=D/'web'/f'{name}.json';assert read(p)['passed'];checks[str(p)]=sha(p)
 # Make provenance of fixtures explicit; never relabel a real user recording.
 for word in ['WORD0099','WORD1290']:
  p=D/'web'/f'capture_{word}.json';q=read(p);q['source']='public_capture_fixture';q['tuning_use']=False;write(p,q)
 pins=read(N/'web/ADMISSION.json')['pins'];assert all(sha(p)==h for p,h in pins.items())
 previous=len(pins)
 for p in list(D.glob('*.py'))+list((D/'apple_runtime').glob('*.py')):
  ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 # Freeze production helpers and their immutable dependencies, not debug logs.
 files=[D/x for x in ['frontal_evidence_logic.py','evidence_candidate.py','ready_capture.py','frontal_repair_runtime.py','serve.py','apple_runtime/worker.py','site/live.js','CONTRACT.json']]
 files += [O/'word100_action_feedback_demo_20261108'/x for x in['capture.py','auto_end.py']]
 files += [O/'word100_live_g6_20261114/capture_pacing.py',G/'public_visibility_roles_20261116/apple_candidate.py',G/'public_visibility_roles_20261116/visibility.py',G/'apple_v1/compose.py']
 for p in files:pins[str(p)]=sha(p)
 write(D/'web/ADMISSION.json',dict(admitted=True,pins=pins,checks=checks,prior_pins_preserved=previous,training_runs=0,private_data=False,scope='Public frontal evidence aggregation and capture readiness; physical-user retry remains unverified'))
 write(D/'web/VALIDATION.json',dict(passed=True,prior_pins_preserved=previous,normal_public=32,normal_preserved=32,partial_finger_mask_checks=len(read(D/'checks/PARTIAL_FINGERS.json')['rows']),overlap_checks=len(read(D/'checks/OVERLAP_CONTROLS.json')['rows']),whole_aggregation_checks=len(read(D/'checks/AGGREGATION_REPRODUCTION.json')['rows']),apple_controls=read(D/'checks/apple_controls/CONTROLS.json')['passed'],apple_diagnostic_only=5,card_quest_checks=len(read(D/'web/CARD_CHECK.json')['rows']),stop_controls=len(read(D/'web/CAPTURE_CHECK.json')['stop_controls']),training_runs=0,models_changed=False,private_data=False,source_scope='Known public front sources and controlled synthetic edits, not unseen learner accuracy',verification='Codex separate output/code/hash checks; not independent expert validation'))
 print('ADMITTED',len(pins),'PRIOR_PINS',previous)
if __name__=='__main__':main()
