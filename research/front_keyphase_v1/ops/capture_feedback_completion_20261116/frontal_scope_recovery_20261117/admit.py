from probe_front import *

def main():
 assert read(N/'AUDIT.json')['passed']
 for name in ['CONSUMER_CHECK','CARD_CHECK']:assert read(N/'web'/(name+'.json'))['passed']
 weights=read(N/'thanks_runtime/LOADED_WEIGHTS.json')+read(N/'apple_runtime/LOADED_WEIGHTS.json');pins={r['path']:r['sha256']for r in weights}
 for path in [N/'frontal_candidate.py',N/'front_geometry.py',N/'frontal_presentation.py',N/'frontal_web_adapter.py',N/'thanks_runtime/worker.py',N/'apple_runtime/worker.py',N/'source_policy.py',N/'AUDIT.json',N/'candidate/RESULT.json',N/'web/CONSUMER_CHECK.json',N/'web/CARD_CHECK.json']:
  pins[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 write(N/'web/ADMISSION.json',dict(admitted=True,pins=pins,scope='Single frontal camera observed lesson requirements. Source-only development regression; not full KSL/learner accuracy.',capture_gate_required_before_live_activation=True,third_party_pose_pretraining_unchanged=True))
 print('FRONT_ADMISSION',len(pins))
if __name__=='__main__':main()
