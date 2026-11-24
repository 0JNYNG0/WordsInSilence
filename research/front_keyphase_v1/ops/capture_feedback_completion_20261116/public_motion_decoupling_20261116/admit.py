from common import *

def main():
 q=read(E/'palm_checks/RESULT.json');assert q['new_warning']==q['lost_supported']==q['failed_controls']==0
 q=read(E/'regression/SHAPE.json');assert q['new_wrong']==q['lost_correct']==0
 assert read(E/'palm_regression/distinctions/RESULT.json')['failed']==0
 assert read(E/'palm_regression/confirmation/PATTERN_CHECK.json')['passed']
 assert read(E/'palm_views/RESULT.json')['passed']and read(E/'unselected_12/RESULT.json')['passed']
 parent=read(D/'web/ADMISSION.json');pins=dict(parent['pins']);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 names=['common.py','event_composition.py','qualified_motion.py','palm_geometry.py','train_palm.py','motion_web_adapter.py','runtime/worker.py','serve.py']
 root=O.parents[2]
 for p in[D/'web/ADMISSION.json']+[E/n for n in names]+list((E/'relation').glob('*.pt'))+list((E/'palm_relation').glob('*.pt'))+list((E/'palm').glob('*.pt'))+[root/'motion_demo_launcher.py']:
  assert p.exists(),p;pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(E/'web/ADMISSION.json',dict(admitted=True,status='Scientific gates passed; isolated consumer/capture tests pending before server activation',pins=pins,private_data=False,scope='Body/palm posture may add observed motion cycles; coarse palm only supports visibly qualified unknowns, no hidden individual finger inference; old cycles and corrections retained.',limitation='Known-side errors/unknowns remain. Code identity-free but lesson-specific endpoint task is not qualified for every KSL word.'))
 print('ADMITTED_FOR_LOCAL_RUNTIME_CHECK',len(pins),flush=True)
if __name__=='__main__':main()
