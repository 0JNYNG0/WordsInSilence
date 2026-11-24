from probe_front import *

def main():
 for kind in ['apple','thanks']:
  dest=N/(kind+'_runtime');dest.mkdir(exist_ok=True)
  src=(G/'public_visibility_roles_20261116'/(kind+'_runtime')/'worker.py').read_text(encoding='utf-8')
  if kind=='thanks':
   src=src.replace('from thanks_candidate import ThanksCandidate','from frontal_candidate import FrontalLesson as ThanksCandidate').replace('from projection_guard import *','from frontal_candidate import *')
   src=src.replace('from repair_presentation import present','from frontal_presentation import present')
  else:
   src=src.replace('from apple_candidate import RoleEvidenceCandidate',"sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'public_visibility_roles_20261116'))\n from apple_candidate import RoleEvidenceCandidate")
  trace='''
import torch,hashlib
_loaded=[];_load=torch.load
def _trace(path,*args,**kwargs):
 p=Path(path).resolve()
 forbidden=['public_view_','public_temporal_visibility','public_cycle_observability','public_motion_decoupling','projection_head','inspection_relation']
 if any(x in str(p) for x in forbidden):raise RuntimeError('Non-frontal feedback weight rejected: '+str(p))
 _loaded.append(str(p));return _load(path,*args,**kwargs)
torch.load=_trace
'''
  src=src.replace('from contextlib import redirect_stdout','from contextlib import redirect_stdout\n'+trace)
  record="Path(__file__).with_name('LOADED_WEIGHTS.json').write_text(json.dumps([dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for p in dict.fromkeys(_loaded)],indent=2),encoding='utf-8')\n"
  src=src.replace('print(json.dumps(dict(ready=True',record+'print(json.dumps(dict(ready=True')
  (dest/'worker.py').write_text(src,encoding='utf-8')
 src=(G/'public_cycle_observability_20261116/serve.py').read_text(encoding='utf-8').replace('from cycle_web_adapter import CaptureService','from frontal_web_adapter import CaptureService')
 (N/'serve.py').write_text(src,encoding='utf-8')
 (N/'web').mkdir(exist_ok=True)
 # Admission initially disabled: tests run isolated before web promotion.
 write(N/'web/ADMISSION.json',dict(admitted=False,pins={},scope='front-only pending integrated verification'))
 print('BUILT_FRONT_RUNTIME')
if __name__=='__main__':main()
