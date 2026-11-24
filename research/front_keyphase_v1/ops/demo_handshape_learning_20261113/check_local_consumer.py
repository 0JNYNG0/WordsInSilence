"""One public integration replay. Python W: reads denied, camera never opened."""
import sys,json,time,traceback
from pathlib import Path
U=Path(__file__).resolve().parent;O=U.parent;seen=set();blocked=[]
sys.dont_write_bytecode=True
def audit(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=args[0].decode(errors='replace') if isinstance(args[0],bytes) else args[0]
  if p.replace('\\','/').lower().startswith(('w:/','//?/w:/')):
   blocked.append(p);raise PermissionError('W: denied by local replay audit')
  seen.add(p)
sys.addaudithook(audit)
sys.path.insert(0,str(O/'word100_action_feedback_demo_20261108'))
result=dict(scope='One public coordinate replay through current consumer, Python W: reads denied',camera_opened=False,physical_disk_disconnected=False,native_filesystem_access_blocked=False,source='NIA_SL_WORD1133_REAL02_F',new_models_used=False)
try:
 import numpy as np
 from consumer import FeedbackConsumer
 r=json.loads((O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD1133_REAL02_F.json').read_text(encoding='utf-8'))
 with np.load(r['raw_path']) as z:raw={k:z[k].copy() for k in z.files}
 c=FeedbackConsumer();t=time.perf_counter();out=c.raw(raw,'WORD1133');result['consumer_seconds']=time.perf_counter()-t
 assert out['word']=='WORD1133' and len(out['tcn_top3'])==3 and len(out['cards'])>0 and not out['correction_validated']
 result.update(status='PUBLIC_CONSUMER_REPLAY_PASS',top1=out['tcn_top3'][0]['label'],cards=len(out['cards']),reference_video_reads=out['reference_video_reads'],rgb_model_calls=out['rgb_model_calls'])
 # Small result fixture for future Unity self-test; no source video or user data.
 fixture={k:out[k] for k in ['word','label','tcn_top3','cards','correction_validated','overall_pass']}
 (U/'PUBLIC_RESULT_FIXTURE.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2),encoding='utf-8')
except Exception:
 result.update(status='FAIL',traceback=traceback.format_exc())
result.update(blocked_w_reads=blocked,dependencies=sorted(p for p in seen if 'WordsInSilence' in p and Path(p).suffix in ['.pt','.json','.npz','.onnx']))
(U/'LOCAL_REPLAY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='dependencies'},ensure_ascii=False))
