from diagnose import *
from observation_quality import finger_observability
from finger_runtime import refine
def main():
 torch.set_num_threads(2);net=ShapeConsensus();checks=[];changes=[]
 def ck(name,condition):
  checks.append(dict(name=name,passed=bool(condition)));assert condition,name
 manifest=json.loads((V/'BASELINE_MANIFEST.json').read_text(encoding='utf-8'))
 for r in manifest:
  z=load(r['raw']);x=channel(z);old=load(r['baseline'])
  for h in ['L','R']:
   q=net.sequence(x,h)
   for k in ['states','fingers','available']:
    diff=q[k]!=old[h+'_'+k]
    if diff.any():changes.append(dict(clip=Path(r['raw']).stem,hand=h,field=k,indices=np.argwhere(diff).tolist()))
 ck('27 public source sequences retain prior hand and finger decisions',not changes)
 # Prior native/camera-sized source results must remain unchanged too.
 for name,path in [('native',U/'source/reference.npz'),('letterbox',U/'seated_diagnosis_20261116/public_video_640_letterbox.npz'),('upper',U/'seated_diagnosis_20261116/public_video_upper_body_640.npz'),('heldout_view',U/'seated_diagnosis_20261116/heldout_crop.npz')]:
  x=channel(load(path));old=load(V/(name+'_predictions.npz'))
  for h in ['L','R']:
   q=net.sequence(x,h);ck(name+' '+h+' states unchanged',np.array_equal(q['states'],old[h+'_states']));ck(name+' '+h+' fingers unchanged',np.array_equal(q['fingers'],old[h+'_fingers']))
 raw=load(U/'source/reference.npz');x=channel(raw);degenerate=[]
 for h,start in [('L',9),('R',30)]:
  normal=net.sequence(x,h)
  for k in [5,9,13,17]:
   xx=x.copy();p=xx[:,:102].reshape(-1,51,2);p[:,start+k:start+k+4]=p[:,start+k:start+k+1]
   q=net.sequence(xx,h);finger=(k-5)//4
   ck(f'{h} collapsed {k} unavailable',not q['available'][:,finger].any())
   ck(f'{h} collapsed {k} no finger verdict',bool((q['fingers'][:,finger]<0).all()))
   ck(f'{h} collapsed {k} no coarse pass',bool((q['states']<0).all()))
   degenerate.append(dict(hand=h,finger=k,reason=str(q['geometry_reasons'][54,finger])))
   # Existing capture-end refinement cannot fabricate a corrective finger fact.
   card=dict(state='attributable_feature_difference',differences=[dict(task='shape',feature=1 if h=='R'else 2,expected=1,span=[0,len(x)-1])])
   ob=net.finger.infer(xx,h);result=refine({'joint_cards':[card]},{h:ob},raw['timestamp_us'])
   bad=[d for d in result['joint_cards'][0]['finger_details']if d['finger']==['index','middle','ring','little'][finger]]
   ck(f'capture-end {h} {k} has no false per-finger correction',not bad)
  xx=x.copy();p=xx[:,:102].reshape(-1,51,2);p[:,start+9]=p[:,start]
  q=net.sequence(xx,h);ck(h+' collapsed palm all unavailable',not q['available'].any())
  xx=x.copy();xx[:,:102]=((xx[:,:102].reshape(-1,51,2)*1.3)+[.2,-.4]).reshape(-1,102)
  a,ar=finger_observability(x,h);b,br=finger_observability(xx,h);ck(h+' scale translation observability invariant',np.array_equal(a,b)and np.array_equal(ar,br))
 # Publicly reproduced disagreement is now diagnosed rather than called bad visibility.
 path=O/'basic_demo_feedback_20261115/local_shape/fresh/raw/NIA_SL_WORD1290_REAL17_F.npz';q=net.sequence(channel(load(path)),'R')
 ck('public open hand frame95 explicitly model disagreement',q['states'][95]==-1 and q['reasons'][95]=='model_disagreement' and q['available'][95].all())
 for r in json.loads((V/'PROTECTED_WEIGHTS.json').read_text(encoding='utf-8')):
  ck('weight unchanged '+r['path'],hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'])
 write(V/'VERIFICATION.json',dict(passed=True,checks=checks,public_sources=len(manifest),public_frames=sum(r['frames']for r in manifest),public_changes=changes,degenerate_controls=degenerate,new_training=False,private_data=False,scope='27 prior public source videos regression, not new signer accuracy. Collapse is controlled input corruption, not prevalence estimate.'))
 print('VERIFIED',len(checks),'checks;27 public sources unchanged')
if __name__=='__main__':main()
