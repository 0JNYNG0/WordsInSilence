"""Independent batch arm oracle, preserved observations, leak/metamorphic tests.
Verification uses no trained classifier and no user/validation data.
"""
import sys,json,hashlib,traceback
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_position_state_audit_20261018'))
from audit import args as batch_args,position
sys.path.insert(0,str(O/'word100_hand_evidence_training_20261018'))
from adapter import frame_inputs
from axes import EvidenceAxes
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def serialize(x):return json.dumps(x,sort_keys=True,allow_nan=False)

def contracts():
 xy=np.full((75,2),np.nan,np.float32);mask=np.zeros(75,bool);conf=np.zeros(33,np.float32)
 for j,p in {11:(50,40),12:(150,40),13:(50,120),14:(150,120),15:(50,155),16:(150,155)}.items():xy[j]=p;mask[j]=True;conf[j]=.9
 def call(x=xy,m=mask,c=conf,**kwargs):return EvidenceAxes().step(x,m,c,0,200,200,**kwargs)
 def obs(x=xy,m=mask,side=0):
  x=x.copy();m=m.copy();x[33+side*21:54+side*21]=np.array([50 if side==0 else 150,155])+np.arange(21)[:,None]*.15;m[33+side*21:54+side*21]=True;return x,m
 checks=[]
 a=call(unassigned_raw=True);assert all(q['observation']['state']=='none' and q['arm_location']['state']=='wrist_interior' and q['assignment']['unassigned_candidate_exists'] for q in a);checks.append('withheld_flag_does_not_erase_arm_or_promote_hand')
 x,m=obs();a=call(x,m,assigned=(True,False),unassigned_raw=True);assert a[0]['observation']['all21_observed'] and a[0]['legacy']['state']=='observed' and a[1]['observation']['state']=='none';checks.append('observed_hand_retained_with_other_ambiguous_candidate')
 x2=x.copy();x2[16]=[150,210];a=call(x2,m,assigned=(True,False),unassigned_raw=True);assert a[1]['arm_location']['state']=='wrist_boundary_or_outside' and a[1]['observation']['state']=='none';checks.append('outside_pose_remains_inferred_not_hand21')
 x2=xy.copy();x2[15]=[50,125];a=call(x2);assert a[0]['arm_location']['reason']=='arm_geometry_uncertain';checks.append('short_projected_forearm_stays_uncertain')
 c=conf.copy();c[15]=.49;assert call(c=c)[0]['arm_location']['state']=='unknown';checks.append('existing_confidence_threshold_retained')
 x,m=obs();m[34:54]=False;m[33+np.array([0,5,9,13,17])]=True;a=call(x,m,assigned=(True,False));assert a[0]['observation']['state']=='palm5_observed' and a[0]['observation']['usable_point_count']==5 and not a[0]['observation']['all21_observed'];checks.append('palm5_is_not_full21')
 x=xy.copy();x[[13,14]]=[[100,110],[100,110]];x[[15,16]]=[[150,155],[50,155]];a=call(x);assert all(q['arm_location']['state']=='wrist_interior' for q in a) and a[0]['arm_location']['wrist_xy_px'][0]>100 and a[1]['arm_location']['wrist_xy_px'][0]<100;checks.append('crossed_arms_not_reassigned_by_screen_half')
 mirror=x.copy();mirror[:,0]=200-mirror[:,0];b=call(mirror);assert [q['side'] for q in a]==[q['side'] for q in b] and [q['arm_location']['state'] for q in a]==[q['arm_location']['state'] for q in b];checks.append('mirrored_position_does_not_change_anatomical_slot')
 a=call(assignment_evidence_available=False);assert all(q['assignment']['unassigned_candidate_exists'] is None for q in a);checks.append('unavailable_assignment_is_not_false_certainty')
 x,m=obs();a=call(x,m);b=call(x,m,assigned=(True,True),unassigned_raw=True);assert [q['arm_location'] for q in a]==[q['arm_location'] for q in b] and [q['observation'] for q in a]==[q['observation'] for q in b];checks.append('assignment_flags_do_not_change_observation_or_arm_axes')
 x,m=obs();x.flags.writeable=False;m.flags.writeable=False;conf.flags.writeable=False;call(x,m);checks.append('readonly_arrays_accepted_without_mutation')
 try:call(unassigned_raw=True,assignment_evidence_available=False)
 except ValueError:checks.append('hidden_assignment_flag_rejected')
 else:raise AssertionError('hidden flag accepted')
 return checks

def main():
 c=read(U/'CONTEXT.json');assert not (U/'RUNNING.json').exists();summary=read(U/'SUMMARY.json');paired=load(U/'PAIRED_DIAGNOSTICS.npz');old=load(O/'word100_position_state_audit_20261018/PAIRED_TRAIN.npz')
 assert np.array_equal(paired['legacy'],old['states'][:,:2]) and np.array_equal(paired['count'],old['counts'])
 tests=contracts();verified_arm=verified_obs=hidden_frames=future_frames=0
 for ri,r in enumerate(c['rows']):
  w,h=r['box'][2:]
  for view in ('masked','sensor'):
   obs,raw,ass=[load(r['views'][view][k]) for k in ('observation','raw','assignment')];out=load(U/'diagnostics'/f"{r['clip_id']}_{view}.npz")
   a=list(batch_args(obs,raw,ass,view,w,h));a[0][:,33:]=np.nan;a[1][:,33:]=False;a[3][:]=False;a[4][:]=False
   p,_=position(a,w,h);expected=np.where(p==2,1,np.where(p==3,2,0));assert np.array_equal(expected,out['arm_state']),(r['clip_id'],view,'arm oracle')
   verified_arm+=p.size
   xy=obs['xy_px'];valid=obs['usable']&np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<=w)&(xy[:,:,1]>=0)&(xy[:,:,1]<=h)
   for s in (0,1):
    hand=valid[:,33+s*21:54+s*21];assert np.array_equal(hand.sum(1),out['observation_count'][:,s]);assert np.array_equal(hand[:,[0,5,9,13,17]].all(1),out['palm5'][:,s]);assert np.array_equal(hand.all(1),out['all21'][:,s])
   verified_obs+=out['observation_count'].size
   if view=='masked':
    # Full-source hidden values and raw assignment cannot affect any sidecar
    # output in the masked view. 3 actual frames in every TRAIN video.
    mo={k:v.copy() for k,v in obs.items()};mr={k:v.copy() for k,v in raw.items()};ma={k:v.copy() for k,v in ass.items()}
    hidden=~valid;mo['xy_px'][hidden]=123456.;mr['pose_conf'][hidden[:,:33]]=.999;mr['pose_xy_px'][:]=-98765.;mr['hand_xy_px'][:]=77.;mr['raw_count'][:]=2;ma['sides'][:]=-1
    ix=np.unique(np.linspace(0,len(xy)-1,min(3,len(xy))).round().astype(int))
    astream,bstream=EvidenceAxes(),EvidenceAxes()
    for i in ix:
     aa=frame_inputs(obs,raw,ass,int(i),'masked',w,h);bb=frame_inputs(mo,mr,ma,int(i),'masked',w,h)
     ra=astream.step(*aa[:3],int(obs['timestamp_us'][i]),w,h,assigned=aa[3],unassigned_raw=aa[4],assignment_evidence_available=False)
     rb=bstream.step(*bb[:3],int(obs['timestamp_us'][i]),w,h,assigned=bb[3],unassigned_raw=bb[4],assignment_evidence_available=False)
     assert serialize(ra)==serialize(rb),(r['clip_id'],'hidden leakage');hidden_frames+=1
   # Causal prefix vs full stream with only future observations changed.
   # Three current frames cover initial/middle/end; past prefixes identical.
   if ri%30==0:
    k=max(1,len(xy)//2);stream=EvidenceAxes();prefix=[]
    for i in range(k):
     a=frame_inputs(obs,raw,ass,i,view,w,h);prefix.append(serialize(stream.step(*a[:3],int(obs['timestamp_us'][i]),w,h,assigned=a[3],unassigned_raw=a[4],assignment_evidence_available=view=='sensor')))
    changed={k2:v.copy() for k2,v in obs.items()};changed['xy_px'][k:]=123456.;changed['usable'][k:]=False;stream=EvidenceAxes()
    for i in range(len(xy)):
     a=frame_inputs(changed,raw,ass,i,view,w,h);q=stream.step(*a[:3],int(obs['timestamp_us'][i]),w,h,assigned=a[3],unassigned_raw=a[4],assignment_evidence_available=view=='sensor')
     if i<k:assert serialize(q)==prefix[i];future_frames+=1
  if (ri+1)%150==0:print('Independently verified',ri+1,'/899',flush=True)
 for p,h in c['pins'].items():assert digest(p)==h,p
 for p,h in read(U/'OUTPUT_PINS.json').items():assert digest(p)==h,p
 result={'status':'PASS','same_operator_separate_calculation':True,'synthetic_and_metamorphic_contracts':tests,'arm_states_independently_recomputed':verified_arm,'actual_hand_states_independently_recomputed':verified_obs,
 'hidden_evidence_mutation_frames':hidden_frames,'future_mutation_prefix_frames':future_frames,'primary_saved_observations_and_legacy_unchanged':len(old['states'])*2,'source_pins_unchanged':len(c['pins']),
 'scope':'TRAIN899 only; numerical invariants, not hand-identity ground truth or recognition accuracy','training':0,'model_inference':0}
 target=U/'INDEPENDENT.json';assert not target.exists();target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'FIRST_VERIFY_ERROR.json'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
