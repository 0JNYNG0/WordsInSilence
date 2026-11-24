"""Separate scalar production replay + direct paired array recount.
Same operator, separate verification method; not independent human review.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_hand_evidence_training_20261018'))
from adapter import frame_inputs,HandVisibility,LOCATION,POSITION
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def equal(a,b):return np.array_equal(a,b,equal_nan=True) if np.asarray(a).dtype.kind=='f' else np.array_equal(a,b)
ctx=read(U/'CONTEXT.json');d=load(U/'PAIRED_TRAIN.npz');summary=read(U/'SUMMARY.json')
assert len(d['states'])==summary['total_hand_frames'];assert len(np.unique(d['clip_index']))==899
m,s,wr,ass,pose,common=d['states'].T;diff=m!=s
assert int(diff.sum())==summary['mismatch']
matrix=[[int(((m==i)&(s==j)).sum()) for j in range(6)] for i in range(6)]
assert matrix==summary['transition_matrix_masked_rows_sensor_columns']
for i,key in enumerate(ctx['variants'][2:],2):
 q=d['states'][:,i];t=summary['counterfactuals'][key]
 assert t['sensor_changed']==int((s!=q).sum())
 assert t['original_mismatch_closed']==int((diff&(m==q)).sum())
 assert t['original_mismatch_remains']==int((diff&(m!=q)).sum())
 assert t['new_mismatch_from_original_agreement']==int((~diff&(m!=q)).sum())
 assert t['total_mismatch_after']==int((m!=q).sum())
assert np.array_equal(wr,pose),'Not assumed; empirical equality only'
assert not ((s==1)&(d['counts'][:,1]==0)).any()
# 3 evenly spaced paired frames per video, both hands, all six variants.
# Production frame adapter and streaming scalar rule, independent of audit's
# vectorized adapter/classifier. Exact same candidate evidence is replayed.
checks=0;original_arguments=0
for ri,r in enumerate(ctx['rows']):
 w,h=r['box'][2:];views={v:[load(r['views'][v][k]) for k in ('observation','raw','assignment')] for v in ['masked','sensor']}
 indices=np.flatnonzero(d['clip_index']==ri)[::2];chosen=np.unique(indices[np.linspace(0,len(indices)-1,min(3,len(indices))).round().astype(int)])
 lookup={int(f):i for i,f in enumerate(views['sensor'][0]['frame_id'])}
 for j in chosen:
  i=lookup[int(d['frame_id'][j])];t=int(d['timestamp_us'][j]);ma=frame_inputs(*views['masked'],i,'masked',w,h);sa=frame_inputs(*views['sensor'],i,'sensor',w,h);ca=frame_inputs(*views['sensor'],i,'masked',w,h)
  wa=[v.copy() if isinstance(v,np.ndarray) else v for v in sa]
  for hand in (0,1):
   p=wa[0][15+hand]
   if not (np.isfinite(p).all() and 0<=p[0]<=w and 0<=p[1]<=h):wa[2][15+hand]=0
  argslist=[ma,sa,tuple(wa),(sa[0],sa[1],sa[2],ca[3],ca[4]),(ca[0],ca[1],ca[2],sa[3],sa[4]),ca]
  for vi,a in enumerate(argslist):
   result=HandVisibility().step(*a[:3],t,w,h,assigned=a[3],unassigned_raw=a[4])
   output=[POSITION.index(LOCATION[x['state']]) for x in result]
   assert output==d['states'][j:j+2,vi].tolist(),(r['clip_id'],int(d['frame_id'][j]),vi,output,d['states'][j:j+2,vi])
   checks+=2
  original_arguments+=4
# Dedicated same-observation policy sentinel: no actual Hand21, trusted
# arm wrist extrapolated below image. Original C masks not promoted.
xy=np.full((75,2),np.nan,np.float32);mask=np.zeros(75,bool);conf=np.zeros(33,np.float32)
for ix,p in {11:(40,20),12:(60,20),13:(40,40),15:(40,65)}.items():xy[ix]=p;conf[ix]=.9;mask[ix]=p[1]<=60
src=(xy,mask,conf)
a=HandVisibility().step(*src,0,100,60)
bconf=conf.copy();bconf[15]=0
b=HandVisibility().step(xy,mask,bconf,0,100,60)
assert a[0]['state']=='boundary_outside_likely' and b[0]['state']=='unknown'
prio=HandVisibility().step(*src,0,100,60,unassigned_raw=True)
assert prio[0]['state']=='assignment_uncertain'
assert not mask[33:].any()
for p,h in ctx['pins'].items():
 with Path(p).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,p
out={'status':'PASS','scalar_hand_states_checked':checks,'sampled_actual_frames':checks//12,'same_evidence_sentinel':3,
 'saved_position_reconstruction_from_separate_vectorized_logic':summary['independent_position_reconstruction'],
 'all_primary_counts_recounted':True,'source_pins_unchanged':len(ctx['pins']),
 'same_operator_separate_calculation_not_external_review':True,'new_training':0,'model_inference':0,
 'partition_original_mismatch':{'closed_under_same_visible_evidence_policy':int((diff&(m==common)).sum()),
 'remaining_observed_or_partial_status_difference':int((diff&(m!=common)&((m<2)|(common<2))).sum()),
 'remaining_arm_evidence_difference':int((diff&(m!=common)&(m>=2)&(common>=2)).sum())},
 'palm_recovery_sensor':int(((m!=0)&(s==0)).sum()),'palm_loss_sensor':int(((m==0)&(s!=0)).sum()),
 'observed_palm_but_not_all21_visible':{v:int(((d['states'][:,i]==0)&(d['counts'][:,i]<21)).sum()) for i,v in enumerate(['masked','sensor'])}}
assert sum(out['partition_original_mismatch'].values())==summary['mismatch']
dest=U/'INDEPENDENT.json';assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
