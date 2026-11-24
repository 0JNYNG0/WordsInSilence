"""Partial witnessed pair order; never full-action completion or absence."""
from readout_model import *

def observe(physical,occurrence,available,times,first,second,condition):
 physical=np.asarray(physical);occurrence=np.asarray(occurrence);av=np.asarray(available,bool);times=np.asarray(times)
 with torch.inference_mode():
  p,mass=order_probability(torch.as_tensor(physical),torch.as_tensor(occurrence),torch.as_tensor(av),first,second,condition)
 p=float(p);mass=float(mass)
 weights=occurrence.copy()
 if condition=='tied_readout':weights=weights*physical[:,:8]
 a=np.flatnonzero(av&(weights[:,first]>=.8));b=np.flatnonzero(av&(weights[:,second]>=.8))
 overlap=np.intersect1d(a,b);status='PAIR_OCCURRENCE_UNCONFIRMED';direction=None;gaps=[]
 if len(a) and len(b):
  scope=np.arange(min(a.min(),b.min()),max(a.max(),b.max())+1);gaps=scope[~av[scope]].tolist()
  if gaps:status='PAIR_OBSERVATION_GAP'
  elif len(overlap):status='PAIR_OVERLAPPING_EVIDENCE'
  elif a.max()<b.min():
   direction=1;status='OBSERVED_A_BEFORE_B' if p>=.8 else 'PAIR_READOUT_DISAGREEMENT'
  elif b.max()<a.min():
   direction=0;status='OBSERVED_B_BEFORE_A' if p<=.2 else 'PAIR_READOUT_DISAGREEMENT'
  else:status='PAIR_INTERLEAVED_EVIDENCE'
 # Preserve strong contrary physical witnesses as disagreement. Weak or
 # absent old class output does not erase newly learned occurrence evidence.
 pa=np.flatnonzero(av&(physical[:,first]>=.8))
 pb=np.flatnonzero(av&(physical[:,second]>=.8))
 physical_direction=None
 if len(pa) and len(pb):
  if pa.max()<pb.min():physical_direction=1
  elif pb.max()<pa.min():physical_direction=0
  elif not np.intersect1d(pa,pb).size:physical_direction='interleaved'
 # Raw classes can include approach/return poses. A different argmax, low
 # score, or interleaving alone is not an opposite-action certificate.
 learned_status=status
 if status in ['OBSERVED_A_BEFORE_B','OBSERVED_B_BEFORE_A'] and physical_direction in [0,1] and physical_direction!=direction:
  status='RETAINED_PHYSICAL_ORDER_DISAGREEMENT'
 return dict(first=int(first),second=int(second),forward_probability=p,pair_mass=mass,
  first_evidence_frames=a.tolist(),second_evidence_frames=b.tolist(),overlap_frames=overlap.tolist(),
  first_evidence_seconds=times[a].tolist(),second_evidence_seconds=times[b].tolist(),
  gaps_inside_evidence=gaps,status=status,observed_direction=direction,
  physical_direction=physical_direction,learned_pair_status=learned_status,
  raw_physical_first_frames=pa.tolist(),raw_physical_second_frames=pb.tolist(),
  raw_physical_first_seconds=times[pa].tolist(),raw_physical_second_seconds=times[pb].tolist(),
  raw_physical_context_qualified=False,
  comparison_frames=len(av),unobserved_frames=np.flatnonzero(~av).tolist(),
  whole_sign_correctness=None,complete_action_verified=False,missing_action_verified=False,
  scope='Only the before/after relation of two observed queried physical events, not their exclusive boundaries or required completion')

def behavior():
 def call(ids,condition='direct_readout',gap=(),extra=None):
  n=len(ids);physical=np.eye(9,dtype=np.float32)[ids];occ=physical[:,:8].copy()
  if extra is not None:occ[extra[0],extra[1]]=1
  av=np.ones(n,bool);av[list(gap)]=False
  return observe(physical,occ,av,np.arange(n)/30,0,1,condition)
 for cond in ['tied_readout','direct_readout']:
  assert call([0,0,1,1],cond)['status']=='OBSERVED_A_BEFORE_B'
  assert call([1,1,0,0],cond)['status']=='OBSERVED_B_BEFORE_A'
  assert call([0]*40+[1]*40+[0],cond)['status']=='PAIR_INTERLEAVED_EVIDENCE'
  assert call([0]*4,cond)['status']=='PAIR_OCCURRENCE_UNCONFIRMED'
  assert call([0,0,1,1],cond,gap=[1])['status']=='PAIR_OBSERVATION_GAP'
  # Missing images outside the evidence are retained, not turned into an
  # error or used to claim completion of the entire sign.
  assert call([8,0,0,1,1,8],cond,gap=[0,5])['status']=='OBSERVED_A_BEFORE_B'
 assert call([0,0,1,1],extra=(1,1))['status']=='PAIR_OVERLAPPING_EVIDENCE'
 empty=observe(np.zeros((4,9),np.float32),np.ones((4,8),np.float32),np.zeros(4,bool),np.arange(4),0,1,'direct_readout')
 assert empty['pair_mass']==0 and empty['status']=='PAIR_OCCURRENCE_UNCONFIRMED'
 physical=np.eye(9,dtype=np.float32)[[1,1,0,0]];occ=np.eye(9,dtype=np.float32)[[0,0,1,1]][:,:8]
 disagreement=observe(physical,occ,np.ones(4,bool),np.arange(4)/30,0,1,'direct_readout')
 assert disagreement['status']=='RETAINED_PHYSICAL_ORDER_DISAGREEMENT' and disagreement['learned_pair_status']=='OBSERVED_A_BEFORE_B'
 return dict(status='PASS',before_after=True,short_contrary_not_averaged_away=True,
  missing_occurrence_not_absence=True,overlap_not_forced=True,missing_images_inside_pair_held=True,
  outer_missing_images_preserved_without_whole_completion=True)
