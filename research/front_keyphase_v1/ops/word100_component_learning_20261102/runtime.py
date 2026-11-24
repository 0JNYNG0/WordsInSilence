"""Additive research adapter. Never substitutes its output for existing observers."""
from common import *
from train_evaluate import status

class ComponentObserver:
 def __init__(self,condition='ordered'):
  assert condition in ['ordered','pooled']
  self.point=PointNet().eval();self.point.load_state_dict(torch.load(U/'runs/point.pt',map_location='cpu',weights_only=True))
  self.temporal=TemporalNet(condition).eval();self.temporal.load_state_dict(torch.load(U/'runs'/f'{condition}.pt',map_location='cpu',weights_only=True))
  self.stats=read(U/'NORMALIZATION.json');self.point_stats=read(V/'NORMALIZATION.json')
  self.catalog={r['clause_id']:r for r in read(Q/'CLAUSE_COMPONENT_LINKS.json')['rows']}
  self.targets=read(S/'DIRECT_TARGETS.json')['rows']
  # Canonical extents are proposal anchors in the already learned prototype.
  # Their mapping is never a label or an asserted exact event boundary.
  self.nominal={}
  for r in self.targets:
   if r['person']==1 and r['role']=='fit' and r['mask'] and r['y']==1 and r['source_clause_ids']:
    key=(r['word_id'],r['target']);e=self.nominal.setdefault(key,dict(frames=[],clauses=set()))
    e['frames']+=r['native_frames'];e['clauses'].update(r['source_clause_ids'])

 @torch.inference_mode()
 def dense_points(self,a):
  p,z=normalized(a,self.point_stats);n=len(p);out=np.full((n,2),np.nan,np.float32)
  for start in range(6,n-6,64):
   centers=np.arange(start,min(start+64,n-6));ix=centers[:,None]+np.arange(-6,7)
   valid=np.all(a['frame_id'][ix]==a['frame_id'][centers,None]+np.arange(-6,7),1)
   centers=centers[valid];ix=ix[valid]
   if len(ix):out[centers]=self.point(torch.from_numpy(p[ix]),torch.from_numpy(z[ix])).sigmoid().numpy()
  return out

 @torch.inference_mode()
 def attach(self,a,word,learned_path,protected_point_probabilities):
  # Old7shape+3relation+6motion fields remain outside and untouched. The caller
  # passes exact saved/new observation probabilities for three equivalent keys.
  original={k:np.asarray(v).copy() for k,v in protected_point_probabilities.items()}
  points=self.dense_points(a);path=np.asarray(learned_path,int);n=len(a['frame_id']);results=[]
  assert path.ndim==2 and path.shape[1]==2 and path[:,0].min()>=0 and path[:,0].max()<n
  for (w,target),nominal in self.nominal.items():
   if w!=word or target not in TEMP+POINT+list(PROTECTED_POINT):continue
   lo,hi=min(nominal['frames']),max(nominal['frames'])
   ix=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0])
   if not len(ix):continue
   full=np.arange(ix.min(),ix.max()+1);frames=a['frame_id'][full];clauses=sorted(nominal['clauses'])
   entry=dict(component_id=target,source_clause_ids=clauses,
    source_phrases=[self.catalog[c]['text'] for c in clauses],
    source_provenance=[self.catalog[c]['source'] for c in clauses],
    native_frames=frames.tolist(),native_seconds=(a['timestamp_us'][full]/1e6).tolist(),
    proposal='existing_learned_prototype_DTW_envelope',proposal_is_truth=False,
    partial_component_not_full_clause=True,whole_sign_correctness=None,
    deployment_status='RESEARCH_ONLY_NOT_VALIDATED_FOR_USER_CORRECTION')
   if target in TEMP:
    rr=dict(target=target,native_frames=[int(frames[0]),int(frames[-1])],y=None,mask=False)
    sample=temporal_sample(rr,a,self.stats);x,l=batch([sample]);prob=float(self.temporal(x,l).sigmoid()[0,TEMP.index(target)])
    entry.update(probability=prob,status=status(prob,sample['usable']),
       geometry_coverage=sample['geometry_coverage'],image_coverage=sample['image_coverage'],
       evidence_mode='current_frame_geometry_temporal_branch',
       missing_joint_support_not_repaired_by_this_branch=True)
   else:
    p=points[full,POINT.index(target)] if target in POINT else original[target][full]
    av=np.asarray(a.get('image_available',np.ones(n,bool)))[full]
    entry['point_evidence']=[dict(frame=int(f),seconds=float(t/1e6),probability=float(q) if np.isfinite(q) else None,
       status=status(float(q),bool(ok and np.isfinite(q)))) for f,t,q,ok in zip(frames,a['timestamp_us'][full],p,av)]
    entry['diagnostic_only']=target==POINT[0]
   results.append(entry)
  for k,v in original.items():assert np.array_equal(v,protected_point_probabilities[k],equal_nan=True)
  return dict(components=results,old_observations_modified=False,individual_reference_comparisons=0,
     whole_sign_correctness=None,automatic_proposals_not_yet_verified=True)
