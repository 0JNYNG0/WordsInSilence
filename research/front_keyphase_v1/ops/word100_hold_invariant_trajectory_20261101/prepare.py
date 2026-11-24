from common import *
import copy
SAVED=['identity','speed_075','speed_125','omit_short_core_5f','swap_adjacent_core_blocks','right_hand_unobserved']
FRESH=['hold_insert5','reverse_first_parent','small_common_coordinate_drift','right_middle_gap5']
R_MISSING=[28,29,34,35,36,37]+list(range(39,59))+[60,65,66,67,68]

def modified(base,parents,kind):
 n=len(base['x']);ids=np.arange(n);core=parents[0]['core'];lo,hi=core;mid=(lo+hi)//2
 changed=[];removed=[]
 if kind=='hold_insert5':
  ids=np.r_[ids[:mid+1],np.full(5,mid),ids[mid+1:]];changed=list(range(mid+1,mid+6))
 elif kind=='reverse_first_parent':
  ids=np.r_[ids[:lo],ids[lo:hi+1][::-1],ids[hi+1:]];changed=np.flatnonzero(ids!=np.arange(n)).tolist()
 q={k:v[ids].copy() for k,v in base.items()}
 if kind in ['hold_insert5','reverse_first_parent']:
  q['frame_id']=np.arange(len(ids),dtype=np.int64);q['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333
 if kind=='small_common_coordinate_drift':
  jitter=.002*np.sin(np.arange(n)*2*np.pi/8)
  for offset in [0,28]:
   for j in range(offset,offset+10):q['x'][:,j]+=np.where(q['mask'][:,j],jitter,0)
  changed=np.flatnonzero(np.any((q['x']!=base['x'])&q['mask'],axis=1)).tolist()
 elif kind=='right_middle_gap5':
  changed=list(range(max(0,mid-2),min(n,mid+3)));q['mask'][np.ix_(changed,R_MISSING)]=False
 m=dict(kind='saved-feature technical transform, not RGB re-extraction or actual KSL error',original_indices=ids.tolist(),
  changed_query_indices=changed,removed_original_frames=removed,intended_reference_core=core,
  derivative_features='copied/reordered with source rows; not recalculated for synthetic timing',
  original_index_used_by_matcher=False)
 return q,m

def main():
 inv=read(U/'review/SOURCE_INVENTORY.json')
 old=read(OPS/'word100_joint_stage_alignment_20261101/TASKS.json')
 info=read(STAGE/'LEARNING_INPUTS.json');allrows={r['alias']:r for r in info['rows']}
 aliases=sorted({p['query'] for p in old['pairs']}|{p['reference'] for p in old['pairs']})
 assert len(aliases)==16
 rows={a:allrows[a] for a in aliases}
 phases={c['alias']:c for c in read(PHASE)['clips'] if c['alias'] in aliases}
 refs=read(TARGET/'REFERENCE_PHASES.json')['clips']
 pins=inv['pins'].copy()
 for f in [STAGE/'LEARNING_INPUTS.json',PHASE,REL/'relational.py',PREV/'NEXT_PLAN_KO.txt',PREV/'COMPLETION.json',PREV/'RESULT_MANIFEST.json']:
  pins[str(f)]=sha(f)
 natural=[]
 for pair in old['pairs']:
  p=Path(pair['baselines']['global_dtw']);d=read(p);pins[str(p)]=sha(p)
  assert refs[pair['reference']]['role']=='fit'
  for parent in refs[pair['reference']]['parents']:
   if parent['role']!='core_candidate':continue
   local,qr,rr,oldpath=crop(d['path'],parent['core'])
   natural.append(dict(query=pair['query'],reference=pair['reference'],role=pair['role'],word_id=pair['word_id'],
    parent=parent,baseline=str(p),query_range=qr,reference_range=rr,seed_aliases=[42,43,44],
    source_readouts=[str(OPS/'word100_alignment_missing_split_20261101/readout_replay'/f"{pair['query']}_{pair['reference']}_{s}.json.gz") for s in [42,43,44]]))
 controls=[];missing=[]
 for alias in aliases:
  row=rows[alias];assert sha(row['input'])==row['input_sha256'];pins[row['input']]=row['input_sha256']
  base=array(row['input']);parents=[p for p in phases[alias]['segments'] if p['role']=='core_candidate'];assert parents
  video=Path(row['source']);assert video.exists() and sha(video)==row['source_sha256'];pins[str(video)]=row['source_sha256']
  for kind in SAVED:
   p=REL/'controls'/f'{alias}_{kind}.json';a=p.with_suffix('.npz')
   if not p.exists():
    assert not a.exists();missing.append(dict(alias=alias,control=kind,status='NO_SAVED_CONTROL',role=row['role'],parents=parents));continue
   d=read(p);assert sha(a)==d['array_sha256'];pins[str(p)]=sha(p);pins[str(a)]=sha(a)
   controls.append(dict(id=f'{alias}_{kind}',alias=alias,role=row['role'],word_id=row['word_id'],control=kind,array=str(a),
    parents=parents,baseline=str(p),baseline_origin='saved_REL_control_path',metadata=d['metadata']))
  for kind in FRESH:
   q,meta=modified(base,parents,kind);p=U/'inputs'/f'{alias}_{kind}.npz';p.parent.mkdir(exist_ok=True)
   assert not p.exists();np.savez_compressed(p,**q);pins[str(p)]=sha(p)
   controls.append(dict(id=f'{alias}_{kind}',alias=alias,role=row['role'],word_id=row['word_id'],control=kind,array=str(p),
    parents=parents,baseline=None,baseline_origin='new_fixed_REL_control_path_only',metadata=meta))
 assert len(natural)==117 and len(controls)==158 and len(missing)==2
 t=dict(pins=pins,protected=inv['protected'],rows=rows,natural=natural,controls=controls,missing_controls=missing,
  component_names=list(COMPONENTS),natural_component_opportunities=len(natural)*5,
  control_component_opportunities=sum(len(x['parents'])*5 for x in controls+missing),
  new_control_baselines=64,saved_controls=94,control_case_denominator=160,
  geometry_independent_of_saved_seeds=True,seed_aliases=[42,43,44],query_truth_runtime=False,
  numerical_tolerance=1e-9,jitter_bound_tolerance=1e-6)
 new(U/'TASKS.json',t)
 new(U/'AUTHORIZATION.json',dict(request='Execute all previously reviewed trajectory plan and make an easy visual prior/current HTML report.',
  scope='One additive trajectory comparator, saved baseline reuse and64 fresh technical-control baseline paths, public sources only',
  html_requested_even_without_performance_gain=True,training=0,model_inference=0,private_access=False,production_change=False))
 update(U/'CURRENT_EXECUTION.json',dict(status='PREPARED_PENDING_FREEZE',active=None))
 print(dict(natural_parents=len(natural),controls=len(controls),missing=len(missing),
  natural_components=t['natural_component_opportunities'],control_components=t['control_component_opportunities']))

if __name__=='__main__':main()
