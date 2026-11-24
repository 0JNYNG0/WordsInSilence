from pathlib import Path
import sys,json,hashlib,time,datetime,os,copy
sys.dont_write_bytecode=True
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
PREV=OPS/'word100_target_known_feedback_20261101';OLD=OPS/'word100_stage_feedback_cycle_20261031'
REL=OPS/'word100_relational_sequence_compare_20261031';CORE=OPS/'word100_core_context_split_20261101'
PHYS=OPS/'word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json'
sys.path.insert(0,str(PREV))
import comparator as original
SEEDS=(42,43,44)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,d):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def init():
 bank=read(PREV/'REFERENCE_BANK.json');tasks=read(PREV/'COMPARATOR_TASKS.json')['tasks'];store=original.RuntimeStore(bank)
 pairs=[];pins={}
 for t in tasks:
  for ref in bank['words']:
   if ref['word_id']!=t['practice_word_id']:continue
   for rr in ref['references']:
    ra=rr['legacy_alias'];qa=t['query_alias']
    if ra==qa:continue
    paths={'global_dtw':REL/'pairs'/f'B_same_path_{qa}_{ra}.json','core_first':CORE/'results'/f'B_same_path_{qa}_{ra}.json','relational_global':REL/'pairs'/f'C_relational_path_{qa}_{ra}.json'}
    for p in paths.values():pins[str(p)]=sha(p)
    pairs.append(dict(query=qa,reference=ra,role=t['role'],word_id=t['practice_word_id'],baselines={k:str(p) for k,p in paths.items()}))
 controls=[]
 for t in tasks:
  qa=t['query_alias']
  row=store.rows[qa];pins[row['input']]=row['input_sha256'];a=store.clip(qa)
  for p in sorted((REL/'controls').glob(qa+'_*.json')):
   d=read(p);ap=p.with_suffix('.npz');assert sha(ap)==d['array_sha256'];pins[str(p)]=sha(p);pins[str(ap)]=sha(ap)
   controls.append(dict(name=p.stem,alias=qa,role=t['role'],control=d['control'],array=str(ap),metadata=d['metadata'],baseline=str(p)))
  controls.append(dict(name=qa+'_small_common_coordinate_drift',alias=qa,role=t['role'],control='small_common_coordinate_drift',array=None,baseline=str(REL/'controls'/(qa+'_identity.json')),metadata={'amplitude':.002,'period_frames':8,'kind':'synthetic observed-coordinate jitter, not a linguistic error'}))
 assert len(pairs)==36 and len(controls)==142
 for p in [PREV/'REFERENCE_BANK.json',PREV/'COMPARATOR_TASKS.json',PREV/'REFERENCE_PHASES.json',PREV/'comparator.py',OLD/'LEARNING_INPUTS.json',PHYS,PREV/'PROJECTION_AUDIT.json']+[OLD/'runs'/f'pose_plus_signrep_{s}'/'probabilities.npy' for s in SEEDS]:pins[str(p)]=sha(p)
 new(U/'TASKS.json',dict(pairs=pairs,controls=controls,clips=tasks,pins=pins,seeds=SEEDS,protected=bank['protected']))
 new(U/'AUTHORIZATION.json',dict(request='User approves joint handshape/relations/movement stage comparison across public words; training only if necessity established; preserveC/T3F and private recordings; noHTMLwithoutclearimprovement',scope='One frozen combined alignment candidate, oldpaths reused, fixed public and technical control evaluation',private_access=False,new_model_inference=False,new_training_pending_evidence=True))
 update(U/'CURRENT_EXECUTION.json',dict(status='PREPARATION_AND_REVIEW',active=None))
 print(json.dumps({'pairs':len(pairs),'controls':len(controls),'candidate_runs':(len(pairs)+len(controls))*3}))
def freeze():
 tasks=read(U/'TASKS.json');pins=tasks['pins'].copy()
 for p in [U/'joint_alignment.py',U/'joint_alignment_selftest.py',U/'run.py',U/'evaluate.py',U/'PLAN_KO.txt',U/'TASKS.json',U/'AUTHORIZATION.json',U/'SELFTEST.json',U/'review/METHOD_REVIEW.json',U/'review/EVALUATION_CONTRACT.json',U/'review/PREEXECUTION_CODE_REVIEW.json']:
  pins[str(p)]=sha(p)
 new(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_PATH_RESULTS',candidate_count=1,methods=['saved_global','saved_core_first','saved_relational_global','joint_full_sequence'],seeds=SEEDS,pairs=36,controls=142,new_alignments=534,reference_full_frames_per_seed=4380,reference_certain_frames_per_seed=1080,query_labels_only_after_output=True,training0=True,pins=pins))
def checkpins():
 d=read(U/'FROZEN.json')
 for p,h in d['pins'].items():assert sha(p)==h,p
 return d
def select(a,seed):
 return {k:v for k,v in a.items() if k not in ('probs',)}|dict(probs=a['probs'][seed].copy(),names=a['names'])
def control_input(c,base,seed):
 if c['array']:
  with np.load(c['array'],allow_pickle=False) as d:q={k:d[k].copy() for k in ('x','mask','frame_id','timestamp_us')}
 else:
  q={k:base[k].copy() for k in ('x','mask','frame_id','timestamp_us')}
  jitter=.002*np.sin(np.arange(len(q['x']))*2*np.pi/8)
  # A common visible-body translation; keep pairwise differences untouched.
  for side in ('L','R'):
   for part in ('hand_wrist','pose_wrist','elbow','index_tip','palm_center'):
    for axis in ('x','y'):
     j=base['names'].index(f'{side}_{part}_{axis}');q['x'][:,j]+=np.where(q['mask'][:,j],jitter,0)
 ids=np.asarray(c['metadata'].get('original_indices',list(range(len(q['x'])))),int)
 assert len(ids)==len(q['x']) and ids.min()>=0 and ids.max()<len(base['x'])
 q['probs']=base['probs'][seed][ids].copy();q['names']=base['names'];q['support']=original.source_support(q,q['names'])
 q['original_indices_evaluation_only']=ids
 return q
def execute():
 import joint_alignment as joint
 checkpins();tasks=read(U/'TASKS.json');store=original.RuntimeStore(read(PREV/'REFERENCE_BANK.json'))
 lock=U/'RUNNING.json';new(lock,dict(pid=os.getpid(),started=time.time()));update(U/'CURRENT_EXECUTION.json',dict(status='RUNNING_FIXED_COMPARISON',active={'pid':os.getpid(),'log':'RUN.log'}))
 records=[]
 # Synthetic-only compilation; no extra real example scored.
 joint.solve(np.zeros((4,4)))
 jobs=[('pair',p,s) for p in tasks['pairs'] for s in SEEDS]+[('control',p,s) for p in tasks['controls'] for s in SEEDS]
 for i,(kind,p,seed) in enumerate(jobs):
  name=(p['query']+'_'+p['reference'] if kind=='pair' else p['name'])+f'_{seed}'
  dest=U/'results'/(kind+'_'+name+'.json');assert not dest.exists(),'No overwrite completed results'
  if kind=='pair':q=select(store.clip(p['query']),seed);r=select(store.clip(p['reference']),seed)
  else:q=control_input(p,store.clip(p['alias']),seed);r=select(store.clip(p['alias']),seed)
  # True original-index mapping is NOT passed to the aligner.
  q.pop('original_indices_evaluation_only',None)
  start=time.perf_counter();out=joint.align(q,r);elapsed=time.perf_counter()-start
  out.update(kind=kind,task=p,seed=seed,name=name,elapsed_seconds=elapsed,frozen_sha256=sha(U/'FROZEN.json'),query_frames=len(q['x']),reference_frames=len(r['x']),query_physical_labels_used=False,new_model_inference=False)
  new(dest,out);records.append(dict(path=str(dest),sha256=sha(dest),kind=kind,name=name))
  update(U/'STATUS.json',dict(done=i+1,total=len(jobs),last=name));
  if i%20==0 or i+1==len(jobs):print(i+1,len(jobs),name,round(elapsed,3),flush=True)
 checkpins();new(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_PENDING_INDEPENDENT_EVALUATION',results=records));lock.unlink();update(U/'CURRENT_EXECUTION.json',dict(status='PATH_EXECUTION_COMPLETE_EVALUATION_PENDING',active=None))
if __name__=='__main__':{'init':init,'freeze':freeze,'run':execute}[sys.argv[1]]()
