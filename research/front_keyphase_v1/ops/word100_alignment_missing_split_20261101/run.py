from pathlib import Path
import sys, json, hashlib, time, os, copy, importlib.util
sys.dont_write_bytecode = True
import numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_joint_stage_alignment_20261101'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
previous=load(OLD/'run.py','previous_run_readonly')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        if p.parent.name=='results':json.dump(d,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        else:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
SEEDS=(42,43,44)
def init():
    old=read(OLD/'TASKS.json');ext=read(U/'review/EXTENSION_ELIGIBILITY.json')
    controls=copy.deepcopy(old['controls'])
    for c in controls:c['scope']='previous16'
    for row in ext['rows']:
        for kind in ('identity','right_hand_unobserved','short_shape_change_5f'):
            controls.append(dict(name=row['alias']+'_expanded_'+kind,alias=row['alias'],role=row['role'],word_id=row['word_id'],
                control=kind,scope='expanded32',array=None,baseline=None,metadata=dict(changed_frames=row['changed_frame_ordinals'] if kind=='short_shape_change_5f' else [],
                    kind='saved-feature synthetic control, no native RGB/model re-extraction')))
    pins=old['pins'].copy();pins.update(ext['pins'])
    for p in [OLD/'TASKS.json',OLD/'FROZEN.json',OLD/'EVALUATION.json',OLD/'RESULT_MANIFEST.json',OLD/'run.py',OLD/'evaluate.py',OLD/'joint_alignment.py',U/'review/EXTENSION_ELIGIBILITY.json']:
        pins[str(p)]=sha(p)
    assert len(controls)==238
    new(U/'TASKS.json',dict(pairs=old['pairs'],controls=controls,seeds=SEEDS,pins=pins,protected=old['protected'],total=822,extended_rows=ext['rows']))
    new(U/'AUTHORIZATION.json',dict(request='User explicitly approves general fixes to two calculation issues, not two-word special cases.',
        scope='One fixed experimental alignment plus two-axis evidence readout; old frozen paths/results reused; public stored inputs only',
        user_mentor_access=False,production_change=False,training=False,new_model_inference=False,html_unless_clear_improvement=False))
    print('prepared',len(old['pairs']),len(controls),822)

def freeze():
    t=read(U/'TASKS.json');pins=t['pins'].copy()
    for f in ['PLAN_KO.txt','AUTHORIZATION.json','TASKS.json','run.py','evaluate.py','split_alignment.py','selftest.py','SELFTEST.json',
        'review/METHOD_PLAN.json','review/EVALUATION_PLAN.json','review/PREEXECUTION_REVIEW.json']:
        pins[str(U/f)]=sha(U/f)
    assert read(U/'SELFTEST.json')['status'].startswith('PASS')
    new(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_PUBLIC_RESULTS',candidate_count=1,pins=pins,protected=t['protected'],runs=822,
        missing_cost_in_alignment=0,missing_is_agreement=False,readout='both axes, observed means/peaks and missing provenance',
        unresolved_missing_island_counterexample=True,training=0,new_inference=0))
    print('frozen',len(pins))

def checkpins():
    f=read(U/'FROZEN.json')
    for section in ('pins','protected'):
        for p,h in f[section].items():assert sha(p)==h,p
    return f

def control_input(task,base,seed):
    if task['scope']=='previous16':return previous.control_input(task,base,seed)
    a=previous.select(base,seed)
    q={k:(v.copy() if isinstance(v,np.ndarray) else copy.deepcopy(v)) for k,v in a.items()}
    if task['control']=='right_hand_unobserved':
        # Same fixed masking intervention as the preserved earlier controls.
        cols=[28,29,34,35,36,37]+list(range(39,59))+[60,65,66,67,68]
        q['mask'][:,cols]=False
    elif task['control']=='short_shape_change_5f':
        changed=task['metadata']['changed_frames'];cols=list(range(39,53));ix=np.ix_(changed,cols)
        before=q['x'][ix].copy();q['x'][ix]=np.where(before<.5,1.,0.)
        task['metadata']['actual_observed_changed_cells']=int((q['mask'][ix] & (q['x'][ix]!=before)).sum())
    elif task['control']!='identity':raise ValueError(task['control'])
    q['support']=previous.original.source_support(q,q['names'])
    return q

def execute():
    checkpins();t=read(U/'TASKS.json')
    lock=U/'RUNNING.json';new(lock,dict(pid=os.getpid(),started=time.time()))
    update(U/'CURRENT_EXECUTION.json',dict(status='RUNNING',active=os.getpid(),log=str(U/'RUN.log')))
    import split_alignment as candidate
    store=previous.original.RuntimeStore(read(previous.PREV/'REFERENCE_BANK.json'))
    candidate.solve(np.zeros((4,4))) # Synthetic JIT warmup; no public input.
    jobs=[('pair',p,s) for p in t['pairs'] for s in SEEDS]+[('control',p,s) for p in t['controls'] for s in SEEDS]
    assert len(jobs)==822
    records=[]
    for idx,(kind,task0,seed) in enumerate(jobs):
        p=copy.deepcopy(task0);name=(p['query']+'_'+p['reference'] if kind=='pair' else p['name'])+f'_{seed}'
        dest=U/'results'/(kind+'_'+name+'.json');assert not dest.exists(),'Do not overwrite result'
        if kind=='pair':q=previous.select(store.clip(p['query']),seed);r=previous.select(store.clip(p['reference']),seed)
        else:q=control_input(p,store.clip(p['alias']),seed);r=previous.select(store.clip(p['alias']),seed)
        q.pop('original_indices_evaluation_only',None)
        start=time.perf_counter();out=candidate.align(q,r);elapsed=time.perf_counter()-start
        out.update(kind=kind,task=p,seed=seed,name=name,elapsed_seconds=elapsed,frozen_sha256=sha(U/'FROZEN.json'),
            query_frames=len(q['x']),reference_frames=len(r['x']),query_labels_used=False,new_model_inference=False)
        new(dest,out);records.append(dict(path=str(dest),sha256=sha(dest),kind=kind,name=name))
        update(U/'STATUS.json',dict(done=idx+1,total=len(jobs),last=name))
        if idx%30==0 or idx+1==len(jobs):print(idx+1,len(jobs),name,round(elapsed,3),flush=True)
    checkpins();new(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_PATH_EXECUTION',results=records))
    lock.unlink();update(U/'CURRENT_EXECUTION.json',dict(status='PATHS_COMPLETE_EVALUATION_PENDING',active=None))

if __name__=='__main__':{'init':init,'freeze':freeze,'run':execute}[sys.argv[1]]()
