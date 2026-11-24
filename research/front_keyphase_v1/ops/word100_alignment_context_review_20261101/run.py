from pathlib import Path
import json,sys,gzip,hashlib,collections,time,os,copy
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;OPS=U.parent
PREV=OPS/'word100_alignment_missing_split_20261101'
REF=OPS/'word100_target_known_feedback_20261101/REFERENCE_PHASES.json'
def read(p):
    p=Path(p)
    if p.suffix=='.gz':return json.loads(gzip.decompress(p.read_bytes()).decode('utf8'))
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def new(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.gz':
        with p.open('xb') as f:f.write(gzip.compress(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8'),mtime=0))
    else:
        with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def init():
    previous=read(PREV/'READOUT_ONLY_RESULT.json');refs=read(REF)['clips']
    assert previous['record_count']==108
    pins={str(REF):sha(REF)}
    for f in ['READOUT_ONLY_RESULT.json','READOUT_ONLY_CONTRACT.json','readout_only.py','EVALUATION.json','COMPLETION.json','review/READOUT_ONLY_VERIFICATION.json']:
        pins[str(PREV/f)]=sha(PREV/f)
    for r in previous['records']:
        assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256']
        pins[r['path']]=r['sha256'];pins[r['source']]=r['source_sha256']
        assert r['query']!=r['reference'] and refs[r['reference']]['role']=='fit'
    protected=read(PREV/'TASKS.json')['protected']
    new(U/'TASKS.json',dict(records=previous['records'],pins=pins,protected=protected,total=108,unique_query_seed_groups=48,
        axis_opportunities=1926120,shared_cells=736800,mixed_cells=3963,unique_query_opportunities=437640))
    new(U/'AUTHORIZATION.json',dict(request='User approves reviewed staged continuation towards trustworthy feedback after the failed alignment candidate.',
        scope='Keep existing alignment, annotate observational temporal context and FIT-reference occurrence context, enumerate all reference support',
        user_mentor_access=False,production_change=False,new_alignment=False,new_inference=False,new_training=False))
    update(U/'CURRENT_EXECUTION.json',dict(status='PREPARING',active=None))
    print('prepared108records48querygroups')
def freeze():
    t=read(U/'TASKS.json');pins=t['pins'].copy()
    for f in ['PLAN_KO.txt','TASKS.json','AUTHORIZATION.json','run.py','evaluate.py','context_evidence.py','selftest.py','SELFTEST.json',
              'review/METHOD_PLAN.json','review/EVALUATION_PLAN.json','review/SYNTHETIC_SUPPLEMENT.json','review/PREEXECUTION_REVIEW.json']:
        pins[str(U/f)]=sha(U/f)
    assert read(U/'SELFTEST.json')['status'].startswith('PASS')
    new(U/'FROZEN.json',dict(status='FROZEN_ADDITIVE_CONTEXT_CONTRACT',pins=pins,protected=t['protected'],new_alignment=False,new_model_inference=False,
        query_annotations_runtime=False,whole_sign_verdict=None,records=108,unique_query_seed_groups=48))
    print('frozen',len(pins))
def checkpins():
    d=read(U/'FROZEN.json')
    for section in ('pins','protected'):
        for p,h in d[section].items():assert sha(p)==h,p
    return d
def execute():
    checkpins();import context_evidence as context
    new(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
    update(U/'CURRENT_EXECUTION.json',dict(status='ANNOTATING_STORED_PATHS',active=os.getpid()))
    tasks=read(U/'TASKS.json');refs=read(REF)['clips'];groups=collections.defaultdict(list)
    for r in tasks['records']:groups[(r['query'],r['seed'])].append(r)
    outputs=[];avoutputs=[];done=0
    for (query,seed),records in sorted(groups.items()):
        availability_inputs=[]
        for rec in records:
            source=read(rec['path']);before=digest(source)
            reference=copy.deepcopy(refs[rec['reference']]);assert reference['role']=='fit' and rec['query']!=reference['alias']
            annotated=context.annotate(source['evidence'],source['path'],reference)
            assert digest(source)==before,'Source evidence/path mutated'
            dest=U/'contexts'/f"{query}_{rec['reference']}_{seed}.json.gz"
            out=dict(query=query,reference=rec['reference'],seed=seed,source=rec['path'],source_sha256=rec['sha256'],
                annotation=annotated,original_evidence_digest=before,original_path_exact=True,query_truth_used=False,frozen_sha256=sha(U/'FROZEN.json'))
            new(dest,out);outputs.append(dict(path=str(dest),sha256=sha(dest),source=rec['path'],query=query,reference=rec['reference'],seed=seed))
            availability_inputs.append(dict(reference_id=rec['reference'],query_id=query,seed=seed,evidence=source['evidence'],
                source_id=rec['path'],annotation_id=str(dest)))
            done+=1;update(U/'STATUS.json',dict(done=done,total=108,last=dest.name))
        before=[digest(x) for x in availability_inputs]
        av=context.availability(availability_inputs)
        assert before==[digest(x) for x in availability_inputs],'Reference evidence mutated'
        dest=U/'availability'/f'{query}_{seed}.json.gz'
        new(dest,dict(query=query,seed=seed,availability=av,source_records=[r['path'] for r in records],frozen_sha256=sha(U/'FROZEN.json')))
        avoutputs.append(dict(path=str(dest),sha256=sha(dest),query=query,seed=seed))
        print(done,108,query,seed,flush=True)
    assert len(outputs)==108 and len(avoutputs)==48
    checkpins();new(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_CONTEXT_COMPUTATION',contexts=outputs,availability=avoutputs,
        new_alignment=0,new_scores=0,new_model_inference=0,new_training=0))
    (U/'RUNNING.json').unlink();update(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_PENDING_VERIFICATION',active=None))
if __name__=='__main__':{'init':init,'freeze':freeze,'run':execute}[sys.argv[1]]()
