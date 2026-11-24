import sys
sys.dont_write_bytecode = True
import argparse, copy, gzip, hashlib, json, math, os, time, traceback
from pathlib import Path
import numpy as np

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
HOLD = OPS / 'word100_hold_invariant_trajectory_20261101'
CONT = OPS / 'word100_continuous_trajectory_20261101'

def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text('utf-8-sig'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(p,value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    if p.exists(): raise FileExistsError(p)
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b)

def data(row,span):
    a,b=span
    with np.load(row['input'],allow_pickle=False) as z:
        out={k:z[k][a:b+1].copy() for k in ('x','mask','frame_id','timestamp_us')}
    out['x']=out['x'].astype(np.float64)
    out['names']=row['feature_names']
    return out

def altered(src,side,condition):
    q={k:(v.copy() if isinstance(v,np.ndarray) else copy.deepcopy(v)) for k,v in src.items()}
    off=0 if side=='L' else 28
    x=q['x']; m=q['mask']; n=len(x)
    axis=[off+25,off+26]; elbow=[off+4,off+5]; wrist=[off+2,off+3]
    theta=np.zeros(n,dtype=np.float64); changed=[]
    if condition in ('axis_rotation','common_arm_hand_rotation'): theta[:]=math.pi/6
    if condition=='short_axis_change':
        lo=max(0,n//2-2); theta[lo:min(n,lo+5)]=math.pi/6
    if condition=='axis_jitter': theta=np.where(np.arange(n)%2==0,1,-1)*math.pi/360
    if np.any(theta):
        c,s=np.cos(theta),np.sin(theta)
        a=x[:,axis].copy(); x[:,axis[0]]=c*a[:,0]-s*a[:,1]; x[:,axis[1]]=s*a[:,0]+c*a[:,1]
        changed.extend(axis)
        if condition=='common_arm_hand_rotation':
            a=x[:,wrist]-x[:,elbow]
            rot=np.column_stack((c*a[:,0]-s*a[:,1],s*a[:,0]+c*a[:,1]))
            x[:,elbow]=x[:,wrist]-rot; changed.extend(elbow)
    if condition=='finger_change':
        j=off+13
        x[:,j]+=np.where(x[:,j]<=0.9,0.1,-0.1);changed.append(j)
    if condition=='common_translation':
        changed=[j for off2 in (0,28) for j in (off2,off2+2,off2+4,off2+6,off2+8)]
        x[:,changed]+=0.2
    if condition=='axis_missing': m[:,axis]=False; changed.extend(axis)
    if condition=='one_finger_missing': m[:,off+13]=False; changed.append(off+13)
    return q,{'condition':condition,'hand':side,'modified_columns':sorted(set(changed)),
        'axis_delta_degrees':np.rad2deg(theta).tolist(),'shape_probabilities_reused':False,
        'feature_space_only_not_physical_video':True}

def prepare():
    if (UNIT/'INPUTS.json').exists(): raise FileExistsError('prepared inputs already exist')
    old=read(HOLD/'TASKS.json'); manifest=read(CONT/'RESULT_MANIFEST.json')
    lookup={r['id']:r for r in manifest['results']}
    pins={}
    def pin(p,expected=None):
        p=str(Path(p).resolve()); got=digest(p)
        if expected is not None and got!=expected: raise AssertionError(('hash',p))
        pins[p]=got
        return {'path':p,'sha256':got}
    for p in (HOLD/'TASKS.json',CONT/'RESULT_MANIFEST.json',CONT/'FINAL_VERIFICATION.json',OPS/'word100_core_stage_evidence_20261031/PHASE_DATA.json'):
        pin(p)
    # Previous code/protected file pins are immutable even when not new runtime inputs.
    for p,h in old['protected'].items(): pin(p,h)
    for row in old['rows'].values(): pin(row['input'],row['input_sha256'])
    natural=[]
    for t in old['natural']:
        eid=f"pair_{t['query']}_{t['reference']}_{t['parent']['segment_id']}"
        links=[]
        for component in old['component_names']:
            r=lookup[eid+'__'+component]; pin(r['path'],r['sha256'])
            links.append(dict(r,component=component))
        first=read(links[0]['path']); src=read(first['source_record']); pin(first['source_record'],first['source_sha256'])
        natural.append(dict(t,id=eid,kind='natural',position_sources=links,
                            source_geometry=pin(first['source_record']),baseline_full_path=src['baseline_full_path']))
        for p in t['source_readouts']: pin(p)
        pin(t['baseline'])
    plan=read(UNIT/'PLAN.json'); phase=read(OPS/'word100_core_stage_evidence_20261031/PHASE_DATA.json')
    technical=[]
    for clip in phase['clips']:
        for parent in clip['segments']:
            if parent['role']!='core_candidate': continue
            r=data(old['rows'][clip['alias']],parent['core'])
            for side in ('L','R'):
                for condition in plan['technical']['conditions']:
                    eid=f"technical_{parent['segment_id']}_{side}_{condition}"
                    q,edit=altered(r,side,condition)
                    p=UNIT/'technical_inputs'/f'{eid}.npz'; p.parent.mkdir(exist_ok=True)
                    if p.exists(): raise FileExistsError(p)
                    np.savez_compressed(p,x=q['x'],mask=q['mask'],frame_id=q['frame_id'],timestamp_us=q['timestamp_us'])
                    technical.append({'id':eid,'kind':'technical','query':clip['alias'],'reference':clip['alias'],
                        'role':clip['role'],'word_id':clip['word_id'],'parent':parent,'query_range':parent['core'],
                        'reference_range':parent['core'],'input':pin(p),'edit':edit,
                        'path_origin':'predeclared_identity_not_estimated_alignment','no_saved_control_model_outputs':True})
    assert len(natural)==117 and len(technical)==810
    write(UNIT/'INPUTS.json',{'rows':old['rows'],'natural':natural,'technical':technical,'pins':pins,
        'denominators':{'clips':16,'pairs':36,'natural_parents':117,'position_records':585,'technical_parents':45,'technical_opportunities':810}})
    print(json.dumps({'prepared_natural':117,'prepared_technical':810,'pinned':len(pins)}))

def freeze():
    inp=read(UNIT/'INPUTS.json')
    inp['pins'].update(read(UNIT/'review/INPUT_AUDIT.json')['pins'])
    inp['pins'].update(read(UNIT/'review/INPUT_AUDIT.json')['protected'])
    for name in ('SELFTEST.json','review/INPUT_AUDIT.json','review/PREEXECUTION_REVIEW.json','review/MATH_SELFTEST_FINAL.json','review/PREPARATION_REVIEW_FINAL.json'):
        report=read(UNIT/name)
        assert str(report['status']).startswith('PASS'),(name,report['status'])
    for p,h in inp['pins'].items(): assert digest(p)==h,p
    files=['PLAN.json','INPUTS.json','pipeline.py','evaluate.py','hand_evidence.py','selftest.py','SELFTEST.json','review/METHOD_PLAN.json','review/INPUT_AUDIT.json','review/SHAPE_GUIDE_LINKS.json','review/RGB_CASE_SELECTION.json','review/RGB_VISIBILITY_REVIEW.json','review/PREEXECUTION_REVIEW.json','review/verify_math.py','review/independent_arithmetic.py','review/MATH_SELFTEST.json','review/MATH_SELFTEST_FINAL.json','review/PREPARATION_REVIEW.json','review/PREPARATION_REVIEW_FINAL.json','review/verify_saved_sources.py']
    write(UNIT/'FROZEN.json',{'pins':{str(UNIT/p):digest(UNIT/p) for p in files},'input_pins':inp['pins'],
        'time_unix':time.time(),'gate':'implementation selftest and independent source/method preflight completed; no outcome tuning'})
    print('FROZEN')

def run():
    import hand_evidence
    if (UNIT/'RESULT_MANIFEST.json').exists() or (UNIT/'RUNNING.json').exists(): raise RuntimeError('existing result or execution lock; inspect before any recovery')
    frozen=read(UNIT/'FROZEN.json')
    for p,h in {**frozen['pins'],**frozen['input_pins']}.items(): assert digest(p)==h,p
    write(UNIT/'RUNNING.json',{'pid':os.getpid(),'started':time.time()})
    inp=read(UNIT/'INPUTS.json'); start=time.perf_counter(); results=[]; compute=0
    shape_links=read(UNIT/'review/SHAPE_GUIDE_LINKS.json')
    shape_index={(r['query'],r['reference'],r['parent_id']):i for i,r in enumerate(shape_links['records'])}
    for task in inp['natural']+inp['technical']:
        r=data(inp['rows'][task['reference']],task['reference_range'])
        if task['kind']=='natural':
            q=data(inp['rows'][task['query']],task['query_range'])
            qa,ra=task['query_range'][0],task['reference_range'][0]
            path=np.array(task['baseline_full_path'],dtype=int)-[qa,ra]
        else:
            with np.load(task['input']['path'],allow_pickle=False) as z: q={k:z[k].copy() for k in ('x','mask','frame_id','timestamp_us')}
            q['names']=r['names']; path=np.column_stack((np.arange(len(q['x'])),np.arange(len(r['x']))))
        started=time.perf_counter(); out=hand_evidence.compare(q,r,path); elapsed=time.perf_counter()-started;compute+=elapsed
        record={'id':task['id'],'kind':task['kind'],'query':task['query'],'reference':task['reference'],
            'parent_id':task['parent']['segment_id'],'query_range':task['query_range'],'reference_range':task['reference_range'],
            'evidence':out,'position_sources':task.get('position_sources',[]),'shape_sources':task.get('source_readouts',[]),
            'shape_guide_link':({'path':str(UNIT/'review/SHAPE_GUIDE_LINKS.json'),'sha256':digest(UNIT/'review/SHAPE_GUIDE_LINKS.json'),
                'json_pointer':'/records/'+str(shape_index[(task['query'],task['reference'],task['parent']['segment_id'])]),
                'source_expectation_not_semantic_label':True} if task['kind']=='natural' else None),
            'technical_edit':task.get('edit'),'elapsed_seconds':elapsed,'frozen_sha256':digest(UNIT/'FROZEN.json'),
            'educational_verdict':None,'whole_sign_verdict':None,'old_global_path_replaced':False,'new_inference':False}
        p=UNIT/'results'/(task['id']+'.json.gz');write(p,record)
        results.append({'id':task['id'],'path':str(p),'sha256':digest(p),'kind':task['kind']})
    for p,h in {**frozen['pins'],**frozen['input_pins']}.items(): assert digest(p)==h,p
    write(UNIT/'RESULT_MANIFEST.json',{'status':'COMPLETE_PENDING_VERIFICATION','results':results,'compute_seconds':compute,
        'wall_seconds':time.perf_counter()-start,'end_to_end_latency':None,'new_training':0,'new_model_inference':0,'new_dtw':0})
    (UNIT/'RUNNING.json').rename(UNIT/'RUN_FINISHED.json')
    print(json.dumps({'results':len(results),'compute_seconds':compute,'wall_seconds':time.perf_counter()-start}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','freeze','run']); args=ap.parse_args()
    try: globals()[args.stage]()
    except Exception:
        p=UNIT/(f'FIRST_ERROR_{args.stage}.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
