"""Frozen independent synthetic/public hand-evidence arithmetic verification."""
from pathlib import Path
from collections import Counter
import json,gzip,hashlib,sys,time,traceback,copy
sys.dont_write_bytecode=True
import numpy as np
from independent_arithmetic import verify_evidence
U=Path(__file__).resolve().parents[1]
def read(p):
    p=Path(p);return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data(path,span=None):
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
    if span is not None:out={k:v[span[0]:span[1]+1] for k,v in out.items()}
    out['x']=out['x'].astype(float);return out


def synthetic_cases():
    def base(n):
        x=np.zeros((n,69));m=np.ones_like(x,dtype=bool)
        for off in [0,28]:
            x[:,off+2]=1;x[:,off+25]=1;x[:,off+11:off+25]=np.linspace(.15,.8,14)
        return {'x':x,'mask':m,'frame_id':np.arange(n,dtype=np.int64),'timestamp_us':np.arange(n,dtype=np.int64)*33333}
    cases=[]
    def add(name,q,r,path=None):cases.append((name,q,r,[[i,i] for i in range(len(q['x']))] if path is None else path))
    r=base(5);add('identity',copy.deepcopy(r),r)
    q=copy.deepcopy(r);q['x'][:,13]+=.1;add('single_finger_change',q,r)
    q=copy.deepcopy(r);q['mask'][:,13]=False;add('one_finger_unknown',q,r)
    q=copy.deepcopy(r);q['mask'][:,25]=False;add('one_axis_coordinate_unknown',q,r)
    q=copy.deepcopy(r);q['x'][:,4:6]=q['x'][:,2:4];add('forearm_degenerate',q,r)
    q=copy.deepcopy(r);q['x'][:,25:27]=[-1,0];add('exact_antipodal',q,r)
    q=copy.deepcopy(r);theta=np.radians(30);q['x'][:,25:27]=[np.cos(theta),np.sin(theta)]
    q['x'][:,4:6]=q['x'][:,2:4]-[np.cos(theta),np.sin(theta)];add('common_arm_hand_rotation',q,r)
    q=copy.deepcopy(r);q['x'][2,25:27]=[0,1];add('one_frame_direction_peak',q,r)
    q=copy.deepcopy(r);q['x'][:,25:27]=[np.cos(np.radians(-179)),np.sin(np.radians(-179))]
    wrap_r=copy.deepcopy(r);wrap_r['x'][:,25:27]=[np.cos(np.radians(179)),np.sin(np.radians(179))];add('wrap_boundary',q,wrap_r)
    q=base(3);rr=base(4);rr['mask'][1,13]=False;rr['mask'][2,25]=False
    q['x'][1,13]+=.1;q['x'][1,25:27]=[0,1];add('partial_shared_and_compressed_peak',q,rr,[[0,0],[1,0],[1,1],[1,2],[2,3]])
    add('partial_fixed_path',copy.deepcopy(r),r,[[1,1]])
    add('empty_fixed_path',copy.deepcopy(r),r,[])
    q=copy.deepcopy(r);q.update(position_distance=0,shape_probability=1,word_top1='arbitrary');add('unused_metadata_does_not_gate',q,r)
    q=copy.deepcopy(r);q['mask'][:,0:28]=False;add('one_hand_missing_preserves_other',q,r)
    rng=np.random.default_rng(20261101)
    for i in range(24):
        q,r=base(5),base(5)
        for src in [q,r]:
            for off in [0,28]:
                src['x'][:,off+11:off+25]=rng.random((5,14))
                angles=rng.uniform(-np.pi,np.pi,5);src['x'][:,off+25:off+27]=np.column_stack((np.cos(angles),np.sin(angles)))
                fore=rng.uniform(-np.pi,np.pi,5);src['x'][:,off+2:off+4]=np.column_stack((np.cos(fore),np.sin(fore)))
            src['mask']&=rng.random((5,69))>.12
        add('fixed_random_'+str(i),q,r)
    return cases


def main(mode):
    start=time.perf_counter();checks=Counter();fail=[];rows=[];possible=0
    def check(name,condition,identity):
        checks[name]+=1
        if not condition:fail.append({'check':name,'id':identity})
    try:
        if mode=='selftest':
            sys.path.insert(0,str(U));import hand_evidence
            for identity,q,r,path in synthetic_cases():
                out=hand_evidence.compare(q,r,path)
                possible+=verify_evidence(q,r,path,out,check,identity)
                rows.append({'id':identity})
        elif mode=='public':
            inp=read(U/'INPUTS.json');manifest=read(U/'RESULT_MANIFEST.json');tasks={t['id']:t for t in inp['natural']+inp['technical']}
            check('full_record_scope',len(manifest['results'])==927 and {r['id'] for r in manifest['results']}==set(tasks),'manifest')
            for index,rec in enumerate(manifest['results']):
                check('output_hash',sha(rec['path'])==rec['sha256'],rec['id']);d=read(rec['path']);t=tasks[rec['id']]
                r=data(inp['rows'][t['reference']]['input'],t['reference_range'])
                if t['kind']=='natural':
                    q=data(inp['rows'][t['query']]['input'],t['query_range'])
                    path=np.asarray(t['baseline_full_path'])-[t['query_range'][0],t['reference_range'][0]]
                else:
                    check('technical_input_hash',sha(t['input']['path'])==t['input']['sha256'],t['id'])
                    q=data(t['input']['path']);path=np.column_stack((np.arange(len(q['x'])),np.arange(len(r['x']))))
                possible+=verify_evidence(q,r,path,d['evidence'],check,t['id'])
                check('position_sources_unchanged',d['position_sources']==t.get('position_sources',[]),t['id'])
                check('shape_sources_unchanged',d['shape_sources']==t.get('source_readouts',[]),t['id'])
                check('fixed_range_and_parent',d['parent_id']==t['parent']['segment_id'] and d['query_range']==t['query_range'] and d['reference_range']==t['reference_range'],t['id'])
                check('frozen_pointer',d['frozen_sha256']==sha(U/'FROZEN.json'),t['id'])
                check('no_model_or_whole_sign_change',not d['new_inference'] and not d['old_global_path_replaced'] and d['educational_verdict'] is None and d['whole_sign_verdict'] is None,t['id'])
                rows.append({'id':t['id'],'kind':t['kind'],'source':rec['path']})
                if (index+1)%100==0:print(index+1,'/',len(manifest['results']),flush=True)
        else:raise ValueError(mode)
    except Exception:fail.append({'exception':traceback.format_exc()})
    names=['hand_evidence.py','review/independent_arithmetic.py','review/verify_math.py']+(['FROZEN.json','INPUTS.json','RESULT_MANIFEST.json'] if mode=='public' else [])
    report={'schema':'wis_independent_hand_arithmetic_verification_v1','status':'PASS' if not fail else 'FAIL','mode':mode,'checks':sum(checks.values()),'checks_by_name':dict(checks),'failure_count':len(fail),'failures':fail,'records':len(rows),'rows':rows,'possible_both_axis_cells':possible,'elapsed_seconds':time.perf_counter()-start,
        'method':'Independent unit-vector cross/dot and complex relative directions; bincount/scatter per-feature counts and absolute residual statistics on both axes. No producer extraction/readout reused by verification. Producer is called only on synthetic cases in selftest mode.',
        'new_public_comparisons':0,'new_model_inference':0,'training':0,'pins':{str(U/n):sha(U/n) for n in names}}
    if mode=='selftest':
        initial=U/'review/MATH_SELFTEST.json'
        report['preserved_initial_failure']={'path':str(initial),'sha256':sha(initial),'cause':'Independent verifier did not restore JSON empty link arrays to shape(0,D). Producer code and public inputs were not changed.','correction':'Only empty verifier arrays recover their known group dimension; all nonempty comparisons remain strict.'}
    target=U/'review'/('MATH_SELFTEST_FINAL.json' if mode=='selftest' else 'MATH_RESULT_VERIFICATION.json')
    with target.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','checks','failure_count','records','possible_both_axis_cells','elapsed_seconds']}),flush=True)
    if fail:print(json.dumps(fail[:8]),flush=True);raise SystemExit(1)

if __name__=='__main__':main(sys.argv[1])
