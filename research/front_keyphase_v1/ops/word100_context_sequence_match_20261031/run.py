from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np
U=Path(__file__).resolve().parent; OLD=U.parent/'word100_relational_sequence_compare_20261031'; PHASE=U.parent/'word100_core_stage_evidence_20261031'
sys.path.insert(0,str(OLD))
from relational import GROUPS,features,observed_summary
from context_match import trace_match,window_dp

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def initialize():
    old=read(OLD/'AUTHORIZATION.json')
    rows=[{k:r[k] for k in ['alias','label','word_id','role','person','input','input_sha256','frames']} for r in old['rows']]
    write(U/'AUTHORIZATION.json',dict(request='2026-10-31 user approves concrete next goal, plan review with agents, then full execution and verification.',goal='Disambiguate repeated stage occurrences with ordered entry/core/exit evidence and test short-change/missing controls without shrinking reference scope.',rows=rows,new_neural_training=False,old_models_and_web_preserved=True,user_mentor_data=False,forbidden=['private recordings','production replacement','downloads','external transfer','Git','past model reevaluation']))
    write(U/'CURRENT_EXECUTION.json',dict(status='PLANNING_IMPLEMENTATION',active=None,previous_server='http://127.0.0.1:8792/'))

def freeze():
    files=[U/'AUTHORIZATION.json',U/'PLAN_KO.txt',U/'context_match.py',U/'run.py',U/'summarize.py',U/'review/PLAN_REVIEW.json',U/'review/SEMANTIC_PLAN_REVIEW.json',U/'review/SYNTHETIC_CHECK.json',U/'review/CAUSE_AUDIT.json',OLD/'relational.py',OLD/'COMPLETION.json',PHASE/'PHASE_DATA.json']
    rows=read(U/'AUTHORIZATION.json')['rows']
    files += [Path(r['input']) for r in rows]
    files += sorted((OLD/'controls').glob('*.json'))+sorted((OLD/'controls').glob('*.npz'))
    files += sorted((OLD/'pairs').glob('B_same_path_*.json'))
    write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_CANDIDATE_RESULTS',candidate='ordered full reference trace core plus 0.20 seconds entry/exit',window_frame_count_ratio=[.5,2],not_pts_duration_ratio=True,steps='diagonal/query-only/reference-only; at most2 consecutive query-only or reference-only',objective='start 2*c; diag2*c, non-diag1*c; minimum sum/(query_window_frames+reference_template_frames)',group_cost='equal mean active reference groups, missing query cells cost1; reference-unavailable groups excluded; all-unobserved reference frame cost0 but separately unsupported',support='per phase/group same fixed 75%,no mapped missing,>=2 observed ref and >=2 distinct query; context edges/unobserved incomplete',order='search each core independently over whole query; no interstage monotonic constraint; disjoint equal-cost alternatives ambiguous; close gaps not confidence',comparisons=36,controls=126,ground_truth='old original-index mapping ONLY for numerical correspondence evaluation; no semantic errors/necessary stage truth',neural_training=False,production_adoption=False,pins={str(p):sha(p) for p in files}))

def correspondence(result,metadata):
    indices=metadata.get('original_indices'); out=[]
    if indices is None:indices=list(range(result['query_frames']))
    indices=np.asarray(indices)
    for a in result['anchors']:
        if a['best'] is None:
            s,e=a['reference_core'];expected=set(range(s,e+1))
            out.append(dict(segment_id=a['segment_id'],center_original=None,center_in_original_core=False,recall=0.,precision=None,original_core_frames=len(expected),original_core_frames_present=len(set(indices.tolist())&expected),query_samples_from_original_core=int(np.isin(indices,list(expected)).sum()),path_mean_original_frame_error=None,path_exact_fraction=None,state=a['state']))
            continue
        s,e=a['reference_core']; b=a['best']; lo,hi=b['query_core']; center=(lo+hi)//2
        pred=set(int(v) for v in indices[lo:hi+1]); expected=set(range(s,e+1)); inter=len(pred&expected)
        pp=np.asarray(b['path']); pp=pp[(pp[:,1]>=s)&(pp[:,1]<=e)]
        error=np.abs(indices[pp[:,0]]-pp[:,1])
        out.append(dict(segment_id=a['segment_id'],center_original=int(indices[center]),center_in_original_core=bool(s<=indices[center]<=e),recall=inter/len(expected),precision=inter/len(pred),original_core_frames=len(expected),original_core_frames_present=len(set(indices.tolist())&expected),query_samples_from_original_core=int(np.isin(indices,list(expected)).sum()),path_mean_original_frame_error=float(error.mean()),path_exact_fraction=float((error==0).mean()),state=a['state']))
    return out

def execute():
    frozen=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in frozen['pins'].items())
    lock=U/'RUNNING.json';write(lock,dict(pid=os.getpid(),started=time.time()))
    rows=read(U/'AUTHORIZATION.json')['rows']; clips={r['alias']:r for r in read(PHASE/'PHASE_DATA.json')['clips']}
    arrays={r['alias']:load(r['input']) for r in rows}; data={a:features(v) for a,v in arrays.items()}
    folder=U/'results';folder.mkdir(exist_ok=True); manifest=[]
    # Compile with synthetic zeros only; not a real-data warmup rerun.
    window_dp(np.zeros((4,4)),0,4)
    jobs=[]
    for row in rows:
        a=row['alias']
        for path in sorted((OLD/'controls').glob(a+'_*.json')):
            prev=read(path);arr=path.with_suffix('.npz');assert sha(arr)==prev['array_sha256']
            jobs.append(dict(name=path.stem,kind='control',query=a,reference=a,control=prev['control'],input_path=arr,baseline=path,metadata=prev['metadata']))
    for path in sorted((OLD/'pairs').glob('B_same_path_*.json')):
        prev=read(path);jobs.append(dict(name=path.stem,kind='person_pair',query=prev['query'],reference=prev['reference'],baseline=path))
    for job in jobs:
        dest=folder/(job['name']+'.json');assert not dest.exists(),'Do not overwrite completed results'
        a,b=job['query'],job['reference'];rawq=load(job['input_path']) if job['kind']=='control' else arrays[a]
        q=features(rawq);start=time.perf_counter()
        d=trace_match(q,data[b],clips[b]['segments'],arrays[b]['timestamp_us'],GROUPS,observed_summary)
        d['seconds']=time.perf_counter()-start; d.update({k:v for k,v in job.items() if k not in ['input_path','baseline','metadata']})
        d['baseline_path']=str(job['baseline']);d['baseline_sha256']=sha(job['baseline']);d['query_array_sha256']=sha(job['input_path']) if 'input_path' in job else next(r['input_sha256'] for r in rows if r['alias']==a)
        d['frozen_sha256']=sha(U/'FROZEN.json'); d['original_query_frames_preserved_in_global_baseline']=True
        if job['kind']=='control': d['numerical_correspondence']=correspondence(d,job['metadata'])
        covered=set()
        for anchor in d['anchors']:
            if anchor['best']: covered.update(range(anchor['best']['query_core'][0],anchor['best']['query_core'][1]+1))
        d['outside_core_candidates']=[i for i in range(len(rawq['x'])) if i not in covered];d['outside_is_not_normal_or_discarded']=True
        write(dest,d);manifest.append(dict(file=str(dest),sha256=sha(dest),kind=job['kind'],name=job['name']))
        (U/'STATUS.json').write_text(json.dumps(dict(done=len(manifest),total=len(jobs),last=job['name'])),encoding='utf8')
        print(len(manifest),len(jobs),job['name'],round(d['seconds'],3),flush=True)
    write(U/'RESULT_MANIFEST.json',dict(status='EXECUTION_COMPLETE_PENDING_VERIFICATION',results=manifest,frozen_sha256=sha(U/'FROZEN.json')))
    lock.unlink()
if __name__=='__main__':{'init':initialize,'freeze':freeze,'run':execute}[sys.argv[1]]()
