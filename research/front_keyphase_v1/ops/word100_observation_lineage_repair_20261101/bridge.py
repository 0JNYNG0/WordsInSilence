"""Versioned recovered geometry -> existing39 event comparison, fixed DTW path.

No baseline results, learned posteriors or position-path scores are recomputed.
Only parents whose geometry input gains observations are compared once anew.
"""
import sys
sys.dont_write_bytecode=True
import argparse,copy,gzip,hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
CASE=O/'word100_action_feedback_casebook_20261101'
HAND=O/'word100_independent_hand_evidence_20261101'
def read(p):
    p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    b=json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    with p.open('xb') as f:f.write(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b)
def module(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ('x','mask','frame_id','timestamp_us')}

def freeze():
    gate=read(U/'RECOVERY_ADOPTION.json')
    assert gate['geometry_bridge_allowed'], 'Input candidate not verified for bridge'
    method=read(U/'review/BRIDGE_METHOD_REVIEW.json');assert method['status'].startswith('PASS')
    old=read(CASE/'INPUTS.json');new=read(U/'RECOVERY_RESULT_MANIFEST.json')
    pins={}
    for p in [Path(__file__),CASE/'INPUTS.json',CASE/'RESULT_MANIFEST.json',
              HAND/'hand_evidence.py',CASE/'case_evidence.py',U/'RECOVERY_ADOPTION.json',
              U/'RECOVERY_RESULT_MANIFEST.json',U/'review/BRIDGE_METHOD_REVIEW.json']:
        pins[str(p)]=sha(p)
    for r in old['natural']:
        for p,h in [(r['hand_record']['path'],r['hand_record']['sha256']),
                    (r['legacy']['path'],r['legacy']['sha256'])]:
            assert sha(p)==h;pins[p]=h
    for r in old['rows'].values():
        assert sha(r['input'])==r['input_sha256'];pins[r['input']]=r['input_sha256']
    for r in read(CASE/'RESULT_MANIFEST.json')['results']:
        if r['kind']=='natural':assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
    for r in new['records']:
        for k in ('output','ledger'):
            assert sha(r[k])==r[k+'_sha256'];pins[r[k]]=r[k+'_sha256']
    write(U/'BRIDGE_FROZEN.json',{'pins':pins,'parents':39,'pairs':12,'references_per_case':3,
        'method':'Only new geometry at selected303 frames; original69 input otherwise exact. Reuse all old paths. Compare changed parents only. Old learned scores remain baseline-only.',
        'raw_RGB_quality_not_educational_truth':True,'new_training':0,'new_classifier_inference':0,'new_dtw':0})
    print('BRIDGE_FROZEN')

def run():
    if (U/'BRIDGE_MANIFEST.json').exists() or (U/'BRIDGE_RUNNING.json').exists():raise RuntimeError('Existing output or execution')
    f=read(U/'BRIDGE_FROZEN.json')
    for p,h in f['pins'].items():assert sha(p)==h,p
    write(U/'BRIDGE_RUNNING.json',{'started':time.time()})
    hand=module(HAND/'hand_evidence.py','bridge_fixed_hand_evidence')
    summarizer=module(CASE/'case_evidence.py','bridge_fixed_case_evidence')
    old=read(CASE/'INPUTS.json');new=read(U/'RECOVERY_RESULT_MANIFEST.json')
    previous={r['id']:r for r in read(CASE/'RESULT_MANIFEST.json')['results'] if r['kind']=='natural'}
    base={a:load(r['input']) for a,r in old['rows'].items()}
    patched={a:{k:v.copy() for k,v in d.items()} for a,d in base.items()}
    new_sources={r['alias']:r for r in new['records']}
    for alias,r in new_sources.items():
        z=load(r['output']);b=base[alias];d=patched[alias];ix={int(k):i for i,k in enumerate(b['frame_id'])}
        for j,fid in enumerate(z['frame_id']):
            i=ix[int(fid)];assert b['timestamp_us'][i]==z['timestamp_us'][j]
            assert np.all(z['mask'][j,b['mask'][i]])
            assert np.array_equal(z['x'][j,b['mask'][i]],b['x'][i,b['mask'][i]])
            d['x'][i]=z['x'][j];d['mask'][i]=z['mask'][j]
    outputs=[];changed=0;old_counts={};new_counts={}
    def tally(dst,evidence):
        for g in evidence['groups'].values():
            for axis in ('query','reference'):
                for k,v in g[axis]['counts'].items():dst[k]=dst.get(k,0)+v
    for t in old['natural']:
        prior=read(t['hand_record']['path']);tally(old_counts,prior['evidence'])
        qa,qb=t['query_range'];ra,rb=t['reference_range']
        def cut(alias,a,b):
            d={k:v[a:b+1].copy() for k,v in patched[alias].items()};d['x']=d['x'].astype(np.float64);d['names']=old['rows'][alias]['feature_names'];return d
        q=cut(t['query'],qa,qb);r=cut(t['reference'],ra,rb)
        affected=any(not np.array_equal(patched[alias][key][a:b+1],base[alias][key][a:b+1])
                     for alias,a,b in [(t['query'],qa,qb),(t['reference'],ra,rb)] for key in ('x','mask'))
        if not affected:
            tally(new_counts,prior['evidence']);outputs.append({'id':t['id'],'query':t['query'],'reference':t['reference'],'parent_id':t['parent_id'],
                'mode':'REUSED_UNCHANGED_BASELINE','result':previous[t['id']],'geometry':t['hand_record']});continue
        changed+=1
        baseline_case=read(previous[t['id']]['path'])
        stage=read(t['legacy']['path'])['cases'][t['legacy']['case_index']]['stages'][t['legacy']['stage_index']]
        path=np.array(stage['path'],dtype=int)-[qa,ra]
        tick=time.perf_counter();out=hand.compare(q,r,path);elapsed=time.perf_counter()-tick;tally(new_counts,out)
        row=copy.deepcopy(prior);row['evidence']=out
        row.update(source_geometry_version='RECOVERY_FIXED303_V1',
            frozen_sha256=sha(U/'BRIDGE_FROZEN.json'),elapsed_seconds=elapsed,
            baseline_geometry=t['hand_record'],new_geometry_sources={a:new_sources[a] for a in (t['query'],t['reference']) if a in new_sources},
            old_global_path_replaced=False,new_inference=False,new_coordinate_comparison=True,
            new_perception_inference_in_upstream_only=True,old_learned_shape_outputs_are_baseline_only=True,
            old_position_results_are_baseline_only=True,educational_verdict=None,whole_sign_verdict=None)
        dest=U/'bridge_geometry'/(t['id']+'.json.gz');write(dest,row)
        summary=summarizer.summarize(row,description=baseline_case['summary']['description_support']['provided'],
            correspondence=baseline_case['summary']['correspondence_support']['provided'],visual_review=None)
        result={'id':t['id'],'kind':'natural','query':t['query'],'reference':t['reference'],'parent_id':t['parent_id'],
            'source':{'path':str(dest),'sha256':sha(dest)},'summary':summary,
            'recovery_review':{'path':str(U/'RECOVERY_ADOPTION.json'),'sha256':sha(U/'RECOVERY_ADOPTION.json')},
            'old_record':previous[t['id']],
            'legacy_shape_and_position_outputs_role':'BASELINE_ONLY_NOT_REINFERRED_FROM_RECOVERED_INPUT',
            'new_training':False,'new_dtw':False,'educational_verdict':None,'whole_sign_verdict':None}
        rp=U/'bridge_results'/(t['id']+'.json.gz');write(rp,result)
        outputs.append({'id':t['id'],'query':t['query'],'reference':t['reference'],'parent_id':t['parent_id'],
            'mode':'NEW_RECOVERED_GEOMETRY_FIXED_OLD_PATH','result':{'path':str(rp),'sha256':sha(rp)},
            'geometry':{'path':str(dest),'sha256':sha(dest)}})
    assert len(outputs)==39
    for p,h in f['pins'].items():assert sha(p)==h,p
    assert old_counts['possible']==new_counts['possible']==30560
    write(U/'BRIDGE_MANIFEST.json',{'status':'COMPLETE_PENDING_VERIFICATION','records':outputs,'parents':39,'pairs':12,
        'changed_parents':changed,'unchanged_reused':39-changed,'baseline_counts':old_counts,'candidate_counts':new_counts,
        'new_coordinate_comparisons':changed,'old_comparisons_rerun':0,'new_dtw':0,'new_training':0,'new_classifier_inference':0,
        'old_learned_outputs_not_updated':True,'normal_reference_difference_is_not_error':True,'educational_accuracy':None})
    (U/'BRIDGE_RUNNING.json').rename(U/'BRIDGE_RUN_FINISHED.json')
    print(json.dumps({'parents':39,'changed':changed,'old':old_counts,'candidate':new_counts}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    try:globals()[a.stage]()
    except Exception:
        p=U/('BRIDGE_FIRST_ERROR_'+a.stage+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
