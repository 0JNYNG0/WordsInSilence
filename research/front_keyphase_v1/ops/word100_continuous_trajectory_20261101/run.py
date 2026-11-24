from prepare import *
import time,os,traceback
def update(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def checkpins():
    d=read(U/'FROZEN.json')
    for group in ['pins','protected']:
        for p,h in d[group].items():assert sha(p)==h,p
    return d
def freeze():
    t=read(U/'INPUTS.json');pins=t['pins'].copy()
    for f in ['INPUTS.json','AUTHORIZATION.json','PLAN_KO.txt','prepare.py','run.py','evaluate.py','continuous.py','selftest.py','SELFTEST.json',
       'review/INPUT_AUDIT.json','review/FEATURE_PRESERVATION.json','review/METHOD_PLAN.json',
       'review/independent_math.py','review/verify_results.py','review/INDEPENDENT_MATH_SELFTEST.json','review/PREEXECUTION_REVIEW.json']:
        pins[str(U/f)]=sha(U/f)
    assert read(U/'SELFTEST.json')['status'].startswith('PASS')
    new(U/'FROZEN.json',dict(status='FROZEN_ONE_CONTINUOUS_GEOMETRY_CANDIDATE',pins=pins,protected=t['protected'],
        atol=t['atol'],max_iterations=t['max_iterations'],model_inference=0,training=0,old_alignment_recalculation=0,
        orientation_or_wrist_bend_classifier_trained=False,shape_features_changed=False,whole_sign_verdict=False))
    print('frozen',len(pins),flush=True)
def execute():
    checkpins();import continuous
    t=read(U/'INPUTS.json');new(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
    update(U/'CURRENT_EXECUTION.json',dict(status='RUNNING_FIXED_CONTINUOUS_COMPARISON',active=os.getpid()))
    out=[];start=time.perf_counter();cost=[];(U/'results').mkdir(exist_ok=True)
    for i,r in enumerate(t['records']):
        d=read(r['path']);q=d['query_plot'];ref=d['reference_plot'];tic=time.perf_counter()
        c=continuous.compare(q['values'],ref['values'],q['mask'],ref['mask'],atol=t['atol'],max_iterations=t['max_iterations'])
        secs=time.perf_counter()-tic;cost.append(secs)
        assert c['complete']==r['old_complete'],r['id']
        record=dict(id=r['id'],source_record=r['path'],source_sha256=r['sha256'],kind=r['kind'],role=r['role'],
            query=r['query'],reference=r['reference'],component=r['component'],control=r.get('control'),parent_id=r['parent_id'],
            candidate=c,old_discrete_distance=r['old_distance'],elapsed_seconds=secs,frozen_sha256=sha(U/'FROZEN.json'),
            unchanged_source_fields=['query_plot','reference_plot','query_range','reference_range','comparison.query','comparison.reference','parent','baseline_full_path','edit','source_readouts'],
            source_values_copied_or_smoothed=False,old_path_replaced=False,educational_verdict=None,
            handshape_orientation_wrist_bend_verdict=None,position_distance_is_not_whole_handshape_agreement=True)
        p=U/'results'/(r['id']+'.json.gz')
        with p.open('xb') as f:f.write(gzip.compress(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf8'),mtime=0))
        out.append(dict(id=r['id'],path=str(p),sha256=sha(p)))
        if (i+1)%100==0 or i+1==len(t['records']):
            update(U/'STATUS.json',dict(completed=i+1,total=len(t['records']),last=r['id']))
            print(i+1,'/',len(t['records']),flush=True)
    checkpins()
    new(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_PENDING_VERIFICATION',results=out,missing=t['missing'],
        elapsed_wall_seconds=time.perf_counter()-start,geometry_compute_seconds=sum(cost),
        new_training=0,new_model_inference=0,old_path_recalculation=0,source_originals_changed=False,
        end_to_end_user_latency_measured=False))
    (U/'RUNNING.json').unlink();update(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_PENDING_VERIFICATION',active=None))
if __name__=='__main__':
    try:{'freeze':freeze,'run':execute}[sys.argv[1]]()
    except Exception:
        if not (U/'FIRST_EXECUTION_FAILURE.json').exists():new(U/'FIRST_EXECUTION_FAILURE.json',dict(error=traceback.format_exc(),argv=sys.argv))
        raise
