"""One frozen public integration job; not a repeat of an old scored evaluation."""
from pathlib import Path
import json, hashlib, time, uuid, traceback, os, sys

U=Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert not (U/'PUBLIC_RUN_STARTED.json').exists(), 'Do not repeat a partially executed public job'
    frozen=read(U/'EXECUTION_FROZEN.json')
    for p,h in frozen['pins'].items(): assert sha(p)==h,p
    review=read(U/'CODE_REVIEW.json')
    assert review['approved_for_one_public_integration'] is True
    item=read(U/'PUBLIC_INPUT_FROZEN.json')
    assert item['source_role']=='public_runtime_inference'
    assert item['role']=='new_exposed_development_integration_not_accuracy_test'
    assert sha(item['source'])==item['source_sha256']
    # Imports are delayed until every executable and the sole public source
    # have been frozen. No private folder discovery or input choice occurs.
    start=time.perf_counter()
    import learner_adapter as adapter
    import numpy as np
    jid=uuid.uuid4().hex
    job=U/'public_runtime_runs'/jid
    job.mkdir(parents=True,exist_ok=False)
    intent=dict(item,job_id=jid,training=False,tuning=False,candidate_selection=False)
    write(job/'INTENT.json',intent)
    write(U/'PUBLIC_RUN_STARTED.json',dict(pid=os.getpid(),job_id=jid,at=time.time(),source_sha256=item['source_sha256']))
    try:
        warm=adapter.warmup()
        write(job/'WARMUP.json',warm)
        # Count the actual learned-prototype DTW call in the unchanged engine.
        engine=sys.modules['engine']
        assert Path(engine.__file__).resolve()==(U.parent/'word100_learned_guide_feedback_20261102/engine.py').resolve()
        original=engine.dtw
        calls=[]
        def counted(query,reference,*args,**kwargs):
            calls.append(dict(query_frames=len(query),learned_prototype_frames=len(reference)))
            return original(query,reference,*args,**kwargs)
        engine.dtw=counted
        try:
            result=adapter.analyze(item['source'],job/'analysis',item['word_id'],provenance=intent,
                progress=lambda v:(job/'PROGRESS.json').write_text(json.dumps(v,ensure_ascii=False),encoding='utf-8'))
        finally:
            engine.dtw=original
        assert len(calls)==1, 'Single learned prototype alignment required'
        assert result['runtime_individual_reference_video_comparisons']==0
        assert result['source_sha256']==item['source_sha256']
        a=dict(np.load(job/'analysis/analysis.npz',allow_pickle=False))
        assert a['frame_id'].tolist()==item['frame_id'] and a['timestamp_us'].tolist()==item['timestamp_us']
        report=dict(status='ONE_NEW_PUBLIC_RUNTIME_COMPLETE',job_id=jid,source_sha256=item['source_sha256'],
            word_id=item['word_id'],public_registered_id=item['registered_id'],new_input_role=item['role'],
            result=str(job/'analysis/LIVE_FEEDBACK.json'),result_sha256=sha(job/'analysis/LIVE_FEEDBACK.json'),
            actual_dtw_calls=calls,individual_reference_video_comparisons=0,
            local_event_forwards=1,training=0,new_accuracy_test=False,
            server_analysis_seconds=result['local_combined_feedback_latency_seconds'],
            cold_job_seconds=time.perf_counter()-start,latency_excludes='browser upload/render and user recording',
            preserved_original_axis=True,live_servers_changed=False,whole_goal_complete=False)
        for p,h in frozen['pins'].items(): assert sha(p)==h,p
        write(U/'PUBLIC_RUN_RESULT.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        adapter.close()
    write(U/'PUBLIC_RUN_FINISHED.json',dict(status='complete',job_id=jid,at=time.time(),result_sha256=sha(U/'PUBLIC_RUN_RESULT.json')))

if __name__=='__main__':
    try:main()
    except Exception:
        with (U/('PUBLIC_RUN_FAILURE_'+str(time.time_ns())+'.txt')).open('x',encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
