"""Four fixed public sources, fresh full pipeline timing, isolated output only."""
import sys,time,traceback
from phase_model import *
from phase_runtime import PhaseObserver

def main():
    assert not (S/'RUNTIME_RESULTS.json').exists();v=read(S/'SAVED_VERIFICATION.json')
    assert v['status']=='PASS_STATE_METRICS_TO_SEQUENCE_INTEGRATION'
    rows=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['person']==12];assert len(rows)==4
    exclusive(S/'RUNTIME_INTENT.json',dict(sources=[dict(clip_id=r['clip_id'],source=r['source'],sha256=r['source_sha256'],word_id=r['word_id']) for r in rows],condition='temporal_readout_42',public_only=True,existing_demo_not_changed=True))
    sys.path.insert(0,str(G));import learner_adapter
    learner_adapter.U=S;learner_adapter.warmup();observer=PhaseObserver();results=[]
    for r in rows:
        assert sha(r['source'])==r['source_sha256'];dest=S/'public_runtime'/r['clip_id'];start=time.perf_counter()
        base=learner_adapter.analyze(r['source'],dest,r['word_id']);a=dict(np.load(dest/'analysis.npz'));out=observer.attach(base,a)
        out['latency']['total_analysis_seconds']=time.perf_counter()-start;out['latency']['phase_head_seconds']=out['learned_state_sequence']['head_seconds']
        exclusive(dest/'SEQUENCE_FEEDBACK.json',out)
        assert out['runtime_individual_reference_video_comparisons']==0 and out['learned_prototype_comparisons']==1 and out['whole_sign_correctness'] is None
        results.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],result=str(dest/'SEQUENCE_FEEDBACK.json'),seconds=out['latency']['total_analysis_seconds'],phase_head_seconds=out['latency']['phase_head_seconds'],status=out['learned_state_sequence']['status'],old_features_and_DTW_preserved=True))
        print(results[-1],flush=True)
    exclusive(S/'RUNTIME_RESULTS.json',dict(rows=results,new_public_pipeline_runs=4,old_web_replaced=False,private_inputs=0,whole_runtime_median_seconds=float(np.median([r['seconds'] for r in results]))))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (S/'RUNTIME_FIRST_FAILURE.txt').exists():(S/'RUNTIME_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
