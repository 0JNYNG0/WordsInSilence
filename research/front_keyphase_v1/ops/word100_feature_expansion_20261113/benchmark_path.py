"""One matched public replay per condition; no camera or repeated source extraction."""
from model_core import *
import statistics

def main():
    assert (U/'EVALUATION_COMPLETE.json').exists();assert not (U/'PIPELINE_TIMING.json').exists()
    # Historical consumer modules expect their own `common`, not this experiment's.
    # Keep the candidate's already-bound globals; isolate the import binding only.
    original_common=sys.modules['common']
    sys.modules['common']=load('_timing_consumer_common',J/'common.py')
    d=O/'word100_action_feedback_demo_20261108';sys.path.insert(0,str(d))
    import consumer
    assert consumer.U==d and consumer.Candidate.__init__.__globals__['U']==J
    sys.modules['common']=original_common
    t=time.perf_counter();base=consumer.FeedbackConsumer();net=FineNet(7);net.load_state_dict(torch.load(U/'runs/C_42/model.pt',map_location='cpu',weights_only=True));net.eval();cold=time.perf_counter()-t
    def extra(raw,baseline):
        # Reuse the original consumer's quarantined observation, not unfiltered joints.
        t0=time.perf_counter();xy=np.asarray(baseline['query_xy'],np.float32)
        z=consumer.A.convert(xy,raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
        hand=np.stack([hand_features(xy[:,sl],z['point_available'][:,sl])[0] for sl in [slice(91,112),slice(112,133)]],1)
        x=old.normalize(z['tcn280'])
        with torch.inference_mode():net(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([0]),torch.from_numpy(hand[None]))
        return time.perf_counter()-t0
    warm=read(J/'records/NIA_SL_WORD1133_REAL01_F.json');z=np.load(warm['raw_path']);raw={k:z[k].copy() for k in z.files};v=base.raw(raw,'WORD1133');extra(raw,v)
    rows=[]
    for j,w in enumerate(old.WORDS):
        cid=f'NIA_SL_{w}_REAL11_F';rec=read(J/'records'/f'{cid}.json');assert sha(rec['raw_path'])==rec['raw_sha256'];z=np.load(rec['raw_path']);raw={k:z[k].copy() for k in z.files}
        measurements={};outputs={}
        for cond in (['current','current_plus_candidate'] if j%2==0 else ['current_plus_candidate','current']):
            t0=time.perf_counter();v=base.raw(raw,w);add=extra(raw,v) if cond!='current' else 0.;dt=time.perf_counter()-t0
            measurements[cond]=dict(seconds=dt,extra_branch_seconds=add,components=v['timing']);outputs[cond]={k:v[k] for k in ['tcn_top3','cards','path','overall_pass','correction_validated']}
        assert outputs['current']==outputs['current_plus_candidate'],'Sidecar must not change existing decisions'
        rows.append(dict(clip_id=cid,frames=len(raw['frame_id']),raw_sha256=rec['raw_sha256'],**measurements,original_decisions_identical=True))
    write(U/'PIPELINE_TIMING.json',dict(status='FOUR_PUBLIC_MATCHED_REPLAYS_COMPLETE',rows=rows,load_seconds=cold,measurement='Cached RTMW observations -> original quarantine/TCN/feedback/DTW/card outputs; added candidate includes guarded conversion+fine network; one timing percondition/clip, alternate order after warmup',concurrent_work='C9 separate CPU2 diagnostic training may still be active; not an idle-machine latency guarantee',not_included=['capture/extraction','browser serialization/display/network','coldload in perclip time'],not_a_user_wait_guarantee=True,not_deployed=True,summary={c:dict(min_seconds=min(r[c]['seconds'] for r in rows),max_seconds=max(r[c]['seconds'] for r in rows),median_seconds=statistics.median(r[c]['seconds'] for r in rows)) for c in ['current','current_plus_candidate']}))

if __name__=='__main__':main()
