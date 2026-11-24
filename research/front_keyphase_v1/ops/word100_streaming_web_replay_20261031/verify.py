"""Saved-artifact arithmetic and invariant checks; never reruns detection or scoring."""
from pipeline import *

def same_npz(a,b):
    out={}
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        names=set(x.files)-{'schema'};assert names==set(y.files)-{'schema'}
        for k in names:
            out[k]=x[k].shape==y[k].shape and x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes()
    return out

def main():
    frozen=read(U/'FROZEN.json');result=read(U/'RESULT.json');rows=result['rows'];assert len(rows)==12
    checks=[];equiv=[];summaries=[]
    for group in ['pins','protected']:
        for p,h in frozen[group].items():assert sha(p)==h,p
    for r in rows:
        d=U/'runs'/r['key'];t=read(d/'TRACE.json');f=read(d/'feedback.json');n=r['frames']
        assert [x['frame_id'] for x in t['rows']]==list(range(n))
        assert len(t['arrivals'])==n
        assert r['end_to_render_ack_seconds']==(r['render_ack_received_ns']-t['capture_end_ns'])/1e9
        assert r['backend_after_end_seconds']==(t['ready_ns']-t['capture_end_ns'])/1e9
        assert t['ready_ns']<=r['render_ack_received_ns']
        assert all(sha(d/p)==h for p,h in r['output_sha256'].items())
        if r['mode']=='during_recording':assert t['capture_end_ns']==t['arrivals'][-1]['available_ns']
        else:assert t['capture_end_ns']<=t['arrivals'][0]['available_ns']
        qrow=next(x for x in frozen['rows'] if x['alias']==r['alias'])
        with np.load(qrow['input'],allow_pickle=False) as orig,np.load(d/'candidate.npz',allow_pickle=False) as actual:
            assert np.array_equal(orig['frame_id'],actual['frame_id']) and np.array_equal(orig['timestamp_us'],actual['timestamp_us'])
        for p in f['result']['pairs']:
            assert p['reference_scope_retained'] and p['reference_expected_cells']==p['compared_cells']+p['missing_cells']
            assert p['educational_verdict'] is None and p['mandatory_omission_verdict'] is None
            assert len(p['query_seconds'])==p['query_frames'] and len(p['reference_seconds'])==p['reference_frames']
        assert f['result']['comparison_is_not_correctness']
        checks.append(dict(key=r['key'],frames=n,source_pts_exact=True,render_arithmetic_exact=True,all_reference_denominators_retained=True,output_hashes=True))
    for repeat in [1,2]:
        for alias in ALIASES:
            a=U/'runs'/f'{alias}_{repeat}_after_end';b=U/'runs'/f'{alias}_{repeat}_during_recording'
            arrays={name:same_npz(a/name,b/name) for name in ['baseline.npz','candidate.npz','raw.npz','assignment.npz']}
            assert all(v for part in arrays.values() for v in part.values())
            pa=np.load(a/'head_probabilities.npy');pb=np.load(b/'head_probabilities.npy');assert pa.tobytes()==pb.tobytes()
            assert read(a/'feedback.json')==read(b/'feedback.json')
            old=STREAM/'captures'/f'{alias}_{repeat}_single_decode_paced'
            old_arrays={name:same_npz(b/name,old/name) for name in arrays}
            assert all(v for part in old_arrays.values() for v in part.values())
            equiv.append(dict(alias=alias,repeat=repeat,arrays_byte_exact=arrays,previous_capture_byte_exact=old_arrays,head_byte_exact=True,feedback_json_exact=True))
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode];s=dict(mode=mode,runs=len(rr))
        for key in ['backend_after_end_seconds','end_to_render_ack_seconds','capture_tail_seconds','feature_seconds','head_seconds','dtw_three_refs_seconds','backend_ready_to_render_ack_seconds','queue_wait_max_seconds']:
            v=[r[key] for r in rr];s[key]=dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),min=min(v),max=max(v))
        summaries.append(s)
    # Synthetic absence control on cached fit reference. No new detection/model run.
    info=read(OLD/'LEARNING_INPUTS.json');r=frozen['references'][0];i=next(i for i,x in enumerate(info['rows']) if x['alias']==r['alias'])
    a=dict(np.load(r['input'],allow_pickle=False));probs=np.load(OLD/'runs/pose_capacity_matched_42/probabilities.npy',mmap_mode='r')
    ref=fb.prepare_case(r,a,probs[i,:len(a['x'])]);full=fb.pair(ref,ref,CONDITION)
    q=copy.deepcopy(ref);q['support'][:]=False;q['full_support'][:]=False;q['seq']['present'][:]=False
    missing=fb.pair(q,ref,CONDITION);assert missing['reference_expected_cells']==full['reference_expected_cells']>0
    assert missing['compared_cells']==0 and missing['missing_cells']==missing['reference_expected_cells']
    agg=fb.aggregate('synthetic_all_missing',[dict(reference=r['alias'],**missing)],CONDITION)
    assert agg['status']=='NO_COMPARISON_EVIDENCE' and agg['distance'] is None and not agg['events']
    output=dict(status='PASS_SAVED_OUTPUT_AND_TIMING',rows=checks,equivalence=equiv,summary=summaries,total_processed_frames=sum(r['frames'] for r in rows),
       all_missing_control=dict(reference_denominator=missing['reference_expected_cells'],compared=0,aggregate_distance=None,status=agg['status']),
       unchanged_protected=len(frozen['protected']),same_operator_check=True,independent_human_truth=False)
    write(U/'VERIFICATION.json',output)
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
