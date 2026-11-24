"""Fixed public diagnostic comparison, not new held-out word accuracy."""
from diagnose_terminal import *
from motion_evidence import movement_evidence,repetition_feedback
import time

def main():
    torch.set_num_threads(2)
    out=HERE/'PROGRESS_COMPARISON.json'
    assert not out.exists(),'Keep completed comparisons immutable'
    model_files=sorted((MOTION/'runs').glob('mlp32_*.pt'))
    protected=read(OPS/'hello_live_coach_20261115/handshape_cause_review_20261116/PROTECTED_WEIGHTS.json')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
    write(HERE/'PROGRESS_FROZEN.json',dict(
        input='Previously-reviewed public gratitude people14–18 only',source_file='plan.json',
        variants=['normal','first_only','constant','missing','fast_timestamps','slow_timestamps','scale_translation','reversed','gap_at_arrival'],
        expected='Normal arrival count preserved, known18 prefix observed progress retained without arrival invention. Fixed/missing0. Every output span actually exists. No lesson pass.',
        original_sha=hashlib.sha256((MOTION/'check_motion_bridge.py').read_bytes()).hexdigest(),model_sha256=hashes,
        candidate_sha=hashlib.sha256((HERE/'motion_evidence.py').read_bytes()).hexdigest(),
        reversal='May still contain approaches: no invented whole-word reverse label. Not a word error recognition claim.',
        independent_final_test=False,training_runs=0,private_data=False))
    ff=Features();base=RelationObserver();ob=Observer('mlp32');results=[]
    for r in sources():
        p=r['person'];raw=dict(np.load(r['raw_path']));end={14:54,15:58,16:51,17:90,18:60}[p];n=len(raw['frame_id'])
        for kind in read(HERE/'PROGRESS_FROZEN.json')['variants']:
            d={k:v.copy() for k,v in raw.items()}
            if kind=='first_only':
                for k in ['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:end+1]
            elif kind=='constant':d['xy_px']=np.repeat(d['xy_px'][end:end+1],n,axis=0)
            elif kind=='missing':d['xy_px'][:,91:]=np.nan
            elif kind=='fast_timestamps':d['timestamp_us']=(d['timestamp_us']*.65).astype(np.int64)
            elif kind=='slow_timestamps':d['timestamp_us']=(d['timestamp_us']*1.5).astype(np.int64)
            elif kind=='scale_translation':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
            elif kind=='reversed':
                d['xy_px']=d['xy_px'][::-1].copy();d['score']=d['score'][::-1].copy()
            elif kind=='gap_at_arrival':d['xy_px'][end-3:end+1,91:]=np.nan
            ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']))
            f,q,_=ff(ch);s,_=base(f,q,ch['timestamp_us']);m=ob.sequence(ch);old=bridge(s,m)
            ids=[91,96,100,104,108,112,117,121,125,129]
            available=ch['point_available'][:,ids].all(1)&np.isfinite(ch['body_relative_xy'][:,ids]).all((1,2))
            t=time.perf_counter();e=movement_evidence(s,m,old,available);feedback=repetition_feedback(e,2);elapsed=time.perf_counter()-t
            checks=dict(no_completion=not feedback['lesson_complete'],
                spans_in_recording=all(0<=c['source_span'][0]<=c['source_span'][1]<len(s) for c in e['movement_candidates']),
                no_invented_arrival=e['confirmed_arrival_count']<=len(old))
            if kind in ['normal','fast_timestamps','slow_timestamps','scale_translation']:
                checks['normal_preserved']=e['confirmed_arrival_count']==len(old)==2
            if kind in ['constant','missing']:
                checks['no_false_motion']=e['confirmed_arrival_count']==0 and len(e['movement_candidates'])==0
            if kind=='first_only':
                checks['no_extra_repeat']=e['confirmed_arrival_count']<=1
                checks['actual_progress_retained']=len(e['movement_candidates'])>0
            results.append(dict(person=p,variant=kind,baseline_arrivals=len(old),evidence=e,feedback=feedback,checks=checks,
                seconds_evidence_assembly_only=elapsed,passed=all(checks.values())))
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
    write(out,dict(rows=results,passed=sum(r['passed'] for r in results),total=len(results),
        training_runs=0,web_admitted=False,whole_word_accuracy=None,
        limitations=['Five known clips; transformations are derived cases, not independent videos.',
                    'No new genuine incorrect performance or expert review.',
                    'Endpoint and handshape uncertainty remain; this preserves progress only.'],model_unchanged=True))
    print('CASES',sum(r['passed'] for r in results),'/',len(results))
    for r in results:
        if not r['passed'] or r['variant']=='first_only':print(json.dumps({k:r[k] for k in ['person','variant','baseline_arrivals','checks','feedback']},ensure_ascii=False))

if __name__=='__main__':main()
