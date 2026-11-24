"""Offline check: can observed movement bridge a short static-state uncertainty?"""
from motion_model import *
import sys,time
sys.path.insert(0,str(U))
from core import Features
from basic_runtime import RelationObserver
sys.path.insert(0,str(U.parent/'word100_rtmw_shared_input_20261108'))
import adapter

def bridge(static,motion):
    """Count observed closer arrivals; require an observed separating leg to rearm.

    The movement window has radius3, so arrival evidence must reach its right end.
    No required repetition count, word identity, static far-label fill, or DTW input.
    """
    events=[];separate=0;closer=0;armed=None;last_arrival=None
    for i,(s,m) in enumerate(zip(static,motion)):
        if s==0:
            separate=closer=0;armed=None
            continue
        if m==2:
            separate+=1;closer=0
            if separate>=2:armed=i-1
        elif m==1:
            closer+=1;separate=0
            if armed is None and last_arrival is None:armed=i
        elif m<0:
            closer=separate=0
        else:
            closer=separate=0
        # Use measured near state only AFTER the motion window, not its future context.
        if closer>=2 and armed is not None:
            end=i+3
            if end < len(static) and static[end]==2:
                if last_arrival is None or armed>last_arrival:
                    events.append(dict(start=int(armed),end=int(end),source='independent_motion_plus_observed_near'))
                    last_arrival=end;armed=None;closer=0
    return events

def main():
    torch.set_num_threads(2);assert not(D/'BRIDGE_RESULTS.json').exists()
    write(D/'BRIDGE_FROZEN.json',dict(model='mlp32 motion; edge features showed no gain and are not selected',
        model_hashes=[sha(D/'runs'/f'mlp32_{seed}.pt')for seed in [42,43,44]],
        code_sha=sha(Path(__file__)),primary='Known public diagnostics14–17; not new final test',
        classes='0 little net change;1 closer;2 separating',confidence=.7,radius=3,
        expected='Normal2; first-arrival-prefix<=1; constant and missing0; transformed speed and camera scale preserve count.',
        no_word_thresholds=True,no_web_write=True))
    rows=read(U/'source_corrected/CONFIRMATION_INPUTS.json')['rows']
    for r in read(U/'local_shape/fresh/EXTRACTED.json')['rows']:
        rows.append(dict(person=r['person'],clip_id=r['clip_id'],raw=r['raw_path'],channels=r['channel_path']))
    base=RelationObserver();ff=Features();ob=Observer('mlp32');results=[]
    first={14:52,15:56,16:49,17:88}
    for r in rows:
        raw=dict(np.load(r['raw']));n=len(raw['frame_id'])
        for kind in ['normal','first_only','constant','missing','fast_timestamps','slow_timestamps','scale_translation']:
            d={k:v.copy()for k,v in raw.items()}
            if kind=='first_only':
                for k in ['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:first[r['person']]+3]
            if kind=='constant':d['xy_px']=np.repeat(d['xy_px'][first[r['person']]:first[r['person']]+1],n,axis=0)
            if kind=='missing':d['xy_px'][:,91:]=np.nan
            if kind=='fast_timestamps':d['timestamp_us']=(d['timestamp_us']*.65).astype(np.int64)
            if kind=='slow_timestamps':d['timestamp_us']=(d['timestamp_us']*1.5).astype(np.int64)
            if kind=='scale_translation':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
            ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']))
            t=time.perf_counter();f,q,_=ff(ch);s,old=base(f,q,ch['timestamp_us']);m=ob.sequence(ch);new=bridge(s,m);elapsed=time.perf_counter()-t
            expected=2 if kind in ['normal','fast_timestamps','slow_timestamps','scale_translation']else 0 if kind in ['constant','missing'] else None
            passed=len(new)<=1 if expected is None else len(new)==expected
            results.append(dict(person=r['person'],kind=kind,old_events=old,new_events=new,diagnostic_pass=passed,seconds=elapsed,
                motion_states=m.tolist() if kind=='normal' else None,static_states=s.tolist() if kind=='normal' else None))
    write(D/'BRIDGE_RESULTS.json',dict(rows=results,passes=sum(r['diagnostic_pass']for r in results),total=len(results),
        web_admitted=False,timing='Already-extracted RTMW, excludes capture/extraction/DTW/render',
        speed_test='Timestamp transform only, no temporal frame dropping; not camera end-to-end speed validation'))
    print(json.dumps([dict(person=r['person'],old=len(r['old_events']),new=len(r['new_events']),passed=r['diagnostic_pass'])for r in results if r['kind']=='normal']))
    print('checks',sum(r['diagnostic_pass']for r in results),len(results))
if __name__=='__main__':main()
