"""Read-only public-input boundary stress audit; no training or live changes.

Stress frames are synthetic, not claims about actual user errors. Original
motion, internal missing frames and the original coordinate/mask arrays remain
intact. A solitary copied hand point at an added boundary tests span sensitivity.
"""
from pathlib import Path
import sys, json, hashlib, time, argparse, os
from collections import Counter
import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('C:/MyUnityProjects/WordsInSilence')
OPS = ROOT/'research/front_keyphase_v1/ops'
B = OPS/'word100_body_boundary_20261017'
T = OPS/'word100_frozen_c_temporal_20261019'
for p in [ROOT/'research/sign_model_pilot_v1/src', B, T]:
    sys.path.insert(0, str(p))
from pilot.data import build_channels
from pilot.web_input import to_native_like, active_span

VIEWS = ('sensor', 'masked')
KINDS = ('empty_wait', 'start_spike', 'end_spike', 'both_spikes')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}

def prepare_native(z):
    xy, present = to_native_like(z['xy_px'], z['usable'])
    lo, hi = active_span(present)
    return xy[lo:hi+1], present[lo:hi+1], z['timestamp_us'][lo:hi+1], (lo, hi)

def transform(xy, present, ts, kind):
    """Add one second each side; preserve all original timestamps relative to one another."""
    step = int(np.median(np.diff(ts)))
    assert step > 0
    n = max(1, round(1_000_000/step))
    left, right = np.repeat(xy[:1], n, axis=0), np.repeat(xy[-1:], n, axis=0)
    lp, rp = np.repeat(present[:1], n, axis=0), np.repeat(present[-1:], n, axis=0)
    left[:, 25:67] = 0; right[:, 25:67] = 0
    lp[:, 25:67] = False; rp[:, 25:67] = False
    picked = {}
    for end, enabled, a, mask, original, origmask, at in [
        ('start', kind in ('start_spike', 'both_spikes'), left, lp, xy[0], present[0], 0),
        ('end', kind in ('end_spike', 'both_spikes'), right, rp, xy[-1], present[-1], -1)]:
        if enabled:
            available = np.flatnonzero(origmask[25:67])+25
            j = next((k for k in (25,46) if origmask[k]), int(available[0]))
            a[at,j] = original[j]; mask[at,j] = True; picked[end] = j
    allxy = np.concatenate([left,xy,right]); allpr = np.concatenate([lp,present,rp])
    allts = np.concatenate([ts[0]-step*np.arange(n,0,-1), ts, ts[-1]+step*np.arange(1,n+1)])
    lo, hi = active_span(allpr)
    out = build_channels(allxy[lo:hi+1], allpr[lo:hi+1], allts[lo:hi+1])
    offset = n-lo
    assert np.array_equal(allxy[n:n+len(xy)],xy)
    assert np.array_equal(allpr[n:n+len(xy)],present)
    assert np.array_equal(allts[n:n+len(xy)],ts)
    # No omitted genuine interior frames, including all-missing ones.
    assert lo <= n and hi >= n+len(xy)-1
    return out, dict(extra_frames=n, start_offset=offset, picked_native_joints=picked,
                     span=[int(lo),int(hi)], original_frames=len(xy))

def nullable_scales(q):
    return {k: float(v) if np.isfinite(v) else None for k,v in q.items()}

def freeze(out):
    assert not (out/'FROZEN.json').exists()
    files = [Path(__file__), B/'training_CONTEXT.json',
             ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',
             ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',
             B/'training_C_20261013_best.pt', B/'training_C_20261013_RESULT.json',
             T/'T3F_20261013_best.pt', T/'T3F_20261013_RESULT.json',
             T/'frozen_temporal.py', B/'training_core.py']
    write(out/'FROZEN.json',dict(
        at=time.time(), schema='boundary_support_stress_v1',
        purpose='Separate truly absent waiting from one-point boundary contamination before changing any model.',
        public_scope='899 TRAIN plus 300 DEV, sensor/masked; no 199/200 heldout access',
        input_checks=KINDS, added_wait_seconds=1.0,
        new_inference='C13 and T3F13 on three synthetic spike conditions, DEV300 only. Saved original logits reused.',
        natural_edge_warning='Sparse edge is a diagnostic, not ground truth of wrong detection or dispensable motion.',
        invariants=['empty_wait exactly equals original channels','all original coordinates/masks and internal time differences preserved',
                    'existing stored channels reconstruct exactly','no hidden coordinates consumed','no origin/model/user changes'],
        no_training=True, no_deployment=True, no_user_recording_reads=True,
        pins={str(p):sha(p) for p in files}))

def check(out):
    ctx=read(out/'FROZEN.json')
    for p,h in ctx['pins'].items(): assert sha(p)==h, p
    return ctx

def audit(out):
    check(out); ctx=read(B/'training_CONTEXT.json'); rows=ctx['ordered_rows']
    assert Counter(r['role'] for r in rows)=={'train_candidate':899,'development_only':300}
    audits=[]; stress=[]; buffers={v:{k:[] for k in KINDS[1:]} for v in VIEWS}
    for i,r in enumerate(rows):
        recfile=B/'data/records'/f"{r['clip_id']}.json"; rec=read(recfile)
        assert rec['role']==r['role'] and rec['signer_id']==r['signer_id']
        for v in VIEWS:
            info=rec['views'][v]; z=arrays(info['observation']); saved=arrays(info['channel_file'])
            assert sha(info['channel_file'])==r['views'][v]['sha256']
            xy,pr,ts,(lo,hi)=prepare_native(z); orig=build_channels(xy,pr,ts)
            assert np.array_equal(orig['x'],saved['x_whole']), (r['clip_id'],v)
            empty,em=transform(xy,pr,ts,'empty_wait')
            assert np.array_equal(empty['x'],orig['x'])
            hidden=z['xy_px'].copy(); hidden[~z['usable']]=123456.75
            native2,pr2=to_native_like(hidden,z['usable'])
            assert np.array_equal(native2[lo:hi+1],xy) and np.array_equal(pr2[lo:hi+1],pr)
            count=pr[:,25:67].sum(axis=1)
            wrist_only=[bool(pr[t,25:67].any() and not np.delete(pr[t,25:67],[0,21]).any()) for t in (0,-1)]
            audits.append(dict(clip_id=r['clip_id'],role=r['role'],view=v,signer_id=r['signer_id'],
                label_index=r['label_index'], observation_sha256=sha(info['observation']), record_sha256=sha(recfile),
                channel_sha256=sha(info['channel_file']),full_frames=len(z['frame_id']),span=[lo,hi],
                frames=len(xy),first_hand_points=int(count[0]),last_hand_points=int(count[-1]),
                wrist_only_edges=wrist_only,internal_no_hand_frames=int((count==0).sum()),
                reconstructed_exact=True,empty_wait_exact=True,hidden_values_ignored=True))
            if r['role']=='development_only':
                for k in KINDS[1:]:
                    q,meta=transform(xy,pr,ts,k); offset=meta['start_offset']
                    central=q['x'][offset:offset+len(xy)]
                    assert np.array_equal(central[:,186:279],orig['x'][:,186:279])
                    assert np.array_equal(central[1:,279],orig['x'][1:,279])
                    stress.append(dict(clip_id=r['clip_id'],label_index=r['label_index'],signer_id=r['signer_id'],
                        view=v,kind=k,meta=meta,body_scale_before=orig['body_scale'],body_scale_after=q['body_scale'],
                        hand_scales_before=nullable_scales(orig['hand_scales']),hand_scales_after=nullable_scales(q['hand_scales']),
                        central_coordinate_max_change=float(np.max(np.abs(central[:,:186]-orig['x'][:,:186]))),
                        central_masks_unchanged=True,central_intervals_unchanged=True))
                    buffers[v][k].append((r,q['x']))
        if (i+1)%100==0: print(f'INPUT AUDIT {i+1}/1199',flush=True)
    write(out/'INPUT_AUDIT.json',dict(status='PASS',rows=audits,views=len(audits),
        original_channel_values=sum(a['frames']*280 for a in audits),
        model_input_invariance='All original channels exactly reproduced; truly absent added waiting removes itself.',
        synthetic_limit='Added point copies a visible edge landmark; this is a controlled stress, not an observed detector failure.'))
    write(out/'STRESS_INPUTS.json',dict(rows=stress,synthetic=True))
    return buffers

def inference(out,buffers):
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    from frozen_temporal import FrozenCTemporal
    from training_core import MODEL,padded_arrays,prediction
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    for name,path,oldfile in [('C13',B/'training_C_20261013_best.pt',B/'training_C_20261013_RESULT.json'),
                              ('T3F13',T/'T3F_20261013_best.pt',T/'T3F_20261013_RESULT.json')]:
        check(out); ck=torch.load(path,map_location='cpu',weights_only=True)
        model=MaskedSignTCN(**MODEL) if name=='C13' else FrozenCTemporal(**MODEL)
        model.load_state_dict(ck['state_dict'],strict=True); model.eval()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        scores={}; saved=read(oldfile)['predictions']; metrics={}
        with torch.inference_mode():
            for v in VIEWS:
                baseline={r['clip_id']:r for r in saved[v]}; scores[v]={}; metrics[v]={}
                assert len(baseline)==300
                for k in KINDS[1:]:
                    series=buffers[v][k]; pred=[]
                    for j in range(0,len(series),32):
                        batch=series[j:j+32]; x,pad=padded_arrays([x for r,x in batch])
                        logits=model(torch.from_numpy(x),torch.from_numpy(pad)).numpy()
                        pred.extend(prediction(r,z) for (r,x),z in zip(batch,logits))
                    assert len(pred)==300
                    scores[v][k]=pred
                    metrics[v][k]=dict(n=300,correct=sum(q['correct'] for q in pred),
                        top5=sum(q['top5_correct'] for q in pred),baseline_correct=sum(q['correct'] for q in baseline.values()),
                        label_changed=sum(q['prediction']!=baseline[q['clip_id']]['prediction'] for q in pred),
                        new_errors=[q['clip_id'] for q in pred if not q['correct'] and baseline[q['clip_id']]['correct']],
                        corrected=[q['clip_id'] for q in pred if q['correct'] and not baseline[q['clip_id']]['correct']])
                    print(name,v,k,metrics[v][k]['correct'],'/300',flush=True)
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        check(out)
        write(out/f'{name}_STRESS_SCORES.json',dict(model=name,checkpoint_sha256=sha(path),
            saved_original_scores=str(oldfile),baseline_reinference=False,synthetic=True,scores=scores,metrics=metrics,
            model_state_unchanged=True,training=0,user_recordings=0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--freeze',action='store_true')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.freeze: freeze(a.out); print('FROZEN'); return
    assert not (a.out/'RUNNING.json').exists() and not (a.out/'INPUT_AUDIT.json').exists()
    write(a.out/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
    started=time.time()
    try:
        buffers=audit(a.out); inference(a.out,buffers); check(a.out)
        write(a.out/'EXECUTION_DONE.json',dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',seconds=time.time()-started))
        (a.out/'RUNNING.json').unlink()
    except Exception:
        import traceback
        write(a.out/'FIRST_ERROR.json',dict(traceback=traceback.format_exc(),at=time.time()))
        raise
if __name__=='__main__': main()
