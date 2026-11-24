"""Frozen motion network queried on an explicit, independently proposed episode.

This module does not find a successful subinterval, infer semantic correctness,
or use handshape correctness to decide which frames count as the episode.
"""
from pathlib import Path
import sys, time
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
R = O/'word100_learned_rgb_motion_20261102'
M = O/'word100_learned_motion_order_20261102'
for p in [R,M,O/'word100_process_contract_alignment_20261102']:
    sys.path.insert(0,str(p))
from rgb_model import RGBMotionNet, rgb_input, predict, read
from motion_model import normalize
from process_rgb_support import build_rgb_support, span_support, window_starts

def integer_axis(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a):
        raise ValueError(name+' must be a nonempty integer axis')
    if (a<0).any() or (np.diff(a.astype(np.int64))<=0).any():
        raise ValueError(name+' must be nonnegative and increasing')
    return a.astype(np.int64)

def prepare_input(a,z,mask,geometry_stats,rgb_stats):
    ids=integer_axis(a['frame_id'],'frame_id')
    ts=integer_axis(a['timestamp_us'],'timestamp_us')
    n=len(ids)
    if len(ts)!=n or np.shape(z)!=(n,73) or np.shape(mask)!=(n,73):
        raise ValueError('Feature/axis shape mismatch')
    if np.asarray(mask).dtype.kind!='b':
        raise ValueError('Feature validity must be boolean')
    if np.shape(a['rgb'])!=(n,768):
        raise ValueError('RGB shape mismatch')
    av=np.asarray(a.get('image_available',np.ones(n,bool)))
    if av.shape!=(n,) or av.dtype.kind!='b':
        raise ValueError('Image availability must be boolean on the native axis')
    finite=np.isfinite(a['rgb']).all(1)
    available=av&finite
    valid=np.asarray(mask)&np.isfinite(z)&available[:,None]
    video=rgb_input(a,rgb_stats)
    x=np.concatenate([normalize(np.where(valid,z,0),valid,geometry_stats),video],axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite normalized input')
    # This lineage has16-frame stride2 encodings and center-time interpolation.
    # Reject a different input route rather than inventing its temporal support.
    ws=np.asarray(a['window_starts'])
    expected=np.asarray(window_starts(n))
    if ws.dtype.kind not in 'iu' or not np.array_equal(ws,expected):
        raise ValueError('Unsupported RGB window provenance')
    centers=(ts[ws]+ts[ws+15])/2
    if not np.array_equal(centers,np.asarray(a['window_center_us'])):
        raise ValueError('RGB center/native time mismatch')
    wf=np.asarray(a['window_features'])
    if wf.shape!=(len(ws),768) or not np.isfinite(wf).all():
        raise ValueError('RGB window feature shape or validity mismatch')
    interpolated=np.stack([np.interp(ts,centers,wf[:,j]) for j in range(768)],axis=1).astype('float32')
    if not np.array_equal(interpolated[available],np.asarray(a['rgb'])[available]):
        raise ValueError('RGB values do not match declared cached windows')
    support=build_rgb_support(ts,av,ids,rgb_finite=finite)
    return dict(x=x,mask=valid,available=available,ids=ids,ts=ts,support=support)

def select_episode(prepared, frame_bounds, target_index):
    bounds=integer_axis(frame_bounds,'episode endpoints') if len(frame_bounds)!=1 else None
    if bounds is None or len(bounds)!=2:
        raise ValueError('Episode requires two increasing endpoints')
    if type(target_index) is not int or target_index not in range(5):
        raise ValueError('Unsupported physical target')
    ids=prepared['ids'];lo,hi=map(int,bounds)
    ix=np.flatnonzero((ids>=lo)&(ids<=hi))
    if not len(ix) or not np.array_equal(ids[ix],np.arange(lo,hi+1)):
        raise ValueError('Requested native frames missing from the input axis')
    return dict(indices=ix,target_index=target_index,
                sequence=prepared['x'][ix],rgb_support=span_support(prepared['support'],ix))

def describe(prepared,episode,probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(6,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Six finite probability values are required')
    ix=episode['indices'];j=episode['target_index']
    im=float(prepared['available'][ix].mean());coverage=float(prepared['mask'][ix].mean())
    value=float(p[j]);continuity=float(p[5]);opposite={2:4,4:2}.get(j)
    # Same gates as the frozen observer; no new tuned thresholds.
    if im<.8 or len(ix)<3 or coverage<.5: status='MOTION_OBSERVATION_INSUFFICIENT'
    elif continuity<.8: status='MOTION_CONTINUITY_UNCONFIRMED'
    elif value>=.8: status='DIRECTED_MOTION_OBSERVED'
    elif value<=.2 and opposite is not None and p[opposite]>=.8: status='OPPOSITE_DIRECTION_CANDIDATE'
    else: status='EXPECTED_MOTION_UNCONFIRMED'
    return dict(target_index=j,frame_ids=prepared['ids'][ix].tolist(),
        timestamp_us=prepared['ts'][ix].tolist(),
        start_seconds=float(prepared['ts'][ix[0]])/1e6,end_seconds=float(prepared['ts'][ix[-1]])/1e6,
        raw_observer_status=status,probabilities=p.tolist(),probability=value,
        continuity_probability=continuity,feature_coverage=coverage,image_coverage=im,
        unavailable_frame_ids=prepared['ids'][ix[~prepared['available'][ix]]].tolist(),
        partial_geometry_frame_ids=prepared['ids'][ix[~prepared['mask'][ix].all(1)]].tolist(),
        rgb_context_support=episode['rgb_support'],
        scope_attribution='NEURAL_CONTEXT_EVIDENCE_NOT_EXACT_FRAME_TRUTH',
        episode_selection='explicit body-context proposal, independent of tested motion score',
        complete_action=None,whole_clause_correctness=None,educational_error=None,
        correction_enabled=False,reference_video_comparisons=0,
        message_ko='선택한 동작 범위에서 학습된 움직임 근거를 확인했습니다. 앞뒤 영상의 영향과 실제 구간 안의 움직임은 구분해서 해석해야 합니다.' if status=='DIRECTED_MOTION_OBSERVED'
                   else '이 범위에서 필요한 움직임을 확실히 확인하지 못했습니다. 동작을 잘못했다는 판정은 아닙니다.')

class EpisodeQuery:
    def __init__(self):
        import torch
        self.model=RGBMotionNet('rgb_residual')
        self.model.load_state_dict(torch.load(R/'runs/rgb_residual_42/model.pt',map_location='cpu',weights_only=True))
        self.model.eval();self.model.requires_grad_(False)

    def query(self,items):
        start=time.perf_counter()
        values=predict(self.model,[e['sequence'] for _,e in items])
        seconds=time.perf_counter()-start
        return [describe(a,e,p) for (a,e),p in zip(items,values)],seconds
