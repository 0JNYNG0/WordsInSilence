"""One public-trained network pipeline, one learned-flow DTW, native user time."""
from functools import lru_cache
from pathlib import Path
import time,numpy as np
from common import G as U,O,P,V,R,read,write
import demo_adapter as shared_capture
from rgb_runtime import RGBMotionObserver
from engine import WORDS

@lru_cache(maxsize=1)
def motion():return RGBMotionObserver('rgb_residual',42)

def qualify(result,arrays):
    """Keep numeric predictions; separate motion occurrence from guide binding."""
    stages={s['guide_id']:s for s in result['stages']}
    for m in result['motion_observations']:
        s=stages[m['guide_id']]
        linked=s['neural_stage_agreement']>=.5 and s['image_time_coverage']>=.8 and s['unique_query_frames']>=2
        m['guide_time_binding']='SUPPORTED' if linked else 'UNCONFIRMED'
        if m['status']=='DIRECTED_MOTION_OBSERVED' and not linked:
            m['display_status']='MOVEMENT_SEEN_STAGE_UNCONFIRMED'
            m['display_message']='비슷한 이동은 보이지만, 설명의 이 구간에서 한 동작인지 확실하지 않습니다.'
        elif m['status']=='DIRECTED_MOTION_OBSERVED':
            m['display_status']=m['status'];m['display_message']='이 방향의 이동이 보입니다. 동작 전체의 순서·완료 여부까지 확인했다는 뜻은 아닙니다.'
        else:
            m['display_status']=m['status'];m['display_message']='해당 이동을 충분히 확인하지 못했습니다. 다른 동작을 했다고 확정하지 않습니다.'
    for s in result['stages']:s['motions']=[m for m in result['motion_observations'] if m['guide_id']==s['guide_id']]
    # Observation inventory only, without filling coordinates or changing grades.
    no_pose=np.flatnonzero(~np.asarray(arrays['mask'],bool)[:,:61].any(1));result['joint_evidence_gaps']=[]
    if len(no_pose):
        for span in np.split(no_pose,np.flatnonzero(np.diff(no_pose)>1)+1):
            if not len(span):continue
            result['joint_evidence_gaps'].append(dict(start_seconds=float(arrays['timestamp_us'][span[0]]/1e6),
                end_seconds=float(arrays['timestamp_us'][span[-1]]/1e6),frames=span.tolist(),
                current_image_available=bool(np.asarray(arrays['image_available'])[span].all()),status='JOINTS_UNAVAILABLE_IMAGE_TIME_ESTIMATE_ONLY'))
    result['sequence_verdict']=dict(status='PARTIAL_FEATURE_FEEDBACK_ONLY',whole_sign_correctness=None,
        message='손모양·관계·이동 일부를 배운 범위와 비교했습니다. 전체 순서와 필수 반복 횟수의 정오 판정은 아직 검증 중입니다.')
    result['observer_input_goal_blind']=True;result['trained_recorded_ids']=list(range(1,10))
    result['runtime_individual_reference_video_comparisons']=0;result['model_version']='learned-guide-rgb-motion-20261102-fixed42'
    return result

def analyze(source,directory,word,condition='learned_motion',progress=None):
    assert condition=='learned_motion'
    shared_capture.U=U
    start=time.perf_counter()
    if progress:progress(dict(stage='detecting'))
    result=shared_capture.analyze(source,directory,word,'rgb_geometry',progress)
    a=dict(np.load(Path(directory)/'analysis.npz'))
    result=motion().attach(result,a,a['retained_probability'],a['relation_probability'])
    result=qualify(result,a)
    result['latency']['base_analysis_seconds']=result['latency']['total_analysis_seconds']
    result['latency']['total_analysis_seconds']=time.perf_counter()-start
    result['latency']['motion_head_seconds']=result['motion_head_seconds']
    result['training']=False;result['tuning']=False
    write(Path(directory)/'FEEDBACK.json',result)
    if progress:progress(dict(stage='complete'))
    return result

def warmup():
    shared_capture.U=U;out=shared_capture.shared.warmup();shared_capture.engine('rgb_geometry');motion();return out
