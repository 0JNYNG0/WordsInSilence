"""Freeze the two candidate-only sensor runs before any MediaPipe inference."""
from pathlib import Path
import json,hashlib,importlib.metadata
U=Path(__file__).resolve().parent;O=U.parent;LAB=O.parent;ROOT=LAB.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (U/'EXECUTION_FROZEN.json').exists()
    review=read(U/'CODE_REVIEW.json');assert review['approved_for_two_candidate_captures'] is True
    synthetic=read(U/'SYNTHETIC_CHECKS.json')
    assert synthetic['status']=='PASS_FAKE_FRAME_PARALLEL_CONTRACT'
    assert synthetic['checks_count']==45
    assert all(synthetic[k]==0 for k in ['actual_model_loads','actual_detector_calls','source_video_reads'])
    for p,h in synthetic['pins'].items():assert sha(p)==h,p
    plan=read(U/'INPUT_PLAN.json');assert len(plan['rows'])==2
    pins={}
    for mapping in [plan['pins'],review['code_pins']]:
        for p,h in mapping.items():
            key=str(Path(p).resolve());assert sha(key)==h,key
            assert key not in pins or pins[key]==h
            pins[key]=h
    paths=[U/n for n in ['CODE_REVIEW.json','INPUT_PLAN.json','SYNTHETIC_CHECKS.json','run_candidate.py','freeze_execution.py','parallel_capture.py']]
    paths += [ROOT/'Tools/models'/n for n in ['pose_landmarker_lite.task','hand_landmarker.task','face_landmarker.task']]
    paths += [LAB/p for p in ['config/contract.json','src/wis_lab/capture/landmarker.py','src/wis_lab/capture/face_landmarks.py',
        'src/wis_lab/capture/video_source.py','src/wis_lab/core/observation.py']]
    paths += [O/p for p in ['word100_seated_web_framing_20261102/adapter.py',
        'word100_hand_assignment_20261016/capture_raw.py','word100_hand_assignment_joint_20261016/assignment.py']]
    for p in paths:
        key=str(p.resolve());h=sha(p);assert key not in pins or pins[key]==h
        pins[key]=h
    result=dict(status='FROZEN_TWO_PUBLIC_SENSOR_CANDIDATES',pins=pins,
        packages={n:importlib.metadata.version(n) for n in ['mediapipe','numpy','av']},
        actual_candidate_runs=2,baseline_reruns=0,frame_skipping=0,resizing=0,changed_detector_options=0,
        gate='Both outputs exact with equal NaN positions, unchanged axes/masks/hand assignment and semantic sidecar fields.',
        speed_gate='No isolated causal speed claim from differently loaded historical baseline.',
        semantic_inference=0,training=0,original_live_changed=False,whole_goal_complete=False)
    with (U/'EXECUTION_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('Frozen',len(pins),'inputs and code pins')
if __name__=='__main__':main()
