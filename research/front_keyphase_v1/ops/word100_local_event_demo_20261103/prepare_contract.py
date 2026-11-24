"""Pin the isolated demonstration and its single public integration input."""
from pathlib import Path
import json, hashlib
U=Path(__file__).resolve().parent; O=U.parent
AK=O/'word100_local_event_runtime_20261103'; AL=O/'word100_capture_parallel_20261103'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'SCOPE.json').exists()
    old=read(AK/'PUBLIC_RUN_RESULT.json'); old_result=Path(old['result'])
    source=AK/'public_input/NIA_SL_WORD1133_REAL10_F.mp4'
    assert sha(source)==old['source_sha256']
    assert read(AL/'FINAL.json')['exact_observation_preserved'] is True
    scope=dict(status='ISOLATED_DEMO_IMPLEMENTATION_IN_PROGRESS',
        objective_ko='학습된 특징·기준흐름 DTW·시간별움직임을 새 영상에서 분석하고 설명 문구와 원래 시간에 연결하는 별도 연구 데모.',
        isolation_ko='C/T3F/기존8804/8808/8809 및 AK/AL 결과·코드는 읽기전용. 새 단위만 수정.',
        proposed_port=8810,
        supported_words=['WORD0739','WORD1133','WORD1567','WORD2523'],
        private_policy=dict(role='user_inference_only',training=False,tuning=False,candidate_selection=False,
            historical_private_access=False,automatic_camera=False,external_transmission=False),
        sensor=dict(same_frame_detectors_parallel=True,frame_skip=0,original_sensor_options=True,full_source_extent=True),
        classifier=dict(existing_frozen_heads=True,new_training=0,new_learned_prototype_DTW_per_input=1,reference_video_comparisons=0),
        display=dict(native_timestamp_us=True,source_clause_or_reference_context_explicit=True,
            local_occurrence_is_whole_clause=False,whole_direction_and_local_occurrence_separate=True,
            missing_is_correct=False,whole_sign_correctness=None,contact=None,required_repetition_count=None),
        integration=dict(planned_new_candidate_runs=1,source=str(source),source_sha256=sha(source),word_id='WORD1133',
            registered_id=10,role='already_exposed_public_development_integration',
            purpose='New AM full pipeline and original-time output preservation, actual total latency; not accuracy or candidate tuning.',
            unchanged_baseline_reruns=0,old_result=str(old_result),old_result_sha256=sha(old_result),
            old_results_read_only=True,new_head_or_threshold_selection=False),
        saved_example=dict(word_id='WORD1133',source=str(source),source_sha256=sha(source),result=str(old_result),result_sha256=sha(old_result),
            provenance='saved_public_replay',new_inference=False),
        gates=['inference-only roles and paths', 'whole analyze/warmup/close serialization',
            'sensor/extract globals restored on error', 'source/target/native-axis identity',
            'local observation versus whole-direction/whole-action separation',
            'browser visibility and correct native media times', 'same public candidate once after frozen review'],
        whole_goal_complete=False)
    with (U/'SCOPE.json').open('x',encoding='utf-8') as f: json.dump(scope,f,ensure_ascii=False,indent=2)
    print('Isolated demo scope and one already-exposed public candidate input fixed; no model run')
if __name__=='__main__': main()
