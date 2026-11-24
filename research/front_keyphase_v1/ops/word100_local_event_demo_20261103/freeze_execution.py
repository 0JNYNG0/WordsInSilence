"""Freeze one new full-pipeline candidate; old evaluation outputs stay read-only."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent; O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (U/'EXECUTION_FROZEN.json').exists()
    review=read(U/'CODE_REVIEW.json');assert review['approved_for_one_public_integration'] is True
    scope=read(U/'SCOPE.json');assert scope['integration']['planned_new_candidate_runs']==1
    assert read(U/'ADAPTER_CHECKS.json')['status']=='PASS_SYNTHETIC_DEMO_ADAPTER_CONTRACT'
    pins={}
    def add(p,h=None):
        key=str(Path(p).resolve());actual=sha(key)
        if h is not None:assert actual==h,key
        assert key not in pins or pins[key]==actual,key
        pins[key]=actual
    for name in ['word100_local_event_runtime_20261103','word100_capture_parallel_20261103']:
        path=O/name/'EXECUTION_FROZEN.json';add(path)
        for p,h in read(path)['pins'].items():add(p,h)
    for p,h in review['code_pins'].items():add(p,h)
    for p,h in review['required_resource_pins'].items():add(p,h)
    for p,h in read(U/'ADAPTER_CHECKS.json')['pins'].items():add(p,h)
    for name in ['learner_adapter.py','order_event_worker.py','local_runtime.py','parallel_capture.py','run_demo.py','demo.html','demo.js',
                 'run_public.py','freeze_execution.py','SCOPE.json','CODE_REVIEW.json','ADAPTER_CHECKS.json']:
        add(U/name)
    old=read(O/'word100_local_event_runtime_20261103/PUBLIC_INPUT_FROZEN.json')
    item=dict(old,role=scope['integration']['role'],source_role='public_runtime_inference',candidate='isolated_demo_full_pipeline',
        purpose=scope['integration']['purpose'])
    assert item['source']==scope['integration']['source'] and item['source_sha256']==scope['integration']['source_sha256']
    assert item['word_id']==scope['integration']['word_id']
    with (U/'PUBLIC_INPUT_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(item,f,ensure_ascii=False,indent=2)
    add(U/'PUBLIC_INPUT_FROZEN.json');add(item['source'],item['source_sha256'])
    value=dict(status='FROZEN_ONE_ISOLATED_DEMO_PUBLIC_CANDIDATE',pins=pins,
        maximum_new_public_jobs=1,unchanged_baseline_reruns=0,training=0,candidate_selection=False,
        expected=dict(native_axis_exact=True,learned_DTW_calls=1,new_local_event_forwards=1,reference_video_comparisons=0),
        performance_claim='Actual new-pipeline latency and integration only; historical load differs; no new accuracy proof.',
        old_web_changed=False,whole_goal_complete=False)
    with (U/'EXECUTION_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print('Frozen new demo public candidate',len(pins),'pins')
if __name__=='__main__':main()
