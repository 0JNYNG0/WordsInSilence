"""Freeze reviewed code and inherited immutable dependencies, no inference."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
O=U.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not (U/'EXECUTION_FROZEN.json').exists()
    review=read(U/'CODE_REVIEW.json')
    assert review['approved_for_one_public_integration'] is True
    pins={}
    def add(path,value):
        key=str(Path(path).resolve())
        assert sha(key)==value,key
        assert key not in pins or pins[key]==value,key
        pins[key]=value
    historical=[O/'word100_partial_order_live_20261102/EXECUTION_FROZEN.json',
        O/'word100_local_event_learning_20261103/FINAL.json',
        O/'word100_local_event_guide_adapter_20261103/FROZEN.json']
    for source in historical:
        add(source,sha(source))
        for p,h in read(source)['pins'].items():add(p,h)
    for p,h in review['code_pins'].items():add(p,h)
    for name in ['learner_adapter.py','order_event_worker.py','local_runtime.py','run_public.py','freeze_execution.py']:
        path=U/name;add(path,sha(path))
    for name in ['CODE_REVIEW.json','PUBLIC_INPUT_FROZEN.json','PUBLIC_INPUT_PLAN.json','STRUCTURAL_CHECKS.json','WRAPPER_CHECKS.json']:
        path=U/name;add(path,sha(path))
    source=read(U/'PUBLIC_INPUT_FROZEN.json')
    add(source['source'],source['source_sha256'])
    value=dict(status='FROZEN_BEFORE_SINGLE_NEW_PUBLIC_INTEGRATION',pins=pins,source=source,
        maximum_new_public_jobs=1,model_choice_or_tuning=False,
        goal='One newly extracted public video through unchanged observation and learned DTW plus frozen local-event head.',
        expected=dict(native_axis_exact=True,learned_DTW_calls=1,new_local_event_forwards=1,
            original_result_fields_preserved=True,reference_video_comparisons=0),
        performance_claim='Integration and timing only; no new accuracy/gold semantics.',
        original_web_changed=False,whole_goal_complete=False)
    with (U/'EXECUTION_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print('Frozen single public integration',len(pins),'pins')

if __name__=='__main__':main()
