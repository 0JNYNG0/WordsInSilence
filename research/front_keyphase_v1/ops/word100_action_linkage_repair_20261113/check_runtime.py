"""Four public cached-joint replays, one neural pass each, camera never opened."""
from pathlib import Path
import sys,json,hashlib,copy,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_action_feedback_demo_20261108'
sys.path.insert(0,str(D));sys.path.insert(0,str(O/'word100_actionable_feedback_20261113'))
from runtime_adapter import FeedbackConsumer,CaptureService,OriginalCapture,OriginalConsumer,attach,bind_dependency
from localized_direction import refine
from component_coverage import apply_coverage
from positive_evidence import refine as old_refine
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(U/'LOCAL_DIRECTION_COMPARISON_V2.json')['passed']
assert OriginalConsumer.channels.__globals__['refine'] is old_refine
assert FeedbackConsumer.channels.__code__ is OriginalConsumer.channels.__code__
assert CaptureService.prepare.__code__ is OriginalCapture.prepare.__code__
for k,v in OriginalConsumer.channels.__globals__.items():
    if k!='refine':assert FeedbackConsumer.channels.__globals__[k] is v,k
for k,v in OriginalCapture.prepare.__globals__.items():
    if k!='FeedbackConsumer':assert CaptureService.prepare.__globals__[k] is v,k
saved={}
def paired(base,*args):
    saved['previous']=apply_coverage(old_refine(base,*args))
    return refine(base,*args)
class ComparisonConsumer(FeedbackConsumer):
    channels=bind_dependency(FeedbackConsumer.channels,'refine',paired)
start=time.perf_counter();consumer=ComparisonConsumer();initialization=time.perf_counter()-start
checks=[];rows=[]
for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
    path=D/'parallel'/word/'rtmw/observations.npz'
    start=time.perf_counter()
    with np.load(path) as raw:result=consumer.raw(raw,word)
    result=attach(result);seconds=time.perf_counter()-start
    assert result['rgb_model_calls']==0 and result['reference_video_reads']==0
    assert result['overall_pass'] is False and result['correction_validated'] is False
    assert not result['coaching']['cues'],word
    for a,b in zip(saved['previous']['cards'],result['cards']):
        assert a['shape_evidence']==b['shape_evidence']
        if b.get('observation_quarantine_overlap'):continue
        assert a['display_status']==b['display_status'],word
    assert result['evidence_version']=='positive_shape_and_contiguous_direction_v2'
    out=U/'public_runtime';out.mkdir(exist_ok=True)
    with (out/(word+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
    rows.append(dict(word=word,frames=len(result['timestamp_us']),seconds=seconds,source_sha256=sha(path),
        states=[c['display_status'] for c in result['cards']],top1=result['tcn_top3'][0]['word'],camera_opened=False,
        neural_passes=1,cues=len(result['coaching']['cues'])))
    print(word,'public replay verified',flush=True)
assert OriginalConsumer.channels.__globals__['refine'] is old_refine
out=dict(passed=True,rows=rows,initialization_seconds=initialization,isolated_dependency_binding=True,
    camera_opened=False,private_data_used=False,network_weights_changed=False,
    meaning='Replay of already exposed public cached joints. Not end-to-end webcam latency or new learner accuracy.')
with (U/'PUBLIC_RUNTIME_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
