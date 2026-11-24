from pathlib import Path
import urllib.request,json,time,hashlib
U=Path(__file__).resolve().parent;B='http://127.0.0.1:8821'
def request(path,token=None,data=None):
    headers={'X-Session-Token':token} if token else {}
    if data is not None:headers['Content-Type']='application/json'
    return json.load(urllib.request.urlopen(urllib.request.Request(B+path,headers=headers,data=None if data is None else json.dumps(data).encode()),timeout=30))
meta=request('/api/meta');token=meta['token'];status=request('/api/status',token)
assert status['state']=='ready' and not status['camera_thread_active']
before=status['quest'];start=time.perf_counter()
replay=request('/api/public-replay',token,dict(client='public-local-direction-check',word='WORD1133'))
elapsed=time.perf_counter()-start
r=replay['result'];assert r['evidence_version']=='positive_shape_and_contiguous_direction_v2'
assert r['coaching']['version']=='evidence_bound_contiguous_coaching_v2' and not r['coaching']['cues']
assert replay['quest']==before and replay['outcome']['advanced'] is False
assert r['source']=='public_integration_replay' and r['rgb_model_calls']==0 and r['reference_video_reads']==0
after=request('/api/status',token);assert not after['camera_thread_active']
for example in meta['examples']:
    saved=request(example['url']);assert saved['coaching']['version']=='evidence_bound_contiguous_coaching_v2'
out=dict(passed=True,url=B,camera_opened=False,quest_unchanged=True,public_replay_seconds=elapsed,
    inference='One current server cached-public-joint replay, no RTMW re-extraction, not webcam end-to-end latency',
    served_saved_examples=len(meta['examples']),server=json.loads((U.parent/'web_tutorial_live_integration_20261113/SERVER.json').read_text('utf-8')))
with (U/'SERVED_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
