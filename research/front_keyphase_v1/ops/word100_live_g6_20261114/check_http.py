from pathlib import Path
import urllib.request,json,time,uuid
U=Path(__file__).resolve().parent;base='http://127.0.0.1:8821'
def request(path,token=None,data=None):
 headers={'X-Session-Token':token} if token else {}
 if data is not None:headers['Content-Type']='application/json'
 with urllib.request.urlopen(urllib.request.Request(base+path,headers=headers,data=None if data is None else json.dumps(data).encode()),timeout=90) as r:return json.load(r)
meta=request('/api/meta');status=request('/api/status',meta['token']);assert status['state']=='ready' and not status['camera_thread_active']
examples=[request(e['url']) for e in meta['examples']];start=time.perf_counter();replay=request('/api/public-replay',meta['token'],dict(client=str(uuid.uuid4()),word='WORD1133'))
checks=dict(latest_status=status['feedback_version']=='G6-frozen-G4',public_examples_latest=all(e['feedback_version']=='G6-frozen-G4' for e in examples),live_replay_latest=replay['result']['feedback_version']=='G6-frozen-G4',public_does_not_advance=replay['outcome']['advanced'] is False,no_camera=request('/api/status',meta['token'])['camera_thread_active'] is False)
out=dict(passed=all(checks.values()),checks=checks,replay_roundtrip_seconds=time.perf_counter()-start,model_seconds=replay['result']['compute_seconds'],camera_opened=False)
(U/'HTTP_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out);assert out['passed']
