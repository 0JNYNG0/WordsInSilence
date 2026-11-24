from pathlib import Path
import urllib.request,json,time
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8821'
def get(path,token=None):
 r=urllib.request.Request(BASE+path,headers={'X-Session-Token':token} if token else {})
 return json.load(urllib.request.urlopen(r,timeout=30))
m=get('/api/meta');s=get('/api/status',m['token']);assert not s['camera_thread_active']
initial_index=s['quest']['index'];checks=[]
for e in m['examples']:
 x=get(e['url']);assert x['coaching']['version']=='evidence_bound_coaching_v1'
 assert x['source']!='user_inference_only'
 checks.append(dict(example=e['id'],status=x['coaching']['status'],cues=len(x['coaching']['cues'])))
# A single actual-model integration call is sufficient for the new output wrapper.
req=urllib.request.Request(BASE+'/api/public-replay',data=json.dumps(dict(client='coaching-integration-public',word='WORD1133')).encode(),headers={'X-Session-Token':m['token'],'Content-Type':'application/json','Origin':BASE})
actual=json.load(urllib.request.urlopen(req,timeout=30));assert actual['result']['coaching']['version']=='evidence_bound_coaching_v1'
assert actual['result']['tcn_top3'] and actual['result']['cards'] and actual['result']['path']
assert not actual['outcome']['advanced'] and actual['quest']['index']==initial_index
end=get('/api/status',m['token']);assert not end['camera_thread_active']
out=dict(passed=True,public_examples=checks,actual_model_wrapper_passed=True,physical_camera_opened=False,quest_not_advanced=True,
 model_training=False,scope='Single public cached-joint inference for modified API integration; not new accuracy evaluation',
 server=json.loads((U.parent/'web_tutorial_live_integration_20261113/SERVER.json').read_text('utf-8-sig')))
(U/'WEB_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
