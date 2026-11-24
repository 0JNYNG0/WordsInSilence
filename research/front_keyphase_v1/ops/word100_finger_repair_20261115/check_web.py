from common import *
import urllib.request,time,uuid
base='http://127.0.0.1:8821'
def get(path,token=None):
 req=urllib.request.Request(base+path,headers={'X-Session-Token':token}if token else {})
 return json.load(urllib.request.urlopen(req,timeout=90))
meta=get('/api/meta');token=meta['token'];before=get('/api/status',token)
assert before['state']=='ready',before['state']
started=time.perf_counter()
req=urllib.request.Request(base+'/api/public-replay',data=json.dumps(dict(client=str(uuid.uuid4()),word='WORD1133')).encode(),headers={'Content-Type':'application/json','X-Session-Token':token},method='POST')
payload=json.load(urllib.request.urlopen(req,timeout=90));elapsed=time.perf_counter()-started
result=payload['result'];write(U/'PUBLIC_WEB_RESULT.json',result)
assert result['feedback_version']=='G6-finger-20261115'
assert result['training_use']is False and payload['outcome']['advanced']is False
assert result['coaching']['whole_sign_correct']is None
after=get('/api/status',token)
assert after['state']=='ready'
write(U/'WEB_CHECK.json',dict(passed=True,public_source=result.get('source'),feedback_version=result['feedback_version'],seconds=elapsed,timing=result['timing'],cards=[dict(id=c['id'],status=c['display_status'])for c in result['cards']],cues=result['coaching']['cues'],camera_started=False,quest_advanced=False,scope='Stored public RTMW input through live TCN/DTW/qualified-evidence HTTP path; excludes capture/extraction/browser rendering'))
print(json.dumps(read(U/'WEB_CHECK.json'),ensure_ascii=False))
