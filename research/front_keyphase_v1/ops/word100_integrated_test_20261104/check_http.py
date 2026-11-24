"""Live local transport checks; never uploads or reads a user recording."""
from pathlib import Path
import http.client,json,hashlib
U=Path(__file__).resolve().parent;checks=[]
def req(method,path,body=None,headers=None):
 c=http.client.HTTPConnection('127.0.0.1',8812,timeout=8);c.request(method,path,body=body,headers=headers or {});r=c.getresponse();s=r.status;data=r.read();c.close();return s,data
def check(n,b):assert b,n;checks.append(n)
s,d=req('GET','/api/state');state=json.loads(d);token=state['token'];check('ready',s==200 and state['ready'])
check('external origin rejected',req('GET','/api/state',headers={'Origin':'https://invalid.example'})[0]==403)
check('foreign host rejected',req('GET','/api/state',headers={'Host':'invalid.example'})[0]==403)
check('old private session unavailable',req('GET','/media/session/'+'0'*32)[0]==404)
check('directory escape rejected',req('GET','/../STATE.json')[0] in [400,403,404])
check('preview no token rejected',req('POST','/api/preview',b'xx')[0]==403)
check('preview query rejected',req('POST','/api/preview?x=1',b'xx',{'X-Session-Token':token})[0]==403)
check('preview bounded before body',req('POST','/api/preview',b'',{'X-Session-Token':token,'Content-Length':'300001'})[0]==413)
check('analysis no token rejected',req('POST','/api/analyze?word=WORD1133&condition=learned_motion',b'xx')[0]==403)
check('unsupported word rejected',req('POST','/api/analyze?word=WORD0003&condition=learned_motion',b'xx',{'X-Session-Token':token})[0]==400)
check('duplicate target rejected',req('POST','/api/analyze?word=WORD1133&word=WORD0739&condition=learned_motion',b'xx',{'X-Session-Token':token})[0]==400)
s,b=req('GET','/media/example/WORD1133',headers={'Range':'bytes=0-63'});check('video range playback',s==206 and len(b)==64)
s,b=req('GET','/api/example/WORD1133');r=json.loads(b);check('example remains public',r['ui_source_identity']['case_kind']=='saved_public_replay')
check('all100 class candidates',all(r['tcn'][k]['candidate_count']==100 for k in ['C','T3F']))
check('TCN target blind',r['tcn']['requested_word_used'] is False)
check('no private job made',(U/'private_sessions').exists() is False)
(U/'HTTP_VERIFIED.json').write_text(json.dumps(dict(status='PASS',count=len(checks),checks=checks,accepted_user_uploads=0,hardware_camera=0),ensure_ascii=False,indent=2),encoding='utf8');print('PASS',len(checks))
