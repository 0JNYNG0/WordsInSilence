from pathlib import Path
import urllib.request,urllib.error,json,time,hashlib
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8821';token=None
def call(path,body=None,auth=True,origin=None):
 h={}
 if auth and token:h['X-Session-Token']=token
 if body is not None:h['Content-Type']='application/json';h['Origin']=origin or BASE
 req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=h)
 try:
  with urllib.request.urlopen(req,timeout=60) as r:raw=r.read();return r.status,json.loads(raw) if 'json' in r.headers.get('Content-Type','') else raw
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
checks=[]
def check(name,value):
 checks.append({'name':name,'passed':bool(value)});assert value,name
code,m=call('/api/meta');check('meta',code==200);token=m['token']
code,s=call('/api/status');check('model_ready',code==200 and s['state']=='ready');check('camera_off',not s['camera_thread_active'])
check('unauthenticated_status_blocked',call('/api/status',auth=False)[0]==403)
client='public-integration-check-1113'
check('cross_origin_command_blocked',call('/api/start',dict(client=client,word='WORD1567',mode='quest'),origin='https://untrusted.example')[0]==403)
check('start_without_camera_rejected',call('/api/start',dict(client=client,word='WORD1567',mode='quest'))[0]==400)
check('private_video_not_exposed',call('/recorded.mp4')[0]==404)
check('path_traversal_not_served',call('/../server.py')[0]==404)
rows=[]
for w in m['words']:
 t=time.perf_counter();code,x=call('/api/public-replay',dict(client=client,word=w['id']));elapsed=time.perf_counter()-t
 check('replay_'+w['id'],code==200)
 r=x['result'];check('real_tcn_'+w['id'],len(r['tcn_top3'])==3);check('feedback_'+w['id'],len(r['cards'])>0);check('alignment_'+w['id'],len(r['path'])>0);check('public_cannot_advance_'+w['id'],not x['outcome']['advanced'] and x['quest']['index']==0)
 check('same_frame_axes_'+w['id'],len(r['query_xy'])==len(r['timestamp_us'])==len(r['frame_id']))
 (U/'public_checks').mkdir(exist_ok=True);(U/'public_checks'/f"{w['id']}.json").write_text(json.dumps(x,ensure_ascii=False),encoding='utf-8')
 rows.append(dict(word=w['id'],label=w['label'],frames=len(r['timestamp_us']),cards=len(r['cards']),top1=r['tcn_top3'][0]['label'],http_and_saved_pose_inference_seconds=elapsed))
code,s=call('/api/status');check('camera_still_off',not s['camera_thread_active']);check('quest_still_zero',s['quest']['index']==0)
for p,sha in json.loads((U/'PROTECTED.json').read_text()).items():check('preserved_'+Path(p).name,hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha)
out=dict(passed=True,checks=checks,public_replays=rows,camera_opened=False,scope='New adapter API and stored public joint inference; not fresh accuracy evaluation, not capture/end-to-screen latency')
(U/'HTTP_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
