from pathlib import Path
import urllib.request,urllib.error,json,uuid,hashlib,ast
U=Path(__file__).resolve().parent;ROOT=U.parents[4];OPS=U.parents[1];BASE='http://127.0.0.1:8821'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def request(path,body=None):
 headers={'Content-Type':'application/json'}
 if 'meta'in globals():headers['X-Session-Token']=meta['token']
 req=urllib.request.Request(BASE+path,data=json.dumps(body).encode()if body else None,headers=headers)
 with urllib.request.urlopen(req,timeout=90)as f:return json.load(f)
meta=request('/api/meta');state=request('/api/status');assert state['state']=='ready'and not state['camera_thread_active']
assert 'SEN0354'not in {x['id']for x in meta['words']}
assert 'SEN0354'not in {x['id']for x in meta['planned_words']}
assert not any('안녕하세요'in x['label']for x in meta['examples'])
assert [r['id']for r in meta['quest']['steps']]==['WORD0099','WORD1290']
rows=[]
for word in ['WORD0099','WORD1290']:
 z=request('/api/public-replay',dict(client=str(uuid.uuid4()),word=word));q=z['result']
 assert q['overall_pass']and q['tcn_top3'][0]['word']==word and not z['outcome']['advanced']and z['quest']['index']==0
 rows.append(dict(word=word,normal_pass=q['overall_pass'],top1=q['tcn_top3'][0]['word'],criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},compute_seconds=q['compute_seconds']))
blocked=[]
for path in ['/ref/SEN0354','/hello-examples/normal.json','/hello-media/reference.mp4']:
 try:request(path);raise AssertionError('Excluded demo route still accessible')
 except urllib.error.HTTPError as e:assert e.code==404;blocked.append(path)
pins={}
for branch in ['apple_v1','completion_v1']:
 a=read(U.parent/branch/'web/ADMISSION.json');assert a['admitted']
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in a['pins'].items());pins[branch]=True
for p in [ROOT/'demo_launcher.py',OPS/'web_tutorial_live_integration_20261113/server.py']:ast.parse(p.read_text('utf-8-sig'))
out=dict(hello_excluded=True,hello_archived=(U.parent/'hello_capture_v1/CHECKPOINT.json').is_file(),rows=rows,blocked_routes=blocked,model_pins_unchanged=pins,camera_opened=False,training=False,user_data=False,quest_not_advanced=True)
(U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
