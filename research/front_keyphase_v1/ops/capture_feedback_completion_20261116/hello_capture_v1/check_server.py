from pathlib import Path
import urllib.request,json,uuid
U=Path(__file__).resolve().parent;root='http://127.0.0.1:8821'
def read(path,body=None):
 headers={'Content-Type':'application/json'}
 if 'meta'in globals():headers['X-Session-Token']=meta['token']
 req=urllib.request.Request(root+path,data=json.dumps(body).encode()if body else None,headers=headers)
 with urllib.request.urlopen(req,timeout=90)as r:return json.load(r)
meta=read('/api/meta');assert read('/api/status')['state']=='ready';rows=[]
for word in ['WORD0099','WORD1290','SEN0354']:
 z=read('/api/public-replay',dict(client=str(uuid.uuid4()),word=word));q=z['result'];ref=read('/ref/'+word)
 assert z['quest']['index']==0 and not z['outcome']['advanced']
 row=dict(word=word,overall=q['overall_pass'],practice_complete=q.get('exercise_complete'),top1=(q['tcn_top3'][0]['word']if q['tcn_top3']else None),criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},seconds=q['compute_seconds'],reference_frames=len(ref['timestamp_us']),version=q['feedback_version'])
 if word=='SEN0354':assert not row['overall']and row['practice_complete']and row['top1']is None
 else:assert row['overall']and row['top1']==word
 rows.append(row)
(U/'web/API_CHECK.json').write_text(json.dumps(dict(rows=rows,quest_not_advanced=True,physical_camera=False),ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
