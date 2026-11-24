from pathlib import Path
import urllib.request,json,time,hashlib
D=Path(__file__).resolve().parent
def request(path,token=None,data=None):
 h={'X-Session-Token':token}if token else{}
 if data is not None:h['Content-Type']='application/json'
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8821'+path,headers=h,data=json.dumps(data).encode()if data is not None else None),timeout=15)as r:return json.load(r)
def main():
 meta=request('/api/meta');deadline=time.monotonic()+45
 while True:
  s=request('/api/status',meta['token'])
  if s['state']!='preparing' or time.monotonic()>deadline:break
  time.sleep(.5)
 assert s['state']=='ready' and s['frontal_scope_active'] and s['evidence_repair_active'] and not s['camera_thread_active'],s
 with urllib.request.urlopen('http://127.0.0.1:8821/live.js')as r:js=r.read()
 assert hashlib.sha256(js).hexdigest()==hashlib.sha256((D/'site/live.js').read_bytes()).hexdigest()
 rows=[]
 for word in ['WORD0099','WORD1290']:
  o=request('/api/public-replay',meta['token'],dict(client='public-evidence-verification',word=word));q=o['result']
  passed=q['overall_pass'] and q['evidence_repair_version']=='public_front_evidence_20261117' and not o['outcome']['advanced']
  rows.append(dict(word=word,passed=passed,version=q['evidence_repair_version'],seconds=q['compute_seconds'],source=q['source'],quest_advanced=o['outcome']['advanced']))
 assert all(r['passed']for r in rows)
 final=request('/api/status',meta['token']);assert not final['camera_thread_active']
 q=dict(passed=True,rows=rows,readiness_ui_served=True,physical_camera=False,camera_active=False,server=json.loads((D.parent.parent/'web_tutorial_live_integration_20261113/SERVER.json').read_text(encoding='utf-8')))
 (D/'web/HTTP_CHECK.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':main()
