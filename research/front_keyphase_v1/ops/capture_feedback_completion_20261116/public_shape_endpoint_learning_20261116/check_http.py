from pathlib import Path
import urllib.request,json,time
D=Path(__file__).resolve().parent
def request(path,token=None,data=None):
 h={'X-Session-Token':token}if token else{}
 if data is not None:h['Content-Type']='application/json'
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8821'+path,headers=h,data=json.dumps(data).encode()if data is not None else None),timeout=10)as r:return json.load(r)
def main():
 m=request('/api/meta');deadline=time.monotonic()+100
 while True:
  s=request('/api/status',m['token'])
  if s['state']!='preparing':break
  if time.monotonic()>deadline:raise RuntimeError('server preparing too long')
  time.sleep(1)
 assert s['state']=='ready'and s['hand_form_active']and s['visibility_active']and not s['camera_thread_active'];rows=[]
 for w in['WORD0099','WORD1290']:
  q=request('/api/public-replay',m['token'],dict(client='public-shape-verify-20261116',word=w))['result'];rows.append(dict(word=w,overall_pass=q['overall_pass'],module=q.get('hand_form_module'),visibility=q.get('visibility_module'),seconds=q['compute_seconds'],criteria={c['id']:c['status']for c in q['lesson_acceptance']['criteria']},source=q['source']))
 assert all(r['overall_pass']for r in rows)
 output=dict(state=s['state'],camera_active=s['camera_thread_active'],hand_form_active=s['hand_form_active'],visibility_active=s['visibility_active'],rows=rows,physical_camera=False,private_data=False)
 (D/'web/HTTP_CHECK.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(output,ensure_ascii=False))
if __name__=='__main__':main()
