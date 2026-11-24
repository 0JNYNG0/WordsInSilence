from pathlib import Path
import urllib.request,json,time
N=Path(__file__).resolve().parent
def request(path,token=None,data=None):
 h={'X-Session-Token':token}if token else{}
 if data is not None:h['Content-Type']='application/json'
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8821'+path,headers=h,data=json.dumps(data).encode()if data is not None else None),timeout=15)as r:return json.load(r)
def main():
 meta=request('/api/meta');s=request('/api/status',meta['token']);assert s['state']=='ready'and s['frontal_scope_active']and not s['camera_thread_active'];rows=[]
 for word in ['WORD0099','WORD1290']:
  output=request('/api/public-replay',meta['token'],dict(client='frontal-source-verification',word=word));q=output['result']
  rows.append(dict(word=word,passed=q['overall_pass'],scope=q.get('feedback_scope'),source=q['source'],quest_advanced=output['outcome']['advanced'],seconds=q['compute_seconds']))
 assert all(r['passed']and r['scope']=='single_front_camera'and not r['quest_advanced']for r in rows)
 final=request('/api/status',meta['token']);assert not final['camera_thread_active']
 result=dict(passed=True,rows=rows,camera_active=False,physical_camera=False,private_data=False,server=json.loads((N.parent.parent/'web_tutorial_live_integration_20261113/SERVER.json').read_text(encoding='utf-8')))
 (N/'web/HTTP_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
