"""Two public normal requests through old and admitted new servers, no camera."""
from pathlib import Path
import json, urllib.request, hashlib, uuid
D=Path(__file__).resolve().parent
def api(port,path,token=None,data=None):
    headers={'X-Session-Token':token} if token else {}
    if data is not None:headers['Content-Type']='application/json'
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,headers=headers,data=json.dumps(data).encode() if data is not None else None)
    with urllib.request.urlopen(req,timeout=40) as r:return json.load(r)
def main():
    client='public-check-'+uuid.uuid4().hex
    tokens={p:api(p,'/api/meta')['token'] for p in [8821,8822]}
    for p in tokens:
        s=api(p,'/api/status',tokens[p]);assert s['state']=='ready' and not s['camera_thread_active']
    assert s['endpoint_role_active'] and s['rest_stability_active'] and s['selective_retry_active']
    rows=[]
    for word in ['WORD0099','WORD1290']:
        q=[api(p,'/api/public-replay',tokens[p],dict(client=client,word=word)) for p in tokens]
        a,b=[x['result'] for x in q]
        ca={c['id']:c['status'] for c in a['lesson_acceptance']['criteria']}
        cb={c['id']:c['status'] for c in b['lesson_acceptance']['criteria']}
        row=dict(word=word,same_criteria=ca==cb,same_tcn=a['tcn_top3']==b['tcn_top3'],
                 overall_pass=b['overall_pass'],before_seconds=a['compute_seconds'],after_seconds=b['compute_seconds'],
                 no_quest_advance=all(x['outcome']['advanced'] is False for x in q),version=b.get('endpoint_role_version'))
        row['passed']=row['same_criteria'] and row['same_tcn'] and row['overall_pass'] and row['no_quest_advance'] and row['after_seconds']<1.5 and bool(row['version'])
        assert row['passed'],row;rows.append(row)
    for name in ['scene.js','hologram.js','live.js','companion.css']:
        with urllib.request.urlopen('http://127.0.0.1:8822/'+name) as r:body=r.read()
        assert hashlib.sha256(body).digest()==hashlib.sha256((D/'site'/name).read_bytes()).digest(),name
    s=api(8822,'/api/status',tokens[8822]);assert s['state']=='ready' and not s['camera_thread_active']
    out=dict(passed=True,rows=rows,physical_camera=False,private_data_read=False,ui_matches_local=True,
             scope='Public normal HTTP smoke check; broad public regressions in separate check files; not physical webcam')
    (D/'web/HTTP_INTEGRATION_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
