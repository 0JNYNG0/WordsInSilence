from pathlib import Path
import sys,json,time,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'public_frontal_evidence_repair_20261117'
sys.path.insert(0,str(P))
from check_http import request

def main():
    meta=request('/api/meta');deadline=time.monotonic()+90
    while True:
        status=request('/api/status',meta['token'])
        if status['state']!='preparing' or time.monotonic()>deadline:break
        time.sleep(.5)
    assert status['selective_retry_version']=='selective_front_rtmw_retry_v2'
    assert status['state']=='ready'and status['selective_retry_active']and status['evidence_repair_active']and status['frontal_scope_active']and not status['camera_thread_active'],status
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8821/live.js')as r:served=r.read()
    assert hashlib.sha256(served).digest()==hashlib.sha256((P/'site/live.js').read_bytes()).digest()
    rows=[]
    for word in ['WORD0099','WORD1290']:
        q=request('/api/public-replay',meta['token'],dict(client='selective-retry-public-check',word=word));result=q['result']
        passed=result['overall_pass']and result['evidence_repair_version']=='public_front_evidence_20261117'and not q['outcome']['advanced']
        rows.append(dict(word=word,passed=passed,seconds=result['compute_seconds'],quest_advanced=q['outcome']['advanced'],source=result['source']))
    final=request('/api/status',meta['token']);assert not final['camera_thread_active']
    q=dict(passed=all(r['passed']for r in rows),rows=rows,selective_retry_active=final['selective_retry_active'],old_readiness_js_preserved=True,physical_camera=False,server=json.loads((D/'SERVER.json').read_text(encoding='utf-8')))
    (D/'web/HTTP_CHECK.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(q,ensure_ascii=False));assert q['passed']
if __name__=='__main__':main()
