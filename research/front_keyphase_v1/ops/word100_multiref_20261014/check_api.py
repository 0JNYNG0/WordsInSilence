from pathlib import Path
import sys,json,hashlib,time,faulthandler
faulthandler.dump_traceback_later(25)
print('IMPORTS_START',flush=True)
from unittest.mock import patch
U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100 as W,word100_analysis as A,word100_multiref as M,app as S
from fastapi.testclient import TestClient
print('IMPORTS_DONE',flush=True)
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];results=[]
def check(n,ok):
    assert ok,n
    checks.append(n)
    print(n,flush=True)
with patch.object(S.CameraWorker,'start',side_effect=AssertionError('NO_CAMERA')),patch.object(W,'recognise',side_effect=AssertionError('NO_INFERENCE')):
    c=TestClient(S.app)
    bank,pin=M.bank();check('baseline activated only after independent verification',pin==read(U/'INDEPENDENT_CHECK.json')['baseline_sha256'])
    rows=[r for r in c.get('/api/v1/practice100/history').json()['items'] if r['session_id']=='ses_79977f57893d']
    check('all original3 attempts still listed',len(rows)==3)
    for row in rows:
        url='/api/v1/practice100/saved/'+row['session_id']+'/'+row['trial_id']
        start=time.perf_counter();r=c.get(url);elapsed=time.perf_counter()-start
        check('analysis endpoint '+row['trial_id'],r.status_code==200)
        data=r.json();comp=data['analysis']['training_comparison']
        check('frozen9-person bank and3 axes '+row['trial_id'],comp['reference_count']==9 and len(comp['metrics'])==3 and comp['baseline_sha256']==pin)
        original,base=A.saved_result(row['session_id'],row['trial_id'])
        check('original model scores unchanged '+row['trial_id'],data['practice100']['logits']==original['practice100']['logits'] and comp['model_sha256']==W.PIN)
        check('saved analysis repeat identical '+row['trial_id'],c.get(url).json()==data)
        check('append-only sidecar exists '+row['trial_id'],base.with_name(base.name+'.training_dtw_'+pin[:12]+'.json').exists())
        results.append(dict(trial_id=row['trial_id'],retrospective=True,elapsed_seconds=elapsed,result=data))
    check('invalid session rejected',c.get('/api/v1/practice100/saved/ses_000000000000/'+rows[0]['trial_id']).status_code==404)
    check('no original files or frozen model modified',all(sha(p)==h for p,h in read(U/'ORIGINAL_GUARD.json').items()))
(U/'SAVED_ANALYSES.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf-8')
(U/'API_CHECK.json').write_text(json.dumps(dict(status='PASS',checks=checks,retrospective_trials=3,new_user_test_trials=0,camera=False,retraining=False,inference=False,baseline_sha256=pin),ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(dict(status='PASS',checks=len(checks),query_seconds=[r['elapsed_seconds'] for r in results],ratios=[[m['ratio'] for m in r['result']['analysis']['training_comparison']['metrics']] for r in results])))
faulthandler.cancel_dump_traceback_later()
