"""Targeted real Store expiry checks using a fake camera and local synthetic sessions."""
import hashlib,json,sys,threading,time
from pathlib import Path
here=Path(__file__).resolve().parent;web=here.parents[1]
sys.path.insert(0,str(web/'src'))
from wis_lab.service import app as A
phase=sys.argv[1];root=here/('synthetic_'+phase);root.mkdir(exist_ok=False)
store=A.Store.__new__(A.Store)
store.lock=threading.RLock();store.sessions={};store.trials={};store.telemetry_write_failures=0
store.camera_session='ses_recorded'
class FakeCamera:
    stopped=0
    def stop(self):self.stopped+=1
camera=FakeCamera();store.camera=camera;A.store=store
for sid,ids,expired in [('ses_idle',[],True),('ses_recorded',['tri_complete','tri_running'],True),('ses_live',[],False)]:
    folder=root/sid;folder.mkdir();(folder/'trials').mkdir()
    s={'session_id':sid,'run_dir':str(folder),'purpose':'development','participant_id':'synthetic','client_id':'synthetic_client',
       'state':'OPEN','camera':sid=='ses_recorded','store_video':False,'trial_ids':list(ids),'request_ids':{'do_not_persist':'private-internal'},
       'created_epoch':time.time(),'lease_touched_monotonic':time.monotonic()-(A.LEASE_TTL_S+10 if expired else 0)}
    store.sessions[sid]=s
    initial={k:v for k,v in s.items() if k!='request_ids'};initial['trial_ids']=[]
    A.write_json(str(folder/'session.json'),initial)
for tid,state in [('tri_complete','FEEDBACK'),('tri_running','RECORDING')]:
    store.trials[tid]={'trial_id':tid,'session_id':'ses_recorded','state':state,'clock_id':'synthetic_clock','error':None,'end_reason':None,'timing':{},'result':{'synthetic':True} if state=='FEEDBACK' else None}
live_before=(root/'ses_live/session.json').read_bytes()
calls=[];writer=A.write_json
def tracked(path,obj):calls.append(path);return writer(path,obj)
A.write_json=tracked
checks=[]
def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
dead=store.reap_expired()
saved={sid:json.loads((root/sid/'session.json').read_text(encoding='utf-8')) for sid in store.sessions}
check('zero-trial expired session persisted CLOSED',saved['ses_idle']['state']=='CLOSED' and saved['ses_idle'].get('closed_reason')=='LEASE_EXPIRED')
check('recorded expired session persisted CLOSED and cameraFalse',saved['ses_recorded']['state']=='CLOSED' and saved['ses_recorded']['camera'] is False)
check('every original trial preserved in exact order',saved['ses_recorded']['trial_ids']==['tri_complete','tri_running'])
check('one snapshot per expired session only',len(calls)==2 and set(dead)=={'ses_idle','ses_recorded'})
check('live session untouched',(root/'ses_live/session.json').read_bytes()==live_before and store.sessions['ses_live']['state']=='OPEN')
check('internal request map excluded',all('request_ids' not in s for s in saved.values()))
events=[json.loads(x) for x in (root/'ses_recorded/timing_events.jsonl').read_text(encoding='utf-8').splitlines()]
check('unfinished trial cancellation event preserved',len(events)==1 and events[0]['trial_id']=='tri_running' and events[0]['outcome']=='CANCELLED')
check('successful trial unchanged',store.trials['tri_complete']['state']=='FEEDBACK')
check('camera released exactly once',camera.stopped==1 and store.camera is None and store.camera_session is None)
count=len(calls);event_bytes=(root/'ses_recorded/timing_events.jsonl').read_bytes();dead_again=store.reap_expired()
check('repeat reaper does not rewrite or duplicate',not dead_again and len(calls)==count and (root/'ses_recorded/timing_events.jsonl').read_bytes()==event_bytes)
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','phase':phase,'passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'app_sha256':hashlib.sha256((web/'src/wis_lab/service/app.py').read_bytes()).hexdigest(),'real_camera_opens':0,'model_inference':0,'human_records':0}
(here/('verification_'+phase+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','phase','passed','total')}))
print(json.dumps({'failed':[c['name'] for c in checks if not c['passed']]}))
sys.exit(0 if report['status']=='PASS' else 1)
