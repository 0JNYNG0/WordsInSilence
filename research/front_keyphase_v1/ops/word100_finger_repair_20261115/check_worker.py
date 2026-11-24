from common import *
import subprocess,time,os
g6=O/'word100_live_g6_20261114';ext=read(O/'word100_public_g4_20261114/EXTRACTED.json')['rows']
stderr=(U/'worker_check_stderr.log').open('w',encoding='utf-8');proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(g6/'worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
rows=[]
try:
 ready=json.loads(proc.stdout.readline());assert ready['ready'];print('ready',round(ready['seconds'],2),flush=True)
 for cid,kind in [('NIA_SL_WORD1133_REAL18_F','normal'),('NIA_SL_WORD1133_REAL18_F','reverse'),('NIA_SL_WORD0739_REAL17_F','normal'),('NIA_SL_WORD0739_REAL17_F','missing')]:
  r=next(r for r in ext if r['clip_id']==cid);assert sha(r['raw_path'])==r['raw_sha256'];z=np.load(r['raw_path']);raw={k:z[k].copy()for k in z.files};raw={k:raw[k]for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']}
  if kind=='reverse':raw['xy_px']=raw['xy_px'][::-1].copy();raw['score']=raw['score'][::-1].copy()
  if kind=='missing':raw['xy_px'][:,91:133]=np.nan;raw['score'][:,91:133]=0
  t=time.perf_counter();proc.stdin.write(json.dumps(dict(word=r['word'],raw={k:v.tolist()for k,v in raw.items()}),allow_nan=True)+'\n');proc.stdin.flush();reply=json.loads(proc.stdout.readline());assert reply['ok'],reply
  q=reply['result'];assert q['runtime']['version']=='G6-finger-20261115';assert q['finger_supplement']['enabled'];rec=dict(clip_id=cid,kind=kind,roundtrip=time.perf_counter()-t,compute=q['runtime']['seconds'],supplement=q['supplement_seconds'],states=[c['state']for c in q['joint_cards']],detail_count=q['finger_supplement']['detail_count'])
  if kind in ['normal','missing']:assert rec['detail_count']==0
  write(U/f'worker_{cid}__{kind}.json',dict(check=rec,result=q));rows.append(rec);print(json.dumps(rec),flush=True)
finally:
 if proc.poll()is None:proc.stdin.write('{"close":true}\n');proc.stdin.flush();proc.wait(timeout=10)
 stderr.close()
write(U/'WORKER_CHECK.json',dict(passed=True,rows=rows,prepare_seconds=ready['seconds'],raw_public_inputs=True,physical_camera=False,TCN_changed=False,latency_scope='full qualified-evidence worker only, excludesRTMW,TCN,DTW,web'))
