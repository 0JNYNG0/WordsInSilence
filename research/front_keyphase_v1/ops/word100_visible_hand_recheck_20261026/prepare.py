from pathlib import Path
import sys,json,hashlib,datetime,importlib.util,traceback
from collections import Counter,defaultdict
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];I=O/'word100_wrist_only_inputs_20261026';B=O/'word100_body_boundary_20261017'
import candidate as C
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,z):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(z,f,ensure_ascii=False,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
def fixtures():
 rng=np.random.default_rng(4);p=rng.normal(0,10,(21,2))+[120,120];x=np.tile(p,(8,2,1,1));x[:,1,:,0]+=200;v=np.ones((8,2,21),bool);t=np.arange(8)*33333;roi=[0,0,640,480];th={'shape_residual':.1,'angle_rate':50.}
 d=C.descriptors(x,v,t,roi);assert np.nanmax(abs(d['shape_residual']))<1e-10 and np.nanmax(d['angle_rate'])<1e-7
 theta=.3;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 y=x.copy();y[3:]=(y[3:]-[200,200])@r*1.2+[240,210];e=C.descriptors(y,v,t,roi);assert e['shape_residual'][3].max()<1e-10 and e['angle_rate'][3].max()<1e-7
 y=x.copy();y[3,0,8]+=[40,20];e=C.descriptors(y,v,t,roi);assert C.schedule(e,th)[3,0]&8
 vv=v.copy();vv[2,0]=False;e=C.descriptors(x,vv,t,roi);assert e['returning'][3,0] and C.schedule(e,th)[3,0]&1
 vv=v.copy();vv[4,0,8]=False;e=C.descriptors(x,vv,t,roi);assert C.schedule(e,th)[4,0]&2 and not e['inroi'][4,0,8]
 y=x.copy();y[:,1]=y[:,0]+[2,0];original=y.copy();e=C.descriptors(y,v,t,roi);assert e['overlap'].all() and (C.schedule(e,th)&4).all();assert np.array_equal(y,original)
 vv=v.copy();vv[:,0]=False;e=C.descriptors(x,vv,t,roi);assert not C.schedule(e,th)[:,0].any()
 y=x.copy();y[:,0,:,1]+=600;e=C.descriptors(y,v,t,roi);assert not C.schedule(e,th)[:,0].any()
 box=C.crop_box(x[0,0],v[0,0],640,480);assert 0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480
 assert C.crop_box(x[0,0],np.zeros(21,bool),640,480) is None
 assert C.descriptors(x,v,t,roi)['returning'][0].all()
 return dict(stable_no_shape_trigger=True,rigid_transform_invariant=True,articulation_requests_not_rejects=True,reappearance=True,partial=True,crossing_preserved=True,whole_missing_separate=True,outside_roi_separate=True)
def prepare():
 old=read(I/'FROZEN.json');rows=[r for r in old['rows'] if r['view']=='masked'];assert len(rows)==899 and {x['signer_id'] for x in rows}==set(range(1,10));check(old['protected'])
 meta={r['clip_id']:r for r in read(B/'PREP_CONTEXT.json')['ordered_rows']};assert all(meta[r['clip_id']]['role']=='train_candidate' for r in rows)
 write(U/'AUTHORIZATION.json',dict(at=now(),goal='Approved autonomous common public recognition/feedback improvement. Existing native observations remain read-only; second image evidence scheduling only.',scope='TRAIN899 thresholds and audit; no DEV score access, no user/mentor/199/200, no training, operating replacement,downloads or transmission.',budget='Collect TRAIN distributions then freeze99th-percentile two numeric request thresholds before new detector comparisons. These are resource-scheduling thresholds,not correctness criteria. No optimization against class/rank/error labels. Deterministic100-word representative subset for image trial,all triggered frames plus3uniform control frames each. No successful-only filtering.'))
 tests=fixtures();write(U/'FIXTURES.json',dict(at=now(),status='PASS',tests=tests));(U/'diagnostics').mkdir();cache={};pins={str(p):sha(p) for p in [Path(__file__),Path(C.__file__),I/'FROZEN.json',B/'PREP_CONTEXT.json']};samples=defaultdict(list);total=Counter()
 for i,r in enumerate(rows):
  paths=r['info']['source_arrays'];n=load(paths['capture']);d=C.descriptors(n['xy_px'][:,33:].reshape(-1,2,21,2),n['usable'][:,33:].reshape(-1,2,21),n['timestamp_us'],r['info']['original_roi']);core=np.zeros(len(n['frame_id']),bool)
  for a,b in meta[r['clip_id']]['intervals']:core|=(n['timestamp_us']/1e6>=a)&(n['timestamp_us']/1e6<=b)
  for k in ['shape_residual','angle_rate']:
   z=d[k][core];samples[k].extend(z[np.isfinite(z)].tolist())
  p=U/'diagnostics'/(r['clip_id']+'.npz');np.savez_compressed(p,**d,frame_id=n['frame_id'],timestamp_us=n['timestamp_us'],core=core);pins[str(p)]=sha(p);pins[paths['capture']]=sha(paths['capture']);cache[r['clip_id']]=d;total['frames']+=len(core);total['visible_hand_frames']+=int(d['visible'].sum());total['core_frames']+=int(core.sum())
  if (i+1)%200==0:print('TRAIN_DIAGNOSTICS',i+1,899,flush=True)
 thresholds={k:float(np.quantile(x,.99,method='linear')) for k,x in samples.items()};by=defaultdict(list);out=[]
 for r in rows:
  cid=r['clip_id'];d=cache[cid];reason=C.schedule(d,thresholds);p=U/'diagnostics'/(cid+'.npz');z=load(p);controls=[];active=np.flatnonzero(z['core'])
  if len(active):controls=sorted(set(int(active[int(round((len(active)-1)*p))]) for p in [.25,.5,.75]))
  targets=[dict(index=int(i),frame_id=int(z['frame_id'][i]),sides=np.flatnonzero(reason[i]).tolist(),reasons=reason[i].tolist(),control=i in controls) for i in sorted(set(np.flatnonzero(reason.any(1)).tolist()+controls))]
  item=dict(clip_id=cid,word_code=meta[cid]['word_code'],label=meta[cid]['canonical_label'],signer_id=r['signer_id'],input=r,targets=targets,controls=controls,trigger_hand_frames=int((reason>0).sum()),trigger_frames=int(reason.any(1).sum()),reason_counts={str(bit):int(((reason&bit)>0).sum()) for bit in [1,2,4,8,16]},diagnostics=str(p));out.append(item);by[item['word_code']].append(item);total['trigger_hand_frames']+=item['trigger_hand_frames'];total['trigger_frames']+=item['trigger_frames']
  for k,x in item['reason_counts'].items():total['reason_'+k]+=x
 # Hash order balances identity independently of detector outcome or scores.
 selected=[min(v,key=lambda x:hashlib.sha256(x['clip_id'].encode()).hexdigest())['clip_id'] for k,v in sorted(by.items())];assert len(selected)==100
 selected_rows=[x for x in out if x['clip_id'] in selected]
 for r in selected_rows:
  check(r['input']['input_pins']);pins.update(r['input']['input_pins'])
 check(old['protected']);write(U/'FROZEN.json',dict(at=now(),rows=out,selected_ids=selected,thresholds=thresholds,calibration_quantile=.99,calibration_samples={k:len(x) for k,x in samples.items()},counts=dict(total),selected_target_frames=sum(len(x['targets']) for x in selected_rows),pins=pins,protected=old['protected'],mp_options=old['mp_options'],
  method='Native full-image hand coordinates before ROI. Schedule only when at least1point is inROI. Reasons1reappearance,2partial hand,4bboxoverlap,8shape residual beyond TRAIN99%,16projected angle rate beyondTRAIN99%. Similarity residual removes2Dtranslation/rotation/scale. All triggers are requests,not proof of error. Preserve all original coordinates,identity,normalization,TCN/DTW. Same-frame MP full/crop and optional saved/newYOLO evidence only; no automatic replacement.',
  controls='3annotation-core quarter/middle/threequarter frames perclip evaluated irrespective trigger. Oneclip per100word chosen by SHA ofID,not outcomes. Allselectedframes retained, failures reported. Labels onlysampling/reference; notintrigger.',
  evaluation='Report actual extra evidence and disagreement, existing normal/crossing coordinate preservation,cost. Same detector agreement notaccuracy. No threshold relaxation to make counts grow. Before deployment require anatomical/effect evidence.',training=0,user_data=0,production_changed=False));print('FROZEN_VISIBLE_RECHECK',thresholds,dict(total),'selected_frames',sum(len(x['targets']) for x in selected_rows),flush=True)
if __name__=='__main__':
 try:prepare()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
