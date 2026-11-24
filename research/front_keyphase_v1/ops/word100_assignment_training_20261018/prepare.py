from common import *
from adapter import *
import argparse,time,traceback,zipfile,zlib,shutil
from collections import Counter

def freeze():
 assert read(U/'AUTHORIZATION.json')['approved'] and not (U/'PREP_CONTEXT.json').exists()
 p=read(PLAN/'PROPOSED_COMPARISON.json');prior=read(O/'word100_remaining_identity_audit_20261018/CONTEXT.json')
 c=read(O/'word100_hand_evidence_training_20261018/PREP_CONTEXT.json')
 rows=c['records'];assert len(rows)==1199
 pins=dict(read(PLAN/'EVIDENCE.json')['source_pins'])
 # All copied observation outputs have an earlier saved checksum; validate, never re-bless.
 upins=read(OBS/'OUTPUT_PINS.json');rpins=prior['pins'];cpins=c['pins']
 for r in rows:
  assert r['signer_id'] in (range(1,10) if r['role']=='train_candidate' else (11,12,13))
  for v in ['masked','sensor']:
   for key in ['raw','observation','channels','assignment']:
    f=r['views'][v][key];h=rpins.get(f,cpins.get(f));assert h,('not_previously_pinned',f);assert sha(f)==h; pins[f]=h
  if r['role']=='train_candidate':
   for v in ['masked','sensor']:
    f=str(OBS/'results'/f"{r['clip_id']}_{v}.npz");assert sha(f)==upins[f];pins[f]=upins[f]
 for f in [U/'AUTHORIZATION.json',PLAN/'PROPOSED_COMPARISON.json',U/'FROZEN_PROTOCOL.json']+list(U.glob('*.py')):pins[str(f)]=sha(f)
 # Pin all transitive local implementation modules actually loaded by the adapter.
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(ROOT).lower() in str(f).lower() and Path(f).suffix=='.py':pins[str(Path(f).resolve())]=sha(Path(f))
 write(U/'PREP_CONTEXT.json',{'at':now(),'rows':rows,'pins':pins,'parameters':{'contour':contour.PARAMS,'duplicate':duplicate.PARAMS,'guard':guard.PARAMS},'train_only_until_visual_gate':True,'training_allowed':False},new=True)
 print('FROZEN',len(rows),'records',len(pins),'pins',flush=True)

def source_video(r):
 cid=r['clip_id'];src=r['source']
 paths=list(O.glob(f'*/videos/{cid}.mp4'))
 for path in paths:
  if path.is_file() and sha(path)==src['sha256']:return path
 path=U/'videos'/f'{cid}.mp4'
 assert not path.exists();path.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(src['archive']) as z:
  entry=z.getinfo(src['member']);blob=z.read(entry)
 assert len(blob)==src['bytes'] and hashlib.sha256(blob).hexdigest()==src['sha256'] and (zlib.crc32(blob)&0xffffffff)==src['crc32']
 path.write_bytes(blob);return path

def upstream(r,v,z):
 if r['role']=='train_candidate':return load(OBS/'results'/f"{r['clip_id']}_{v}.npz")
 assert (U/'TRAIN_VISUAL_READY.json').exists() and read(U/'TRAIN_VISUAL_READY.json')['status']=='PASS'
 a=obsmod.HandAssigner();T=len(z['raw_count']);out={'sides':np.full((T,2),-1,np.int8),'identity_side':np.full((T,2),-1,np.int8),'point_mask':np.zeros((T,2,21),bool)}
 for t,n0 in enumerate(z['raw_count']):
  n=int(n0);ss,d=a.assign(z['hand_xy_px'][t,:n],z['raw_label'][t,:n].tolist(),z['raw_score'][t,:n],z['pose_xy_px'][t],z['pose_usable'][t],int(z['timestamp_us'][t]),int(z['width'][t]),int(z['height'][t]))
  out['sides'][t,:n]=[-1 if ss[i] is None else ss[i] for i in range(n)]
  for k in ['identity_side','point_mask']:out[k][t,:n]=d['observation'][k]
 return out

def process(r,contextsha):
 cid=r['clip_id'];recpath=U/'records'/f'{cid}.json'
 if recpath.exists():
  rec=read(recpath);assert rec['context_sha256']==contextsha;check(rec['outputs']);return rec
 for v in ['masked','sensor']:
  for arm in ['P','Q']:
   assert not (U/'inputs'/f'{cid}_{v}_{arm}.npz').exists(),'Orphan input, review before recovery'
 zmap={v:load(r['views'][v]['raw']) for v in ['masked','sensor']};bs={v:upstream(r,v,z) for v,z in zmap.items()}
 needs={v:[t for t in range(len(z['raw_count'])) if contour_eligible(z,bs[v],t)] for v,z in zmap.items()}
 contour_out={v:{} for v in zmap};video=None
 if any(needs.values()):
  video=source_video(r);cap=cv2.VideoCapture(str(video));assert cap.isOpened()
  for v,indices in needs.items():
   z=zmap[v];b=bs[v]
   for t in indices:
    fid=int(z['frame_id'][t]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok,(cid,v,t)
    out,_=contour_step(im,z,b,t)
    contour_out[v][t]={'sides':out['sides'],'evidence_side':out['evidence_side'],'reason':out['reason'],'supported':[[q.get('supported',0) for q in hand] for hand in out['trials']]}
  cap.release()
 rec={k:r[k] for k in ['clip_id','role','signer_id','label_index','word_code','label']};rec.update(context_sha256=contextsha,views={},outputs={},changes=[],feature_info={},counts={},source_video=str(video) if video else None)
 for v,z in zmap.items():
  b=bs[v];old=legacy.Observation.load(r['views'][v]['observation']);original=load(r['views'][v]['channels']);N=len(z['raw_count'])
  assert np.array_equal(z['frame_id'],old.frame_id) and np.array_equal(z['timestamp_us'],old.timestamp_us)
  ss={k:b['sides'].copy() for k in ['P','Q']};trace=[]
  for t,n0 in enumerate(z['raw_count']):
   n=int(n0);before=b['sides'][t,:n].tolist();qs=contour_out[v].get(t,{}).get('sides',before)
   for arm,s in [('P',before),('Q',qs)]:
    after,g=add_duplicate(z,b,t,s);ss[arm][t,:n]=after
    if after!=before:rec['changes'].append({'view':v,'arm':arm,'index':t,'frame_id':int(z['frame_id'][t]),'before':before,'after':after,'contour':contour_out[v].get(t),'duplicate':g})
   trace.append({'index':t,'contour':contour_out[v].get(t)}) if t in contour_out[v] else None
  sp=U/'observations'/f'{cid}_{v}.npz';assert not sp.exists();sp.parent.mkdir(exist_ok=True)
  np.savez_compressed(sp,P_sides=ss['P'],Q_sides=ss['Q'],base_sides=b['sides'],point_mask=b['point_mask'],identity_side=b['identity_side'],frame_id=z['frame_id'],timestamp_us=z['timestamp_us'])
  rec['outputs'][str(sp)]=sha(sp);rec['views'][v]={};rec['feature_info'][v]={}
  oldass=load(r['views'][v]['assignment'])
  for arm in ['P','Q']:
   ob,unused=route(old,z,b,ss[arm],r['box']);arr,info=legacy.features(ob,r)
   frames=np.flatnonzero(np.any(old.usable!=ob.usable,axis=1)|np.any(np.nan_to_num(old.xy_px)!=np.nan_to_num(ob.xy_px),axis=(1,2)))
   info.update(unrouted_roi_points=unused,observation_changed_frames=frames.tolist(),old_whole_frames=original['whole_frame_id'].tolist(),new_valid_hand_points=int(ob.usable[:,33:].sum()),old_valid_hand_points=int(old.usable[:,33:].sum()))
   if arr is not None:
    equal=all(k in original and np.array_equal(x,original[k]) for k,x in arr.items());info['equals_C_tensor']=equal
    arr.update(xy_px=ob.xy_px,usable=ob.usable,frame_id=ob.frame_id,timestamp_us=ob.timestamp_us)
    out=U/'inputs'/f'{cid}_{v}_{arm}.npz';out.parent.mkdir(exist_ok=True);np.savez_compressed(out,**arr);rec['outputs'][str(out)]=sha(out);rec['views'][v][arm]=str(out)
   else:rec['views'][v][arm]=None
   rec['feature_info'][v][arm]=info
  # Routing changes vs original C, separately from changes of latest guards vs U.
  rec['counts'][v]={'frames':N,'contour_frames':len(needs[v]),'raw_candidates':int(z['raw_count'].sum()),'sides_changed_from_C':{a:int((ss[a]!=oldass['sides']).sum()) for a in ['P','Q']}}
  tp=U/'observations'/f'{cid}_{v}_contour.json';write(tp,trace,new=True);rec['outputs'][str(tp)]=sha(tp)
 write(recpath,rec,new=True);return rec

def main(role,limit):
 c=read(U/'PREP_CONTEXT.json');check(c['pins']);lock=U/'PREP_RUNNING.json';assert not lock.exists()
 write(lock,{'pid':os.getpid(),'at':now(),'role':role},new=True);start=time.monotonic();completed=[];new=0
 try:
  for r in c['rows']:
   if r['role']!=role:continue
   if not (U/'records'/f"{r['clip_id']}.json").exists():
    if new>=limit:break
    new+=1
   rec=process(r,sha(U/'PREP_CONTEXT.json'));completed.append(rec)
   if len(completed)%10==0:event(stage='INPUT_PREPARATION',role=role,complete=len(completed),new=new,seconds=round(time.monotonic()-start,1),training=0)
  summary={'status':'PREPARED_PENDING_VISUAL_AND_INDEPENDENT_CHECK','role':role,'completed':len(completed),'new':new,'seconds':time.monotonic()-start,'changed_events':sum(len(r['changes']) for r in completed),'feature_failures':[{'clip_id':r['clip_id'],'view':v,'arm':a,'reason':r['feature_info'][v][a]['reason']} for r in completed for v in ['masked','sensor'] for a in ['P','Q'] if r['views'][v][a] is None],'record_pins':{str(U/'records'/f"{r['clip_id']}.json"):sha(U/'records'/f"{r['clip_id']}.json") for r in completed},'training':0}
  write(U/f'{role}_PREP_RESULT.json',summary);event(**{k:v for k,v in summary.items() if k!='record_pins'})
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'traceback':traceback.format_exc(),'role':role,'completed':len(completed)},new=True)
  raise
 finally:
  if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--role',default='train_candidate');p.add_argument('--limit',type=int,default=1199);a=p.parse_args()
 freeze() if a.action=='freeze' else main(a.role,a.limit)
