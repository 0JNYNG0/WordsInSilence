"""Replay new assignment state causally. Original captures and C stay read-only."""
import os,sys,json,hashlib,time,traceback,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def call(a,r,f):
 n=int(r['raw_count'][f]);return a.assign(r['hand_xy_px'][f,:n],r['raw_label'][f,:n],r['raw_score'][f,:n],r['pose_xy_px'][f],r['pose_usable'][f],int(r['timestamp_us'][f]),int(r['width'][f]),int(r['height'][f]))
def pinscheck(pins):
 for p,h in pins.items():assert sha(p)==h,p
def main():
 c=read(U/'CONTEXT.json');pinscheck(c['pins']);assert not (U/'SUMMARY.json').exists();write(U/'RUNNING.json',{'pid':os.getpid(),'task':'assignment replay, no training'});start=time.monotonic()
 old=module('original_assignment',c['baseline']);new=module('candidate_assignment',U/'assignment.py');tot=Counter();changes=[];guards=[];cross_examples=[];outputs={};per=[]
 try:
  for ri,r in enumerate(c['rows']):
   for view in ('masked','sensor'):
    raw=load(r['views'][view]['raw']);saved=load(r['views'][view]['assignment']);origtrace=read(Path(r['views'][view]['assignment']).with_name('trace.json'));a=old.HandAssigner();b=new.HandAssigner();res=np.full_like(saved['sides'],-1);changed_traces=[];local=Counter()
    for arr in raw.values():arr.flags.writeable=False
    for f in range(len(raw['raw_count'])):
     out,d=call(a,raw,f);out2,e=call(b,raw,f);n=int(raw['raw_count'][f]);prev=[-1 if out[i] is None else out[i] for i in range(n)]
     assert np.array_equal(prev,saved['sides'][f,:n]),(r['clip_id'],view,f,'baseline_mismatch')
     assert d['reasons']==origtrace[f]['reasons'],(r['clip_id'],view,f,'reason_mismatch')
     assert np.array_equal(saved['frame_id'],raw['frame_id']) if 'frame_id' in saved else True
     curr=[-1 if out2[i] is None else out2[i] for i in range(n)];res[f,:n]=curr
     tot['frames']+=1;tot['raw_candidates']+=n;tot[view+'_frames']+=1
     for i in range(n):
      if prev[i]>=0:tot['baseline_assigned']+=1
      if curr[i]>=0:tot['candidate_assigned']+=1
      if prev[i]==curr[i]:tot['unchanged_candidate_slots']+=1
      else:
       typ='withheld' if curr[i]<0 else 'newly_assigned' if prev[i]<0 else 'side_flip'
       tot[typ]+=1;local[typ]+=1
       changes.append({'clip_id':r['clip_id'],'label':r['label'],'view':view,'index':f,'frame_id':int(raw['frame_id'][f]),'timestamp_us':int(raw['timestamp_us'][f]),'raw_index':i,'old_side':prev[i],'new_side':curr[i],'old_reason':d['reasons'][i],'new_reason':e['reasons'][i],'kind':typ})
      if d['reasons'][i]!='label_without_close_pose' and prev[i]>=0:
       tot['supported_baseline_assignments']+=1
       if curr[i]==prev[i]:tot['supported_baseline_preserved']+=1
      if d['reasons'][i]=='label_without_close_pose':tot[view+'_label_only']+=1
     if e['label_conflicts']:
      guards.append({'clip_id':r['clip_id'],'view':view,'index':f,'frame_id':int(raw['frame_id'][f]),'trace':e});tot['guard_frames']+=1
     if prev!=curr or e['label_conflicts']:changed_traces.append({'index':f,'frame_id':int(raw['frame_id'][f]),'baseline':d,'candidate':e})
     # Geometry only selects review candidates, never determines identities.
     if n==2 and set(prev)=={0,1}:
      ix={s:prev.index(s) for s in (0,1)};xy=raw['hand_xy_px'][f,:,0,:];pose=raw['pose_xy_px'][f];pu=raw['pose_usable'][f]
      if pu[[11,12]].all() and (xy[ix[0],0]-xy[ix[1],0])*(pose[11,0]-pose[12,0])<0:
       tot['crossed_projection_frames']+=1
       if prev==curr:tot['crossed_projection_preserved']+=1
       if len(cross_examples)<8 and not any(x['clip_id']==r['clip_id'] for x in cross_examples):cross_examples.append({'clip_id':r['clip_id'],'label':r['label'],'view':view,'index':f,'frame_id':int(raw['frame_id'][f]),'sides':prev,'reasons':d['reasons']})
    target=U/'results'/f"{r['clip_id']}_{view}.npz";assert not target.exists();np.savez_compressed(target,sides=res,frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us']);outputs[str(target)]=sha(target)
    if changed_traces:
     target=target.with_suffix('.json');write(target,changed_traces);outputs[str(target)]=sha(target)
    per.append({'clip_id':r['clip_id'],'view':view,'changes':dict(local)})
   if (ri+1)%100==0:print('Replayed',ri+1,'/899',flush=True)
  pinscheck(c['pins']);write(U/'CHANGES.json',changes);write(U/'GUARDS.json',guards);write(U/'CROSSING_SELECTION.json',cross_examples);write(U/'PER_CLIP.json',per);write(U/'OUTPUT_PINS.json',outputs)
  write(U/'SUMMARY.json',{'status':'REPLAY_DONE_PENDING_INDEPENDENT_AND_VISUAL_REVIEW','counts':dict(tot),'changed_clips':len(set(x['clip_id'] for x in changes)),'seconds':time.monotonic()-start,'training':0,'user_data':0,'production_changed':False});(U/'RUNNING.json').unlink();print(json.dumps(dict(tot)),flush=True)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':main()
