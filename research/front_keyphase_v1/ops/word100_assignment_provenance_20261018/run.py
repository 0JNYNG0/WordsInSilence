"""Frozen public TRAIN replay, with old C/production untouched."""
import os,sys,json,hashlib,time,traceback,shutil
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_assignment_conflict_20261018';A=O/'word100_label_only_audit_20261018';ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pinscheck(pins):
 for p,h in pins.items():assert sha(p)==h,p
def prepare():
 assert read(A/'FINAL_AUDIT.json')['status']=='PASS' and read(U/'CONTRACTS.json')['status']=='PASS'
 assert not (A/'RUNNING.json').exists() and not (P/'RUNNING.json').exists()
 prior=read(P/'CONTEXT.json');pins=dict(read(A/'CONTEXT.json')['pins']);pins.update(read(P/'OUTPUT_PINS.json'))
 for p in [P/'assignment.py',P/'SCOPE.json',P/'POINT_CONTRACT.json',A/'VISUAL_REVIEW.json',A/'EVENTS.json',A/'INDEPENDENT.json',U/'candidate.py',U/'contracts.py',U/'CONTRACTS.json',U/'run.py']:pins[str(p)]=sha(p)
 (U/'results').mkdir();(U/'before').mkdir()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 write(U/'CONTEXT.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'authorization':'User approved isolated provisional/confirmed provenance and duplicate-prevention candidate; public validation only, no training until joint review.','rows':prior['rows'],'pins':pins,'baseline':str(P/'assignment.py'),'annotated_development_set':str(A/'VISUAL_REVIEW.json'),'protocol':{
 'single_candidate':'One fixed candidate; no post-result threshold search. Known TRAIN456 audit informed design, not independent heldout accuracy.',
 'parameters':'Existing gates/ratio/time/speed retained. New full21 smaller-hull containment0.80 fixed engineering screen, not fitted/swept. Pre-freeze diagnostic geometry on known examples documented; no universal validity claim.',
 'state':'Baseline proposal preserved; current unique pose or independently confirmed bounded history required for confirmation. Label-seeded temporal chains remain provisional. Raw observations unchanged.',
 'guards':'Only provisional conflicts withheld: recent<=100ms confirmed-track unreachable by max(.4 shoulder,12shoulders/s*dt), or>=80% region containment with confirmed opposite anchor/history and no reachable own confirmed history. Never flip identities or fill absent slots.',
 'preservation':'Audit441 normal cases and315 geometric-crossing frames must retain association; every hard change inspected in original RGB; report all regression/uncertain cases. Do not equate preserved association with correct finger geometry.',
 'validation':'Public899 TRAIN both detector streams; compare saved previous candidate exact proposals, separate arithmetic/provenance check, permutation/reflection/empty/reset contracts, changed clips/prefix verification.','stop':'Report input-processing outcome and remaining issues; no model training or production adoption.'},
 'prohibited':['user/mentor/DEV/heldout data','training','classifier inference','camera','download','external transmission','Git','production C/TCN/DTW/web edits','word/frame-specific exceptions','future frames/outside input pixels'],
 'implementation':'Codex local; independent scalar/saved-array checks by same operator, not independent human ground truth.'})
 print('Context frozen',len(prior['rows']),'TRAIN',len(pins),'protected pins',flush=True)

STATES={'unassigned':0,'provisional':1,'confirmed':2,'withheld_conflict':3}
def main():
 from candidate import HandAssigner
 c=read(U/'CONTEXT.json');pinscheck(c['pins']);assert not (U/'SUMMARY.json').exists()
 write(U/'RUNNING.json',{'pid':os.getpid(),'task':'public association replay only'});start=time.monotonic()
 ann=read(A/'VISUAL_REVIEW.json')['records'];lookup={(r['clip_id'],r['view'],r['index'],r['raw_index']):r for r in ann}
 totals=Counter();changes=[];annotated=[];outputs={};cross=[];per=[]
 try:
  for rn,row in enumerate(c['rows'],1):
   assert row['role']=='train_candidate' and 1<=row['signer_id']<=9
   for view in ('masked','sensor'):
    raw=load(row['views'][view]['raw']);previous=load(P/'results'/f"{row['clip_id']}_{view}.npz");model=HandAssigner();N=len(raw['raw_count']);local=Counter()
    for ar in raw.values():ar.flags.writeable=False
    sides=np.full((N,2),-1,np.int8);states=np.zeros((N,2),np.int8);roots=np.full((N,2),-1,np.int32);parents=roots.copy();direct=np.zeros((N,2),bool);hist=direct.copy();guards=np.zeros((N,2),np.int8)
    pc=np.full((N,2,2),np.nan);ct=pc.copy();overlap=np.zeros(N);scales=np.full(N,np.nan)
    for f in range(N):
     n=int(raw['raw_count'][f]);out,d=model.assign(raw['hand_xy_px'][f,:n],raw['raw_label'][f,:n],raw['raw_score'][f,:n],raw['pose_xy_px'][f],raw['pose_usable'][f],int(raw['timestamp_us'][f]),int(raw['width'][f]),int(raw['height'][f]))
     old=[-1 if s is None else s for s in d['proposal_sides']];assert old==previous['sides'][f,:n].tolist(),(row['clip_id'],view,f,'baseline_mismatch')
     new=[-1 if out[i] is None else out[i] for i in range(n)]
     totals['frames']+=1;totals[view+'_frames']+=1;totals['raw_candidates']+=n
     overlap[f]=d['region_overlap_min_area'];scales[f]=d['baseline']['scale'] if d['baseline']['scale'] is not None else np.nan
     for i in range(n):
      sides[f,i]=new[i];states[f,i]=STATES[d['identity_state'][i]];roots[f,i]=d['origin'][i]['frame'] if d['origin'][i] else -1;parents[f,i]=d['parent_frame'][i] if d['parent_frame'][i] is not None else -1
      direct[f,i]=d['current_pose_confirmation'][i];hist[f,i]=d['confirmed_history_match'][i]
      pc[f,i]=[np.nan if x is None else x for x in d['baseline']['pose_distance'][i]]
      ct[f,i]=[np.nan if x is None else x for x in d['confirmed_track_distance'][i]]
      totals['state_'+d['identity_state'][i]]+=1
      if old[i]>=0:totals['baseline_assigned']+=1
      if new[i]>=0:totals['candidate_assigned']+=1
      if old[i]==new[i]:totals['unchanged_candidate_slots']+=1
      else:
       kind='withheld' if new[i]<0 else 'side_flip' if old[i]>=0 else 'newly_assigned';totals[kind]+=1;local[kind]+=1
       changes.append({'clip_id':row['clip_id'],'word':row['label'],'view':view,'index':f,'frame_id':int(raw['frame_id'][f]),'raw_index':i,'old_side':old[i],'new_side':new[i],'kind':kind,'trace':d})
      if old[i]>=0 and d['baseline']['reasons'][i]!='label_without_close_pose':
       totals['non_label_baseline_assignments']+=1
       if old[i]==new[i]:totals['non_label_preserved']+=1
      key=(row['clip_id'],view,f,i)
      if key in lookup:
       truth=lookup[key];annotated.append({'id':truth['id'],'verdict':truth['verdict'],'actual_side':truth['actual_side'],'old_side':old[i],'new_side':new[i],'state':d['identity_state'][i],'origin':d['origin'][i],'trace':d})
     for g in d['guards']:
      guards[f,g['raw_index']]=1 if g['kind'].startswith('unconfirmed_jump') else 2
      totals[g['kind']]+=1
     if n==2 and set(old)=={0,1}:
      ix={s:old.index(s) for s in (0,1)};xy=raw['hand_xy_px'][f,:,0,:];pose=raw['pose_xy_px'][f];pu=raw['pose_usable'][f]
      if pu[[11,12]].all() and (xy[ix[0],0]-xy[ix[1],0])*(pose[11,0]-pose[12,0])<0:
       totals['crossed_projection_frames']+=1
       if old==new:totals['crossed_projection_preserved']+=1
       cross.append({'clip_id':row['clip_id'],'view':view,'index':f,'preserved':old==new})
    path=U/'results'/f"{row['clip_id']}_{view}.npz";assert not path.exists()
    np.savez_compressed(path,sides=sides,state=states,origin_frame=roots,parent_frame=parents,current_pose=direct,confirmed_history=hist,guard=guards,pose_distance=pc,confirmed_track_distance=ct,region_overlap=overlap,scale=scales,frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us'])
    outputs[str(path)]=sha(path);per.append({'clip_id':row['clip_id'],'view':view,'changes':dict(local)})
   if rn%100==0:print('Public replay',rn,'/899',flush=True)
  assert len(annotated)==456
  pinscheck(c['pins'])
  write(U/'CHANGES.json',changes);write(U/'ANNOTATED_COMPARISON.json',sorted(annotated,key=lambda r:r['id']));write(U/'CROSSING_COMPARISON.json',cross);write(U/'OUTPUT_PINS.json',outputs);write(U/'PER_CLIP.json',per)
  breakdown={v:dict(Counter((('kept' if r['new_side']==r['old_side'] else 'withheld')+'_'+r['state']) for r in annotated if r['verdict']==v)) for v in ('normal_identity','error_identity','uncertain')}
  write(U/'SUMMARY.json',{'status':'REPLAY_DONE_PENDING_VERIFICATION_AND_VISUAL_REVIEW','counts':dict(totals),'audited456':breakdown,'changed_clips':len({r['clip_id'] for r in changes}),'seconds':time.monotonic()-start,'training':0,'production_changed':False,'user_mentor_dev_heldout_reads':0})
  (U/'RUNNING.json').unlink();print(json.dumps({'counts':dict(totals),'audited456':breakdown},ensure_ascii=False),flush=True)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
  raise

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':main()
