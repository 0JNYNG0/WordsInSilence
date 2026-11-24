from pathlib import Path
import sys,json,hashlib,time,importlib.util
import cv2,numpy as np
from candidate import PARAMS,region_flow,nearest_anchors,assess
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_mano_identity_recheck_20261026';G=O/'word100_mano_duplicate_guard_20261025';H=O/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
spec=importlib.util.spec_from_file_location('contour',O/'word100_forearm_contour_20261018/candidate.py');ct=importlib.util.module_from_spec(spec);spec.loader.exec_module(ct)
sys.path.insert(0,str(P));from identity_v3 import build_anchors,conflict_mask
saved_candidate=sys.modules['candidate']
spec=importlib.util.spec_from_file_location('candidate',G/'candidate.py');olddu=importlib.util.module_from_spec(spec);spec.loader.exec_module(olddu);sys.modules['candidate']=olddu
spec=importlib.util.spec_from_file_location('duplicate',G/'candidate_v2.py');du=importlib.util.module_from_spec(spec);spec.loader.exec_module(du)
sys.modules['candidate']=saved_candidate

def images(path,ids):
 cap=cv2.VideoCapture(str(path));assert cap.isOpened();assert cap.get(cv2.CAP_PROP_FRAME_WIDTH)==1920
 wanted=set(map(int,ids));frames={};fps=cap.get(cv2.CAP_PROP_FPS)
 for j in range(int(max(ids))+1):
  ok,im=cap.read();assert ok,(path,j)
  if j in wanted:frames[j]=cv2.cvtColor(cv2.resize(im,(640,360)),cv2.COLOR_BGR2GRAY)
 cap.release();return [frames[int(i)] for i in ids],fps

def save_track(path,tr):
 keys=['previous','forward','backward','status','back_status','error','valid']
 arr={k:np.stack([x[k] for x in tr['steps']]) if tr['steps'] else np.empty((0,)) for k in keys}
 np.savez_compressed(path,initial=tr['initial'],xy=tr['xy'],final_valid=tr['valid'],**arr)

def main():
 f=read(U/'FROZEN.json');assert sha(U/'candidate.py')==f['candidate_sha256'];assert sha(U/'run.py')==f['runner_sha256'];assert not (U/'RESULT.json').exists()
 for p in f['prior_pins']:assert sha(p['path'])==p['sha256'],p['path']
 (U/'tracks').mkdir();(U/'outputs').mkdir();diags=[];index=[];clipcounts=[];start=time.perf_counter()
 strict={(x['clip_index'],x['frame'],x['side']):x['raw_assessment'] for x in read(P/'DIAGNOSTICS_V3.json')}
 cases=[]
 for k,m in enumerate(f['manifest']):
  assert sha(m['prepared'])==m['prepared_sha256'] and sha(m['rgb_path'])==m['rgb_sha256']
  z=dict(np.load(m['prepared']));r=dict(np.load(P/'outputs_v3'/(m['clip_id']+'.npz')))
  cases.append((k,m['clip_id'],m['rgb_path'],z,r['anchors'],None))
 for n,br in enumerate(read(P/'RESULT_V3.json')['broad_diagnostics']):
  cid=br['clip'];m=next(x for x in f['controls'] if x['clip_id']==cid);assert sha(m['path'])==m['sha256']
  a=dict(np.load(m['path']));pose=a['xy_px'][:,:33];pv=a['usable'][:,:33];xy=a['xy_px'][:,33:75].reshape(-1,2,21,2);v=a['usable'][:,33:75].reshape(-1,2,21)
  rgb=G/'source_clips'/(cid+'.mp4');assert sha(rgb)==m['source']['sha256']
  z=dict(full_xy=xy,full_valid=v,pose=pose,pose_valid=pv,frame_id=a['frame_id'],timestamp_us=a['frame_id']/30*1e6,requested=np.zeros(v.shape[:2],bool))
  anchors=build_anchors(xy,v,pose,pv,du.inspect_pair)
  cases.append((7+n,cid,str(rgb),z,anchors,br['frame']))
 for k,cid,rgb,z,anchors,only in cases:
  gray,fps=images(rgb,z['frame_id']);times=z['timestamp_us']/1e6;cache={};outdir=U/'tracks'/cid;outdir.mkdir();local=[]
  targets=list(range(len(gray))) if only is None else [only]
  for t in targets:
   xy=z['full_xy'];valid=z['full_valid'];pose=z['pose'][t];pv=z['pose_valid'][t];sc=float(np.linalg.norm(pose[11]-pose[12])) if pv[[11,12]].all() else 0
   if sc<=1:continue
   hands=[('original',s,xy[t,s],None) for s in (0,1) if valid[t,s].all()]
   if only is None:
    for s in (0,1):
     if z['requested'][t,s]:hands.append(('hamer',s,np.load(H/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px'],strict[(k,t,s)]))
   if not hands:continue
   near=nearest_anchors(t,anchors,times);tracks={0:[],1:[]}
   for s in (0,1):
    for a in near[s]:
     key=(a,t,s)
     if key not in cache:
      asc=float(np.linalg.norm(z['pose'][a,11]-z['pose'][a,12]));other=xy[a,1-s] if valid[a,1-s].all() else None
      tr=region_flow(gray,a,t,xy[a,s],other,asc);dest=outdir/f'{a:04d}_{t:04d}_{s}.npz';save_track(dest,tr);cache[key]=tr
      index.append(dict(clip=cid,anchor=a,target=t,side=s,path=str(dest),sha256=sha(dest),scale=asc,qualified=tr['qualified'],reason=tr['reason'],steps=[{q:st[q] for q in ['frame_from','frame_to','residual','qualified']} for st in tr['steps']]))
     tracks[s].append(cache[key])
   _,gx,gy,threshold=ct.fields(gray[t])
   for source,s,hand,old in hands:
    contour=[]
    for side in (0,1):
     e=pose[13+side]/3 if pv[13+side] else None;w=hand[0]/3 if np.isfinite(hand[0]).all() else None
     c=ct.tube(gx,gy,threshold,e,w,sc/3);contour.append({q:v for q,v in c.items() if q not in ['slices','offsets']})
    rec=dict(clip=cid,clip_index=k,frame=t,side=s,source=source,assessment=assess(hand,pose,pv,sc,tracks,contour),previous_strict=old)
    diags.append(rec);local.append(rec)
  if only is None:
   r=dict(np.load(P/'outputs_v3'/(cid+'.npz')));np.savez_compressed(U/'outputs'/(cid+'.npz'),**r)
  clipcounts.append(dict(clip=cid,frames=len(targets),assessments=len(local),tracks=len(cache)))
  print(cid,'assessments',len(local),'region paths',len(cache),flush=True)
 write(U/'EVIDENCE.json',diags);write(U/'TRACK_INDEX.json',index)
 ham=[x for x in diags if x['source']=='hamer'];conf=[x for x in diags if x['clip_index']==0 and 77<=x['frame']<=86]
 report=dict(status='REPLAY_COMPLETE_REVIEW_REQUIRED',primary_frames=812,primary_clips=7,public_people=4,broad_scenes=2,assessments=len(diags),region_paths=len(index),hamer_requests=len(ham),
  old_strict_supported=sum(x['previous_strict']['supported_side']>=0 for x in ham),position_evidence=sum(x['assessment']['position_evidence'] for x in ham),
  identity_proposals=sum(x['assessment']['identity_proposal']>=0 for x in ham),new_coarse_when_strict_unknown=sum(x['assessment']['identity_proposal']>=0 and x['previous_strict']['supported_side']<0 for x in ham),
  conflicts=conf,clip_counts=clipcounts,elapsed_seconds=time.perf_counter()-start,coordinates_changed=0,training=0,new_neural_inference=0,user_data_access=0,production_change=False)
 write(U/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['conflicts','clip_counts']},ensure_ascii=False))
if __name__=='__main__':main()
