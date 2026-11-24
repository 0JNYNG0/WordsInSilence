from pathlib import Path
import sys,json,hashlib,time,copy
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'));import server
sys.path.insert(0,str(D));from evidence_candidate import EvidenceLesson,RoleEvidenceCandidate,adapter,hand_evidence,np
from frontal_evidence_logic import partial_match,overlap_bridges
sys.path.insert(0,str(N));from source_policy import require_front
from required_role_candidate import torch
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
def main():
 torch.set_num_threads(2);old=RoleEvidenceCandidate();new=EvidenceLesson();rows=[];cache={}
 for r in read(N/'INPUTS.json')['rows']:
  require_front(r)
  if r['word']!='WORD0099':continue
  z=dict(np.load(r['raw']));a=old.analyze(z);b=new.analyze(z);cache[r['id']]=(r,z,a,b)
  aa={c['id']:c['status'] for c in a['criteria']};bb={c['id']:c['status'] for c in b['criteria']};c=next(c for c in b['criteria'] if c['id']=='wipe_shape')
  rows.append(dict(id=r['id'],before=aa,after=bb,bridges=c.get('bridged_projected_overlap'),passed=not any(v=='supported' and bb[k]!='supported' for k,v in aa.items()) and not any(v=='difference' and aa[k]!='difference' for k,v in bb.items())))
 write(D/'checks/NORMAL_APPLE.json',dict(rows=rows,passed=all(r['passed'] for r in rows)));print('NORMAL_APPLE',len(rows),sum(r['passed'] for r in rows),flush=True)
 # Real public finger source annotations: classify observations first, then compare
 # known target contradictions. Masking another finger tests aggregation only.
 labels=read(O/'word100_finger_repair_20261115/SOURCE_REVIEW_INPUTS.json')['rows'];pred={};fchecks=[]
 for r in labels:
  require_front(r);key=(r['clip_id'],r['side'])
  if key not in pred:
   raw=dict(np.load(r['raw_path']));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));pred[key]=(new.shape.sequence(ch,r['side']),raw['frame_id'])
  sh,ids=pred[key];i=int(np.flatnonzero(ids==r['frame'])[0]);truth=np.asarray(r['y'][1:]);f=np.asarray(sh['fingers'][i]);available=np.asarray(sh['available'][i],bool)
  # Only existing confidently correct observed fingers with visible source labels
  # are used for this controlled loss-of-one-other-prediction check.
  mapped=np.where(truth==0,1,np.where(truth==2,0,-1));valid=(mapped>=0)&available&(f==mapped)
  for mismatch in np.flatnonzero(valid):
   for unknown in range(4):
    if unknown==mismatch:continue
    target=mapped.copy()
    if (target<0).any():continue
    target[mismatch]=1-target[mismatch];v=f.copy();v[unknown]=-1
    before=1 if (v==target).all() else 0 if (v>=0).all() else -1
    after=int(partial_match(v[None],available[None],target)[0]);fchecks.append(dict(clip=r['clip_id'],side=r['side'],frame=r['frame'],mismatch=int(mismatch),unknown=unknown,before=before,after=after,passed=after==0,synthetic_prediction_mask=True))
 write(D/'checks/PARTIAL_FINGERS.json',dict(rows=fchecks,public_label_rows=len(labels),passed=bool(fchecks)and all(r['passed']for r in fchecks)));print('FINGER_MASK',len(fchecks),sum(r['passed']for r in fchecks),flush=True)
 # Controlled masks on otherwise valid public observed trajectories; labelled
 # as synthetic, never as genuine newly collected learner errors.
 checks=[];source_frames=[]
 for cid,(r,z,a,b) in cache.items():
  if r.get('role')=='fit':continue
  pair=next(e for e in a['events']if e['region']==2);lo,hi=pair['context'];ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));s=new.surface.sequence(ch['tcn280'],'R');facts=hand_evidence(ch,'R')
  # Center a mask inside an observed, stable palm run, not on a private timestamp.
  centers=[i for i in range(lo+4,hi-4)if np.all(s[i-3:i+4]==0)and not facts['uncertain'][i-3:i+4].any()]
  if not centers:continue
  mid=centers[len(centers)//2];ts=np.asarray(ch['timestamp_us']);source_frames.append(dict(id=cid,video=r.get('video',r.get('video_path')),frame=mid))
  for kind in ['brief_projection','true_missing','change_inside','changed_after','long_projection','unbounded_projection']:
   f={k:v.copy()for k,v in facts.items()};v=s.copy();t=ts.copy();start,end=mid-2,mid+2;f['projected_overlap'][start:end+1]=True
   if kind=='true_missing':f['own_missing'][mid]=True
   elif kind=='change_inside':v[mid]=1
   elif kind=='changed_after':v[end+1]=1
   elif kind=='long_projection':t[end+1:]+=1000000
   elif kind=='unbounded_projection':f['projected_overlap'][lo:start]=True
   f['uncertain']=f['projected_overlap']|f['own_missing'];accepted,rejected=overlap_bridges(v,f,t,lo,hi);found=any(x<=mid<=y for x,y in accepted)
   checks.append(dict(id=cid,kind=kind,accepted=found,passed=found==(kind=='brief_projection'),synthetic_mask=True))
 write(D/'checks/OVERLAP_CONTROLS.json',dict(rows=checks,source_frames=source_frames,passed=bool(checks)and all(r['passed']for r in checks)));print('OVERLAP',len(checks),sum(r['passed']for r in checks),flush=True)
 # Unknown-only and partial support must never imply mismatch or a full match.
 unit=[]
 for e in [[1,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,1,1]]:
  for unknown in range(4):
   f=np.array(e);f[unknown]=-1;got=int(partial_match(f[None],np.ones((1,4),bool),e)[0]);unit.append(got==-1)
 write(D/'checks/UNKNOWN_NOT_ERROR.json',dict(conditions=len(unit),passed=all(unit)))
 # Existing public synthetic order/direction/shape/partial/normal-speed controls.
 import check_controls
 dest=D/'checks/apple_controls';(dest/'controls').mkdir(parents=True,exist_ok=True)
 check_controls.main(EvidenceLesson,dest,True)
 write(D/'checks/SUMMARY.json',dict(normal_passed=all(r['passed']for r in rows),finger_passed=bool(fchecks)and all(r['passed']for r in fchecks),overlap_passed=bool(checks)and all(r['passed']for r in checks),unknown_passed=all(unit),control_failed=len(read(dest/'CONTROLS.json')['failed']),private_data=False,training_runs=0))
if __name__=='__main__':main()
