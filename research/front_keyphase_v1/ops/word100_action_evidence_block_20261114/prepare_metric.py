from pathlib import Path
import json,hashlib,collections
import numpy as np
from sequence_core import features
U=Path(__file__).resolve().parent;D=U/'metric_recovery';D.mkdir(exist_ok=True);O=U.parent;P=O/'word100_terminal_relation_targets_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(D/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
first=read(U/'EVALUATION.json');assert not first['summary']['preliminary_gate']
# All labels below are from the six FIT review sheets actually viewed, not model predictions.
opening={2:[0,2,1,1,1,1],3:[0,0,1,1,1,1],4:[0,None,1,1,1,1],5:[0,2,1,1,1,1],6:[0,2,None,1,1,1],7:[0,2,1,1,1,1]}
cross={2:{'R':[1,None,1,1,None,1],'L':[1,None,1,1,None,1]},
3:{'R':[1,None,1,1,None,1],'L':[1,None,1,1,None,1]},
4:{'R':[1,None,1,1,None,1],'L':[1,None,1,1,None,1]},
5:{'R':[None,1,1,1,None,1],'L':[1,1,1,1,None,1]},
6:{'R':[1,1,1,1,None,1],'L':[1,1,1,1,None,1]},
7:{'R':[1,1,1,1,None,1],'L':[1,1,1,1,None,1]}}
old=read(U/'LABELS.json');bykey={(r['clip_id'],r['hand'],r['frame']):r for r in old['point_rows']}
new=[];conflicts=[];duplicates=[];catalog=[]
for r in read(U/'REVIEW_INPUTS.json')['rows']:
 rec=r['record_id'];cid=r['clip_id'];hand=r['hand']
 if r['origin']=='RGB_open_to_closed':labels=opening[rec]
 elif r['origin']=='RGB_closed_shape_during_forearm_raise':labels=[1]*6
 elif 'WORD1133'in cid:labels=[0 if hand=='R'else 1]*6
 else:labels=cross[rec][hand]
 catalog.append(dict(pair_index=r['pair_index'],frames=r['frames'],labels=labels))
 for frame,target in zip(r['frames'],labels):
  if target is None:continue
  key=cid,hand,frame
  item=dict(clip_id=cid,record_id=rec,split='fit',hand=hand,frame=frame,target=target,
       label=['four_open','four_closed','transition'][target],origin='direct_FIT_multi_view_pose_review',
       source=r['source'],source_sha256=r['source_sha256'],sheet=r['review_sheet'])
  if key in bykey:
   if bykey[key]['target']!=target:conflicts.append(dict(previous=bykey[key],new=item))
   else:duplicates.append(key)
  else:new.append(item);bykey[key]=item
conflictkeys={(r['new']['clip_id'],r['new']['hand'],r['new']['frame'])for r in conflicts}
points=[r for k,r in bykey.items()if k not in conflictkeys]
manifest={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']};cache={};pins={};xx=[]
for r in points:
 cid=r['clip_id']
 if cid not in cache:
  m=manifest[cid];p=Path(m['channel']);assert sha(p)==m['channel_sha256'];z=np.load(p);cache[cid]=(z['feedback'if'feedback'in z.files else'full'],z['frame_id']);pins[str(p)]=sha(p)
 x,fid=cache[cid];ids=np.flatnonzero(fid==r['frame']);assert len(ids)==1;f,ok=features(x[ids],r['hand']);xx.append(f[0]);r['available']=bool(ok[0])
np.save(D/'point_features.npy',np.stack(xx))
write('LABELS.json',dict(rows=points,added=new,conflicts_excluded=conflicts,duplicate_annotations=len(duplicates),catalog=catalog,pins=pins,
 target='Visible configuration only; curved morphology not whole-action progress',
 new_labels_are_FIT_only=all(r['record_id']<=7 for r in new),review='Same Codex, not independent educator',
 preserve_old_results=True,do_not_train_on_probe=True))
write('PLAN.json',dict(reason='Sequence-only candidate returns five uncertain probe cases; FIT view-invariant configuration supervision was sparse.',
 remedy='Use explicit same-configuration contrasts across FIT palm views and records; retain unknown/occluded labels and original endpoint outputs.',
 no_score_threshold_changes=True,old_candidate_retained=str(U/'EVALUATION.json'),
 gate='Original 24 condition change gate and original endpoint preservation; no all-withholding improvement. G2 integration still required.'))
print(json.dumps(dict(new_labels=len(new),unique_point_labels=len(points),conflicts_excluded=len(conflicts),duplicates=len(duplicates))))
