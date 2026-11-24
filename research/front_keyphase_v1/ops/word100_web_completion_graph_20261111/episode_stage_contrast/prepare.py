"""Add directly viewed confusable non-target moments, not word-specific inference rules."""
from pathlib import Path
import sys,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent; U=W.parent; P=U/'episode_stage'
sys.path.insert(0,str(U))
from graph import read,write,stamp
F=[2,3,4,5,7]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Actual RGB overview samples reviewed by root. Values are source annotations only.
# entry / deliberate other held presentation / returning arms. No interval filling.
S={
 '0739':{2:([36,40],[48,52,56,60],[104,108]),4:([60,64],[72,76,80,84],[136,140]),5:([32,36],[44,48,52],[96,100]),6:([44,48],[56,60,64,68],[112,116]),8:([36,40],[48,52,56],[108,112]),9:([32,36],[40,44,48],[88,92])},
 '1133':{2:([40],[],[80,84]),4:([64,68],[],[112,116]),5:([36,40],[],[80,84]),6:([32,36],[],[68,72]),8:([64,68],[],[112,116]),9:([36,40],[],[76,80])},
 '1567':{2:([36,40],[],[108,112]),4:([56,60],[],[128,132]),5:([36,40],[],[100,104]),6:([40,44],[],[100,104]),8:([52,56],[],[124,128]),9:([48,52],[],[108,112])},
 '2523':{2:([40,44],[52],[88,92]),4:([68,72],[80,84],[112,116]),5:([48,52],[60],[88,92]),6:([48,52],[60,64],[92,96]),8:([36,40],[48,52],[84,88]),9:([52,56],[60,64],[88,92])}}

def main():
 assert not (W/'INPUTS_COMPLETE.json').exists()
 srcpath=U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json';src=read(srcpath);sm={r['clip_id']:r for r in src['rows']};old=read(P/'INPUTS_DENSE_COMPLETE.json');rows=[];observations=[];added=collections.Counter()
 for sr in src['rows']:
  sh=next(s for s in sr['sheets'] if s['path'].endswith('_overview.jpg'));assert sha(sh['path'])==sh['sha256']
  entry,other,ret=S[sr['word_id'][4:]][sr['person']];direct=set(sh['frames']);assert set(entry+other+ret)<=direct
  assert all(f<sr['groups'][0]['onset'][0] for f in entry+other)
  assert all(f>max(b for a,b in sr['holds']) for f in ret)
  observations.append(dict(clip_id=sr['clip_id'],source_sha256=sr['source_sha256'],sheet=sh,entry=entry,other_held_presentation=other,returning=ret,meaning='Not this queried movement or its terminal held posture. NOT an incorrect sign; a held presentation may be important to another clause.',reviewer='root Codex actual sampled RGB review; no independent human/KSL confirmation'))
 obsmap={r['clip_id']:r for r in observations}
 for r in old['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['cache'])==r['cache_sha256'];sr=sm[r['clip_id']];d=np.load(r['path']);z=np.load(r['cache'])['z'];target={(t['frame'],t['family']):dict(t) for t in r['labels']};ob=obsmap[r['clip_id']];new=[]
  def put(f,family,basis):
   key=(f,family)
   if key in target:assert target[key]['label']==0;return
   t=dict(frame=f,family=family,label=0,basis=basis);target[key]=t;new.append(t);added[r['split']]+=1
  for g in sr['groups']:
   for role in ['entry','other_held_presentation','returning']:
    for f in ob[role]:put(f,g['family'],'direct_reviewed_'+role)
   # A different action's terminal hold is a different action, not this one's hold.
   if g['family']!=sr['groups'][-1]['family']:
    for f in sr['hold_sample_frames']:put(f,g['family'],'different_reviewed_action_terminal_hold')
  labels=sorted(target.values(),key=lambda t:(t['frame'],t['family']));ids=sorted({t['frame'] for t in labels});y=np.full((len(ids),5),-1,np.int64);pos={f:i for i,f in enumerate(ids)}
  for t in labels:y[pos[t['frame']],F.index(t['family'])]=t['label']
  dest=W/'inputs'/Path(r['path']).name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,x=z[ids].astype(np.float32),y=y,frame_id=np.array(ids))
  counts=collections.Counter(f'{t["family"]}/{t["label"]}' for t in labels)
  rows.append(dict(r,path=str(dest),sha256=sha(dest),labels=labels,counts=dict(counts),targets=len(labels),sample_points=len(ids),new_labels=new))
 write(W/'RGB_CONTRAST_LABELS.json',dict(at=stamp(),rows=observations,training_labels_are_not_sign_error=True,interpolation=False))
 write(W/'INPUTS_COMPLETE.json',dict(rows=rows,added_targets=dict(added),families=F,pins={str(p):sha(p) for p in [Path(__file__),srcpath,P/'INPUTS_DENSE_COMPLETE.json',W/'RGB_CONTRAST_LABELS.json']},limits=old['limits']+['New negatives are same-video observed entry/return and other held presentations, not unreviewed unknowns.','Probe expanded before comparison; old340 diagnostic remains separately reportable. Not a new independent test.'],training_allowed=False))
 print('Prepared directly reviewed context contrasts in24 public videos. Added',dict(added))
if __name__=='__main__':main()
