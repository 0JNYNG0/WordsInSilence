"""Sparse source-reviewed motion/terminal-hold supervision; no semantic correctness."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
FAMILIES=[2,3,4,5,7];MAP={0:2,1:3,2:4,3:7,4:5}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
 assert not (W/'INPUTS_COMPLETE.json').exists()
 source_path=U/'episode_membership/coverage_review/SOURCE_EPISODES.json';src=read(source_path)
 cachepath=U/'context_role_training/INPUTS_COMPLETE.json';cache={(r['clip_id'],r['view']):r for r in read(cachepath)['rows']}
 rows=[];conflicts=[];checks=[]
 for r in src['rows']:
  assert r['role']=='fit' and r['person'] in [2,4,5,6,8,9]
  seen=set()
  for s in r['sheets']:
   assert s['viewed'] and sha(s['path'])==s['sha256'];seen.update(s['frames'])
  groups=r['groups'];assert all(g['required_repetition_count'] is None for g in groups)
  for view in ['full','upper']:
   c=cache[r['clip_id'],view];assert sha(c['path'])==c['sha256'];z=np.load(c['path'],allow_pickle=False)['z'];assert len(z)==r['frames']
   target=np.full((len(z),5),-1,np.int64);reason={};q=z[:,292:392].reshape(-1,10,10)
   def put(frame,family,label,basis):
    j=FAMILIES.index(family)
    if target[frame,j] not in [-1,label]:conflicts.append(dict(clip=r['clip_id'],view=view,frame=frame,family=family,old=int(target[frame,j]),new=label));target[frame,j]=-2
    elif target[frame,j]!=-2:target[frame,j]=label;reason[frame,j]=basis
   for g in groups:
    axis=g['axis'];family=g['family'];valid=q[:,axis*2,8]>.5
    # Group inner points, including turns, are action-in-progress evidence.
    # Source boundary bands are unknown; one-frame or arbitrary dense labels are not created.
    points=[f for f in sorted(seen) if g['onset'][1]<f<g['offset'][0]]
    for f in points:
     if valid[f]:put(f,family,1,'direct_sample_inside_reviewed_action_group')
    # Only the final group's adjacent held posture is labelled terminal-hold.
    if g is groups[-1]:
     for f in r['hold_sample_frames']:
      if f>g['offset'][1] and valid[f]:put(f,family,2,'direct_sample_terminal_hold_after_this_group')
    # Rest before first action and after all holds: direct viewed, broad margins.
    # Restrict to frames0..20 and final three source samples to avoid entry/return.
    rest=[f for f in sorted(seen) if f<=20 or f>=r['frames']-12]
    for f in rest:
     if all(not(gg['onset'][0]<=f<=gg['offset'][1]) for gg in groups) and not any(a<=f<=b for a,b in r['holds']):put(f,family,0,'direct_viewed_rest_sample')
    # Other reviewed guide actions in this recording provide association negatives,
    # not errors. Do not label unreviewed different families.
    for other in groups:
     if other['family']==family:continue
     for f in sorted(seen):
      if other['onset'][1]<f<other['offset'][0] and valid[f]:put(f,family,0,'different_reviewed_action_in_same_source')
   target[target==-2]=-1
   ids=np.flatnonzero((target>=0).any(1));assert len(ids)
   p=W/'inputs'/f"{r['clip_id']}_{view}.npz";p.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(p,x=z[ids].astype(np.float32),y=target[ids],frame_id=ids)
   counts=collections.Counter(f'{FAMILIES[j]}/{int(target[f,j])}' for f in ids for j in range(5) if target[f,j]>=0)
   rows.append(dict(clip_id=r['clip_id'],person=r['person'],source_sha256=r['source_sha256'],view=view,split='probe' if r['person'] in [8,9] else 'train',path=str(p),sha256=sha(p),cache=c['path'],cache_sha256=c['sha256'],counts=dict(counts),sample_points=len(ids),targets=sum(counts.values()),labels=[dict(frame=int(f),family=FAMILIES[j],label=int(target[f,j]),basis=reason[f,j]) for f in ids for j in range(5) if target[f,j]>=0]))
  checks.append(dict(clip_id=r['clip_id'],source_sha256=r['source_sha256'],direct_seen=len(seen),groups=len(groups)))
 assert not conflicts
 report=dict(rows=rows,counts=dict(collections.Counter(k for r in rows for k in [r['split']] for _ in range(r['targets']))),families=FAMILIES,classes={'0':'other reviewed action/rest','1':'action in progress including internal turns','2':'terminal held posture'},source_checks=checks,pins={str(p):sha(p) for p in [Path(__file__),source_path,cachepath]},limits=['24 already-owned public FIT videos;16 continuation train,8 held from this head only. Earlier shared encoder has seen these people; not fresh-person generalization.','Only directly viewed sampled points. Unknown boundaries are masked. No dense interpolation.','Source occurrence extent is not semantic completion; no required repeat count/gesture amplitude target.','No inferred hidden finger/contact truth; handshape/direction modules remain frozen.','No word ID, source frame number, or desired order enters x.','Four pilot words supply labels, common100 observation remains frozen; no all-word accuracy claim.','Full and upper versions correlated, not independent recordings.'],training_allowed=False,conflicts=conflicts)
 write(W/'INPUTS_COMPLETE.json',report);print('Prepared',len(rows),'views',report['counts'])
if __name__=='__main__':main()
