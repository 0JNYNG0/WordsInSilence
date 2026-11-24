"""New FIT-only label edition, with a same-schedule continuation control."""
from graph import *
import numpy as np
O=U.parent;R=O/'word100_candidate_role_learning_20261111'


def main():
 target=U/'occurrence_supervision/INPUTS_COMPLETE.json'
 assert not target.exists()
 admission=read(U/'occurrence_supervision/ROOT_ADMISSION.json')
 assert admission['approved_for_isolated_fit'] is True
 audit=read(U/'occurrence_supervision/DIRECT_ROLE_AUDIT.json')
 assert admission['audit_sha256']==hashlib.sha256((U/'occurrence_supervision/DIRECT_ROLE_AUDIT.json').read_bytes()).hexdigest()
 accepted={(r['clip_id'],r['sequence_index']):r for r in audit['rows'] if r['category']!='uncertain'}
 assert {r['category'] for r in accepted.values()}<={'new_addition','duplicate','conflict'}, {r['category'] for r in accepted.values()}
 labels={r['clip_id']:r for r in read(R/'ROLE_LABELS.json')['rows']}
 sources=[s for s in read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'] if not s['reverse'] and s['clip_id'] in labels]
 write(U/'occurrence_supervision/FROZEN.json',dict(
  conditions=['continued_old_direct','reviewed_roles'],seeds=[42,43,44],primary_seed=42,
  epochs=25,lr=.0003,weight_decay=.01,teacher_kl_weight=.2,
  trainable='R role MLP only; common100 observer, attribute, motion, quality/pair, TCN and DTW frozen',
  supervision='Only old w=1 or explicitly admitted new RGB frames; no unseen-frame interpolation.',
  comparison='Same inputs, initialization, schedule and seeds. Label edition is the condition difference.',
  weak_labels_used=False,source_reversal_training=False,
  teacher='Old matching-seed role output, only outside current direct supervision; not ground truth.',
  selection='Final epoch. No seed/threshold selection from DEV. Preserve all conditions.',
  validation='Existing exposed DEV plus new source-audited diagnostics; reserved REAL17/18 untouched.',
  admission_sha256=hashlib.sha256((U/'occurrence_supervision/ROOT_ADMISSION.json').read_bytes()).hexdigest(),
  private_used=False,web_changed=False))
 records=[]
 for s in sources:
  cid=s['clip_id'];assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256']
  z=np.load(s['path']);old=labels[cid];assert old['source_sha256']==s['source_sha256']
  y=z['y'].copy();w=(z['w']==1).astype(np.float32)
  for condition in ['continued_old_direct','reviewed_roles']:
   yy=y.copy();ww=w.copy();changes=[]
   if condition=='reviewed_roles':
    for (clip,i),r in accepted.items():
     if clip!=cid:continue
     assert r['source_sha256']==s['source_sha256']
     if r['category']=='conflict':assert f'{cid}:{i}' in admission['accepted_conflicts']
     assert int(y[i])==r['old_family']
     changes.append(dict(index=i,old_y=int(y[i]),old_w=float(w[i]),new_y=r['proposed_family']))
     yy[i]=r['proposed_family'];ww[i]=1
   # Ambiguous new observations remain outside the NEW label edition; existing
   # source labels are preserved unless an explicit admitted correction exists.
   p=U/'occurrence_supervision/inputs'/condition/f"{cid}_{s['view']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(p,z=z['z'],y=yy,w=ww)
   records.append(dict(condition=condition,clip_id=cid,person=s['person'],view=s['view'],source_sha256=s['source_sha256'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),direct_frames=int(ww.sum()),changes=changes))
 write(target,dict(rows=records,public_source_videos=len({s['clip_id'] for s in records}),no_feature_reinference=True))
 print('Prepared',len(records),'condition/view inputs')

if __name__=='__main__':main()
