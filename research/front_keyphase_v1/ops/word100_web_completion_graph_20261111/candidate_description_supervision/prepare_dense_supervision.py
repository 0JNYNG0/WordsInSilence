"""Broaden training window positions without adding invented motion truth."""
from core import *
from collections import defaultdict
from dense_proposal_probe import pooled
import time,traceback
N=W/'dense_window_supervision'

def main():
 assert not (N/'INPUTS_COMPLETE.json').exists();N.mkdir(exist_ok=True)
 sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];co={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};sup=read(W/'short_action_review/REVIEW.json')['rows']+read(W/'short_action_review/FOREARM_REVIEW.json')['rows'];old=read(W/'INPUTS_COMPLETE.json');aa=np.load(old['arrays']);rows=list(old['rows']);xx=list(aa['x']);yy=list(aa['y']);mm=list(aa['mask']);oldkeys={(r['clip_id'],r['view'],tuple(r['span'])) for r in rows};cache={(r['clip_id'],r['view']):r for r in read(U/'context_role_training/INPUTS_COMPLETE.json')['rows']};rng=np.random.default_rng(20261112)
 write(N/'PREPARATION_CONTRACT.json',dict(at=stamp(),reason='Dense proposals exposed confident short/partial selections. Existing training saw limited candidate placements. Expand FIT window positions while retaining original labels and model.',per_supervision_signature_max=16,normal_sources_only=True,partial_motion_is_positive_presence=True,coverage_not_linguistic_completion=True,unknown_masked=True,no_eval_labels_used=True))
 for s in sources:
  for r in sup:
   if r['clip_id']==s['clip_id']:
    e=next(e for e in s['events'] if e['event_id']==r['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+r['additional_progression_frames']))
  groups,bg=source_witnesses(s,co[s['clip_id']]);split='probe' if s['person'] in [8,9] else 'train'
  for view in ['full','upper']:
   cr=cache[s['clip_id'],view];assert sha(cr['path'])==cr['sha256'];z=np.load(cr['path'])['z'];q=z[:,292:392].reshape(-1,10,10);buckets=defaultdict(list)
   for a in range(len(z)-3):
    for b in range(a+3,len(z)):
     if (s['clip_id'],view,(a,b)) in oldkeys:continue
     y=np.zeros((5,3),np.float32);m=np.zeros_like(y);basis={}
     for family in groups:
      valid=q[:,AX[family]*2,8]>.5
      if not valid[a] or not valid[b] or valid[a:b+1].mean()<.8:continue
      y0,m0,why=labels(a,b,family,groups,bg,valid);j=F.index(family);y[j]=y0;m[j]=m0;basis[str(family)]=why
     if m.any():buckets[tuple(np.r_[y.ravel(),m.ravel()].astype(int))].append((a,b,y,m,basis))
   chosen=[]
   for signature,items in sorted(buckets.items()):
    indices=rng.choice(len(items),min(16,len(items)),replace=False);chosen.extend(items[i] for i in sorted(indices))
   if chosen:
    feats=pooled(z,np.array([[v[0],v[1]] for v in chosen]))
    for f,(a,b,y,m,basis) in zip(feats,chosen):
     xx.append(f);yy.append(y);mm.append(m);rows.append(dict(clip_id=s['clip_id'],person=s['person'],word_id=s['word_id'],source_sha256=s['source_sha256'],view=view,split=split,span=[a,b],origin='dense_reviewed_witness_stratified',basis=basis,cache=cr['path'],cache_sha256=cr['sha256']))
  print('dense source',s['clip_id'],len(rows),flush=True)
 p=N/'inputs.npz';np.savez_compressed(p,x=np.array(xx),y=np.array(yy),mask=np.array(mm),train=np.array([r['split']=='train' for r in rows]));meta=dict(old,at=stamp(),rows=rows,arrays=str(p),arrays_sha256=sha(p),old_windows_retained=len(old['rows']),new_windows=len(rows)-len(old['rows']),source_truth=old['source_truth'],pins={str(p):sha(p) for p in [Path(__file__),W/'core.py',N/'PREPARATION_CONTRACT.json',W/'INPUTS_COMPLETE.json']});write(N/'INPUTS_COMPLETE.json',meta)
 cfg=read(W/'FROZEN.json');cfg.update(at=stamp(),data_sha256=sha(p),pins={str(p):sha(p) for p in [N/'INPUTS_COMPLETE.json',N/'PREPARATION_CONTRACT.json',W/'core.py',W/'train.py']},selection=dict(conditions=['baseline','previous_normal_body','dense_candidates']),training_change='Only FIT window coverage; same network/seeds/steps/optimizer. Development results did not select thresholds, seeds, or epochs.');write(N/'FROZEN.json',cfg)
 a=np.load(p);assert np.isfinite(a['x']).all();tr={r['person'] for r in rows if r['split']=='train'};va={r['person'] for r in rows if r['split']=='probe'};assert tr=={2,4,5,6} and va=={8,9} and not tr&va
 for r,y,m in zip(rows,a['y'],a['mask']):
  for fam,b in r['basis'].items():
   j=F.index(int(fam));assert m[j,2]==0 or y[j,2]==float(set(b['required_observed_progression'])<=set(b['own_progression']))
 write(N/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),pass_=True,split_disjoint=True,original_arrays_preserved=bool(np.array_equal(a['x'][:len(old['rows'])],aa['x'])),windows=len(rows),coverage_label_recomputed=True,no_private=True,no_final8=True));print('ready',len(rows),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:N.mkdir(exist_ok=True);(N/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
