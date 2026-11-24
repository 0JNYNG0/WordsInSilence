from core import *
from collections import Counter
meta=read(W/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);rows=meta['rows'];train=a['train'];y=a['y'];m=a['mask'];assert sha(meta['arrays'])==meta['arrays_sha256']
assert np.isfinite(a['x']).all() and set(np.unique(y))<={0,1} and set(np.unique(m))<={0,1}
assert len(rows)==len(a['x'])==1137 and a['x'].shape[1]==912
assert set(r['source_sha256'] for r in rows if r['split']=='train').isdisjoint(r['source_sha256'] for r in rows if r['split']=='probe')
assert set(r['person'] for r in rows if r['split']=='train')=={2,4,5,6};assert set(r['person'] for r in rows if r['split']=='probe')=={8,9}
assert all(h==sha(p) for p,h in meta['pins'].items())
# Separate arithmetic reconstruction from saved witness sets, no learned predictions.
for i,r in enumerate(rows):
 for fam,b in r['basis'].items():
  j=F.index(int(fam));own=set(b['own_progression']);other=set(b['other_reviewed']);required=set(b['required_observed_progression'])
  assert y[i,j,0]==bool(own)
  if m[i,j,1]:assert y[i,j,1]==(not other)
  if m[i,j,2]:assert y[i,j,2]==(required<=own)
  if b['missing_observation']:assert m[i,j,1]==0
for fam in range(5):
 for head in range(3):assert all(((y[:,fam,head]==v)&(m[:,fam,head]>0)&train).any() for v in [0,1])
audit=dict(at=stamp(),status='PASS_LIMITED_SOURCE_LABELS',source_split=meta['source_person_split'],source_clips=24,views=48,windows=len(rows),unknown_labels_masked=True,counts=meta['counts'],limitations=['Sparse witnesses are not dense gold. Purity means no reviewed conflicting witness, not proof of every intervening frame.','Coverage is this source occurrence, not linguistic minimum duration or valid completion.','Probe people excluded from this head but previously seen upstream.','Family2 coverage probe has no negative; do not report balanced accuracy for that cell.'],verifier='root Codex separate arithmetic; not independent human/expert')
write(W/'PRETRAIN_VERIFICATION.json',audit)
cfg=dict(at=stamp(),seeds=[42,43,44],steps=1200,lr=.0003,weight_decay=.01,gradient_clip=2.,std_floor=.05,heads=meta['heads'],normalization='train windows only',loss='masked BCE, each family/head/class receives equal mass; each source clip equal within cell; two correlated views share source mass',no_word_id_input=True,upstream_frozen=True,web_changed=False,selection=dict(conditions=['baseline','presence_purity','presence_purity_coverage','coverage_joint'],eligibility='unchanged saved stage has_motion; no correct-direction/handshape filter',scores='sum log independent head probabilities; fixed unit weights; coverage_joint adds frozen parent-model cross-action log probability, no internal pair reward',relation_contract=[[2,3]],order_constraint=False,completion=None),gates=['Preserve all known normal progression witnesses and normal/speed candidate retention.','Reduce wrong-action selection on known reordered cases across all seeds without missing witnesses.','Retain wrong-direction/shape attempts for downstream comparison; missing-hand evidence never becomes observed.','No web adoption before downstream semantic regression and reserved final evaluation.'],data_sha256=meta['arrays_sha256'],pins={str(p):sha(p) for p in [W/'core.py',W/'prepare.py',W/'INPUTS_COMPLETE.json',W/'PRETRAIN_VERIFICATION.json',Path(__file__)]})
assert not (W/'FROZEN.json').exists();write(W/'FROZEN.json',cfg);print('AUDIT PASS; fixed experiment ready',flush=True)
