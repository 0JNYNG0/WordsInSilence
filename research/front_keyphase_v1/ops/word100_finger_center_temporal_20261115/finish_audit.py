from common import *
import copy
from datetime import datetime
audit=dict(at=datetime.now().isoformat(),reviewer='Codex visual review of original frontal and side RGB, not independent KSL expert',labels_frozen_unchanged=True,predictions_unchanged=True,post_prediction=True,not_fresh_generalization=True,
 corrections=[dict(clip_id='NIA_SL_WORD0071_REAL13_F',hand='L',frame=82,old=[1,1,1,1],audited=[1,0,0,-1],reason='Original F and nominal same-take side frames show left middle/ring folded, not all four extended. Pinky individual posture at this crossing is not sufficiently visible: exclude rather than impute. Right hand at same frame remains open. Original coarse positive_four_extended_RGB_label was applied to both hands.',evidence='review/regression/NIA_SL_WORD0071_REAL13_F_multiview.jpg')],
 retained_errors=[dict(clip_id='NIA_SL_WORD0739_REAL12_F',hand='R',frame=61,finger='index',truth=1,base=-1,new=0,evidence='review/regression/NIA_SL_WORD0739_REAL12_F_multiview.jpg',observation='Before the visible closing at f63, the fingers are still extended together at f61. Frontal index chain bends more than the projected hand contour; new center classifiers commit flexion. Projection/chain geometry and classification interact; sole causal attribution not established.'),dict(clip_id='NIA_SL_WORD2551_REAL12_F',hand='R',frame=35,finger='index',truth=1,base=0,new=0,observation='Prior independent-view source audit confirms brief extension. Center-chain seeds disagree and whole-hand center still predicts flexion. Temporal branch alone cannot recover the state.')],
 phase_false_changes=[dict(clip_id='NIA_SL_WORD2551_REAL16_F',hand='R',frames=[38],fingers=['index','middle'],observation='Curled shape maintained while hand turns; front projection changes and phase branch says changing.'),dict(clip_id='NIA_SL_WORD2748_REAL11_F',hand='R',frames=[71,73],fingers=['index'],observation='Hooked index maintained in side views while hand moves/turns; phase branch says changing.')],
 interpretation='A change flag is not proof of the exact central posture. Source truth, observation reliability and posture inference need separate handling. No threshold, weights or runtime changed after confirmation.')
write(U/'SOURCE_TRUTH_AUDIT.json',audit)
d=read(U/'v2_historic_RESULTS.json');rows=copy.deepcopy(d['rows'])
for r in rows:
 for c in audit['corrections']:
  if (r['clip_id'],r['hand'],r['frame'])==(c['clip_id'],c['hand'],c['frame']):assert r['shape']==c['old'];r['shape']=c['audited']
y=np.array([r['shape']for r in rows]);a=np.array([r['predictions']['available']for r in rows]);mask=(y>=0)&a;base=np.array([r['predictions']['base']for r in rows]);oldcorrect=(base==y)&mask;metrics={}
for mode in d['metrics']:
 p=np.array([r['predictions'][mode]for r in rows]);metrics[mode]=dict(known=int(mask.sum()),correct=int(((p==y)&mask).sum()),wrong=int(((p>=0)&(p!=y)&mask).sum()),unknown=int(((p<0)&mask).sum()),lost_previous_correct=int(((p!=y)&oldcorrect).sum()),new_correct=int(((p==y)&mask&~oldcorrect).sum()),new_confident_errors=int(((p>=0)&(p!=y)&mask&~((base>=0)&(base!=y))).sum()))
write(U/'HISTORIC_SOURCE_AUDITED.json',dict(metrics=metrics,rows=rows,source_audit=sha(U/'SOURCE_TRUTH_AUDIT.json'),scope='Saved predictions rescored symmetrically, no retraining. Original967 report preserved; uncertain left little excludes one item, denominator966. Post-prediction audit cannot be claimed as independent test.'))
for mode,m in metrics.items():print(mode,m)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[
 'Initial exposure inventory treated a dictionary like a list; corrected before source selection; no experiment artifact changed.',
 'Source extraction finished WORD2551p10 arrays then metadata duplicate-key error. Verified decoded138 frames/source SHA and recovered metadata; no re-extraction, runtime timing unavailable for that run. Other3 extractions completed after fix.',
 'Panel run before READY failed with FileNotFound; rerun only after dependencies completed.',
 'Old channel cache uses full key rather than tcn280; reader fallback corrected before training.',
 'Temporary Japanese next-action text corrected to Korean before training.',
 'Optional multiview diagnostic script had missing space in conditional expression; syntax corrected before script ran.',
 'Original historic open-hand label wrongly covered both hands at WORD0071p13f82. Frozen files preserved; source-audited symmetric recalculation excludes ambiguous little and corrects middle/ring. This does not make candidate eligible.'
 ],actor='Codex implementation and separate checks, no subagent/expert review this round'))
