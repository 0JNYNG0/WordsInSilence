"""Preserve the first input-contract error; create a separate corrected experiment."""
from core import *
bad=W/'transfer_local_candidate';good=W/'transfer_local_timefix'
assert not good.exists();good.mkdir()
write(bad/'INPUT_SEMANTICS_CORRECTION.json',dict(at=stamp(),status='EXCLUDED_FROM_PRETRAINED_COMPARISON_CLAIMS',cause='Channel279 is inter-frame delta seconds, not normalized word progress. Candidate-local encoder incorrectly replaced it with linspace(0,1).',affected='Only new transfer_local_candidate encoder preparation/evaluation; C/T3-F/shared100/deployed web unchanged.',preserve_old_files=True,corrective_action='Separate transfer_local_timefix with actual internal delta seconds and first candidate delta zero. Reverse uses reversed interval durations.',evidence=[str(O/'word100_positive_direction_20261109/compare.py')],old_artifacts={str(p):sha(p) for p in [bad/'FROZEN.json',bad/'TRAINING_COMPLETE.json',bad/'evaluation/COMPARISON.json',W/'transfer_local_core.py']}))
src=(W/'transfer_local_core.py').read_text(encoding='utf-8')
src=src.replace("TQ=W/'transfer_local_candidate'","TQ=W/'transfer_local_timefix'")
old="""  px=pose[a:b+1].copy();qq=q[a:b+1].copy()
  if reverse:px=px[::-1].copy();qq=qq[::-1].copy()
  # Candidate-relative clock, never the word's original absolute progress.
  px[:,279]=np.linspace(0,1,len(px));seq.append(px);local.append(np.c_[qq,px[:,102:186],px[:,237:279]]);lengths.append(len(px))"""
new="""  px=pose[a:b+1].copy();qq=q[a:b+1].copy();intervals=px[1:,279].copy()
  if reverse:px=px[::-1].copy();qq=qq[::-1].copy();intervals=intervals[::-1].copy()
  # Channel279 is elapsed seconds since the preceding frame, NOT progress.
  px[:,279]=np.r_[0.,intervals]
  assert np.isfinite(intervals).all() and (intervals>=0).all()
  seq.append(px);local.append(np.c_[qq,px[:,102:186],px[:,237:279]]);lengths.append(len(px))"""
assert old in src;src=src.replace(old,new);(W/'transfer_local_timefix_core.py').write_text(src,encoding='utf-8')
for name in ['prepare_transfer_local','train_transfer_local','evaluate_transfer_local','check_transfer_components']:
 text=(W/(name+'.py')).read_text(encoding='utf-8').replace('from transfer_local_core import *','from transfer_local_timefix_core import *').replace("W/'transfer_local_core.py'","W/'transfer_local_timefix_core.py'")
 text=text.replace('candidate_relative_clock=True',"input_time_feature='actual inter-frame delta seconds; first zero; reverse intervals reversed'")
 dest=name.replace('transfer_local','transfer_timefix').replace('transfer_components','timefix_components')
 (W/(dest+'.py')).write_text(text,encoding='utf-8')
write(good/'RECOVERY_CONTRACT.json',dict(at=stamp(),reason='Correct verified feature-unit implementation error, not score-driven parameter tuning',old_experiment=str(bad),fixed_feature=279,normalization_changed=False,seeds_epochs_labels_candidates_unchanged=True,first_failure_preserved=True,private=False,final8=False))
print('Prepared separately versioned delta-time recovery')
