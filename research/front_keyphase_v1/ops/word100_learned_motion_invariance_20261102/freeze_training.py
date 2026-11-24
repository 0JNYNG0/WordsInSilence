from prepare_real_fit import N,M,P,O,read,sha,exclusive
def main():
 # These conditions are based on physical-role separation and old failure causes,
 # before seeing any new DEV/locked-ID output. Input preparation still running.
 keep=list(range(0,11))+list(range(28,39))+list(range(56,73))
 exclusive(N/'TRAINING_CONTRACT.json',dict(conditions=['real_variants','real_variants_motion_inputs'],seeds=[42,43,44],epochs=120,lr=.0003,
  initialization='Corresponding fixed ordered_gru seed checkpoint in previous motion unit; same architecture',
  conditions_description={'real_variants':'Fine-tune existing motion head using actual pixel-speed variants and replay of original episodes',
   'real_variants_motion_inputs':'Same data; hide per-finger angle/size/palm-axis channels from the motion branch only. Retain location, arm, handshape and relation probability inputs.'},
  motion_value_indices=keep,shape_and_relation_branches_unchanged=True,
  normalization='Keep FIT-only previous normalization for controlled fine-tuning comparison',
  data='Only public IDs01-09. Native reviewed episodes plus exact source-frame mapping for actual original/0.8/1.25 speed videos.',
  training_controls=['reverse temporal feature sequence with opposite physical label','remove middle half with direction label masked and continuity0','freeze first or last observed feature to teach no directed movement'],
  controls_limit='Feature-level controls only in training; not natural learner errors, nor independent images/people. Actual-video evaluation remains separate.',
  losses={'masked_supervised_BCE':1.0,'paired_real_speed_known_label_probability_consistency':.1},
  unseen_semantics='No target word, guide, signer ID, absolute timestamp input to motion network',
  temporal_labels='Event-level direction; no per-frame semantic stage extrapolation. Unknown dimensions remain masked.',
  checkpoint='Final epoch only; no DEV early stop, best seed or threshold selection',
  evaluation='Keep original learnedDTW and feedback thresholds. Report all six fixed models, normal-speed false holds, actual cuts, blackout, reverse and preservation. Existing baseline results reused.',
  promotion='No production/default replacement. New research demo only if normal preservation and important missing-event handling both improve; otherwise preserve last usable8806 and address proven cause.',
  next_boundary_work='Only after same-path evaluation: determine whether event boundary or observed states need joint supervision. Do not silently widen each problematic DEV word window.',
  new_backbone_update=False,old_motion_source_sha256=sha(M/'motion_model.py'),motion_labels_sha256=sha(M/'MOTION_TARGETS.json')))
 print('Six fine-tuning conditions fixed before new DEV output')
if __name__=='__main__':main()
