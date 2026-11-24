"""Separate saved-output/source verification. No training or neural forwards."""
from prepare import U,O,read,sha,write,Path
import numpy as np,collections,sys
G=O/'word100_web_completion_graph_20261111';I=O/'word100_goal_binding_integration_20261114'
sys.path.insert(0,str(I));from core_correspondence import core_order
checks=[]
def check(name,test):
 checks.append(dict(name=name,passed=bool(test)))
 if not test:raise AssertionError(name)
base=read(U/'FROZEN.json');ext=read(U/'EXTRACTED.json');auto=read(U/'AUTO_RESULTS_V2.json');ct=read(U/'CONTROL_COMPLETE.json');rev=read(U/'CONTINUOUS_COMPLETE.json');tr=read(U/'TRAIT_RESULTS_V2.json')
for p,h in read(G/'PROTECTIONS.json').items():check('protected:'+p,Path(p).exists()and sha(p)==h)
for name in ['FROZEN.json','AUTO_FROZEN.json','CONTROL_FROZEN.json','CONTINUOUS_FROZEN.json']:
 d=read(U/name)
 for p,h in d.get('model_pins',d.get('pins',{})).items():check('frozen:'+p,sha(p)==h)
for r in ext['rows']:
 check('source:'+r['clip_id'],sha(r['path'])==r['sha256']);check('raw:'+r['clip_id'],sha(r['raw_path'])==r['raw_sha256']);check('channels:'+r['clip_id'],sha(r['channel_path'])==r['channel_sha256'])
 z=np.load(r['raw_path']);c=np.load(r['channel_path']);check('native_frames:'+r['clip_id'],np.array_equal(z['frame_id'],np.arange(r['frames']))and np.all(np.diff(z['timestamp_us'])>0));check('finite_channels:'+r['clip_id'],np.isfinite(c['tcn280']).all())
 check('hand_schema:'+r['clip_id'],c['tcn280'].shape==(r['frames'],280)and z['xy_px'].shape==(r['frames'],133,2))
check('8unique public sources',len(ext['rows'])==8 and len({r['sha256']for r in ext['rows']})==8)
check('181active metadata no source exposure',not read(U/'EXPOSURE_AUDIT.json')['feedback_matches'])
check('private not used',base['no_private']and auto['training']==False and ct['training']==False and rev['training']==False)
spec=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];cases=auto['rows']+ct['rows']+rev['rows']
summary=[]
for seed in [42,43,44]:
 for mode in ['old_local_full','current_g3']:
  normal=[r for r in auto['rows']if r['seed']==seed and r['condition']==mode and r['kind']!='prefix_before_goal'];partial=[r for r in auto['rows']if r['seed']==seed and r['condition']==mode and r['kind']=='prefix_before_goal']
  missing=[r for r in ct['rows']if r['seed']==seed and r['condition']==mode and r['kind']=='all_hands_missing'];reverse=[r for r in rev['rows']if r['seed']==seed and r['condition']==mode]
  warnings=lambda r:any(c['differences']or any(p['state']=='reverse'for p in c['order_evidence'])for c in r['joint_cards'])
  pos=lambda r:any(c['state']in['measured_features_agree','measured_features_and_goal_agree']for c in r['joint_cards'])
  check(f'normal24_no_new_warning:{seed}:{mode}',len(normal)==24 and not any(map(warnings,normal)))
  check(f'normal18_orders:{seed}:{mode}',sum(len(r['original_orders'])for r in normal)==18 and all(p['state']=='forward'for r in normal for p in r['original_orders']))
  check(f'prefix2_no_positive:{seed}:{mode}',len(partial)==2 and not any(map(pos,partial)))
  check(f'missing8_no_positive_or_correction:{seed}:{mode}',len(missing)==8 and not any(pos(r)or warnings(r)for r in missing))
  # Count the actual errors/unknowns, do not silently assume all runs agree.
  check(f'continuous_reverse6_included:{seed}:{mode}',len(reverse)==6)
  summary.append(dict(seed=seed,condition=mode,normal_conditions=24,normal_warning_cases=sum(map(warnings,normal)),normal_forward_pairs=18,partial_not_completed=2,missing_not_positive=8,reverse_difference_cases=sum(map(warnings,reverse)),reverse_source_count=6))
for r in cases:
 check('no_whole_sign_claim',r['all_required_actions_complete']is None and all(c['show_as_correct_sign']==False and c['whole_action_complete']is None for c in r['joint_cards']))
 for c in r['joint_cards']:
  if c['body_relation']:
   e=c['body_relation'];a,b=e['span'];check('source_address',len(e['source_frames'])==len(e['timestamps_us'])==b-a+1 and all(y>x for x,y in zip(e['timestamps_us'],e['timestamps_us'][1:])))
  if c['state']=='action_link_uncertain':check('no_attributed_unbound_error',not c['differences'])
 # Every displayed pair has an explicit source witness and established guide orientation.
 fs=[f for f in spec[r['clip_id'].split('_')[2]]if f['required']];allowed={tuple([a['id'],b['id']])for a,b in zip(fs,fs[1:])if a['families']!=b['families']}
 check('expected_order_not_selector',all(tuple(p['cards'])in allowed and p['expected_order_used_for_selection']==False for p in r['original_orders']))
for p in(U/'continuous').glob('*.json'):
 r=read(p);m=np.array(r['source_mapping']);check('continuous_reverse_without_splice',np.all(np.diff(m)==-1))
point_summary={};dense=np.load(U/'TRAIT_DENSE.npz');compatible=[]
for key,dom in [('shape_old','shape'),('region_old','region'),('region_new','region')]:
 pp=[p for r in tr['rows']for p in r['points']if p[dom]is not None];cc=collections.Counter('correct'if p['predictions'][key]==p[dom]else'unknown'if p['predictions'][key]is None else'wrong'for p in pp);point_summary[key]=dict(total=len(pp),**cc)
for r in tr['rows']:
 for p in r['points']:
  if p['shape']is None:continue
  key=r['clip_id']+'__'+p['hand']+'__';q=dense[key+'positions'][:,p['frame']];votes=abs(q[:,None]-np.array([0,1,.5])[None]).argmin(1);state=['four_open','four_closed','curved'][votes[0]]if len(set(votes))==1 else None
  state=state if state==p['predictions']['shape_old']else None
  compatible.append('correct'if state==p['shape']else'unknown'if state is None else'wrong')
point_summary['compatible_shape']=dict(total=len(compatible),**collections.Counter(compatible))
check('new_region54correct',point_summary['region_new']==dict(total=54,correct=54))
check('conflicting_shape_not_corrected',point_summary['compatible_shape'].get('wrong',0)==0 and point_summary['compatible_shape']['correct']==51)
for r in tr['rows']:
 for c in r['controls']:
  expected=c['kind']in['normal','speed075','speed125'];check('specified_sequence_control',c['joint']['observed_feature_sequence']==expected)
guard=read(U/'CONFLICT_REGRESSION.json')
check('development82_conflict_regression_preserved',guard['summary']['development']['scenarios']==82 and not guard['summary']['development']['changed'])
check('normal_public_conflict_regression_preserved',not guard['summary']['public_normal']['changed'])
qualified=[r for r in guard['rows']['public_continuous']if r['condition']=='current_g3']
qcounts=collections.Counter('conflict'if any(c['state']=='evidence_conflict'for c in r['joint_cards'])else'difference'if any(c['state']=='attributable_feature_difference'for c in r['joint_cards'])else'other'for r in qualified)
check('all_conflict_disclosed_no_positive',len(qualified)==6 and all(c['state']not in['measured_features_agree','measured_features_and_goal_agree']for r in qualified for c in r['joint_cards']))
write(U/'VERIFIED.json',dict(passed=True,checks=checks,check_count=len(checks),protected_files=len(read(G/'PROTECTIONS.json')),source_frames=sum(r['frames']for r in ext['rows']),
 summary=summary,point_summary=point_summary,verification_method='Codex separate saved-output arithmetic and source/hash audit, no neural forwards; not independent human expert review',
 g4_scope='Frozen models evaluated on8 sources unused in current feedback development; prior TCN exposure disclosed. Supports feature-level feedback only; no real learner error accuracy or whole-sign correctness established.',
 g4_checks=dict(exposure_audit=True,frozen_final_test=True,verification=True),web_deployment_approved=False,
 raw_all_seeds_reverse_detection_pass=False,qualified_reverse_counts=dict(qcounts),
 release_scope='Limited evidence display only. G4 evaluation completed, not every movement judged reliably. Preserve one newly found contradictory reverse case as uncertain. The post-score conflict guard is regression-verified, not independently retested.',
 limitations=['8videos/2recordIDs, not a population accuracy estimate','Partial reverse splice diagnostic failed to isolate direction; full continuous reverse separately reported, no source dropped','Some direction details remain uncertain despite source-level difference recognition','Exact thumb/fingertip/contact/repetition/whole completion unsupported','Do not reuse this8 as fresh test after any model adjustment']))
print('VERIFIED',len(checks),'checks;',point_summary)
