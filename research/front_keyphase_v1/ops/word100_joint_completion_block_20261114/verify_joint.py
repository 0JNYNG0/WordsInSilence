"""Separate saved-result verification. No detector/old-model inference or tuning."""
from pathlib import Path
import json,hashlib,collections,copy
import numpy as np,torch
from region_extent import features
from joint_core import joint_sequence
from feedback_draft import draft
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_action_evidence_block_20261114';G=O/'word100_web_completion_graph_20261111';V=U/'extent_augmented';S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def ck(name,ok,case=''):checks.append(dict(name=name,passed=bool(ok),case=case))
for n in ['FROZEN.json']:
 for p,h in read(V/n)['pins'].items():ck('training_pin',sha(Path(p))==h,p)
for p,h in read(U/'ASSEMBLY_FROZEN_V2.json')['pins'].items():ck('assembly_pin',sha(Path(p))==h,p)
protected=read(G/'PROTECTIONS.json')
for p,h in protected.items():ck('protected_original',sha(Path(p))==h,p)
labels=read(V/'LABELS.json')['rows'];rr=read(V/'EVALUATION.json')['rows'];old_labels=read(U/'EXTENT_LABELS.json')['rows']
ck('original_probe_labels_unchanged',[(r['clip_id'],r['frame'],r['hand'],r['label'])for r in labels if r['split']=='exposed_probe']==[(r['clip_id'],r['frame'],r['hand'],r['label'])for r in old_labels if r['split']=='exposed_probe'])
ck('added_only_fit',all(r['split']=='fit'and 2<=int(r['clip_id'].split('REAL')[1][:2])<=7 for r in read(V/'LABELS.json')['added']))
for r in read(V/'TRAINED.json')['rows']:
 ck('model_hash',sha(Path(r['path']))==r['sha256'],str(r['seed']))
 ck('fit_indices_only',all(labels[i]['split']=='fit'and labels[i]['target']>=0 for i in r['fit_indices']),str(r['seed']))
 state=torch.load(r['path'],map_location='cpu',weights_only=True);x=np.array([r['feature']for r in labels],dtype=np.float32)
 logits=x@state['net.weight'].numpy().T+state['net.bias'].numpy();logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
 ck('independent_linear_probability_arithmetic',np.max(abs(p-np.load(V/f'{r["seed"]}_prob.npy')))<1e-6,str(r['seed']))
probe=[r for r in rr if r['split']=='exposed_probe'and r['truth']is not None]
ck('region_fixed_probe_all_known',len(probe)==27 and all(r['correct']for r in probe))
ck('failed_candidates_not_admitted',not read(U/'BODY_EVALUATION.json')['preliminary_pass']and not read(U/'EXTENT_EVALUATION.json')['preliminary_pass'])
base=read(B/'ASSEMBLED_EVIDENCE.json');final=read(U/'JOINT_FINAL_V2.json');mapping=read(U/'JOINT_COMPARISON.json')['provenance']
ck('all_saved_scenarios_present',len(final['rows'])==246 and len(mapping)==82)
normal_order=[];combined=[];missing=[];normal_joint=[]
for before,after in zip(base['rows'],final['rows']):
 key=f"{after['seed']}/{after['clip_id']}/{after['kind']}"
 for k in before:ck('base_field_preserved',before[k]==after[k],key+'/'+k)
 ck('all_original_cards_retained',len(after['joint_cards'])==len(before['original_candidates']),key)
 ck('no_whole_sign_completion',after['all_required_actions_complete']is None,key)
 for c in after['joint_cards']:
  ck('no_web_or_unqualified_correct',not c['show_as_correct_sign']and not c['web_enabled']and c['whole_action_complete']is None,key+'/'+c['card'])
  old=next((q for q in before['linked_observations']if q['card']==c['card']),None)
  ck('joint_observations_preserved',c['observations']==(old['observations']if old else[]),key+'/'+c['card'])
  ck('all_attributable_differences_preserved',c['differences']==(old['attributable_differences']if old else[]),key+'/'+c['card'])
  if c['body_relation']:
   e=c['body_relation'];a,b=e['span'];native=next(q for q in mapping if q['clip_id']==after['clip_id']and q['kind']==after['kind'])
   ck('original_source_frames_in_span',e['source_frames']==native['source_frames'][a:b+1],key+'/'+c['card'])
   ck('monotonic_observed_time',all(y>x for x,y in zip(e['timestamps_us'],e['timestamps_us'][1:])),key+'/'+c['card'])
   if e['joint']:
    j=e['joint'];ck('witness_frame_time_match',j['witness_source_frames']==[j['source_frames'][i]for i in j['witness_indices']]and j['witness_times_us']==[j['timestamps_us'][i]for i in j['witness_indices']],key)
  if after['kind']in ['full_normal','speed075','speed125']:
   ck('normal_no_new_correction',not any(d['kind']in ['observed_difference','observed_order']for d in c['feedback_drafts']),key+'/'+c['card'])
   if c['body_relation']and c['body_relation']['joint']:normal_joint.append(c['body_relation']['joint']['observed_feature_sequence'])
  if after['kind']=='closed_hand_and_reversed':combined.append(dict(key=key,tasks=[d['task']for d in c['differences']]))
  if after['kind']=='all_hand_missing':
   missing.append(c);ck('missing_no_correction',not c['differences']and not any(d['kind']in ['observed_difference','observed_order']for d in c['feedback_drafts']),key)
  if c['state']in ['action_link_uncertain','component_link_uncertain','unimplemented_sequence_requirement']:
   ck('uncertain_not_coaching',not any(d['kind']in ['observed_difference','observed_order']for d in c['feedback_drafts']),key)
 if after['kind']in ['full_normal','speed075','speed125']:
  normal_order.extend(after['original_orders'])
ck('normal_order_pairs',len(normal_order)==81 and all(p['state']=='forward'for p in normal_order))
ck('combined_shape_and_direction_retained',len(combined)==9 and sum('shape'in c['tasks']for c in combined)==9 and sum('direction'in c['tasks']for c in combined)==6)
ck('normal_joint_positive_not_abstention_only',len(normal_joint)==27 and all(normal_joint))
positive=[];partial=[]
for r in final['controls']:
 j=r['joint'];kind=r['variant'];key=r['clip_id']+'/'+kind
 if kind in ['normal','short_goal_prefix','speed075','speed125']:positive.append(j['observed_feature_sequence']);ck('normal_short_speed_joint_positive',j['observed_feature_sequence'],key)
 if kind in ['held_start','goal_only','reverse','missing_tail','missing_all','reviewed_partial_prefix','middle_long_gap']:
  ck('partial_missing_pose_not_completion',not j['observed_feature_sequence'],key)
 if kind=='reviewed_partial_prefix':partial.append(j)
 if kind in ['missing_all','missing_tail','middle_long_gap']:ck('missing_reason_correct',j['status']=='observation_insufficient',key)
ck('five_reviewed_partial_prefixes',len(partial)==5)
# Physical-coordinate invariance checks: no source label/word/time features.
arr=np.load(S/'source_arrays.npz');faces=np.load(U/'face_arrays.npz');errors=[]
for cid in sorted({r['clip_id']for r in labels if r['split']=='fit'and r['clip_id']+'_face'in faces}):
 x=arr[cid+'_x'];face=faces[cid+'_face'];ok=faces[cid+'_face_ok']
 for hand in ['L','R']:
  a,av=features(x,face,ok,hand)
  changed=x.copy();p=changed[:,:102].reshape(-1,51,2);p[:]=p*.7+np.array([2.1,-1.4]);b,bv=features(changed,face,ok,hand)
  err=float(np.max(abs(a-b)));errors.append(err);ck('translation_scale_invariant',np.array_equal(av,bv)and err<2e-5,cid+hand)
  # Swap anatomical sides and reflect x, preserving each hand's own observation.
  changed=x.copy();p=changed[:,:102].reshape(-1,51,2);idx=np.r_[0,2,1,4,3,6,5,8,7,np.arange(30,51),np.arange(9,30)]
  p[:]=p[:,idx];p[:,:,0]*=-1;changed[:,186:237]=x[:,186+idx]
  f=face.copy();f[:,0]*=-1;f[:,2:4]=-face[:,3:1:-1]
  b,bv=features(changed,f,ok,'R'if hand=='L'else'L');ck('both_hands_same_function',np.array_equal(av,bv)and np.max(abs(a-b))<2e-5,cid+hand)
positive_template=next(r for r in final['controls']if r['variant']=='normal');j=positive_template['joint'];shape=positive_template['old_shape_evidence']
blocked=joint_sequence(shape,j['region_states'],j['available'],j['timestamps_us'],j['source_frames'],False)
ck('uncertain_action_never_completed',not blocked['observed_feature_sequence']and blocked['status']=='action_link_uncertain')
f=draft(dict(state='action_link_uncertain',differences=[],order_evidence=[dict(state='reverse',witness_spans=[])]),'안내')
ck('uncertain_order_has_no_correction',all(d['kind']=='uncertain'for d in f))
failed=[r for r in checks if not r['passed']]
gates=dict(component_preserved=not any(not c['passed']for c in checks if c['name']in ['base_field_preserved','joint_observations_preserved','normal_no_new_correction','normal_order_pairs']),
 combined_errors=not any(not c['passed']for c in checks if c['name']in ['combined_shape_and_direction_retained','all_attributable_differences_preserved']),
 missing_not_pass=not any(not c['passed']for c in checks if c['name']in ['missing_no_correction','partial_missing_pose_not_completion','missing_reason_correct','uncertain_not_coaching']),
 original_time_evidence=not any(not c['passed']for c in checks if c['name']in ['original_source_frames_in_span','monotonic_observed_time','witness_frame_time_match']))
result=dict(passed=not failed,g3_checks=gates,g3_development_pass=not failed and all(gates.values()),checks=checks,failures=failed,
 counts=dict(protected_files=len(protected),probe_known_region=len(probe),probe_region_correct=sum(r['correct']for r in probe),joint_rows=246,source_scenarios=82,normal_speed_order_pairs=len(normal_order),normal_joint_positive=sum(normal_joint),normal_joint_n=len(normal_joint),controls=len(final['controls']),positive_control_n=len(positive),partial_prefixes=len(partial),combined_shape=9,combined_direction=6,combined_total=9),
 scope='G3 evidence integration development gate only. Not all 4 signs complete, no fresh heldout accuracy, real learner-error accuracy or release.',
 reviewer='Separate saved-output/geometry audit by same Codex; not independent KSL human review',
 no_old_neural_inference=True,final8_opened=False,web_changed=False,private_access=False,
 source_pins={str(p):sha(p)for p in[U/'JOINT_FINAL_V2.json',U/'JOINT_COMPARISON.json',V/'EVALUATION.json',Path(__file__)]})
with(U/'VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(passed=not failed,checks=len(checks),counts=result['counts'],failures=failed[:12]),ensure_ascii=False))
