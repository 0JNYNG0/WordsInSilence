"""Stored result/annotation arithmetic only; no model import or inference."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,numpy as np
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(U/'BOUND_COMPONENTS_RESULTS.json');frozen=read(U/'BOUND_COMPONENTS_FROZEN.json')
selector={(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'INTERVAL_PRESERVATION_RESULTS.json')}
R=O/'word100_candidate_role_learning_20261111'
saved={(r['seed'],r['clip_id'],r['kind']):r for r in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']}
inputs={(r['clip_id'],r['kind']):r for r in read(R/'EVALUATION_INPUTS.json')['rows']}
source_rows={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
attempts={(r['clip_id'],r['kind']):r for r in read(O/'word100_attempt_event_supervision_20261109/INPUTS.json')['rows']}
families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
definitions=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words']
cfg=read(O/'word100_positive_direction_20261109/FROZEN.json')['positive']
cache={};source_index_cache={};old_cache={};checks=Counter();errors=[];states=[];exception_details=[];seams=[]
def inp(key):
 if key not in cache:
  m=inputs[key];assert sha(m['path'])==m['sha256']
  with np.load(m['path'],allow_pickle=False) as z:cache[key]={k:z[k].copy() for k in z.files}
 return cache[key]
def idx(key):
 if key not in attempts:return None
 if key not in source_index_cache:
  m=attempts[key];assert sha(m['path'])==m['sha256']
  with np.load(m['path'],allow_pickle=False) as z:source_index_cache[key]=z['source_indices'].copy()
 return source_index_cache[key]
def old(key):
 if key not in old_cache:
  m=saved[key];assert sha(m['result'])==m['sha256'];old_cache[key]=read(m['result'])
 return old_cache[key]
def distinct(ss):return sorted(set((int(a),str(b)) for a,b in ss))
def slim_motion(m):
 if m is None:return None
 keys=['kind','state','span','signed_change','start_progress','end_progress','trend','deadband','learned_class','learned_confidence','learned_eligible','learned_model']
 return {k:m[k] for k in keys if k in m}
def slim_direction(d):
 if d is None:return None
 return dict(state=d.get('state'),expected=d.get('expected'),observed_signs=d.get('observed_signs'),
  evidence=[{k:e.get(k) for k in ['span','axis','prediction','confidence','eligible','complete']} for e in d.get('evidence',[])])
def inside(f,span):return span is not None and span[0]<=f<=span[1]
for r in rows:
 key=(r['clip_id'],r['kind']);triple=(r['seed'],*key);b=old(triple);z=inp(key);source_index=idx(key);word=r['clip_id'].split('_')[2]
 if source_index is not None:assert len(source_index)==len(z['x'])
 fs={s['id']:s for s in families[word]};ds={s['id']:s for s in definitions[word]}
 for c in r['cards']:
  sc=next(k for k in selector[triple]['cards'] if k['id']==c['id']);bc=next(k for k in b['cards'] if k['id']==c['id']);spec=fs[c['id']];desc=ds[c['id']]
  checks['cards']+=1
  if c['span']!=sc['span'] or c['evidence_intervals']!=sc['evidence_intervals']:errors.append([triple,c['id'],'selected_interval_mismatch'])
  if c['action_completed'] is not None or c['whole_sign_correct'] is not None:errors.append([triple,c['id'],'completion_claim'])
  shape_n=len(desc['features']);expected_n=shape_n*len(c['evidence_intervals'])
  if len(c['shape'])!=expected_n:errors.append([triple,c['id'],'shape_count'])
  for i,s in enumerate(c['shape']):
   checks['shape_states']+=1;span=c['evidence_intervals'][i//shape_n];a,bb=span
   if desc.get('shape_scope')=='late':a+=max(0,(bb-a)*2//3)
   j=s['feature'];base=9 if j in [2,4] else 30
   points=[base]+[base+k for k in range(5,21)]
   available=z['x'][:,186:237].astype(bool)[:,points].all(axis=1)
   ratio=float(available[a:bb+1].mean())
   if abs(ratio-s['available_fraction'])>1e-9:errors.append([triple,c['id'],j,'availability'])
   sr=s['supported_spans'];br=s['contradicted_spans'];nr=s['neural_contradicted_spans'];mean=s['mean_expected_probability']
   state='unknown'
   if ratio>=cfg['min_available_fraction']:
    if br:state='contradicted'
    elif sr and mean is not None and mean>=cfg['mean_support_threshold'] and not s['conflicting_predictions'] and not nr:state='supported'
   if state!=s['state']:errors.append([triple,c['id'],j,'shape_state'])
   for interval in sr+br+nr:
    if not(a<=interval[0]<=interval[1]<=bb):errors.append([triple,c['id'],j,'shape_support_outside_selected'])
  signs=set()
  for m in c['movement']:
   checks['movement_records']+=1
   if m['span'] not in c['evidence_intervals']:errors.append([triple,c['id'],'motion_outside_support'])
   if m['geometry'].get('contact_verified') is not False:errors.append([triple,c['id'],'physical_contact_claim'])
   g=m['geometry'];trajectory=g.get('trajectory',[])
   if trajectory and abs((trajectory[-1]-trajectory[0])-g['signed_change'])>1e-8:errors.append([triple,c['id'],'signed_change_arithmetic'])
   if m['eligible'] and m['prediction'] in [0,2] and np.sign(g['signed_change'])!=m['prediction']-1:errors.append([triple,c['id'],'geometry_sign_disagreement_accepted'])
   if m['eligible'] and m['confidence']>=.65 and m['prediction']!=1:signs.add(m['prediction']-1)
   if source_index is not None:
    a,bb=m['span'];seq=source_index[a:bb+1];jumps=[dict(query_index=a+i+1,source_before=int(seq[i]),source_after=int(seq[i+1])) for i in range(len(seq)-1) if abs(int(seq[i+1])-int(seq[i]))>2]
    if jumps:seams.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],movement_span=m['span'],movement_state=c['movement_state'],source_indices=seq.tolist(),jumps=jumps))
  wanted=spec['sign'];ms='not_required' if spec['axis'] is None else 'unknown'
  if spec['axis'] is not None and wanted is not None:ms='supported' if signs=={wanted} else 'opposite_direction' if signs=={-wanted} else 'mixed' if len(signs)>1 else 'movement_not_established'
  elif spec['axis'] is not None:ms='opposite_directions_observed' if signs=={-1,1} else 'insufficient_alternation_evidence'
  if ms!=c['movement_state']:errors.append([triple,c['id'],'movement_state_arithmetic'])
  source_spans=[]
  if source_index is not None:
   source_spans=[dict(query_span=[a,bb],source_frames=sorted(set(map(int,source_index[a:bb+1])))) for a,bb in c['evidence_intervals']]
  pairs=[]
  for p in source_rows[r['clip_id']]['pairs']:
   if p['axis']!=spec['axis']:continue
   per=[]
   for s in source_spans:
    frames=s['source_frames'];per.append(dict(query_span=s['query_span'],both_exact_source_endpoints_present=p['frame_a'] in frames and p['frame_b'] in frames,
      both_inside_source_range=bool(frames) and min(frames)<=p['frame_a']<=p['frame_b']<=max(frames)))
   pairs.append(dict(**p,comparison_scope='Original public RGB endpoint truth; not automatically copied semantic truth for edited/reversed motion',selected_support=per))
  distinct_old=distinct(c['before_shape_states']);distinct_new=distinct(c['after_shape_states'])
  rec=dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],span=c['span'],evidence_intervals=c['evidence_intervals'],
   before_shape_distinct=distinct_old,after_shape_distinct=distinct_new,list_changed=c['before_shape_states']!=c['after_shape_states'],distinct_state_changed=distinct_old!=distinct_new,
   movement_state=ms,observed_correspondence=c['observed_correspondence'],observed_signs=sorted(signs),
   before_motion_evidence=slim_motion(bc.get('motion_evidence')),before_guide_link_direction=slim_direction(bc.get('guide_link_direction')),
   before_display=bc.get('display_status'),before_status=bc.get('status'),before_card_span=bc.get('query_indices'),
   original_rgb_direction_pairs=pairs,source_support=source_spans,
   missing_correspondence_not_same_as_absent_motion=c['span'] is None)
  states.append(rec)
  if (r['kind'] in ['full_normal','speed075','speed125'] and ms in ['movement_not_established','insufficient_alternation_evidence']) or (r['kind']=='omit_final_event' and ms=='opposite_direction'):
   exception_details.append(dict(**rec,new_movement=c['movement'],interpretation_pending_expert_correctness=True))

summary={str(s):{k:dict(Counter(c['movement_state'] for r in rows if r['seed']==s and r['kind']==k for c in r['cards'])) for k in sorted({r['kind'] for r in rows})} for s in [42,43,44]}
shape_summary={str(s):{k:dict(cards=sum(x['seed']==s and x['kind']==k for x in states),list_changes=sum(x['seed']==s and x['kind']==k and x['list_changed'] for x in states),distinct_state_changes=sum(x['seed']==s and x['kind']==k and x['distinct_state_changed'] for x in states)) for k in sorted({r['kind'] for r in rows})} for s in [42,43,44]}
normal_duplicates=[r for r in states if r['kind']=='full_normal' and r['list_changed']]
assert len(normal_duplicates)==9 and all(not r['distinct_state_changed'] for r in normal_duplicates)
assert summary==read(U/'BOUND_COMPONENTS_COMPLETE.json')['summary']
assert sha(U/'INTERVAL_PRESERVATION_RESULTS.json')==frozen['selector_sha256']
assert sha(U/'evaluate_bound_components.py')==frozen['code_sha256']
assert not errors,errors

result=dict(status='OFFLINE_BOUND_COMPONENT_ARITHMETIC_VERIFIED_WITH_REMAINING_DIRECTION_GAPS_NOT_G2_PASS',
 reviewer='Separate Codex arithmetic/source-trace audit; no model rerun, no independent human labels',
 model_inference=False,training=False,private_access=False,web_changed=False,threshold_tuning=False,
 checks=dict(checks),errors=errors,shape_comparison=shape_summary,movement_summary=summary,
 normal_shape_list_changes=normal_duplicates,
 findings=[
  'The reported9normal_shape_changes are duplicate same-feature/same-state entries from the two ancestor intervals. There are0distinct handshape-state changes on normal, slow or fast in every seed. Do not call these9errors or9improvements.',
  'Primary normal has12signed-direction cards:11supported,1not established. These are4signed clauses across3public clips, not11/12whole signs correct. The3alternation cards expose both signs, not verified repeat count or full temporal pattern.',
  'The normal unresolved case is obedience REAL12 close_chest. Learned direction predicts down with high confidence but the selected interval geometry has opposite signed change, so the guard withholds it. Original RGB pair57->69 is down; selected60..73 misses its beginning. This identifies a window/observation-evidence limitation, not evidence that the person moved incorrectly.',
  'Its previous guide_link_direction=supported includes movement outside the card window, notably70..86; old motion_evidence for the card is null. That old broad linked cue and this strict selected-window cue are different tests. A supported->not-established label is not an apples-to-apples model accuracy regression.',
  'Slow well-done REAL12 brush_arm has an actual speed-sensitive verification gap: selected78..96 (source58..72) is not established. The original RGB motion69->79 extends past it. Old motion_evidence used77..117 (a wider movement+arrival range). Neither replacing a valid cue with silence nor borrowing unrelated motion should be hidden; physical parent movement/arrival linkage remains needed.',
  'Slow ancestor REAL11 loses the second accepted direction: geometric delta is negative but learned prediction is positive. Guard correctly rejects the contradictory sign, yet available normal alternation feedback is incomplete. Old guide_link_direction was not_specified, so it never provided the same full alternation test.',
  'Closed-hand, reversed-direction and their compound controls keep their distinct expected component cues: right closed shape contradiction, reversed brush direction contradiction, or both. They are synthetic coordinate controls; they do not establish accuracy on real learner errors.',
  'Omitted and masked final clauses mostly have no selected interval; shape support drops rather than inventing completion. Missing observation and absent performed action remain different: movement_not_established must render as unknown, never as a proved omission or a request to move more.',
  'Seed43 omission obedience12 opposite-direction cue crosses a synthetic cut seam, source72,73,74,117. It is an editing jump, not validated natural opposite movement; exclude it from semantic direction accuracy and retain as artifact sensitivity.',
  'All action_completed and whole_sign_correct outputs remain null. A supported direction or matching terminal shape is not proof the full action was completed.',
  'Only attributes0..4 are checked: index/four-finger open/closed. No thumb aperture, exact contact, full3D palm orientation, independent image understanding or face-location correctness is newly established.',
  'Shape and movement model probabilities were not regenerated in this audit. It verifies recorded state logic, masks, support bounds, geometry arithmetic, source-index traces and unchanged-code pins; not an independent neural forward reproduction.'
 ],
 normal_and_speed_exceptions=exception_details,synthetic_source_jump_diagnostics=seams,component_rows=states,
 next_gate_requirements=[
  'Keep normal/slow/fast feedback uncertainty visible and do not promote unchanged shape/order support to full correctness.',
  'Bind actual parent movement and arrival supports before judging a stroke from a prematurely ending phase alone; validate from FIT occurrence labels without DEV boundary tuning.',
  'Resolve source/motion-model disagreement and full occurrence evidence for slow alternating motion; do not flip geometry signs merely to recover a score.',
  'Keep synthetic cut seams and direction-defined reversed semantics outside a blanket error denominator.',
  'Maintain separate unknown reasons for observation, association and incomplete motion evidence in the eventual renderer; current aggregate movement label conflates them.',
  'G2 remains unpassed: multiple candidate competitors, physical child-parent binding and semantic completion contracts remain open.'
 ],
 sources={str(p):sha(p) for p in [U/'BOUND_COMPONENTS_RESULTS.json',U/'BOUND_COMPONENTS_COMPLETE.json',U/'BOUND_COMPONENTS_FROZEN.json',U/'evaluate_bound_components.py',U/'INTERVAL_PRESERVATION_RESULTS.json',R/'safety_reconcile/FULLPATH_COMPLETE.json',O/'word100_positive_direction_20261109/positive_evidence.py',O/'word100_short_motion_completion_20261111/motion_core.py',O/'word100_action_episode_supervision_20261110/integration.py',O/'word100_shared_motion_learning_20261109/READY.json',O/'word100_attempt_event_supervision_20261109/INPUTS.json',O/'word100_attempt_event_supervision_20261109/prepare.py']})
out=V/'bound_components_verification.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),checks=dict(checks),errors=len(errors),normal_duplicates=len(normal_duplicates),shape=shape_summary,exceptions=len(exception_details),seam_records=len(seams)),ensure_ascii=True))
