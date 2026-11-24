"""Independent saved-array arithmetic audit. Never imports a model or performs inference."""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
U = HERE.parent
O = U.parent
R = O / 'word100_candidate_role_learning_20261111'
D = O / 'word100_action_feedback_demo_20261108'
S = O / 'word100_shared_motion_learning_20261109'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
new = read(U/'INTERVAL_PRESERVATION_RESULTS.json')
old = {(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'JOINT_ORDER_RESULTS.json')}
old_selected = {(r['clip_id'],r['kind']):r for r in read(U/'UNORDERED_SELECTION_RESULTS.json')}
inputs = {(r['clip_id'],r['kind']):r for r in read(R/'EVALUATION_INPUTS.json')['rows']}
roles = {(r['seed'],r['clip_id'],r['kind']):r for r in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']}
cards = read(D/'CARDS.json')['words']
families = read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
phase_cfg = read(D/'FROZEN.json')['phase']
ready = {r['clip_id']:r for r in read(S/'READY.json')['rows']}
link = {r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
audit = read(HERE/'sequence_truth_audit.json')
reverse_anchors = {r['clip_id']:r for r in audit['clips']}
raw_cache = {}; base_cache = {}; source_cache = {}
errors = []; arithmetic_rows=[]; normal_checks=[]; pair_changes=[]; role_paths=[]

def array_input(key):
    if key not in raw_cache:
        m=inputs[key]; assert sha(m['path'])==m['sha256']
        with np.load(m['path'],allow_pickle=False) as z: raw_cache[key]={k:z[k].copy() for k in z.files}
    return raw_cache[key]

def baseline(key):
    if key not in base_cache:
        m=inputs[key]; assert sha(m['baseline'])==m['baseline_sha256']; base_cache[key]=read(m['baseline'])
    return base_cache[key]

def source(cid):
    if cid not in source_cache:
        m=ready[cid]; assert sha(m['path'])==m['sha256']
        with np.load(m['path'],allow_pickle=False) as z: source_cache[cid]={k:z[k].copy() for k in z.files}
    return source_cache[cid]

def independent_runs(mask,times):
    runs=[]; current=[]
    delta=float(np.median(np.diff(times))) if len(times)>1 else 33333.
    for i,flag in enumerate(mask):
        if flag: current.append(i)
        if current and (not flag or i==len(mask)-1):
            a,b=current[0],current[-1]
            if (float(times[b])-float(times[a])+delta)/1000>=phase_cfg['min_evidence_ms']: runs.append([a,b])
            current=[]
    return runs

def inside(f,span): return span is not None and span[0]<=f<=span[1]
stage_context={0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:6,8:7,9:7,10:8,12:9}
context_family={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}
def normal_anchors(cid):
    z=source(cid); result={}
    def add(frame,context,origin):
        key=(int(frame),int(context)); result.setdefault(key,[]).append(origin)
    for i,s in enumerate(z['stage_gold']):
        s=int(s)
        if s in stage_context: add(z['frame_id'][i],stage_context[s],'direct_sparse_stage')
    for a in link.get(cid,{}).get('anchors',[]): add(a['frame'],a['label'],'LINK_LABELS_RGB_anchor')
    for p in ready[cid]['orders']:
        for suffix in ['a','b']:
            g=p['guide_'+suffix]
            if g in stage_context: add(p['frame_'+suffix],stage_context[g],'direct_order_endpoint')
    return [dict(frame=f,context=c,family=context_family[c],provenance=orig) for (f,c),orig in sorted(result.items())]

def inspect_span(anchors,span,wanted_contexts):
    selected=[a for a in anchors if inside(a['frame'],span)]
    return dict(span=span,matched=[a for a in selected if a['context'] in wanted_contexts],
                conflicting=[a for a in selected if a['context'] not in wanted_contexts])


first_joint = {(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'JOINT_CORRESPONDENCE_RESULTS.json')}
child_checks=[]; multi_candidates=[]; union_gap_checks=[]
def all_runs(flags):
    out=[];start=None
    for i,v in enumerate(flags):
        if v and start is None:start=i
        if start is not None and (not v or i==len(flags)-1):
            out.append([start,i if v else i-1]);start=None
    return out
for r in new:
    key=(r['clip_id'],r['kind']);triple=(r['seed'],*key);b=baseline(key);z=array_input(key)
    m=roles[triple];assert sha(m['result'])==m['sha256'];rr=read(m['result']);role_paths.append(m['result'])
    phase=np.asarray(b['phase_probability']);role=np.asarray(rr['role_probabilities']);hand=np.asarray(b['observed_hand_mask']);ts=z['ts'];dt=float(np.median(np.diff(ts))) if len(ts)>1 else 0.
    word=r['clip_id'].split('_')[2];rcards={c['id']:c for c in cards[word]};fspec={c['id']:c for c in families[word]}
    for c in r['cards']:
        spec=rcards[c['id']];f=fspec[c['id']]['families'][0];valid=hand[:,[0 if h=='L' else 1 for h in spec['hands']]].all(axis=1)
        chosen_spans=[]
        for group in c['groups']:
            pp=phase[:,[k+1 for k in group['group']]].sum(axis=1);joint=np.sqrt(np.clip(pp,0,1)*np.clip(role[:,f],0,1))
            raw=all_runs((pp>=phase_cfg['threshold'])&valid);calc=[]
            for a,bb in raw:
                duration=float((ts[bb]-ts[a]+dt)/1000);eligible=(bb>a) if c['parent'] else duration>=phase_cfg['min_evidence_ms']
                calc.append(dict(span=[a,bb],score=float(max(joint[a:bb+1])),duration_ms=duration,eligible=bool(eligible),anchor_indices=[i for i in range(a,bb+1) if joint[i]>=phase_cfg['threshold']]))
            if len(calc)!=len(group['candidates']): errors.append(dict(key=triple,card=c['id'],error='candidate_count'))
            else:
                for a,bb in zip(calc,group['candidates']):
                    if a['span']!=bb['span'] or a['eligible']!=bb['eligible'] or a['anchor_indices']!=bb['anchor_indices'] or abs(a['score']-bb['score'])>1e-12 or abs(a['duration_ms']-bb['duration_ms'])>1e-7:errors.append(dict(key=triple,card=c['id'],error='candidate_arithmetic',computed=a,saved=bb))
            acceptable=[q for q in calc if q['eligible'] and q['score']>=phase_cfg['threshold']]
            selected=acceptable[[q['score'] for q in acceptable].index(max(q['score'] for q in acceptable))] if acceptable else None
            span=selected['span'] if selected else None;anchor=selected['anchor_indices'] if selected else []
            if span!=group['span'] or anchor!=group['anchors']:errors.append(dict(key=triple,card=c['id'],error='selected_span_or_anchor'))
            if span is not None: chosen_spans.append(span)
            if len(acceptable)>1:multi_candidates.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],group=group['group'],selected=span,supported_candidates=acceptable))
        envelope=[min(a for a,b in chosen_spans),max(b for a,b in chosen_spans)] if len(chosen_spans)==len(c['groups']) else None
        observed=sorted(set(i for a,b in chosen_spans for i in range(a,b+1)))
        if c['evidence_intervals']!=chosen_spans or c['span']!=envelope or c['observed_indices']!=observed:errors.append(dict(key=triple,card=c['id'],error='union_or_envelope'))
        if envelope and len(observed)<envelope[1]-envelope[0]+1:union_gap_checks.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],span=envelope,observed_count=len(observed),envelope_count=envelope[1]-envelope[0]+1,unobserved_gap_excluded=True))
        if c['semantic_completion'] is not None: errors.append(dict(key=triple,card=c['id'],error='semantic_completion_not_null'))
        if c['parent']:
            parent=next(k for k in r['cards'] if k['id']==c['parent']);parent_idx=set(parent['observed_indices'])
            child_checks.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],span=c['span'],parent=c['parent'],parent_span=parent['span'],selected=c['span'] is not None,all_child_indices_within_parent_support=bool(c['observed_indices']) and set(c['observed_indices'])<=parent_idx,not_required=not c['required']))
    for p in r['pairs']:
        a,bb=p['spans'];prob=p['order_probability'];verdict='association_unknown'
        if a is not None and bb is not None:verdict='supported' if a[1]<bb[0] and prob>=.65 else 'reversed' if bb[1]<a[0] and prob<=.35 else 'ambiguous'
        if verdict!=p['state']:errors.append(dict(key=triple,error='pair_state'))
    arithmetic_rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],cards=len(r['cards']),pairs=len(r['pairs'])))
    if r['kind']!='full_normal': continue
    a=normal_anchors(r['clip_id']); src=source(r['clip_id'])
    assert list(src['frame_id'])==b['frame_id'],r['clip_id']
    frame_to_i={int(f):i for i,f in enumerate(src['frame_id'])}
    for c in r['cards']:
        oc=next(k for k in old_selected[key]['cards'] if k['id']==c['id']); f=fspec[c['id']]
        before=inspect_span(a,oc['span'],f['contexts']); after=inspect_span(a,c['span'],f['contexts'])
        shape=[]
        for feature,expected in rcards[c['id']].get('features',{}).items():
            fi=int(feature); known=[]
            for source_frame,i in frame_to_i.items():
                if src['direct_ym'][i,fi]:known.append(dict(frame=source_frame,label=float(src['y'][i,fi]),expected=expected))
            shape.append(dict(feature=fi,expected=expected,before=[v for v in known if inside(v['frame'],oc['span'])],after=[v for v in known if inside(v['frame'],c['span'])]))
        direction=[]
        if f['axis'] is not None:
            for p in ready[r['clip_id']]['pairs']:
                if p['axis']==f['axis']:
                    direction.append(dict(**p,before_both_endpoints_inside=inside(p['frame_a'],oc['span']) and inside(p['frame_b'],oc['span']),after_both_endpoints_inside=inside(p['frame_a'],c['span']) and inside(p['frame_b'],c['span'])))
        normal_checks.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['id'],required=f['required'],parent=f.get('parent'),
            before=before,after=after,lost_direct_context_anchors=[x for x in before['matched'] if x not in after['matched']],
            gained_direct_context_anchors=[x for x in after['matched'] if x not in before['matched']],
            shape_label_coverage=shape,direction_endpoint_coverage=direction,span_changed=oc['span']!=c['span']))

summaries={}
normal_summary={}
for seed in [42,43,44]:
    summaries[str(seed)]={k:dict(Counter(p['state'] for r in new if r['seed']==seed and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in new})}
    rows=[r for r in normal_checks if r['seed']==seed]
    normal_summary[str(seed)]=dict(cards=len(rows),span_changed=sum(r['span_changed'] for r in rows),
      old_null=sum(r['before']['span'] is None for r in rows),new_null=sum(r['after']['span'] is None for r in rows),
      old_context_anchor_covered=sum(bool(r['before']['matched']) for r in rows),new_context_anchor_covered=sum(bool(r['after']['matched']) for r in rows),
      cards_losing_any_context_anchor=sum(bool(r['lost_direct_context_anchors']) for r in rows),
      old_conflicting_anchor_cards=sum(bool(r['before']['conflicting']) for r in rows),new_conflicting_anchor_cards=sum(bool(r['after']['conflicting']) for r in rows),
      shape_cards_losing_any_direct_label=sum(any(any(x not in s['after'] for x in s['before']) for s in r['shape_label_coverage']) for r in rows),
      newly_lost_direction_pairs=sum(d['before_both_endpoints_inside'] and not d['after_both_endpoints_inside'] for r in rows for d in r['direction_endpoint_coverage']))

conflicts=[]
for cid in ['NIA_SL_WORD0739_REAL12_F','NIA_SL_WORD0739_REAL13_F']:
    ar=reverse_anchors[cid]
    for seed in [42,43,44]:
        current=next(r for r in new if r['seed']==seed and r['clip_id']==cid and r['kind']=='reversed_core_auto_order')
        previous=old[seed,cid,'reversed_core_auto_order']
        detail=[]
        for card in ['close_chest','raise_forearm']:
            f=next(k for k in families['WORD0739'] if k['id']==card)['families'][0]
            c=next(k for k in current['cards'] if k['id']==card)
            op=next(p for p in previous['pairs'] if p['cards']==['close_chest','raise_forearm']); os=op['spans'][op['cards'].index(card)]
            def check(span):
                selected=[x for x in ar['reviewed_source_anchors'] if inside(x['query_index'],span)]
                return dict(span=span,matched=[x for x in selected if x['source_reviewed_family']==f],conflicting=[x for x in selected if x['source_reviewed_family']!=f])
            detail.append(dict(card=card,before=check(os),after=check(c['span']),all_new_runs=c['groups']))
        conflicts.append(dict(seed=seed,clip_id=cid,before_pair=next(p for p in previous['pairs'] if p['cards']==['close_chest','raise_forearm']),
             after_pair=next(p for p in current['pairs'] if p['cards']==['close_chest','raise_forearm']),spans=detail))

speed_details=[]
for seed in [42,43,44]:
    for kind in ['full_normal','speed125']:
        cid='NIA_SL_WORD1133_REAL11_F'; key=(cid,kind);rr=read(roles[seed,cid,kind]['result']);b=baseline(key);zz=array_input(key)
        pp=np.asarray(b['phase_probability'])[:,5];role=np.asarray(rr['role_probabilities'])[:,4]
        valid=np.asarray(b['observed_hand_mask']).all(axis=1);joint=np.sqrt(pp*role)
        short=all_runs((joint>=phase_cfg['threshold'])&valid);tt=zz['ts'];dt=float(np.median(np.diff(tt)))
        row=next(r for r in new if r['seed']==seed and r['clip_id']==cid and r['kind']==kind)
        child=next(c for c in row['cards'] if c['id']=='reach_wrist')
        speed_details.append(dict(seed=seed,kind=kind,frames=len(tt),median_dt_us=dt,
           joint_threshold_runs=[dict(span=[a,b],duration_ms=float((tt[b]-tt[a]+dt)/1000)) for a,b in short],
           previous_intersection_span=next(c['span'] for c in first_joint[seed,cid,kind]['cards'] if c['id']=='reach_wrist'),
           preserved_phase_span=child['span'],joint_anchor_indices=child['groups'][0]['anchors']))

unknown_summary={}
for seed in [42,43,44]:
    unknown_summary[str(seed)]={}
    for kind in sorted({r['kind'] for r in new}):
        selected=[r for r in new if r['seed']==seed and r['kind']==kind]
        before=[p for r in selected for p in old[seed,r['clip_id'],kind]['pairs']]
        after=[p for r in selected for p in r['pairs']]
        oldcards=[c for r in selected for c in old_selected[r['clip_id'],kind]['cards']]
        jointcards=[c for r in selected for c in first_joint[seed,r['clip_id'],kind]['cards']]
        aftercards=[c for r in selected for c in r['cards']]
        unknown_summary[str(seed)][kind]=dict(old_order_unknown=sum(p['state'] in ['association_unknown','ambiguous'] for p in before),new_order_unknown=sum(p['state'] in ['association_unknown','ambiguous'] for p in after),
            old_null_cards=sum(c['span'] is None for c in oldcards),joint_intersection_null_cards=sum(c['span'] is None for c in jointcards),new_null_cards=sum(c['span'] is None for c in aftercards))

frozen=read(U/'INTERVAL_PRESERVATION_FROZEN.json');code_hash_match={k:sha(U/k)==v for k,v in frozen['code_sha256'].items()}
normal_child=[c for c in child_checks if c['kind']=='full_normal']
result=dict(status='SAVED_INTERVAL_PRESERVATION_VERIFIED_NOT_FULL_FEEDBACK_PASS',
 reviewer='Separate Codex saved-array audit; no model execution or independent human truth',
 numeric_verification=dict(rows=len(new),unique_inputs=len(inputs),card_checks=sum(x['cards'] for x in arithmetic_rows),pair_checks=sum(x['pairs'] for x in arithmetic_rows),
 error_count=len(errors),errors=errors,source_and_saved_role_hashes_verified=True,implementation_hashes_match_freeze=code_hash_match),
 normal_anchor_summary=normal_summary,normal_anchor_details=normal_checks,
 saved_summary_recomputed=summaries,unknown_comparison=unknown_summary,
 prior_two_selected_span_conflicts=conflicts,fast_arrival_details=speed_details,
 multi_supported_candidate_groups=multi_candidates,disjoint_support_gap_checks=union_gap_checks,
 child_parent_link_checks=child_checks,
 child_parent_summary=dict(normal_selected_children=len(normal_child),normal_children_outside_parent_selected_support=sum(c['selected'] and not c['all_child_indices_within_parent_support'] for c in normal_child),
 interpretation='Parent metadata avoids duplicate required order count, but there is no implemented physical parent-occurrence containment/association. Different parent movement and arrival support spans can be legitimate; the current code does not yet establish that they belong to the same physical action.'),
 correction_of_prior_review=dict(file=str(HERE/'joint_correspondence_verification.json'),
 erroneous_claim='speed125 makes the same two frames shorter than60ms (spoken example66.7->53.3ms).',
 corrected='speed125 is a resampled/reindexed input at median33333us, not merely a timestamp scale. Joint threshold support is one local frame58 (~33.333ms); normal has local72..73 (~66.666ms). Existing60ms floor removed the former. Whole phase support58..61 is preserved in the new candidate.',
 first_report_preserved=True,scope='This corrects the duration mechanism and corresponding narrative, not the directly verified fact that the child was null under intersection and is recovered under preservation.'),
 findings=[
  'Selected support is exactly an original phase/validity run; role evidence only selects/rejects it. All candidate boundaries and stored support unions were independently checked.',
  'The new code keeps each candidate, including ineligible short runs, in candidates metadata. Main short runs remain ineligible under60ms; endpoint children require at least2 phase-observed frames. A one-frame arrival remains unavailable for selection, although not erased from metadata.',
  'New method restores the previously cropped Egypt13 spread interval and its direct frame anchors. Keeping the interval is evidence coverage restoration, not proof that the aperture change or whole sign is correct.',
  'Both prior obedience12/13 wrong-source-role raise_forearm assignments are now rejected rather than falsely assigned. Their close->raise relation becomes association_unknown; this is repaired abstention, not recovered complete action.',
  'In obedience12 the joint-only method selected unedited tail98..99 even though phase itself was below its candidate threshold there; interval preservation does not invent that extra tail run.',
  'Multiple supported phase occurrences are retained but the largest joint peak still selects one. Order verdicts do not inspect competing/repeated/interleaved episodes. No full sequence correctness gate follows from this.',
  'Parent arrival children are flagged non-required and excluded from standalone order, but not geometrically/temporally bound to a specific parent episode. The wrapper only passes endpoint_child=True; it never receives the parent occurrence bounds.',
  'Disjoint group gaps are excluded from observed_indices. Consumers must use those indices/support intervals for feature scoring and rendering; the envelope span still exists and could wrongly reintroduce gaps if used alone.',
  'Full feedback shape/direction/coverage/render outputs were not recomputed by evaluate_interval_preservation.py. Anchor/direction-endpoint preservation cannot substitute for that missing regression.',
  'No word-specific thresholds or conditional exceptions are introduced. Word lookup selects the declared descriptors; coverage beyond the tested guide set is unproven.',
  'Phase baseline remains fixed existing42; role/order remain seed-specific auxiliary models. Combining their scores is not independent evidence.',
  'Partial-core time reversal remains a compound, partially preserved source challenge; no blanket9-pair semantic error accuracy is reported.'
 ],
 design_gaps=[
  'Single-peak rank does not implement competitor-aware ambiguity or occurrence count.',
  'Child-parent physical association is still metadata only.',
  'One-observed-frame child is retained in candidates but cannot be selected.',
  'Preserved phase support can still mix preparation/return or omit true movement start.',
  'Only stored order predictions and input arithmetic audited; no new model inference or full-path feature result.'
 ],
 code_evidence=[dict(file=str(U/'occurrence_binding.py'),function='select_occurrence',fact='Candidate generated by phase/valid; role max gates selection; endpointchild is b>a, no parent argument.'),
 dict(file=str(U/'occurrence_binding.py'),function='union_indices',fact='Returns selected subinterval union without gaps.'),
 dict(file=str(U/'evaluate_interval_preservation.py'),function='main',fact='Child flag derives from spec parent; required pairs exclude parent children; descriptor midpoint order only.')],
 model_inference=False,training=False,private_access=False,web_changed=False,threshold_tuning=False,
 source_hashes={str(p):sha(p) for p in [U/'occurrence_binding.py',U/'evaluate_interval_preservation.py',U/'INTERVAL_PRESERVATION_FROZEN.json',U/'INTERVAL_PRESERVATION_RESULTS.json',U/'INTERVAL_PRESERVATION_COMPLETE.json',U/'JOINT_CORRESPONDENCE_RESULTS.json',HERE/'joint_correspondence_verification.json',HERE/'sequence_truth_audit.json',HERE/'occurrence_selection_design.json',R/'EVALUATION_INPUTS.json',R/'FROZEN.json',S/'READY.json',O/'word100_event_guide_linkage_20261110/LINK_LABELS.json']} )
assert not errors and all(code_hash_match.values())
out=HERE/'interval_preservation_verification.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),numeric=result['numeric_verification'],normal=normal_summary,child_parent=result['child_parent_summary'],fast=speed_details,competing_groups=len(multi_candidates),gaps=len(union_gap_checks)),ensure_ascii=True))
