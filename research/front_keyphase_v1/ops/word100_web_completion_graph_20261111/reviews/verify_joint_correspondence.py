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
new = read(U/'JOINT_CORRESPONDENCE_RESULTS.json')
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

for r in new:
    key=(r['clip_id'],r['kind']); triple=(r['seed'],*key); b=baseline(key); z=array_input(key)
    meta=roles[triple]; assert sha(meta['result'])==meta['sha256']; rr=read(meta['result'])
    role_paths.append(meta['result']); phase=np.asarray(b['phase_probability']); role=np.asarray(rr['role_probabilities'])
    observed=np.asarray(b['observed_hand_mask']); word=r['clip_id'].split('_')[2]
    rcards={c['id']:c for c in cards[word]}; fspec={c['id']:c for c in families[word]}
    assert len(phase)==len(role)==len(z['ts'])
    for c in r['cards']:
        spec=rcards[c['id']]; family=fspec[c['id']]['families'][0]
        valid=np.all(observed[:,[0 if h=='L' else 1 for h in spec['hands']]],axis=1)
        inferred=[]
        for group in c['groups']:
            p=phase[:,[g+1 for g in group['group']]].sum(axis=1)
            joint=np.sqrt(p*role[:,family]); runs=independent_runs((joint>=phase_cfg['threshold'])&valid,z['ts'])
            chosen=None
            if runs:
                values=[float(max(joint[a:b+1])) for a,b in runs]
                chosen=runs[values.index(max(values))]
            if runs!=group['runs'] or chosen!=group['selected']: errors.append(dict(key=triple,card=c['id'],group=group,actual=runs,selected=chosen))
            inferred.append(chosen)
        span=[min(a[0] for a in inferred),max(a[1] for a in inferred)] if all(a is not None for a in inferred) else None
        if span!=c['span']: errors.append(dict(key=triple,card=c['id'],span=span,stored=c['span']))
    for p in r['pairs']:
        a,bb=p['spans']; prob=p['order_probability']; verdict='association_unknown'
        if a is not None and bb is not None:
            verdict='supported' if a[1]<bb[0] and prob>=.65 else 'reversed' if bb[1]<a[0] and prob<=.35 else 'ambiguous'
        if verdict!=p['state']: errors.append(dict(key=triple,pair=p,computed=verdict))
        previous=next(x for x in old[triple]['pairs'] if x['cards']==p['cards'])
        if p['state']!=previous['state'] or p['spans']!=previous['spans']:
            pair_changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],cards=p['cards'],before=previous,after=p))
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

sources=[U/'evaluate_joint_correspondence.py',U/'JOINT_CORRESPONDENCE_RESULTS.json',U/'JOINT_CORRESPONDENCE_FROZEN.json',U/'JOINT_CORRESPONDENCE_COMPLETE.json',
 U/'JOINT_ORDER_RESULTS.json',U/'UNORDERED_SELECTION_RESULTS.json',HERE/'sequence_truth_audit.json',S/'READY.json',R/'EVALUATION_INPUTS.json',R/'safety_reconcile/FULLPATH_COMPLETE.json',
 O/'word100_event_guide_linkage_20261110/LINK_LABELS.json',O/'word100_gru_semantic_finetune_20261109/ANNOTATIONS.json',O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json',D/'FROZEN.json',D/'CARDS.json']
result=dict(status='SAVED_ARITHMETIC_AND_SPARSE_RGB_ANCHOR_AUDIT_NOT_DEPLOYMENT_PASS',reviewer='Separate Codex agent; independent code arithmetic, not independent human labels',
 model_inference=False,training=False,private_access=False,web_changed=False,threshold_tuning=False,
 numeric_verification=dict(rows=len(new),unique_inputs=len(inputs),card_checks=sum(x['cards'] for x in arithmetic_rows),pair_checks=sum(x['pairs'] for x in arithmetic_rows),
                           error_count=len(errors),errors=errors,source_and_saved_role_hashes_verified=True),
 saved_summary_recomputed=summaries,normal_anchor_summary=normal_summary,normal_anchor_details=normal_checks,changed_pair_details=pair_changes,
 prior_two_selected_span_conflicts=conflicts,
 seed_contract=dict(baseline_phase_uses_only_existing_seed42=all('guarded\\42\\' in str(m['baseline']) or 'guarded/42/' in str(m['baseline']) for m in inputs.values()),
                   baseline_raw_model_seed_values=sorted(set(str(b.get('raw_model_seed')) for b in base_cache.values())),
                   role_model_seed_specific=True,order_head_seed_specific=True,primary_seed42_not_selected_from_dev=True,
                   caveat='Existing phase/raw observer is fixed seed42; these are three auxiliary role/order seeds, not three fully independent whole pipelines.'),
 findings=[
 'All candidate run boundaries, maximum-peak choices, merged card envelopes and stored final order state arithmetic are recomputed from saved arrays without invoking a model.',
 'Normal/speed order support preservation is not a regression test of the complete shape/direction/coverage/render pipeline. This script only computes correspondence and order.',
 'REAL13 previous wrong-family raise_forearm span becomes null for all seeds. This removes that specific unsupported assignment but does not restore an observed complete action.',
 'REAL12 previous wrong-family raise_forearm span is replaced by source frames98-99 in the unedited tail. Sparse gold has no anchor inside these two frames. Existing RGB frames97 and102 show the already-upright closed-hand terminal posture, not the earlier forearm-turn trajectory. It cannot establish movement completion.',
 'Existing REAL12 RGB69/74 and REAL13 RGB86/93 show near-horizontal chest-level closed forearm; source-index mapping supports the old selected-span conflict diagnosis. New terminal hold selection is a different failure risk, not proof that the full action has been repaired.',
 'The partial-core reverse condition leaves source main/posture observations outside the edited span and reverses within-action direction. Do not score all9 pairs as mandatory reversed semantic truth or call all remaining supported cases false positives.',
 'Joint score sqrt(phase*role) at .65 is equivalent to product>=.4225. It is not the same as requiring each head>=.65. The two heads are correlated learned evidence; not two independent correctness witnesses.',
 'Code retains every qualifying run in JSON but order uses only one maximum-peak run per group. Runs shorter than60ms are discarded. Therefore all_occurrences_preserved=true in FROZEN is too broad if interpreted as all physical occurrences retained/used; short changes and repeated competing episodes remain unhandled.',
 'No word-specific conditional branch, per-word threshold or exception exists in evaluate_joint_correspondence.py. Target word indexes declared guide/card descriptors; four-word coverage is not all-word generalization proof.',
 'Selection considers phase and body role only, not expected shape match. However shrinking/moving windows can change downstream shape/direction support, so full downstream regression is required before any adoption.'
 ],
 visual_recheck=dict(existing_public_panels=[str(O/'word100_gru_semantic_finetune_20261109/review'/f'NIA_SL_WORD0739_REAL{p}_F.jpg') for p in [12,13]],
  exactly_rechecked_source_frames={'REAL12':[46,52,57,63,69,74,80,85,91,97,102,108],'REAL13':[54,60,67,73,80,86,93,99,106,112,119,125]},
  selected_source98_99_directly_reviewed=False,new_dense_labels=False,comparison_to_fit_review='Prior FIT01/04/07 occurrence review independently distinguishes main forearm turning from its terminal hold; no DEV labels were added to FIT.'),
 source_hashes={str(p):sha(p) for p in sources})
for p in result['visual_recheck']['existing_public_panels']: result['source_hashes'][p]=sha(p)
result['unknown_comparison']={}
for seed in [42,43,44]:
    result['unknown_comparison'][str(seed)]={}
    for kind in sorted({r['kind'] for r in new}):
        selected=[r for r in new if r['seed']==seed and r['kind']==kind]
        old_pairs=[p for r in selected for p in old[seed,r['clip_id'],kind]['pairs']]
        new_pairs=[p for r in selected for p in r['pairs']]
        old_cards=[c for r in selected for c in old_selected[r['clip_id'],kind]['cards']]
        new_cards=[c for r in selected for c in r['cards']]
        result['unknown_comparison'][str(seed)][kind]=dict(
            before_order_unknown_or_ambiguous=sum(p['state'] in ['association_unknown','ambiguous'] for p in old_pairs),
            after_order_unknown_or_ambiguous=sum(p['state'] in ['association_unknown','ambiguous'] for p in new_pairs),
            before_null_cards=sum(c['span'] is None for c in old_cards),after_null_cards=sum(c['span'] is None for c in new_cards))
result['normal_risk_highlight']={
    'clip_id':'NIA_SL_WORD2523_REAL13_F','card':'spread_hands','seeds':[42,43,44],
    'old_span':[64,81],'new_span':[64,67],
    'lost_all_direct_spread_anchors':[69,74,79],
    'source64_label':'transition (stage11), not direct spread completion gold',
    'interpretation':'Direct-anchor coverage fell from24/24 to23/24 normal card windows in each seed even though all9 order pairs stayed supported. This is a coverage diagnostic, not a new measured educational accuracy score.',
    'adoption':'No deployment pass. Recompute the full frozen feedback feature/coverage logic before any adoption, and obtain occurrence-level truth rather than tuning this window on these DEV results.'}
speed_key=('NIA_SL_WORD1133_REAL11_F','speed125')
normal_key=(speed_key[0],'full_normal')
durations={}
for key in [normal_key,speed_key]:
    tt=array_input(key)['ts'];durations[key[1]]=float((tt[73]-tt[72]+np.median(np.diff(tt)))/1000)
result['speed_risk_highlight']=dict(clip_id=speed_key[0],card='reach_wrist',required=False,parent='brush_arm',seeds=[42,43,44],
    normal_joint_span=[72,73],speed125_joint_span=None,two_frame_durations_ms=durations,minimum_ms=phase_cfg['min_evidence_ms'],
    interpretation='Speed125 loses an optional arrival card because its qualifying2-frame evidence no longer meets60ms. Required order-pair counts exclude this child and therefore do not reveal this speed regression. This is evidence availability loss, not proof of a new false correction.')
result['code_evidence']=[
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[7,10],fact='Legacy60ms qualifying-run floor retained.'),
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[15],fact='FROZEN all_occurrences_preserved wording exceeds actual single-selected-run decision contract.'),
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[17,18,22],fact='Seed-specific order models and saved role outputs, fixed baseline input path.'),
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[23,24,26,27],fact='Common descriptor-selected product score; no chronological previous_end filter; maximum peak per group and envelope across selected groups.'),
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[29,30,31],fact='Only required non-parent pairs; same-family pair skipped to avoid parent/arrival double count.'),
    dict(file=str(U/'evaluate_joint_correspondence.py'),lines=[34,35,36],fact='Order inference on selected midpoints and .65/.35 agreement; no shape/direction full-path re-evaluation in this script.')]
for path in [R/'FROZEN.json',O/'word100_action_episode_supervision_20261110/episode_core.py']:
    result['source_hashes'][str(path)]=sha(path)
assert not errors,errors
out=HERE/'joint_correspondence_verification.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),numeric=result['numeric_verification'],normal=normal_summary,summary=summaries),ensure_ascii=True))
