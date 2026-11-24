"""Separate arithmetic reconstruction. Never reruns or writes the saved comparison."""
from pathlib import Path
import json, hashlib, copy, importlib.util, itertools
from collections import Counter
from PIL import Image

U=Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}; counts=Counter()
def ck(k,v):
    checks[k]=bool(v)
    if not v: raise AssertionError(k)
F=('index_only','open_four'); M=('A','B','C')
frozen=read(U/'FROZEN.json'); idx=read(U/'visual_inputs/INDEX.json')
labels=read(U/'review/IMAGE_LABELS.json'); out=read(U/'COMPARISON.json')
center_labels=read(U/'independent/CENTER_LABELS.json')
lut={(r['case_id'],r['image_key']):r for r in labels['rows']}
source_by_id={r['case_id']:r for r in frozen['rows']}
independent={r['case_id']:r for r in center_labels['rows']}
saved_seq={s['case_id']:s for s in out['sequences']}
saved_masks={(r['case_id'],r['index'],r['feature']):r for r in out['masked_trials']}
saved_centers={(r['case_id'],r['feature']):r for r in out['centers']}

def reconstruct(values,ts,i,mode):
    if values[i] is not None: return dict(value=values[i],kind='direct',support_indices=[i])
    if mode=='A': return dict(value=None,kind='unknown',reason='no_current_evidence',support_indices=[])
    past=[]; future=[]
    for j,(v,t) in enumerate(zip(values,ts)):
        if v is None or j==i or abs(t-ts[i])>200001: continue
        (past if j<i else future).append(j)
    def unknown(reason,indices): return dict(value=None,kind='unknown',reason=reason,support_indices=indices)
    if len(past)<2:return unknown('past_insufficient',past)
    if any(values[j]!=values[past[0]] for j in past):return unknown('past_conflict',past)
    value=values[past[0]]
    if mode=='B':return dict(value=value,kind='temporal_estimate',reason='past_agreement',support_indices=past)
    if len(future)<2:return unknown('future_insufficient',past+future)
    if any(values[j]!=value for j in future):return unknown('direction_conflict',past+future)
    return dict(value=value,kind='temporal_estimate',reason='bidirectional_agreement',support_indices=past+future)

def outputs(states,ts,i,mode):
    raw={f:reconstruct(states[f],ts,i,mode) for f in F}
    use=copy.deepcopy(raw)
    if all(use[f]['value']==1 for f in F):
        for f in F:
            if use[f]['kind']=='temporal_estimate':use[f].update(value=None,kind='unknown',reason='feature_conflict')
    return raw,use

ck('frozen_and_saved_input_pins',all(sha(p)==h for p,h in {**frozen['pins'],**out['pins']}.items()))
ck('own_first_labels_unchanged',sha(U/'review/IMAGE_LABELS.json')==sha(U/'review/IMAGE_LABELS_FIRST_PASS.json')=='2ed98e18222376f62b23babce353459d2df1df4bb6667da52ff93c8cd1f45250')
ck('complete_unique_input_labels',len(labels['rows'])==len(lut)==234)
ck('same_18_frozen_cases',set(saved_seq)==set(source_by_id)=={r['case_id'] for r in idx['rows']})
masks=[]; centers=[]; all_unknown=[]; transitions=[]
for case in idx['rows']:
    cid=case['case_id']; sf=source_by_id[cid]
    for key in ('source','source_sha256','clip_id','side','recorded_person_id','center_frame_id','center_pts_us','inherited_role'):
        assert case[key]==sf[key],(cid,key)
    frames=sorted(case['frames'],key=lambda r:r['actual_pts_us'])
    ts=[r['actual_pts_us'] for r in frames]
    assert len(frames)==13 and [r['offset_frames'] for r in frames]==list(range(-6,7))
    assert all(b['frame_id']-a['frame_id']==1 for a,b in zip(frames,frames[1:]))
    assert all(b>a for a,b in zip(ts,ts[1:]))
    assert frames[6]['frame_id']==sf['center_frame_id'] and ts[6]==sf['center_pts_us']
    states={f:[lut[cid,r['image_key']][f] for r in frames] for f in F}
    baseline=copy.deepcopy(states)
    saved=saved_seq[cid]
    assert saved['center_index']==6 and saved['person_id']==case['recorded_person_id']
    for i,fr in enumerate(frames):
        assert Image.open(fr['full_file']).mode=='RGB'
        assert hashlib.sha256(Image.open(fr['full_file']).tobytes()).hexdigest()==fr['rgb_sha256']
        counts['decoded_RGB_hashes']+=1
        sr=saved['frames'][i]
        assert (sr['image_key'],sr['frame_id'],sr['pts_us'],sr['offset_frames'])==(fr['image_key'],fr['frame_id'],ts[i],i-6)
        assert sr['direct']=={f:states[f][i] for f in F}
        for mode in M:
            raw,use=outputs(states,ts,i,mode)
            assert raw==sr['raw_outputs'][mode] and use==sr['outputs'][mode],(cid,i,mode,'frame output')
            counts['unmasked_raw_and_usable_output_pairs']+=2
            for f in F:
                if states[f][i] is not None:
                    assert use[f]['value']==states[f][i] and use[f]['kind']=='direct'
                    counts['direct_preserved_'+mode]+=1
        for f in F:
            if states[f][i] is None:all_unknown.append((sr,f))
            if i==6:
                cr=saved_centers[cid,f]
                assert cr['direct']==states[f][i] and cr['independent_direct']==independent[cid][f]
                assert cr['outputs']=={mode:sr['outputs'][mode][f] for mode in M}
                centers.append(cr)
            if states[f][i] is None or i in (0,12):continue
            # New copy from immutable original state for EVERY target; only this scalar masked.
            masked={ff:[None if (ff==f and j==i) else x for j,x in enumerate(states[ff])] for ff in F}
            mr=saved_masks[cid,i,f]
            assert mr['hidden_draft']==states[f][i] and mr['frame_id']==fr['frame_id'] and mr['pts_us']==ts[i]
            for mode in M:
                raw,use=outputs(masked,ts,i,mode)
                assert raw[f]==mr['raw_outputs'][mode] and use[f]==mr['outputs'][mode],(cid,i,f,mode,'mask output')
                counts['masked_raw_and_usable_outputs']+=2
                if mode=='B':
                    # Change every future scalar in both streams; neither raw nor usable causal output may change.
                    changed={ff:[x if j<=i else (1 if x in (None,0) else 0) for j,x in enumerate(masked[ff])] for ff in F}
                    assert outputs(changed,ts,i,mode)==(raw,use)
                    counts['B_future_mutations_unchanged']+=1
                if use[f]['kind']=='temporal_estimate':
                    sp=use[f]['support_indices']
                    assert i not in sp and len(set(sp))==len(sp)
                    assert all(masked[f][j] is not None and abs(ts[j]-ts[i])<=200001 for j in sp)
                    assert sum(j<i for j in sp)>=2
                    if mode=='C':assert sum(j>i for j in sp)>=2
                    assert all(masked[f][j]==use[f]['value'] for j in sp)
            masks.append(mr)
            assert states==baseline
    for feature in F:
        visible=[i for i,v in enumerate(states[feature]) if v is not None]
        for a,b in zip(visible,visible[1:]):
            if states[feature][a]!=states[feature][b]:
                transitions.append(dict(case_id=cid,feature=feature,from_index=a,to_index=b,**{'from':states[feature][a],'to':states[feature][b]},elapsed_us=ts[b]-ts[a],unknown_frames_between=b-a-1,meaning='Change of image-review label; not measured finger movement duration.'))

ck('234_images_original_RGB_hashes',counts['decoded_RGB_hashes']==234)
ck('all_saved_direct_raw_usable_outputs_reconstructed',counts['unmasked_raw_and_usable_output_pairs']==1404)
ck('279_known_interior_targets_no_duplicate_no_omission',len(masks)==len(saved_masks)==279 and len(out['masked_trials'])==279)
ck('independent_mask_raw_and_usable_outputs_exact',counts['masked_raw_and_usable_outputs']==1674)
ck('B_no_future_leak_in_279_counterfactual_checks',counts['B_future_mutations_unchanged']==279)
ck('334_direct_states_preserved_each_condition',all(counts['direct_preserved_'+m]==out['known_states_preserved'][m]==334 for m in M))
ck('observed_draft_transition_rows_exact',transitions==out['observed_label_transitions'])
def aggregate(rows):
    d={}
    for m in M:
        accepted=[r for r in rows if r['outputs'][m]['value'] is not None]
        good=sum(r['outputs'][m]['value']==r['hidden_draft'] for r in accepted)
        d[m]=dict(total=len(rows),estimated=len(accepted),draft_match=good,draft_conflict=len(accepted)-good,withheld=len(rows)-len(accepted))
    return d
ck('masked_summary_exact',aggregate(masks)==out['masked_summary'])
ck('group_summary_exact',{g:aggregate([r for r in masks if r['group']==g]) for g in {r['group'] for r in masks}}==out['masked_by_group'])
ck('person_summary_exact',{str(p):aggregate([r for r in masks if r['person_id']==p]) for p in (1,2,3)}==out['masked_by_person'])
ck('feature_summary_exact',{f:aggregate([r for r in masks if r['feature']==f]) for f in F}==out['masked_by_feature'])
unknown_centers=[r for r in centers if r['direct'] is None]
center_summary={m:dict(centers=len(centers),direct_known=len(centers)-len(unknown_centers),original_unknown=len(unknown_centers),new_estimates=sum(r['outputs'][m]['value'] is not None for r in unknown_centers),remaining_unknown=sum(r['outputs'][m]['value'] is None for r in unknown_centers)) for m in M}
ck('real_unknown_center_summary_exact',center_summary==out['center_summary'])
ck('all_unknown_frame_summary_exact',{m:dict(original_unknown=len(all_unknown),new_estimates=sum(r['outputs'][m][f]['value'] is not None for r,f in all_unknown)) for m in M}==out['all_frames_unknown_summary'])
agree=Counter('both_unknown' if r['direct'] is None and r['independent_direct'] is None else 'one_unknown' if r['direct'] is None or r['independent_direct'] is None else 'known_agree' if r['direct']==r['independent_direct'] else 'known_conflict' for r in centers)
ck('center_same_model_agreement_exact',dict(agree)==out['center_review_agreement'])
ck('no_training_or_accuracy_promotion',out['training_ready'] is False and out['status']=='COMPUTED_NOT_HUMAN_VALIDATED')

# Bounded implementation tests against a separate reconstruction, without re-running comparison.
spec=importlib.util.spec_from_file_location('frozen_support',U/'temporal_support.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fixture_count=0
for values in itertools.product((None,0,1),repeat=5):
    for target in range(5):
        for mode in M:
            assert module.support(list(values),[0,33333,66667,100000,133333],target,mode)==reconstruct(list(values),[0,33333,66667,100000,133333],target,mode)
            fixture_count+=1
ck('exhaustive_5_frame_values_modes_match',fixture_count==3645)
for gap,expected in ((200000,1),(200001,1),(200002,None)):
    # Two past supports: far support only remains valid at the inclusive rounding boundary.
    ts=[0,1,gap,2*gap-1,2*gap]
    result=module.support([1,1,None,1,1],ts,2,'C')
    assert result['value']==expected,(gap,result)
ck('200000_plus_1us_boundary_enforced',True)
ck('single_hidden_pulse_can_be_mistaken_and_is_not_claimed_solved',module.support([0,0,None,0,0],[0,33333,66667,100000,133333],2,'C')['value']==0)
conflicts={m:[{k:r[k] for k in ('case_id','feature','index','frame_id','hidden_draft')}|{'estimated':r['outputs'][m]['value']} for r in masks if r['outputs'][m]['value'] is not None and r['outputs'][m]['value']!=r['hidden_draft']] for m in M}
result={
 'status':'PASS_WITH_SCOPE_LIMITS','checks':checks,'counts':dict(counts),
 'saved_comparison_sha256':sha(U/'COMPARISON.json'),
 'masked_summary_recomputed':aggregate(masks),'real_center_summary_recomputed':center_summary,
 'masked_conflicts':conflicts,'exhaustive_scalar_fixtures':fixture_count,
 'scope_findings':[
  'Arithmetic and fixed-rule implementation match saved outputs. This does not validate the direct RGB drafts as human truth.',
  'C fills zero genuinely unknown centers and zero genuinely unknown fields across these windows. No demonstrated real-unknown recovery from C in this sample.',
  'C reduces disagreement in artificial masking but still overwrites one hidden transient open-four draft with surrounding closed state; exact finger-motion correctness is unverified.',
  'Known-state preservation is by construction. Passing it does not establish good recovery of unknown fast changes.',
  'All trial values are hand-authored image-review states, not detector outputs; no TCN/DTW recognition or feedback accuracy was measured.',
  'Identity concerns are supported by function but default false in these outputs; current numeric comparison alone does not assert image-based identity verification.'
 ],
 'frozen_coefficients_and_labels_changed':False,'source_outputs_recomputed_or_overwritten':False
}
(U/'review'/'COMPARISON_VERIFY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':len(checks),'masked_summary':result['masked_summary_recomputed'],'conflicts':conflicts},ensure_ascii=False))
