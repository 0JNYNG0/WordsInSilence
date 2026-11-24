"""Inventory existing FIT target semantics only; no model/media/result access."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U=Path(__file__).resolve().parent
C=U.parent/'word100_process_contract_alignment_20261102'
VIEW=C/'FIT_VIEW_TARGETS_V2.json'
NOMINAL=C/'NOMINAL_PROCESS_TARGETS.json'
CONTRACT=U/'NEXT_SUPERVISION_CONTRACT.json'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# These are an explicit index of the sixteen full source reasons read by the
# reviewer. No edit-name/keyword classifier assigns a semantic target.
UNKNOWN_REASONS={
 'NIA_SL_WORD0739_REAL01_F_directed_60_70__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD0739_REAL08_F_directed_57_69__native':('relation_boundary_unclear','neck_shoulder_region'),
 'NIA_SL_WORD0739_REAL09_F_directed_52_64__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD1133_REAL01_F_directed_49_64__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD1133_REAL09_F_directed_50_59__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD1567_REAL01_F_directed_62_67__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD1567_REAL01_F_directed_72_78__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD1567_REAL09_F_directed_66_72__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD2523_REAL01_F_directed_38_44__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD2523_REAL09_F_directed_68_77__missing_rgb':('observation_unavailable','hidden_connecting_images'),
 'NIA_SL_WORD0739_REAL07_F_directed_52_63__runtime_proposal__frozen_first_review':('relation_boundary_unclear','neck_shoulder_region'),
 'NIA_SL_WORD0739_REAL08_F_directed_57_69__runtime_proposal__frozen_first_review':('relation_boundary_unclear','neck_shoulder_region'),
 'NIA_SL_WORD1133_REAL06_F_directed_42_54__runtime_proposal__frozen_first_review':('relation_boundary_unclear','foreshortened_arm_and_hand_overlap'),
 'NIA_SL_WORD1567_REAL03_F_directed_42_47__runtime_proposal__frozen_first_review':('relation_boundary_unclear','hand_overlap_at_crossing'),
 'NIA_SL_WORD1567_REAL06_F_directed_50_61__runtime_proposal__frozen_first_review':('relation_boundary_unclear','hand_overlap_at_crossing'),
 'NIA_SL_WORD1567_REAL09_F_directed_72_82__runtime_proposal__frozen_first_review':('relation_boundary_unclear','hand_overlap_at_crossing'),
}
CLASS={1:'direct_physical_process',0:'partial_observed',None:'unknown'}

def summary(rows):
    by=Counter(r['inventory_class'] for r in rows)
    return dict(rows=len(rows),direct_physical_process=by['direct_physical_process'],partial_observed=by['partial_observed'],unknown=by['unknown'],
       source_clips=len({r['source_sha256'] for r in rows}),source_queries=len({tuple(r['source_group'].values()) for r in rows}),
       recorded_person_ids=sorted({r['person'] for r in rows}),
       unknown_reasons=dict(Counter(r['unknown_reason']['category'] for r in rows if r['original_process_label'] is None)),
       guide_feature_contradiction_known=sum(r['guide_feature_contradiction'] is not None for r in rows),
       educational_error_known=sum(r['educational_error'] is not None for r in rows))

def indexed_reason(identity,label,text,pointer):
    if label is not None:
        assert identity not in UNKNOWN_REASONS
        return dict(category=None,subtype=None,basis='not_applicable_to_known_process_label',source_pointer=pointer,source_quote=None)
    if identity not in UNKNOWN_REASONS:
        return dict(category='unclassified',subtype=None,basis='source_reason_not_explicitly_classified',source_pointer=pointer,source_quote=text)
    cat,sub=UNKNOWN_REASONS[identity]
    return dict(category=cat,subtype=sub,basis='explicit_source_reason_read_and_indexed_by_Codex_not_keyword_inference',source_pointer=pointer,source_quote=text)

def main():
    v,n,contract=read(VIEW),read(NOMINAL),read(CONTRACT)
    assert len(v['rows'])==217 and len(n['rows'])==118
    view_by_id={r['view_id']:r for r in v['rows']}
    by_query={r['query_id']:r for r in v['rows'] if r['construction_kind_audit_only']=='native'}
    assert len(by_query)==45
    pins={str(p):sha(p) for p in [VIEW,NOMINAL,CONTRACT]}
    rows=[]
    for i,r in enumerate(v['rows']):
        assert r['role']=='fit' and 1<=r['recorded_person_id']<=9
        label=r['observed_process_evidence'];assert label in CLASS
        assert r['whole_word_correctness'] is None and r['learner_error'] is None
        fm=r['ordered_frame_map'];ident=r['view_id'];reasonptr=f'/rows/{i}/evidence_reason'
        point_counts={side:dict(Counter(str(t[side]) for t in r['sparse_relation_targets'])) for side in ('setup_relation','terminal_relation')}
        interval_counts=dict(Counter(str(t['label']) for t in r['fully_retained_motion_interval_targets']))
        rows.append(dict(inventory_id='view:'+ident,source_population='reviewed_view',source_row_id=ident,
           source_artifact=dict(path=str(VIEW),sha256=pins[str(VIEW)],json_pointer=f'/rows/{i}'),
           query_id=r['query_id'],clip_id=r['clip_id'],word_id=r['word_id'],person=r['recorded_person_id'],role='fit',physical_query=r['physical_query'],
           source_sha256=r['source_sha256'],source_group=dict(person=r['recorded_person_id'],source_sha256=r['source_sha256'],physical_query=r['physical_query']),
           audit_family=r['construction_kind_audit_only'],binding_kind='existing_reviewed_view',
           original_process_label=label,inventory_class=CLASS[label],label_changed=False,
           original_visible_chain=r['visible_chain'],source_reason=r['evidence_reason'],
           unknown_reason=indexed_reason(ident,label,r['evidence_reason'],reasonptr),
           exact_support=dict(output_indices=[f['output_index'] for f in fm],source_frames=[f['source_native_frame'] for f in fm],
               source_timestamps_us=[f['source_native_timestamp_us'] for f in fm],input_timestamps_us=[f['view_timestamp_us'] for f in fm],image_available=[f['image_available'] for f in fm]),
           predicate_definitions=r['predicate_definitions'],original_review_reference=r['review_source'],
           direct_review_pointer=f'/rows/{i}/direct_review',
           auxiliary_scope=dict(sparse_relation_targets_pointer=f'/rows/{i}/sparse_relation_targets',sparse_relation_counts=point_counts,
              exact_netmotion_targets_pointer=f'/rows/{i}/fully_retained_motion_interval_targets',exact_netmotion_counts=interval_counts,
              process_zero_with_preserved_positive_netmotion=label==0 and any(t['label']==1 for t in r['fully_retained_motion_interval_targets']),
              unlisted_targets_remain_unknown=True),
           temporal_inference_truth=None,guide_feature_contradiction=None,educational_error=None,whole_word_correctness=None,required_repetitions=None,
           semantic_absence_reason='These source records establish physical observation scope only; no independently supported guide contradiction or educational error target is supplied.'))
    for i,r in enumerate(n['rows']):
        assert r['role']=='fit' and 1<=r['person']<=9
        label=r['observed_process_evidence'];assert label in CLASS
        assert r['whole_word_correctness'] is None and r['learner_error'] is None
        original=by_query[r['query_id']]
        assert original['source_sha256']==r['source_sha256'] and original['physical_query']==r['physical_query']
        parent=view_by_id.get(r['case_id'])
        if parent is None:
            assert r['case_id']==r['query_id']+'__runtime_proposal'
            family='runtime_proposal'
        else:family=parent['construction_kind_audit_only']
        ident=r['span_id'];reasonptr=f'/rows/{i}/direct_review/reason'
        ref={k:r['direct_review'][k] for k in ['path','sha256']}
        rows.append(dict(inventory_id='nominal:'+ident,source_population='nominal_span',source_row_id=ident,
           source_artifact=dict(path=str(NOMINAL),sha256=pins[str(NOMINAL)],json_pointer=f'/rows/{i}'),
           case_id=r['case_id'],query_id=r['query_id'],clip_id=original['clip_id'],word_id=original['word_id'],person=r['person'],role='fit',physical_query=r['physical_query'],
           source_sha256=r['source_sha256'],source_group=dict(person=r['person'],source_sha256=r['source_sha256'],physical_query=r['physical_query']),
           audit_family=family,binding_kind=r['binding_kind'],original_process_label=label,inventory_class=CLASS[label],label_changed=False,
           original_visible_chain=None,source_reason=r['direct_review']['reason'],
           unknown_reason=indexed_reason(ident,label,r['direct_review']['reason'],reasonptr),
           exact_support=dict(output_indices=r['output_indices'],source_frames=r['source_frames'],source_timestamps_us=r['source_timestamps_us'],
               input_timestamps_us=None,image_available=r['image_available']),
           predicate_definitions=r['predicate_definitions'],original_review_reference=ref,direct_review_pointer=f'/rows/{i}/direct_review',
           auxiliary_scope=dict(existing_chain_pointer=f'/rows/{i}/original_sequence' if r['binding_kind']=='explicit_direct_chain' else None,
               explicit_chain_witness_output_indices=r.get('explicit_chain_witness_output_indices'),
               parent_reviewed_absence_reference=r.get('parent_case_id'),new_dense_or_motion_targets_created=False),
           nominal_binding_scope=dict(label_scope=r['label_scope'],label_inferred_from_whole_view=r['label_inferred_from_whole_view'],
               support_containment_verified=r.get('support_containment_verified'),rgb_context_pointer=f'/rows/{i}/rgb_context_support',
               rgb_context_is_direct_nominal_observation=False),
           temporal_inference_truth=None,guide_feature_contradiction=None,educational_error=None,whole_word_correctness=None,required_repetitions=None,
           semantic_absence_reason='A source-explicit chain or reviewed absence can label this physical observation scope; it does not supply educational or guide-contradiction truth.'))
    assert len(rows)==335 and len({r['inventory_id'] for r in rows})==335
    unknown_ids={r['source_row_id'] for r in rows if r['original_process_label'] is None}
    assert unknown_ids==set(UNKNOWN_REASONS)
    views=[r for r in rows if r['source_population']=='reviewed_view'];spans=[r for r in rows if r['source_population']=='nominal_span']
    families={k:summary([r for r in views if r['audit_family']==k]) for k in sorted({r['audit_family'] for r in views})}
    equal_macro={cl:sum(f[cl]/f['rows'] for f in families.values())/len(families) for cl in CLASS.values()}
    out=dict(status='FIT_SOURCE_TARGET_SEMANTICS_INVENTORIED_LABELS_UNCHANGED_NOT_TRAINING_RELEASE',
       reviewer='Codex retention_direction_review; read-only source-label audit, no new visual annotation or independent expert truth.',
       source_pins=pins,inventory_code_sha256=sha(Path(__file__)),
       contract_review=dict(status='COMPATIBLE_WITH_CURRENT_EVIDENCE_AND_USER_GOAL',
          rationale=['The contract preserves physical observations and their original times while separating their evidence limits from educational feedback.',
             'It preserves the failed AB/AC evaluations and prohibits turning short synthetic edits into linguistic-error targets.',
             'Separate output meanings do not require a new generic model or replacement of already useful frozen physical/RGB observations.'],
          limits=['This inventory adds no learner-error or guide-contradiction supervision; none is directly supplied in the two audited artifacts.',
             'Reason-indexing unknown labels does not itself define a new abstention loss. Boundary ambiguity and unavailable images remain distinct.',
             'Partial observed means the requested connected process was not fully directly observed in the scoped input. It does not assert zero movement, wrong direction, wrong sign, or inadequate repetition.',
             'Do not train merely because an inventory exists; the next single design must identify the remaining learned capability it addresses.']),
       class_definitions=dict(direct_physical_process='Original process label1: scoped directly reviewed connected physical process, including explicit reuse of a directly reviewed chain.',
          partial_observed='Original process label0: requested connected process not directly supported in this scope; may contain real motion and positive point/interval observations. No educational error implication.',
          unknown='Original process labelnull: preserve uncertainty and its explicit reason. No new negative or semantic truth.',
          temporal_inference_truth='Not separately annotated in these artifacts; null remains null rather than assigning a neighboring-state inference target.'),
       scope_accounting=dict(reviewed_views=217,nominal_spans=118,total_dependent_records=335,source_clips=36,source_query_groups=45,recorded_ids=list(range(1,10)),
          actual_read_sources=[str(VIEW),str(NOMINAL),str(CONTRACT)],original_review_references_copied_not_reopened=True,
          media_arrays_predictions_DEV_read=False,nominal_and_view_rows_not_independent=True),
       summaries=dict(all_dependent_records=summary(rows),reviewed_views=summary(views),nominal_spans=summary(spans),
          all_eight_view_families=families,equal_family_macro_class_fractions_descriptive_only=equal_macro,
          nominal_by_binding={k:summary([r for r in spans if r['binding_kind']==k]) for k in sorted({r['binding_kind'] for r in spans})},
          per_physical_query={q:dict(reviewed_views=summary([r for r in views if r['physical_query']==q]),nominal_spans=summary([r for r in spans if r['physical_query']==q])) for q in sorted({r['physical_query'] for r in rows})},
          partial_views_with_explicit_positive_netmotion=sum(r['auxiliary_scope']['process_zero_with_preserved_positive_netmotion'] for r in views)),
       unknown_reason_categories=['observation_unavailable','relation_boundary_unclear','other_explicit','unclassified'],
       unknown_reason_method='The sixteen existing full reasons were individually read and explicitly indexed by source record ID. No keyword/edit-kind classifier, score, or new image interpretation assigned categories. Any future unmatched unknown stays unclassified.',
       educational_error_known_targets=0,guide_feature_contradiction_known_targets=0,
       same_source_controls=dict(all_edit_families_included=True,no_failed_DEV_case_imported=True,
          positive_alternative_occurrences_preserved=True,retimes_not_independent_execution_or_fast_motion_proof=True,
          family_counts_not_equalized_by_sampling=True),
       rows=rows,new_labels=0,changed_label_values=0,new_images_reviewed=0,new_model_calls=0,new_training=0,new_scores=0,
       training_release=False,whole_goal_complete=False)
    p=U/'FIT_TARGET_SEMANTICS_INVENTORY.json';assert not p.exists()
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'path':str(p),'sha256':sha(p),'counts':out['summaries']['all_dependent_records'],'view_families':families,'partial_with_positive_netmotion':out['summaries']['partial_views_with_explicit_positive_netmotion']},ensure_ascii=False))

if __name__=='__main__':main()
