"""Independent source-to-inventory checks. No builder/model/media import."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U=Path(__file__).resolve().parent
C=U.parent/'word100_process_contract_alignment_20261102'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def aggregate(rows):
    return dict(rows=len(rows),
       direct_physical_process=sum(r['original_process_label']==1 for r in rows),
       partial_observed=sum(r['original_process_label']==0 for r in rows),
       unknown=sum(r['original_process_label'] is None for r in rows),
       source_clips=len(set(r['source_sha256'] for r in rows)),
       source_queries=len(set((r['person'],r['source_sha256'],r['physical_query']) for r in rows)),
       recorded_person_ids=sorted(set(r['person'] for r in rows)),
       unknown_reasons=dict(Counter(r['unknown_reason']['category'] for r in rows if r['original_process_label'] is None)),
       guide_feature_contradiction_known=sum(r['guide_feature_contradiction'] is not None for r in rows),
       educational_error_known=sum(r['educational_error'] is not None for r in rows))

def main():
    invpath=U/'FIT_TARGET_SEMANTICS_INVENTORY.json'
    inv=read(invpath);view=read(C/'FIT_VIEW_TARGETS_V2.json');nom=read(C/'NOMINAL_PROCESS_TARGETS.json')
    byview={r['view_id']:r for r in view['rows']}
    native={r['query_id']:r for r in view['rows'] if r['construction_kind_audit_only']=='native'}
    checks=[]
    def ck(name,test):checks.append(dict(check=name,passed=bool(test)))
    ck('335_unique_rows',len(inv['rows'])==335 and len({r['inventory_id'] for r in inv['rows']})==335)
    ck('three_metadata_sources_only',len(inv['source_pins'])==3 and set(Path(p).name for p in inv['source_pins'])=={'FIT_VIEW_TARGETS_V2.json','NOMINAL_PROCESS_TARGETS.json','NEXT_SUPERVISION_CONTRACT.json'})
    for p,h in inv['source_pins'].items():ck('source_hash:'+p,sha(Path(p))==h)
    ck('builder_hash',inv['inventory_code_sha256']==sha(U/'inventory_fit_target_semantics.py'))
    expected_ids={'view:'+r['view_id'] for r in view['rows']}|{'nominal:'+r['span_id'] for r in nom['rows']}
    ck('all_sources_once',set(r['inventory_id'] for r in inv['rows'])==expected_ids)
    seen=set()
    for row in inv['rows']:
        isview=row['source_population']=='reviewed_view'
        srcdoc=view if isview else nom
        artifact=C/('FIT_VIEW_TARGETS_V2.json' if isview else 'NOMINAL_PROCESS_TARGETS.json')
        ptr=row['source_artifact']['json_pointer'];idx=int(ptr.split('/')[-1]);src=srcdoc['rows'][idx]
        idkey='view_id' if isview else 'span_id';person=src['recorded_person_id'] if isview else src['person'];ident=src[idkey]
        seen.add((isview,idx));prefix=row['inventory_id']+':'
        ck(prefix+'source_identity',row['source_row_id']==ident and row['source_artifact']['path']==str(artifact) and row['source_artifact']['sha256']==sha(artifact))
        ck(prefix+'FIT_only',src['role']==row['role']=='fit' and person==row['person'] and person in range(1,10))
        ck(prefix+'same_process_label',type(row['original_process_label']) is type(src['observed_process_evidence']) and row['original_process_label']==src['observed_process_evidence'] and row['label_changed'] is False)
        label=src['observed_process_evidence'];expected_class='unknown' if label is None else 'direct_physical_process' if label==1 else 'partial_observed'
        ck(prefix+'classification_only',row['inventory_class']==expected_class)
        for k in ['query_id','source_sha256','physical_query','predicate_definitions']:ck(prefix+k,row[k]==src[k])
        ck(prefix+'group_exact',row['source_group']==dict(person=person,source_sha256=src['source_sha256'],physical_query=src['physical_query']))
        ck(prefix+'no_semantic_error_invention',row['guide_feature_contradiction'] is None and row['educational_error'] is None and row['whole_word_correctness'] is None and row['required_repetitions'] is None and row['temporal_inference_truth'] is None and src['whole_word_correctness'] is None and src['learner_error'] is None)
        reason=src['evidence_reason'] if isview else src['direct_review']['reason']
        ck(prefix+'reason_exact',row['source_reason']==reason)
        ur=row['unknown_reason']
        ck(prefix+'reason_pointer',ur['source_pointer']==f'/rows/{idx}/'+('evidence_reason' if isview else 'direct_review/reason'))
        if label is None:
            ck(prefix+'unknown_quote_exact',ur['source_quote']==reason and bool(reason))
            ck(prefix+'unknown_reason_explicit_or_unclassified',ur['category'] in ('observation_unavailable','relation_boundary_unclear','other_explicit','unclassified') and ur['basis'] in ('explicit_source_reason_read_and_indexed_by_Codex_not_keyword_inference','source_reason_not_explicitly_classified'))
        else:ck(prefix+'no_unknown_reason_on_known',ur['category'] is None and ur['source_quote'] is None)
        if isview:
            axis=src['ordered_frame_map']
            expected=dict(output_indices=[f['output_index'] for f in axis],source_frames=[f['source_native_frame'] for f in axis],source_timestamps_us=[f['source_native_timestamp_us'] for f in axis],input_timestamps_us=[f['view_timestamp_us'] for f in axis],image_available=[f['image_available'] for f in axis])
            ck(prefix+'family_exact',row['audit_family']==src['construction_kind_audit_only'])
            ck(prefix+'original_review_exact',row['original_review_reference']==src['review_source'])
            ck(prefix+'visible_chain_exact',row['original_visible_chain']==src['visible_chain'])
            for side in ('setup_relation','terminal_relation'):
                ck(prefix+'sparse_counts_'+side,row['auxiliary_scope']['sparse_relation_counts'][side]==dict(Counter(str(p[side]) for p in src['sparse_relation_targets'])))
            ck(prefix+'netmotion_counts',row['auxiliary_scope']['exact_netmotion_counts']==dict(Counter(str(p['label']) for p in src['fully_retained_motion_interval_targets'])))
            ck(prefix+'preserved_positive_netmotion_fact',row['auxiliary_scope']['process_zero_with_preserved_positive_netmotion']==(label==0 and any(p['label']==1 for p in src['fully_retained_motion_interval_targets'])))
        else:
            expected=dict(output_indices=src['output_indices'],source_frames=src['source_frames'],source_timestamps_us=src['source_timestamps_us'],input_timestamps_us=None,image_available=src['image_available'])
            parent=byview.get(src['case_id'])
            family=parent['construction_kind_audit_only'] if parent else 'runtime_proposal'
            ck(prefix+'nominal_family_metadata_only',row['audit_family']==family and (parent is not None or src['case_id']==src['query_id']+'__runtime_proposal'))
            ck(prefix+'nominal_binding_exact',row['binding_kind']==src['binding_kind'] and row['nominal_binding_scope']['label_inferred_from_whole_view']==src['label_inferred_from_whole_view'] and row['nominal_binding_scope']['label_scope']==src['label_scope'])
            ck(prefix+'original_review_exact',row['original_review_reference']=={k:src['direct_review'][k] for k in ('path','sha256')})
            ck(prefix+'clip_metadata_only',row['clip_id']==native[src['query_id']]['clip_id'] and row['word_id']==native[src['query_id']]['word_id'])
            ck(prefix+'no_new_dense_motion',row['auxiliary_scope']['new_dense_or_motion_targets_created'] is False)
        ck(prefix+'full_exact_support',row['exact_support']==expected)
        if label is None and ur['category']=='observation_unavailable':
            ck(prefix+'unavailable_consistent_with_axis',not all(expected['image_available']))
        if label is None and ur['category']=='relation_boundary_unclear':
            ck(prefix+'visible_boundary_not_blackout',all(expected['image_available']))
    ck('all217plus118_rowindices',len(seen)==335)
    rows=inv['rows'];vr=[r for r in rows if r['source_population']=='reviewed_view'];nr=[r for r in rows if r['source_population']=='nominal_span']
    for name,subset in [('all_dependent_records',rows),('reviewed_views',vr),('nominal_spans',nr)]:ck('summary:'+name,aggregate(subset)==inv['summaries'][name])
    families=sorted({r['construction_kind_audit_only'] for r in view['rows']})
    ck('all_eight_families',len(families)==8 and set(inv['summaries']['all_eight_view_families'])==set(families))
    for family in families:ck('family:'+family,aggregate([r for r in vr if r['audit_family']==family])==inv['summaries']['all_eight_view_families'][family])
    for bind in {r['binding_kind'] for r in nr}:ck('binding:'+bind,aggregate([r for r in nr if r['binding_kind']==bind])==inv['summaries']['nominal_by_binding'][bind])
    for query in {r['physical_query'] for r in rows}:
        for name,subset in [('reviewed_views',vr),('nominal_spans',nr)]:ck('query:'+query+':'+name,aggregate([r for r in subset if r['physical_query']==query])==inv['summaries']['per_physical_query'][query][name])
    for cls in ['direct_physical_process','partial_observed','unknown']:
        ratio=sum(sum(r['inventory_class']==cls for r in vr if r['audit_family']==f)/sum(r['audit_family']==f for r in vr) for f in families)/len(families)
        ck('equal_family_descriptive_fraction:'+cls,abs(ratio-inv['summaries']['equal_family_macro_class_fractions_descriptive_only'][cls])<1e-12)
    ck('no_training_or_new_truth',inv['new_labels']==inv['changed_label_values']==inv['new_images_reviewed']==inv['new_model_calls']==inv['new_training']==inv['new_scores']==0 and inv['training_release'] is False)
    failed=[r for r in checks if not r['passed']]
    report=dict(status='PASS_SOURCE_LABEL_SEMANTICS_INVENTORY_NO_NEW_TRUTH' if not failed else 'FAIL',
       inventory_sha256=sha(invpath),verifier_sha256=sha(Path(__file__)),source_pins=inv['source_pins'],
       checks=len(checks),failures=len(failed),failed_checks=failed,
       counts=aggregate(rows),view_counts=aggregate(vr),nominal_counts=aggregate(nr),
       source_label_values_changed=0,new_semantic_ground_truth=0,model_calls=0,training_steps=0,
       independent_check_scope='Separate reconstruction without importing the inventory builder. Verifies exact source values, axes, references, quotations and complete family/group accounting; not independent human verification of the physical labels or reason meaning.',
       original_source_images_or_arrays_or_DEV_opened=False,training_allowed=False,all_checks=checks)
    out=U/'FIT_TARGET_SEMANTICS_VERIFIED.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','inventory_sha256','checks','failures','counts']},ensure_ascii=False))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
