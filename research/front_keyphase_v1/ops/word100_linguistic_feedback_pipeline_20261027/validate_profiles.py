"""Bounded behavioral checks. These check contracts, not Korean sign correctness."""
import hashlib
from pathlib import Path
from feature_profiles import read, dump, source_claims, describe_comparison as compare, check_reference_identity, ROOT


def run():
    profiles = read(ROOT / 'FEATURE_PROFILES.json')
    source_entries = read(ROOT / 'source_audit/SOURCE_ENTRY_DETAILS.json')['entries']
    checks = {}
    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True
    base = dict(query_value=10., reference_values=[8., 11.], observed=True, references_observed=[True, True])
    a = compare(**base, variant_verified=True, sequence_alignment_verified=True)
    require('G01_descriptive_values_without_correctness', a['differences_per_reference'] == [2., -1.] and a['educational_judgment'] is None)
    b = compare(**base)
    require('G02_no_automatic_variant_promotion', b['source_link_status'] == 'lexical_candidate_only')
    star = next(p for p in profiles['profiles'] if p['word_id'] == 'WORD2551')
    require('G03_separate_star_entries', len([e for e in star['candidate_entries'] if e['source_id'] == 'NIKL_nuri']) == 2 and star['verified_variant_id'] is None)
    missing = compare(None, [8.,11.], observed=False, references_observed=[True,True])
    require('G04_query_missing_preserves_plan', missing['planned_units'] == b['planned_units'] == 1 and missing['observed_pair_units'] == 0 and missing['differences_per_reference'] == [None,None])
    mr = compare(10., [None,11.], observed=True, references_observed=[False,True])
    require('G05_reference_missing_preserves_plan', mr['planned_units'] == 1 and mr['reference_observed'] == [False,True] and mr['differences_per_reference'] == [None,-1.])
    rest = compare(**base, expected_from_reference=False)
    require('G07_inactive_is_reference_input_not_query_absence', rest['planned_units'] == 0 and missing['planned_units'] == 1)
    changed = compare(14.,[8.,11.],observed=True,references_observed=[True,True])
    require('G10_real_numeric_difference_not_suppressed', changed['differences_per_reference'] == [6.,3.] and changed['educational_judgment'] is None)
    contact = compare(**base, source_claim={'family':'contact_description','text':'코 접촉 기술'})
    require('G11_contact_description_not_physical_contact_truth', contact['educational_judgment'] is None and 'physical_contact' not in contact)
    require('G12_preserve_reference_values', b['reference_values'] == [8.,11.] and len(b['differences_per_reference']) == 2)
    empty = compare(None,[None,None],observed=False,references_observed=[False,False])
    require('G13_empty_is_insufficient', empty['measurement_status'] == 'insufficient_observation' and empty['educational_judgment'] is None)
    q={'recorded_id':1,'content_sha256':'a'}
    valid=[{'recorded_id':2,'content_sha256':'b'},{'recorded_id':3,'content_sha256':'c'}]
    check_reference_identity(q,valid)
    rejected=0
    for bad in [[q,valid[1]],[{'recorded_id':2,'content_sha256':'a'},valid[1]],[valid[0],valid[0]],
                [valid[0], {'recorded_id':3,'content_sha256':'b'}]]:
        try:check_reference_identity(q,bad)
        except ValueError:rejected+=1
    require('G14_self_duplicate_reference_rejection',rejected==4)
    claims=[c for p in profiles['profiles'] for e in p['candidate_entries'] for c in e['claims']]
    require('G15_no_invented_tolerance_or_requiredness', all(c['requiredness']=='unknown' and c['tolerance'] is None for c in claims))
    ready=read(ROOT/'TRAINING_READINESS.json')
    require('G16_no_labels_no_training', ready['trained_model_count']==0 and not ready['descriptive_feature_training_gate']['pass'] and not ready['pedagogical_correctness_training_gate']['pass'])
    require('G18_unverified_alignment_retains_point_measurements', b['sequence_status']=='alignment_unverified' and b['differences_per_reference']==[2.,-1.])
    require('all_100_profiles_not_only_selected_good_words', len(profiles['profiles'])==100)
    require('no_metadata_first_handshape_to_whole_video', all(c['temporal_scope']=='initial_handshape_only' for c in claims if c.get('source_field')=='signHandList'))
    keys={(e['source_id'],str(e['entry_id'])):e for e in source_entries}
    position_claims=[c for c in claims if c.get('source_field')=='signHandPositionView']
    require('no_null_position_name_invented',bool(position_claims) and all(
        c['source_value']['handNm']==keys[(c['source']['source_id'],str(c['source']['entry_id']))]['typed_metadata']['signHandPositionView']['handNm']
        for c in position_claims))
    text_claims=[c for c in claims if c['kind']=='text_mention_candidate']
    require('all_literal_spans_match_source',all(c['text'][s['start']:s['end']]==s['term'] for c in text_claims for s in c['spans']))
    require('source_sha_and_url_traceable',all(c['source']['url']==keys[(c['source']['source_id'],str(c['source']['entry_id']))]['url'] and c['source']['content_sha256']==keys[(c['source']['source_id'],str(c['source']['entry_id']))]['content_sha256'] for c in claims))
    out={'status':'PASS_PROGRAM_CONTRACTS_NOT_LINGUISTIC_VALIDATION','checks':checks,
         'root_checks_count':len(checks),'delegated_public_checks':['G06','G08','G09','G17'],
         'no_error_truth_metrics':{'verified_errors':0,'error_recall':None,'normal_false_correction_rate':None},
         'implementation':'Codex root; separate review pending'}
    dump(ROOT/'PROFILE_CONTRACTS.json',out)
    dump(ROOT/'COMPARISON_EXAMPLES.json',{'unit':'synthetic scalar contract examples, not real user or sign accuracy','observed':a,'missing_query':missing,'one_missing_reference':mr,'unverified_alignment':b,'changed_value':changed,'all_missing':empty})
    return out


if __name__=='__main__':
    try:
        result=run();print(result['status'],result['root_checks_count'])
    except Exception as exc:
        path=ROOT/'PROFILE_FIRST_ERROR.json'
        if not path.exists():dump(path,{'type':type(exc).__name__,'message':str(exc)})
        raise
