"""Bind evaluation-only direct DEV process reviews; never run the FIT compiler.

The immutable FIT module contributes pure binding/hash helpers only. Its main()
is not invoked. Every output of this driver has a distinct DEV filename.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import statistics
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import compile_reviewed_views as helpers

A = Path(__file__).resolve().parent
OUT = A / 'DEV_PROCESS_VIEW_TARGETS.json'
SELF = A / 'DEV_PROCESS_VIEW_TARGETS_SELF_CHECK.json'
FILES = [A / f'REVIEW_DEV_PARTIAL_{w}.json' for w in ('0739','1133','1567','2523')]
check = helpers.require
read = helpers.read
sha = helpers.sha
pin = helpers.pin


def validate_explicit_motion_arrays(motion, view):
    fm = view['ordered_frame_map']
    entries = [fm[i] for i in motion['output_indices']]
    aliases = {
        'source_timestamps_us':'source_native_timestamp_us',
        'source_native_timestamps_us':'source_native_timestamp_us',
        'native_timestamps_us':'source_native_timestamp_us',
        'view_timestamps_us':'view_timestamp_us',
        'source_frame_hashes':'source_frame_sha256',
        'image_available':'image_available',
    }
    for key, actual in aliases.items():
        if key in motion:
            check(motion[key] == [f[actual] for f in entries],
                  'Explicit motion array mismatch: '+key, 'witness_time')


def main():
    for dest in (OUT, SELF):
        check(not dest.exists(), f'Preserve existing output: {dest}')
    for p in FILES:
        check(p.is_file(), f'Direct review not ready: {p.name}')
    manifest_path=A/'DEV_PARTIAL_VIEWS_MANIFEST.json'
    target_path=A/'DEV_PROCESS_TARGETS.json'
    source_path=A/'DEV_PROCESS_REVIEW_MANIFEST.json'
    manifest, targets, source_manifest=map(read,(manifest_path,target_path,source_path))
    views=manifest['views']; view_by={v['view_id']:v for v in views}
    target_by={r['id']:r for r in targets['rows']}
    source_by={r['id']:r for r in source_manifest['rows']}
    check(len(views)==len(view_by)==80 and len(target_by)==len(source_by)==15, 'DEV coverage')
    check(targets['training_allowed'] is False and manifest['training_allowed'] is False, 'DEV is not FIT')
    for doc in (manifest, targets):
        for p,h in doc['pins'].items():
            pin(p,h)
    input_paths=[manifest_path,target_path,source_path,*FILES,Path(__file__).resolve(),Path(helpers.__file__).resolve()]
    input_hashes={str(p.resolve()):sha(p) for p in input_paths}

    reviews={}
    for path in FILES:
        doc=read(path)
        check(len(doc['rows'])==(26 if '1567' in path.name else 13), 'Partial review coverage', 'review')
        for k in ('source_manifest_sha256','manifest_sha256'):
            if k in doc:pin(manifest_path,doc[k])
        for k in ('input_manifest','manifest'):
            if k in doc:
                check(Path(doc[k]['path']).resolve()==manifest_path.resolve(), 'Review manifest path', 'review')
                pin(doc[k]['path'],doc[k]['sha256'])
        for row in doc['rows']:
            check(row['view_id'] not in reviews, 'Duplicate direct view review','review')
            reviews[row['view_id']]=(row,path,sha(path))
    check(set(reviews)=={v['view_id'] for v in views if v['construction_kind_audit_only']!='native'},
          'Expected 65 directly reviewed non-native views','review')

    page_catalog={}
    for v in views:
        for p in v['review_pages']:page_catalog[p['path']]=p
    # The original manifest owns page-to-frame mapping. Some direct reviews
    # intentionally preserve only page path/SHA, so do not erase its frames.
    for src in source_manifest['rows']:
        for p in src['pages']:page_catalog[p['path']]=p
    for q in targets['rows']:
        for p in q['reviewed_pages']:
            check(p['path'] in page_catalog and page_catalog[p['path']]['sha256']==p['sha256'],
                  'Native reviewed page maps to exact source-manifest page','page')
        pin(q['source_annotation'],q['source_annotation_sha256'])
    compiled=[]; excluded=collections.Counter()
    for v in views:
        q=target_by[v['query_id']]; src=source_by[v['query_id']]
        check(v['role']==q['role']==src['role']=='development', 'Development role','scope')
        check(v['recorded_person_id']==q['person']==src['person'] and q['person'] in (11,12,13), 'DEV IDs','scope')
        check(q['word_id'] in ('WORD0739','WORD1133','WORD1567','WORD2523'), 'Word scope','scope')
        check(q['training_allowed'] is False, 'No DEV training','scope')
        for key in ('clip_id','physical_query','source_sha256'):
            check(v[key]==q[key]==src[key], 'Source agreement: '+key,'source')
        check(Path(v['source_path']).resolve()==Path(q['source']).resolve()==Path(src['source']).resolve(), 'Source path','source')
        pin(v['source_path'],v['source_sha256'])
        pin(v['original_review_path'],v['original_review_sha256'])
        native={f['frame']:f for f in src['frames']}
        fm=v['ordered_frame_map']; frames=[f['source_native_frame'] for f in fm]
        check([f['output_index'] for f in fm]==list(range(len(fm))), 'Output axis','mapping')
        check(frames==sorted(set(frames)), 'Source order/uniqueness','mapping')
        seams=[i for i in range(1,len(fm)) if frames[i]!=frames[i-1]+1]
        check(seams==v['seam_output_indices'], 'Seams','mapping')
        for f in fm:
            n=native[f['source_native_frame']]
            check((f['source_native_timestamp_us'],f['source_frame_path'],f['source_frame_sha256'])==
                  (n['timestamp_us'],n['path'],n['sha256']), 'Source mapping','mapping')
            check(f['pixel_operation'] in ('copy','blackout') and f['image_available']==(f['pixel_operation']=='copy'),
                  'Availability mapping','mapping')
            pin(n['path'],n['sha256'])
        times=[f['view_timestamp_us'] for f in fm]
        check(times[0]==0 and all(b>a for a,b in zip(times,times[1:])), 'View clock monotonic','mapping')
        native_times=[f['timestamp_us'] for f in src['frames']]
        period=Fraction(str(statistics.median(b-a for a,b in zip(native_times,native_times[1:]))))
        speed=Fraction(str(v['speed'])); running=Fraction(0)
        for i in range(1,len(fm)):
            delta=period if i in seams else Fraction(fm[i]['source_native_timestamp_us']-fm[i-1]['source_native_timestamp_us'])
            running+=delta/speed
            check(abs(times[i]-round(running))<=1,'View clock binding','mapping')
        page_indices=[]
        for p in v['review_pages']:
            pin(p['path'],p['sha256']);page_indices.extend(p['output_indices'])
            check([fm[i]['source_native_frame'] for i in p['output_indices']]==p['source_native_frames'], 'Artifact page frames','page')
        check(page_indices==list(range(len(fm))), 'Artifact page coverage','page')

        is_native=v['construction_kind_audit_only']=='native'
        if is_native:
            review=helpers.native_review(q,v);rp=target_path;rh=sha(target_path)
        else:
            review,rp,rh=reviews[v['view_id']]
        check(review['query_id']==v['query_id'] and review['physical_query']==v['physical_query']
              and review['pixel_view_key']==v['pixel_view_key'], 'Review identity','review')
        if not is_native:
            for k in ('clip_id','source_sha256','role','recorded_person_id'):
                if k in review:check(review[k]==v[k], 'Review lineage: '+k,'review')
        check(review['reviewed_output_indices']==list(range(len(fm))), 'Reviewed all outputs','review')
        value=review['observed_process_evidence']
        check(value is None or type(value) is int and value in (0,1), 'Review tri-state','review')
        check(review['visible_chain'] in ('yes','partial','uncertain'), 'Chain schema','review')
        check(value!=1 or review['visible_chain']=='yes', 'Positive requires direct chain','review')
        for k in ('whole_word_correctness','learner_error','required_repetitions'):
            check(review[k] is None, 'No linguistic truth: '+k,'scope')

        viewed=[]
        for p in review['reviewed_pages']:
            pin(p['path'],p['sha256'])
            check(p['path'] in page_catalog and page_catalog[p['path']]['sha256']==p['sha256'], 'Viewed page catalog','page')
            entry=page_catalog[p['path']]
            viewed.extend(entry.get('source_native_frames',entry.get('frames',[])))
        check(viewed==frames, 'Viewed exact native pixel order','page')
        own={p['path'] for p in v['review_pages']};seen={p['path'] for p in review['reviewed_pages']}
        reuse=None
        if own!=seen:
            if is_native:
                check(all(f['image_available'] for f in fm), 'Native review pixel condition','page')
                reuse={'kind':'native_original_review_pages','same_native_pixels_verified':True}
            else:
                check(v['construction_kind_audit_only']=='speed_1_25','Only documented speed-page reuse','page')
                peer=view_by[v['query_id']+'__speed_0_8']
                check(helpers.pixel_signature(peer)==helpers.pixel_signature(v), 'Exact peer pixel signature','page')
                check(seen=={p['path'] for p in peer['review_pages']},'Reviewed peer pages','page')
                check({p['path']:p['sha256'] for p in review['artifact_pages_verified_not_opened']}==
                      {p['path']:p['sha256'] for p in v['review_pages']}, 'Own artifacts explicitly not opened','page')
                reuse={'kind':'identical_native_pixel_order_speed_variant','peer_view_id':peer['view_id'],
                       'actual_retimed_video_or_speed_performance_review':False}
        points={key:[helpers.bind_point(p,v) for p in review[key]] for key in ('setup_witnesses','terminal_witnesses')}
        motions=[]
        for m in review['motion_witnesses']:
            w=helpers.bind_motion(m,v);validate_explicit_motion_arrays(m,v);motions.append(w)
        check(value!=1 or any(w['visible_contiguous_witness_validated'] and
              (w['label']==1 or not w['label_was_explicit']) for w in motions), 'Positive motion witness','review')
        for rel in review.get('relations',[]):helpers.bind_point(rel,v)

        by_source={f['source_native_frame']:f for f in fm}
        relations=[];missing_rel=[];intervals=[];missing_intervals=[]
        for i,rel in enumerate(q['relations']):
            f=by_source.get(rel['frame'])
            reason='not_retained' if f is None else ('missing_or_changed_pixels' if not f['image_available'] or f['pixel_operation']!='copy' else None)
            if reason:
                missing_rel.append({'source_relation_index':i,'source_frame':rel['frame'],'reason':reason})
                excluded['relation_'+reason]+=1;continue
            check(rel['timestamp_us']==f['source_native_timestamp_us'] and rel['input_image']['sha256']==f['source_frame_sha256'], 'Sparse relation source','target')
            relations.append({'source_relation_index':i,'source_annotation':copy.deepcopy(rel),**helpers.binding(f),
                              'setup_relation':rel['setup_relation'],'terminal_relation':rel['terminal_relation']})
        for i,m in enumerate(q['motion_intervals']):
            selected,reason=helpers.complete_interval(fm,m['start'],m['end'])
            if reason:
                missing_intervals.append({'source_motion_interval_index':i,'source_interval':[m['start'],m['end']],
                                         'reason':reason,'inherited_target':None})
                excluded['motion_'+reason]+=1;continue
            oi=[f['output_index'] for f in selected]
            check(not any(oi[0]<s<=oi[-1] for s in seams),'No inherited interval across seam','target')
            intervals.append({'source_motion_interval_index':i,'source_annotation':copy.deepcopy(m),'label':m['label'],
              'source_frames':[f['source_native_frame'] for f in selected],'output_indices':oi,
              'native_timestamps_us':[f['source_native_timestamp_us'] for f in selected],
              'view_timestamps_us':[f['view_timestamp_us'] for f in selected],
              'supervision_unit':'one fully retained reviewed interval; not frame-wise labels'})
        compiled.append({'view_id':v['view_id'],'query_id':v['query_id'],'clip_id':v['clip_id'],'word_id':q['word_id'],
          'recorded_person_id':v['recorded_person_id'],'role':'development','training_allowed':False,
          'grouping_key':'recorded_person_'+str(v['recorded_person_id']),
          'physical_query':v['physical_query'],'predicate_definitions':copy.deepcopy(q['predicate_definitions']),
          'pixel_view_key':v['pixel_view_key'],'query_review_key':v['query_review_key'],
          'construction_kind_audit_only':v['construction_kind_audit_only'],'source_path':v['source_path'],
          'source_sha256':v['source_sha256'],'ordered_frame_map':copy.deepcopy(fm),'seam_output_indices':seams,
          'original_scope_variant':copy.deepcopy(v.get('original_scope_variant')),'artifact_pages':copy.deepcopy(v['review_pages']),
          'direct_review':copy.deepcopy(review),'review_source':{'path':str(rp),'sha256':rh},'page_content_reuse':reuse,
          'observed_process_evidence':value,'visible_chain':review['visible_chain'],'evidence_reason':review['evidence_reason'],
          'bound_setup_witnesses':points['setup_witnesses'],'bound_terminal_witnesses':points['terminal_witnesses'],
          'bound_direct_motion_witnesses':motions,'sparse_relation_targets':relations,
          'fully_retained_motion_interval_targets':intervals,'excluded_source_relations':missing_rel,
          'excluded_source_motion_intervals':missing_intervals,'whole_word_correctness':None,'learner_error':None,
          'required_repetitions':None,'unlisted_is_unknown':True,'dense_frame_targets_created':False})

    native=collections.Counter(str(r['observed_process_evidence']) for r in compiled if r['construction_kind_audit_only']=='native')
    check(native=={'1':15},'Preserve original 15 direct native chains','scope')
    counts={'views':len(compiled),'source_queries':15,'source_clips':len({r['clip_id'] for r in compiled}),
      'sparse_relation_records':sum(len(r['sparse_relation_targets']) for r in compiled),
      'fully_retained_motion_intervals':sum(len(r['fully_retained_motion_interval_targets']) for r in compiled),
      'observed_process_evidence':dict(collections.Counter(str(r['observed_process_evidence']) for r in compiled)),
      'by_construction_audit_only':{k:dict(collections.Counter(str(r['observed_process_evidence']) for r in compiled if r['construction_kind_audit_only']==k)) for k in sorted(manifest['counts'])},
      'excluded_source_targets':dict(excluded)}
    result={'at':datetime.now(timezone.utc).isoformat(),'status':'COMPILED_EVALUATION_ONLY_DEV_PROCESS_VIEWS',
      'pins':input_hashes,'rows':compiled,'counts':counts,'training_allowed':False,'repeated_development':True,
      'independent_final_test':False,'source_targets_unchanged':True,'review_records_embedded_unchanged':True,
      'unknown_and_variant_fields_preserved':True,'partly_retained_intervals_inherited':False,
      'seam_or_blackout_known_motion_inherited':False,'dense_labels_generated':False,
      'schema_compatibility_note':'Unlabelled direct motion witnesses remain nonnumeric witnesses. Explicit interval0/1/null remains interval-level and can coexist with different whole-view process evidence.',
      'supervision_scope':'Observed bounded physical process in the shown material, not whole-sign completion, required repetitions or learner correctness.',
      'dependent_views_are_not_new_people':True,'actual_speed_performance_evaluated':False,
      'independent_human_expert':False,'new_inference':0,'new_training':0,'whole_goal_complete':False}
    for p,h in input_hashes.items():
        check(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Source changed while compiling','final_pin')
    helpers.write_new(OUT,result)
    self_report={'at':datetime.now(timezone.utc).isoformat(),'status':'COMPILER_SELF_CHECK_PASS_NOT_INDEPENDENT_SEMANTIC_VALIDATION',
      'output_path':str(OUT),'output_sha256':sha(OUT),'input_pins':input_hashes,'counts':counts,
      'checks':dict(helpers.CHECKS),'checks_total':sum(helpers.CHECKS.values()),'unique_hashed_files':len(helpers.HASHES),
      'source_video_hashes_verified':12,'training_allowed':False,'no_semantic_or_performance_validation_claim':True}
    helpers.write_new(SELF,self_report)
    print(json.dumps({'status':self_report['status'],'counts':counts,'sha256':sha(OUT),'checks':self_report['checks_total']},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception:
        p=A/('DEV_VIEW_COMPILE_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.txt')
        with p.open('x',encoding='utf8') as f:f.write(traceback.format_exc())
        raise
