"""Reconstruct DEV saved bindings independently; imports no compiler or model.

This is a second arithmetic/schema check by Codex, not new visual adjudication
or independent human/KSL-expert annotation.
"""
import hashlib
import json
import statistics
import traceback
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

A=Path(__file__).resolve().parent
OUT=A/'DEV_PROCESS_VIEW_TARGETS_VERIFIED.json'
COUNTS=Counter()
HASHES={}

def require(condition, message, category='schema'):
    COUNTS[category]+=1
    if not condition:raise AssertionError(message)

def load(name):
    return json.loads((A/name).read_text(encoding='utf-8-sig'))

def sha(path):
    p=Path(path).resolve()
    if str(p) not in HASHES:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
        HASHES[str(p)]=h.hexdigest()
    return HASHES[str(p)]

def main():
    require(not OUT.exists(),'Preserve completed verification')
    result=load('DEV_PROCESS_VIEW_TARGETS.json')
    sources={r['id']:r for r in load('DEV_PROCESS_REVIEW_MANIFEST.json')['rows']}
    originals={r['id']:r for r in load('DEV_PROCESS_TARGETS.json')['rows']}
    manifest=load('DEV_PARTIAL_VIEWS_MANIFEST.json')
    views={r['view_id']:r for r in manifest['views']}
    reviews={r['view_id']:r for w in ['0739','1133','1567','2523']
             for r in load('REVIEW_DEV_PARTIAL_'+w+'.json')['rows']}
    require(len(result['rows'])==len(views)==80 and len(originals)==15 and len(reviews)==65,'Coverage')
    require(result['training_allowed'] is False and result['independent_final_test'] is False,'Evaluation-only contract')
    require({r['view_id'] for r in result['rows']}==set(views),'One target per view')
    for p,h in result['pins'].items():require(sha(p)==h,p,'hash')

    old=load('PROCESS_VIEW_TARGETS_INDEPENDENT.json')
    require(sha(A/'PROCESS_VIEW_TARGETS.json')==old['compiled_hash'],'Preserve frozen FIT compiled targets','fit_immutable')
    require(sha(A/'PROCESS_TARGETS.json')==old['source_target_hash'],'Preserve FIT source targets','fit_immutable')
    require(sha(A/'verify_view_targets_independent.py')==old['code_hash'],'Preserve FIT verification script','fit_immutable')

    pages={p['path']:p for src in sources.values() for p in src['pages']}
    pages.update({p['path']:p for v in views.values() for p in v['review_pages']})
    labels=Counter();by_kind={};relation_count=interval_count=0
    partial_with_positive_motion=[];unknown_with_retained_points=[]
    direct_label_counts=Counter();inherited_label_counts=Counter()
    clock_error=0;mapping_count=0
    for row in result['rows']:
        v=views[row['view_id']];q=originals[row['query_id']];src=sources[row['query_id']]
        fm=v['ordered_frame_map'];frames=[f['source_native_frame'] for f in fm]
        require(row['ordered_frame_map']==fm,'Frame map unchanged','mapping')
        require(row['original_scope_variant']==v.get('original_scope_variant'),'Variant preserved','scope')
        require(row['role']==q['role']==v['role']=='development' and row['recorded_person_id'] in (11,12,13),'Role/split','scope')
        require(row['training_allowed'] is False,'No DEV training','scope')
        for k in ['source_sha256','clip_id','physical_query','pixel_view_key']:
            require(row[k]==v[k],'Metadata '+k,'mapping')
        require(row['source_sha256']==src['source_sha256']==q['source_sha256'],'Source hash lineage','hash')
        require(sha(row['source_path'])==row['source_sha256'],'Source media hash','hash')
        native={f['frame']:f for f in src['frames']}
        for i,f in enumerate(fm):
            n=native[f['source_native_frame']]
            require(f['output_index']==i,'Index binding','mapping')
            require(f['source_native_timestamp_us']==n['timestamp_us'] and f['source_frame_path']==n['path']
                    and f['source_frame_sha256']==n['sha256'],'Original frame/PTS','mapping')
            require(sha(n['path'])==n['sha256'],'Original pixel hash','hash')
            require(f['image_available']==(f['pixel_operation']=='copy'),'Image availability','mapping')
            mapping_count+=1
        require(frames==sorted(set(frames)),'No repeated source frame','mapping')
        gaps=[i for i in range(1,len(fm)) if frames[i]-frames[i-1]!=1]
        require(gaps==row['seam_output_indices']==v['seam_output_indices'],'Native gap/seam','mapping')
        dts=[b['timestamp_us']-a['timestamp_us'] for a,b in zip(src['frames'],src['frames'][1:])]
        ordinary_dt=Decimal(str(statistics.median(dts)))
        running=Decimal(0);speed=Decimal(str(v['speed']))
        require(fm[0]['view_timestamp_us']==0,'Clock origin','clock')
        for i in range(1,len(fm)):
            step=ordinary_dt if i in gaps else Decimal(fm[i]['source_native_timestamp_us']-fm[i-1]['source_native_timestamp_us'])
            running+=step/speed
            expected=int(running.to_integral_value(rounding=ROUND_HALF_EVEN))
            err=abs(expected-fm[i]['view_timestamp_us']);clock_error=max(clock_error,err)
            require(err<=1,'Independently accumulated output PTS','clock')

        if row['view_id'] in reviews:
            direct=reviews[row['view_id']]
            require(row['direct_review']==direct,'Original direct review exact','review')
            value=direct['observed_process_evidence']
        else:
            direct=row['direct_review'];chain=q['sequence_links'][v['sequence_link_index']]
            value=1 if chain['visible_chain']=='yes' else None
            require(direct['source_sequence_link']==chain,'Native original chain exact','review')
            require(direct['source_annotation']==q['source_annotation'] and direct['source_annotation_sha256']==q['source_annotation_sha256'],
                    'Native review provenance','review')
            require([w['source_frame'] for w in direct['setup_witnesses']]==chain['setup_frames']
                    and [w['source_frame'] for w in direct['terminal_witnesses']]==chain['terminal_frames'],
                    'Native witnesses not relabelled','review')
        require(row['observed_process_evidence']==value,'Direct tri-state unchanged','review')
        require(row['visible_chain']==direct['visible_chain'],'Chain value unchanged','review')
        require(row['evidence_reason']==direct['evidence_reason'],'Reason unchanged','review')
        require(all(row[k] is None and direct[k] is None for k in ['whole_word_correctness','learner_error','required_repetitions']),
                'No whole-sign truth','scope')
        require(row['dense_frame_targets_created'] is False and row['unlisted_is_unknown'] is True,'Sparse scope','scope')
        labels[str(value)]+=1;kind=v['construction_kind_audit_only']
        by_kind.setdefault(kind,Counter())[str(value)]+=1

        viewed=[]
        for p in direct['reviewed_pages']:
            require(sha(p['path'])==p['sha256']==pages[p['path']]['sha256'],'Reviewed sheet hash','hash')
            mapped=pages[p['path']]
            viewed+=mapped.get('frames',mapped.get('source_native_frames',[]))
        require(viewed==frames,'Actually reviewed ordered image coverage','review')
        require(direct['reviewed_output_indices']==list(range(len(fm))),'Reviewed output slots','review')
        own={p['path'] for p in v['review_pages']};seen={p['path'] for p in direct['reviewed_pages']}
        if kind!='native' and own!=seen:
            peer=views[row['query_id']+'__speed_0_8']
            fields=['source_native_frame','source_native_timestamp_us','source_frame_sha256','pixel_operation','image_available']
            require([[f[k] for k in fields] for f in fm]==[[f[k] for k in fields] for f in peer['ordered_frame_map']],
                    'Speed reuse pixel/order identity','review')
            require(kind=='speed_1_25' and seen=={p['path'] for p in peer['review_pages']},'Only documented same-pixel reuse','review')
            require({p['path']:p['sha256'] for p in direct['artifact_pages_verified_not_opened']}==
                    {p['path']:p['sha256'] for p in v['review_pages']},'No false own-page inspection claim','review')
        for p in v['review_pages']:require(sha(p['path'])==p['sha256'],'Artifact sheet hash','hash')

        for name,key in [('bound_setup_witnesses','setup_witnesses'),('bound_terminal_witnesses','terminal_witnesses')]:
            require(len(row[name])==len(direct[key]),'Point witness count','witness')
            for w,original in zip(row[name],direct[key]):
                require(w['original_witness']==original,'Point original exact','witness')
                f=fm[w['output_index']]
                require(w['source_frame']==f['source_native_frame'] and f['image_available'] and f['pixel_operation']=='copy','Visible point','witness')
                require(w['source_native_timestamp_us']==f['source_native_timestamp_us'] and w['view_timestamp_us']==f['view_timestamp_us'],
                        'Point clocks','witness')
        positive_direct=False
        require(len(row['bound_direct_motion_witnesses'])==len(direct['motion_witnesses']),'Motion witness count','witness')
        for w,original in zip(row['bound_direct_motion_witnesses'],direct['motion_witnesses']):
            require(w['original_witness']==original,'Motion original exact','witness')
            explicit='label' in original
            require(w['label_was_explicit']==explicit and w['label']==original.get('label'),'No invented numeric label','witness')
            require(w['numeric_interval_target_created_from_unlabelled_witness'] is False,'No unlabelled-to-target promotion','scope')
            sf,oi=w['source_frames'],w['output_indices']
            require(sf==[fm[i]['source_native_frame'] for i in oi],'Motion mapping','witness')
            require(w['native_timestamps_us']==[fm[i]['source_native_timestamp_us'] for i in oi]
                    and w['view_timestamps_us']==[fm[i]['view_timestamp_us'] for i in oi],'Motion clocks','witness')
            if not explicit or w['label'] is not None:
                require(sf==list(range(sf[0],sf[-1]+1)) and oi==list(range(oi[0],oi[-1]+1)),'Known contiguous interval','witness')
                require(all(fm[i]['image_available'] and fm[i]['pixel_operation']=='copy' for i in oi)
                        and not any(oi[0]<s<=oi[-1] for s in gaps),'No known motion across blank/seam','witness')
                positive_direct|=not explicit or w['label']==1
            direct_label_counts[str(w['label']) if explicit else 'unlabelled_witness']+=1
        require(value!=1 or positive_direct,'Process-positive requires visible motion','scope')

        expected_rel=[];excluded_rel=[]
        by_frame={f['source_native_frame']:f for f in fm}
        for j,ann in enumerate(q['relations']):
            f=by_frame.get(ann['frame'])
            if f is None or not f['image_available'] or f['pixel_operation']!='copy':
                excluded_rel.append(j);continue
            expected_rel.append((j,ann['frame'],f['output_index'],ann['setup_relation'],ann['terminal_relation']))
        got=[(r['source_relation_index'],r['source_frame'],r['output_index'],r['setup_relation'],r['terminal_relation']) for r in row['sparse_relation_targets']]
        require(got==expected_rel,'Sparse relation reconstruction','target')
        require([r['source_relation_index'] for r in row['excluded_source_relations']]==excluded_rel,'Excluded relation complement','target')
        for rel in row['sparse_relation_targets']:
            require(rel['source_annotation']==q['relations'][rel['source_relation_index']],'Relation original exact','target')

        expected_intervals=[];excluded_intervals=[]
        for j,ann in enumerate(q['motion_intervals']):
            span=list(range(ann['start'],ann['end']+1))
            selected=[(i,f) for i,f in enumerate(fm) if ann['start']<=f['source_native_frame']<=ann['end']]
            oi=[i for i,f in selected]
            valid=([f['source_native_frame'] for i,f in selected]==span and
                   all(f['image_available'] and f['pixel_operation']=='copy' for i,f in selected) and
                   oi==list(range(oi[0],oi[-1]+1)) if oi else False)
            if valid:expected_intervals.append((j,span,oi,ann['label']))
            else:excluded_intervals.append(j)
        got=[(r['source_motion_interval_index'],r['source_frames'],r['output_indices'],r['label']) for r in row['fully_retained_motion_interval_targets']]
        require(got==expected_intervals,'Whole interval reconstruction without clipping','target')
        require([r['source_motion_interval_index'] for r in row['excluded_source_motion_intervals']]==excluded_intervals,'Excluded interval complement','target')
        for interval in row['fully_retained_motion_interval_targets']:
            require(interval['source_annotation']==q['motion_intervals'][interval['source_motion_interval_index']],'Interval original exact','target')
            oi=interval['output_indices']
            require(interval['view_timestamps_us']==[fm[i]['view_timestamp_us'] for i in oi] and
                    interval['native_timestamps_us']==[fm[i]['source_native_timestamp_us'] for i in oi],'Inherited clocks','target')
            require(not any(oi[0]<s<=oi[-1] for s in gaps),'Inherited seam exclusion','target')
            inherited_label_counts[str(interval['label'])]+=1
        if value==0 and (positive_direct or any(a[3]==1 for a in expected_intervals)):
            partial_with_positive_motion.append(row['view_id'])
        if value is None and expected_rel:unknown_with_retained_points.append(row['view_id'])
        relation_count+=len(expected_rel);interval_count+=len(expected_intervals)

    require(dict(labels)==result['counts']['observed_process_evidence'],'Label count reconstruction','count')
    require(relation_count==result['counts']['sparse_relation_records'] and interval_count==result['counts']['fully_retained_motion_intervals'],
            'Target count reconstruction','count')
    require({k:dict(v) for k,v in by_kind.items()}==result['counts']['by_construction_audit_only'],'By-kind summary only','count')
    require(by_kind['native']=={'1':15},'Native labels unchanged','count')
    require(len({r['clip_id'] for r in result['rows']})==12,'Dependent source count','count')
    report={'at':datetime.now(timezone.utc).isoformat(),'status':'PASS_SEPARATE_STORED_VALUE_RECONSTRUCTION',
      'compiled_path':str(A/'DEV_PROCESS_VIEW_TARGETS.json'),'compiled_sha256':sha(A/'DEV_PROCESS_VIEW_TARGETS.json'),
      'verification_code_sha256':sha(__file__),'views':80,'source_queries':15,'source_clips':12,
      'process_evidence_counts':dict(labels),'sparse_relation_records':relation_count,'whole_retained_intervals':interval_count,
      'direct_motion_label_counts':dict(direct_label_counts),'inherited_interval_label_counts':dict(inherited_label_counts),
      'source_output_time_mappings_checked':mapping_count,'maximum_view_clock_difference_us':clock_error,
      'partial_process_views_with_positive_motion_preserved':partial_with_positive_motion,
      'unknown_views_with_visible_sparse_relations_preserved':unknown_with_retained_points,
      'all_direct_reviews_embedded_exact':True,'unknown_not_changed_to_negative':True,'partly_cut_intervals_not_inherited':True,
      'seam_or_blank_known_motion_not_inherited':True,'whole_sign_correctness_or_completion_labels_created':False,
      'fit_outputs_match_prior_frozen_hashes':True,'training_allowed':False,'repeated_development':True,
      'independent_final_test':False,'independent_human_expert':False,'compiler_modules_imported':False,
      'same_implementer_second_algorithm_not_independent_semantic_review':True,
      'actual_speed_performance_evaluated':False,'new_inference':0,'new_training':0,
      'checks':dict(COUNTS),'checks_total':sum(COUNTS.values()),'unique_hashed_files':len(HASHES),
      'whole_goal_complete':False}
    with OUT.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'views':80,'counts':dict(labels),'relations':relation_count,
                     'intervals':interval_count,'checks':report['checks_total'],'sha256':sha(OUT)},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception:
        failure=A/('DEV_VIEW_VERIFY_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.txt')
        with failure.open('x',encoding='utf8') as stream:stream.write(traceback.format_exc())
        raise
