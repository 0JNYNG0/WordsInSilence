"""Bind bounded physical-process labels; never synthesize positive from endpoints."""
from pathlib import Path
from collections import Counter
import json
from process_data import read, sha, view_record, runtime_record
from process_rgb_support import span_support

U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'
REVIEW_FILES=['REVIEW_FIRST_1133.json','REVIEW_FIRST_2523.json','REVIEW_FIRST_0739_1567.json']


def main():
    dest=U/'NOMINAL_PROCESS_TARGETS.json'
    assert not dest.exists()
    manifest_path=U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json'
    manifest=read(manifest_path)
    for p,h in manifest['pins'].items():assert sha(p)==h,p
    rgb=read(U/'RGB_SUPPORT_MANIFEST.json')
    assert read(U/'RGB_SUPPORT_VERIFIED.json')['status'].startswith('PASS')
    inputs={r['pixel_view_key']:r for r in read(AB/'PROCESS_INPUT_MANIFEST.json')['rows']}
    views=read(U/'FIT_VIEW_TARGETS_V2.json')['rows']
    runtime=read(AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json')['rows']
    records={r['view_id']:view_record(r,inputs[r['pixel_view_key']]) for r in views}
    records.update({b['query_id']+'__runtime_proposal':runtime_record(b) for b in runtime if b['role']=='fit'})
    pins={str(p.resolve()):sha(p) for p in [Path(__file__),manifest_path,U/'RGB_SUPPORT_MANIFEST.json',
        U/'RGB_SUPPORT_VERIFIED.json',U/'FIT_VIEW_TARGETS_V2.json',AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json',
        AB/'PROCESS_AUTO_EVALUATION.json',AB/'PROCESS_INPUT_MANIFEST.json',AB/'PROCESS_TARGETS.json']}
    pins.update(manifest['pins'])
    direct={}
    for name in REVIEW_FILES:
        p=U/name;d=read(p);pins[str(p.resolve())]=sha(p)
        for r in d['rows']:
            assert r['review_id'] not in direct
            direct[r['review_id']]=(r,p)
    assert len(direct)==45
    rows=[];excluded=[]

    def bind(case_id, indices, label, binding, review_path, reason, suffix):
        r=records[case_id];ii=list(map(int,indices))
        assert r['role']=='fit' and r['person'] in range(1,10)
        assert len(ii)>=2 and ii==list(range(ii[0],ii[-1]+1))
        aa=[r['axis'][j] for j in ii]
        support=rgb['inputs'][r['input_sha256']]['support']
        mapper={f:i for i,f in enumerate(support['source_frame_ids'])}
        cache_ii=[mapper[a['source_frame']] for a in aa]
        ctx=span_support(support,cache_ii)
        assert ctx['nominal_source_frames']==[a['source_frame'] for a in aa]
        assert ctx['nominal_source_timestamps_us']==[a['source_timestamp_us'] for a in aa]
        assert label in (0,1,None) and reason
        p=Path(review_path);pins[str(p.resolve())]=sha(p)
        return dict(span_id=case_id+'__'+suffix,case_id=case_id,role='fit',person=r['person'],
            query_id=r['query_id'],physical_query=r['physical_query'],source_sha256=r['source_sha256'],
            input_sha256=r['input_sha256'],predicate_definitions=r['predicate_definitions'],
            output_indices=ii,source_frames=[a['source_frame'] for a in aa],
            source_timestamps_us=[a['source_timestamp_us'] for a in aa],
            image_available=[a['image_available'] for a in aa],
            observed_process_evidence=label,binding_kind=binding,
            label_scope='nominal_visible_process_only',nominal_scope_reviewed=True,
            label_inferred_from_whole_view=binding=='contained_reviewed_absence',
            rgb_context_outside_nominal_possible=True,rgb_context_support=ctx,
            direct_review=dict(path=str(p.resolve()),sha256=sha(p),reason=reason),
            whole_word_correctness=None,learner_error=None,required_repetitions=None,
            exact_completion_boundary=None,independent_human_expert=False)

    for m in manifest['rows']:
        d,p=direct[m['review_id']]
        assert d['case_id']==m['case_id']
        assert d['source_frames']==m['nominal_source_frames']
        assert d['source_timestamps_us']==m['nominal_source_timestamps_us']
        pages=m['review_pages']
        expected={x['path']:x['sha256'] for x in pages}
        actual={x['path']:x['sha256'] for x in d['reviewed_pages']}
        assert expected==actual
        for path,h in actual.items():
            assert sha(path)==h; pins[str(Path(path).resolve())]=h
        r=bind(m['case_id'],m['nominal_output_indices'],d['observed_process_evidence'],
               'direct_nominal_review',p,d['reason'],'frozen_first_review')
        r['review_record_id']=d['review_id'];r['reviewed_pages']=d['reviewed_pages']
        rows.append(r)
        for j,s in enumerate(m['existing_direct_positive_span_candidates']):
            original=s['original_sequence']
            assert original['visible_chain']=='yes' and s['label']==1
            r=bind(m['case_id'],s['output_indices'],1,'explicit_direct_chain',
                AB/'PROCESS_TARGETS.json',original['reason'],'existing_direct_chain_'+str(j))
            assert r['source_frames']==s['source_frames'] and r['source_timestamps_us']==s['source_timestamps_us']
            required=set(original['setup_frames']+original['terminal_frames'])
            if original.get('motion_interval'):
                lo,hi=original['motion_interval'];required.update(range(lo,hi+1))
            assert required<=set(r['source_frames']) and all(r['image_available'])
            assert sha(s['source_annotation'])==s['source_annotation_sha256']
            pins[str(Path(s['source_annotation']).resolve())]=s['source_annotation_sha256']
            local_by_source={a['source_frame']:a['output_index'] for a in records[m['case_id']]['axis']}
            r.update(original_sequence=original,required_source_frames=sorted(required),
                     explicit_chain_witness_output_indices=[local_by_source[f] for f in sorted(required)],
                     source_annotation_path=s['source_annotation'],source_annotation_sha256=s['source_annotation_sha256'],
                     original_sequence_index=s['original_sequence_index'],explicit_chain_inside_nominal=True)
            rows.append(r)
    # A reviewed existential absence is hereditary only within that same actual
    # pixel stream. Whole-positive clips never create positive subwindow labels.
    saved=read(AB/'PROCESS_AUTO_EVALUATION.json')['view_rows']
    for old in saved:
        if old['role']!='fit' or old['kind']!='middle_splice':continue
        candidates=old['qualification']['interval_candidates']
        if not candidates:continue
        record=records[old['view_id']]
        assert old['truth']==record['observed_process_evidence']
        if old['truth']!=0:
            excluded.append(dict(case_id=old['view_id'],reason='Positive/unknown full view does not label arbitrary candidate'))
            continue
        candidate=candidates[0]
        r=bind(old['view_id'],candidate['output_indices'],0,'contained_reviewed_absence',
            U/'FIT_VIEW_TARGETS_V2.json',record['original_target']['evidence_reason'],
            'contained_absence_first_candidate')
        support=r['rgb_context_support']['consumed_context_cache_rows']
        parent=list(range(len(record['axis'])))
        assert set(support)<=set(parent)
        assert rgb['inputs'][r['input_sha256']]['route']=='actual_materialized_view_cache'
        r.update(parent_case_id=old['view_id'],parent_whole_process_truth=0,
            support_containment_verified=True,support_output_indices=support,
            parent_reviewed_output_indices=parent,
            inherited_scope='Same materialized edited pixels, never original unedited neighbor cache',
            new_motion_zero_labels=0)
        rows.append(r)
    assert len({r['span_id'] for r in rows})==len(rows)
    counts=dict(Counter(r['binding_kind'] for r in rows))
    assert counts['direct_nominal_review']==45 and counts['explicit_direct_chain']==44
    for p,h in pins.items():assert sha(p)==h,p
    obj=dict(status='DIRECT_NOMINAL_PROCESS_TARGETS_READY',rows=rows,excluded=excluded,
        counts=counts,labels=dict(Counter(str(r['observed_process_evidence']) for r in rows)),
        independent_observation_count_claim=False,pins=pins,new_inference=0,new_training=0,
        same_codex_review_not_independent_expert=True,whole_goal_complete=False)
    with dest.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'counts':counts,'labels':obj['labels'],'sha256':sha(dest)}))

if __name__=='__main__':main()
