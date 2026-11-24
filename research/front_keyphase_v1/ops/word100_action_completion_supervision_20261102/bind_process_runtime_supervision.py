"""Bind existing direct supervision to fixed full-native runtime proposals.
Preparation only: no scores/results read, no new proposal, inference or training.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,traceback
import numpy as np

U=Path(__file__).resolve().parent
PINS={};INPUTS={};CHECKS=0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def check(x,message):
    global CHECKS
    CHECKS+=1
    if not x:raise AssertionError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def pin(p,expected=None):
    p=str(Path(p).resolve())
    if p not in PINS:PINS[p]=sha(p)
    if expected is not None:check(PINS[p]==expected,'pin '+p)
    return PINS[p]
def fragments(frames,lookup):
    groups=[]
    for f in frames:
        if not groups or f!=groups[-1][-1]+1 or lookup[f]['output_index']!=lookup[groups[-1][-1]]['output_index']+1:
            groups.append([])
        groups[-1].append(f)
    return groups
def main():
    out=U/'PROCESS_RUNTIME_SUPERVISION_BINDING.json'
    check(not out.exists(),'Refuse overwrite of binding')
    names=['PROCESS_LABEL_PROTOCOL.json','PHASE_LABEL_ADDENDUM.json','PROCESS_AUTO_PROPOSAL_PLAN.json',
           'PROCESS_AUTO_NATIVE_MANIFEST.json','PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json',Path(__file__).name]
    for name in names:pin(U/name)
    m=read(U/'PROCESS_AUTO_NATIVE_MANIFEST.json')
    targets={r['id']:r for fn in ['PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json'] for r in read(U/fn)['rows']}
    check(len(m['rows'])==len(targets)==60,'60 fixed queries')
    check(len({r['query_id'] for r in m['rows']})==60,'query uniqueness')
    check({r['query_id'] for r in m['rows']}==set(targets),'query sets')
    rows=[]
    for meta in m['rows']:
        t=targets[meta['query_id']];role=meta['role'];person=meta['person'];clip=meta['clip_id']
        check(role in ['fit','development'],'role')
        check(person in (range(1,10) if role=='fit' else [11,12,13]),'public IDs')
        check(meta['word_id'] in ['WORD0739','WORD1133','WORD1567','WORD2523'],'word scope')
        check(clip==f"NIA_SL_{meta['word_id']}_REAL{person:02d}_F",'clip allowlist')
        check(all(t[k]==meta[k] for k in ['clip_id','word_id','person','role','physical_query','source_sha256']),'source identity')
        check(Path(t['source']).resolve()==Path(meta['source']).resolve(),'source paths')
        pin(meta['source'],meta['source_sha256']);pin(meta['input'],meta['input_sha256'])
        pin(t['source_annotation'],t['source_annotation_sha256'])
        if meta['input'] not in INPUTS:
            with np.load(meta['input'],allow_pickle=False) as z:
                INPUTS[meta['input']]={k:z[k].copy() for k in ['frame_id','timestamp_us','image_available','x','mask','rgb']}
        a=INPUTS[meta['input']];n=len(a['frame_id'])
        check(a['frame_id'].tolist()==meta['frame_ids'] and a['timestamp_us'].tolist()==meta['timestamp_us'],'full axis')
        check(a['x'].shape==a['mask'].shape==(n,69) and a['rgb'].shape==(n,768),'input structure')
        check(a['mask'].dtype==bool and a['image_available'].dtype==bool and a['image_available'].shape==(n,),'masks')
        check(np.all(np.diff(a['frame_id'])>0) and np.all(np.diff(a['timestamp_us'])>0),'monotonic full axes')
        check(np.isfinite(a['x'][a['mask']]).all() and np.isfinite(a['rgb']).all(),'input finiteness')
        ii=meta['proposal']['output_indices']
        check(len(ii)>=2 and ii==list(range(ii[0],ii[-1]+1)) and 0<=ii[0]<=ii[-1]<n,'fixed proposal slice')
        check(meta['proposal']['max_proposals_per_query']==1 and meta['proposal']['exact_event_boundary'] is False,'proposal limit')
        axis=[dict(output_index=j,full_native_index=i,source_frame=int(a['frame_id'][i]),
                   source_timestamp_us=int(a['timestamp_us'][i]),image_available=bool(a['image_available'][i]))
              for j,i in enumerate(ii)]
        lookup={x['source_frame']:x for x in axis}
        full_lookup={int(f):i for i,f in enumerate(a['frame_id'])}
        check(len(lookup)==len(axis),'unique source frames')
        def bound(frame):
            return dict(lookup[frame]) if frame in lookup else None
        points=[];unbound_points=[]
        for idx,relation in enumerate(t['relations']):
            f=int(relation['frame']);entry=bound(f)
            check(f in full_lookup,'source relation frame in native input')
            stamp=int(a['timestamp_us'][full_lookup[f]])
            if 'timestamp_us' in relation:check(int(relation['timestamp_us'])==stamp,'relation native time')
            check(relation['setup_relation'] in [0,1,None] and relation['terminal_relation'] in [0,1,None],'direct point labels')
            if entry is not None and entry['image_available']:
                points.append(dict(source_relation_index=idx,**entry,
                              setup_relation=relation['setup_relation'],terminal_relation=relation['terminal_relation'],
                              component_loss_eligible={'setup_relation':relation['setup_relation'] is not None,
                                                       'terminal_relation':relation['terminal_relation'] is not None},
                              original_annotation=relation,direct_labels_unchanged=True))
            else:
                unbound_points.append(dict(source_relation_index=idx,source_frame=f,source_timestamp_us=stamp,
                    mapped_axis=entry,reason='outside_fixed_proposal' if entry is None else 'image_unavailable',
                    bound_supervision=None,loss_eligible=False,original_annotation_audit_only=relation))
        intervals=[];partial=[]
        for idx,interval in enumerate(t['motion_intervals']):
            lo,hi=int(interval['start']),int(interval['end']);check(hi>lo,'interval length')
            fs=list(range(lo,hi+1));check(all(f in full_lookup for f in fs),'source interval full native coverage')
            stamps=[int(a['timestamp_us'][full_lookup[lo]]),int(a['timestamp_us'][full_lookup[hi]])]
            if 'native_timestamps_us' in interval:check(interval['native_timestamps_us']==stamps,'direct interval clock')
            present=[f for f in fs if f in lookup];available=[f for f in present if lookup[f]['image_available']]
            whole=(len(available)==len(fs) and [lookup[f]['output_index'] for f in fs]==list(range(lookup[fs[0]]['output_index'],lookup[fs[-1]]['output_index']+1))) if len(available)==len(fs) else False
            if whole:
                intervals.append(dict(source_interval_index=idx,source_frames=fs,
                    output_indices=[lookup[f]['output_index'] for f in fs],
                    full_native_indices=[lookup[f]['full_native_index'] for f in fs],
                    source_timestamps_us=[lookup[f]['source_timestamp_us'] for f in fs],
                    label=interval['label'],loss_eligible=interval['label'] is not None,
                    original_annotation=interval,supervision_scope='one whole directly reviewed interval, never dense frame labels',
                    direct_labels_unchanged=True))
            else:
                partial.append(dict(source_interval_index=idx,source_native_extent=[lo,hi],
                    retained_visible_fragments=[[bound(f) for f in g] for g in fragments(available,lookup)],
                    retained_unavailable_frames=[bound(f) for f in present if not lookup[f]['image_available']],
                    missing_source_frames=[f for f in fs if f not in lookup],
                    label=None,loss_eligible=False,original_annotation_audit_only=interval,
                    reason='Not every original interval image survives visibly and contiguously in this fixed proposal; no shortened interval target assigned.'))
        sequence=[];supports=[]
        for idx,link in enumerate(t['sequence_links']):
            required=set(link['setup_frames'])|set(link['terminal_frames'])
            if link.get('motion_interval'):
                lo,hi=link['motion_interval'];required.update(range(lo,hi+1))
            required=sorted(required)
            check(all(f in full_lookup for f in required),'sequence native frame support')
            covered=bool(required) and all(f in lookup and lookup[f]['image_available'] for f in required)
            linked_motion=link.get('motion_interval')
            if covered and linked_motion:
                lo,hi=linked_motion
                covered=[lookup[f]['output_index'] for f in range(lo,hi+1)]==list(range(lookup[lo]['output_index'],lookup[hi]['output_index']+1))
            supported=link['visible_chain']=='yes' and covered
            sequence.append(dict(source_sequence_index=idx,original_annotation=link,
                setup_witnesses=[bound(f) for f in link['setup_frames']],
                terminal_witnesses=[bound(f) for f in link['terminal_frames']],
                required_source_frames=required,required_mapped_axis=[bound(f) for f in required],
                missing_or_unavailable_source_frames=[f for f in required if f not in lookup or not lookup[f]['image_available']],
                all_required_pixels_available_contiguously=covered,
                supports_observed_process_positive=supported,
                binding_value=1 if supported else None,
                scope='Direct physical chain containment only; no exact onset/offset, linguistic extent or word completion.'))
            if supported:supports.append(idx)
        native_truth=1 if any(s['visible_chain']=='yes' for s in t['sequence_links']) else None
        value=1 if supports else None
        rows.append(dict(query_id=meta['query_id'],clip_id=clip,word_id=meta['word_id'],person=person,role=role,
            physical_query=meta['physical_query'],grouping_key=clip,
            source=meta['source'],source_sha256=meta['source_sha256'],input=meta['input'],input_sha256=meta['input_sha256'],
            input_selection_indices=ii,proposal=meta['proposal'],proposal_axis=axis,
            predicate_definitions=t['predicate_definitions'],source_annotation=t['source_annotation'],
            source_annotation_sha256=t['source_annotation_sha256'],
            native_direct_process_evidence=native_truth,observed_process_evidence=value,
            process_supervision_mask=value is not None,supported_sequence_indices=supports,
            sparse_relation_targets=points,unbound_relation_evidence=unbound_points,
            fully_retained_motion_interval_targets=intervals,partial_or_unavailable_motion_evidence=partial,
            sequence_bindings=sequence,
            unlisted_point_label=None,unlisted_motion_label=None,dense_frame_targets_created=0,
            wholly_unlabelled_proposal_output_indices=[x['output_index'] for x in axis if not any(p['output_index']==x['output_index'] for p in points)],
            training_role_eligible=role=='fit',training_executed=False,development_training_allowed=False,
            repeated_development=role=='development',independent_final_test=False,
            whole_word_correctness=None,learner_error=None,required_repetitions=None,
            exact_event_boundaries=None,learner_performed_process_absent=None,
            inference_labels_allowed=False,scope='Targets are for later offline supervised learning/evaluation only. Runtime prediction must not see these labels.'))
    counts={}
    for role in ['fit','development']:
        rr=[r for r in rows if r['role']==role]
        counts[role]=dict(queries=len(rr),unique_source_clips=len({r['clip_id'] for r in rr}),
            native_positive=sum(r['native_direct_process_evidence']==1 for r in rr),
            native_unknown=sum(r['native_direct_process_evidence'] is None for r in rr),
            proposal_positive=sum(r['observed_process_evidence']==1 for r in rr),
            proposal_unknown=sum(r['observed_process_evidence'] is None for r in rr),new_negative_process_labels=0,
            sparse_relation_points=sum(len(r['sparse_relation_targets']) for r in rr),
            relation_component_labels=dict(Counter(str(p[k]) for r in rr for p in r['sparse_relation_targets'] for k in ['setup_relation','terminal_relation'])),
            fully_retained_intervals=sum(len(r['fully_retained_motion_interval_targets']) for r in rr),
            interval_labels=dict(Counter(str(p['label']) for r in rr for p in r['fully_retained_motion_interval_targets'])),
            unbound_relation_points=sum(len(r['unbound_relation_evidence']) for r in rr),
            partial_or_unavailable_intervals=sum(len(r['partial_or_unavailable_motion_evidence']) for r in rr))
    check(counts['fit']['queries']==45 and counts['fit']['native_positive']==44 and counts['fit']['native_unknown']==1,'FIT original direct chain count')
    check(counts['development']['queries']==15 and counts['development']['native_positive']==15,'DEV direct chain count')
    for p,h in PINS.items():check(sha(p)==h,'original pin unchanged '+p)
    obj=dict(status='FIXED_RUNTIME_INPUTS_BOUND_TO_EXISTING_DIRECT_SUPERVISION_NO_TRAINING',
        at=datetime.now(timezone.utc).isoformat(),rows=rows,counts=counts,pins=PINS,
        input_contract=dict(input_context='Existing frozen full-native RGB windows and full-frame MediaPipe geometry, then the original fixed proposal indices. No edited-context feature substitution.',
            future_feature_projection='Existing process_head.make_input contract: first59 body-relative geometry +59 observed masks +768RGB +1image availability; 887 dimensions. No word/person/time-percent/source-index/proposal metadata as numeric inputs.',
            normalization='No new normalization fitted here; any later normalization must use FIT only under a new frozen learning comparison.',
            labels_available_at_runtime=False,actual_source_inputs_rewritten=False),
        preparation_checks=CHECKS,unique_existing_inputs=len(INPUTS),
        source_protocol_unchanged=True,existing_proposals_unchanged=True,scores_read=False,
        no_negative_process_labels_created=True,partial_intervals_not_shortened_or_relabelled=True,
        gaps_not_filled=True,new_proposals=0,new_model_inference=0,new_DTW=0,new_training=0,
        direct_review_is_codex_not_independent_expert=True,performance_improvement_claim=False,
        training_execution_contract_possibility='Same existing full-native feature context and fixed crop can now be used in a future supervised comparison; this does not establish learning success.',
        whole_goal_complete=False)
    with out.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=obj['status'],counts=counts,checks=CHECKS),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'PROCESS_RUNTIME_BINDING_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise

