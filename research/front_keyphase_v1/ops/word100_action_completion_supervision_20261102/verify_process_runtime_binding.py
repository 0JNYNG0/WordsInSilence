"""Pure saved-binding verifier: no builder/model imports, no score fields read."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,traceback
import numpy as np
U=Path(__file__).resolve().parent
CHECKS=0;CACHE={};HASHES={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def check(ok,message):
    global CHECKS
    CHECKS+=1
    if not ok:raise AssertionError(message)
def sha(p):
    key=str(Path(p).resolve())
    if key not in HASHES:
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for part in iter(lambda:f.read(1048576),b''):h.update(part)
        HASHES[key]=h.hexdigest()
    return HASHES[key]
def axis_for(meta):
    if meta['input'] not in CACHE:
        with np.load(meta['input'],allow_pickle=False) as z:
            CACHE[meta['input']]={k:z[k] for k in ['frame_id','timestamp_us','image_available']}
    return CACHE[meta['input']]
def main():
    out=U/'PROCESS_RUNTIME_SUPERVISION_BINDING_VERIFIED.json'
    check(not out.exists(),'Refuse repeat completed verification')
    b=read(U/'PROCESS_RUNTIME_SUPERVISION_BINDING.json')
    m=read(U/'PROCESS_AUTO_NATIVE_MANIFEST.json')
    nt={r['id']:r for name in ['PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json'] for r in read(U/name)['rows']}
    nm={r['query_id']:r for r in m['rows']}
    check(len(b['rows'])==len(nt)==len(nm)==60,'60 queries')
    check({r['query_id'] for r in b['rows']}==set(nt)==set(nm),'all fixed queries')
    for p,h in b['pins'].items():check(sha(p)==h,'pin '+p)
    check(b['pins'][str((U/'PROCESS_AUTO_NATIVE_MANIFEST.json').resolve())]==sha(U/'PROCESS_AUTO_NATIVE_MANIFEST.json'),'manifest pin')
    process_counts={};annotation_counts=Counter()
    for r in b['rows']:
        n=nm[r['query_id']];t=nt[r['query_id']]
        for key in ['clip_id','word_id','person','role','physical_query','source','source_sha256','input','input_sha256']:
            check(r[key]==n[key],'identity '+key)
        check(r['source_sha256']==t['source_sha256'],'annotation source binding')
        check(sha(r['input'])==r['input_sha256'],'input pin')
        check(sha(r['source_annotation'])==r['source_annotation_sha256'],'direct annotation pin')
        check(r['proposal']==n['proposal'],'proposal immutable')
        check(r['input_selection_indices']==n['proposal']['output_indices'],'fixed indices')
        a=axis_for(n);ii=n['proposal']['output_indices']
        expected_axis=[dict(output_index=j,full_native_index=i,source_frame=int(a['frame_id'][i]),
                           source_timestamp_us=int(a['timestamp_us'][i]),image_available=bool(a['image_available'][i]))
                       for j,i in enumerate(ii)]
        check(r['proposal_axis']==expected_axis,'original native axis without retiming')
        byframe={z['source_frame']:z for z in expected_axis}
        frame_to_full={int(f):i for i,f in enumerate(a['frame_id'])}
        eligible_points={i for i,p in enumerate(t['relations']) if p['frame'] in byframe and byframe[p['frame']]['image_available']}
        check({p['source_relation_index'] for p in r['sparse_relation_targets']}==eligible_points,'sparse point selection')
        check(len(r['sparse_relation_targets'])==len(eligible_points),'sparse point uniqueness')
        for p in r['sparse_relation_targets']:
            original=t['relations'][p['source_relation_index']]
            check(p['original_annotation']==original,'direct point unchanged')
            check(all(p[k]==byframe[original['frame']][k] for k in expected_axis[0]),'point native/local/time map')
            for component in ['setup_relation','terminal_relation']:
                check(p[component]==original[component],'point label unchanged')
                check(p['component_loss_eligible'][component]==(original[component] is not None),'point unknown loss mask')
            check(p['direct_labels_unchanged'] is True,'point preservation flag')
        outside=set(range(len(t['relations'])))-eligible_points
        check({p['source_relation_index'] for p in r['unbound_relation_evidence']}==outside,'unbound points accounted')
        for p in r['unbound_relation_evidence']:
            original=t['relations'][p['source_relation_index']];f=original['frame']
            check(p['original_annotation_audit_only']==original and p['bound_supervision'] is None and not p['loss_eligible'],'unbound point remains unlabelled')
            check(p['mapped_axis']==byframe.get(f),'unbound point map')
            check(p['source_timestamp_us']==int(a['timestamp_us'][frame_to_full[f]]),'unbound point native time')
        complete=set()
        for i,motion in enumerate(t['motion_intervals']):
            fs=list(range(motion['start'],motion['end']+1))
            available=[f for f in fs if f in byframe and byframe[f]['image_available']]
            if len(available)==len(fs):
                indices=[byframe[f]['output_index'] for f in fs]
                if all(y-x==1 for x,y in zip(indices,indices[1:])):complete.add(i)
        check({v['source_interval_index'] for v in r['fully_retained_motion_interval_targets']}==complete,'whole interval eligibility')
        check(len(r['fully_retained_motion_interval_targets'])==len(complete),'whole interval uniqueness')
        for v in r['fully_retained_motion_interval_targets']:
            original=t['motion_intervals'][v['source_interval_index']]
            fs=list(range(original['start'],original['end']+1))
            check(v['original_annotation']==original and v['label']==original['label'],'interval label unchanged')
            check(v['loss_eligible']==(original['label'] is not None),'interval unknown remains unknown')
            check(v['source_frames']==fs,'whole original interval retained')
            for key,axis_key in [('output_indices','output_index'),('full_native_indices','full_native_index'),('source_timestamps_us','source_timestamp_us')]:
                check(v[key]==[byframe[f][axis_key] for f in fs],'interval '+key)
        partial=set(range(len(t['motion_intervals'])))-complete
        check({v['source_interval_index'] for v in r['partial_or_unavailable_motion_evidence']}==partial,'partial intervals accounted')
        for v in r['partial_or_unavailable_motion_evidence']:
            original=t['motion_intervals'][v['source_interval_index']]
            check(v['original_annotation_audit_only']==original and v['label'] is None and not v['loss_eligible'],'partial not relabelled')
            fs=list(range(original['start'],original['end']+1))
            visible=[f for f in fs if f in byframe and byframe[f]['image_available']]
            flattened=[z for group in v['retained_visible_fragments'] for z in group]
            check(flattened==[byframe[f] for f in visible],'retained partial pixel/time evidence')
            for group in v['retained_visible_fragments']:
                check(all(y['source_frame']==x['source_frame']+1 and y['output_index']==x['output_index']+1 for x,y in zip(group,group[1:])),'partial fragment continuity')
            check(v['missing_source_frames']==[f for f in fs if f not in byframe],'missing interval frames')
            check(v['retained_unavailable_frames']==[byframe[f] for f in fs if f in byframe and not byframe[f]['image_available']],'unavailable interval frames')
        supporting=[]
        check(len(r['sequence_bindings'])==len(t['sequence_links']),'source sequence preservation')
        for i,(link,source) in enumerate(zip(r['sequence_bindings'],t['sequence_links'])):
            check(link['source_sequence_index']==i and link['original_annotation']==source,'sequence unchanged')
            required=set(source['setup_frames'])|set(source['terminal_frames'])
            if source.get('motion_interval'):
                lo,hi=source['motion_interval'];required.update(range(lo,hi+1))
            fs=sorted(required)
            present=bool(fs) and all(f in byframe and byframe[f]['image_available'] for f in fs)
            if present and source.get('motion_interval'):
                lo,hi=source['motion_interval'];indices=[byframe[f]['output_index'] for f in range(lo,hi+1)]
                present=all(y-x==1 for x,y in zip(indices,indices[1:]))
            positive=source['visible_chain']=='yes' and present
            check(link['required_source_frames']==fs,'required source witnesses')
            check(link['required_mapped_axis']==[byframe.get(f) for f in fs],'sequence shared source clock')
            check(link['setup_witnesses']==[byframe.get(f) for f in source['setup_frames']],'setup witness bindings')
            check(link['terminal_witnesses']==[byframe.get(f) for f in source['terminal_frames']],'terminal witness bindings')
            check(link['missing_or_unavailable_source_frames']==[f for f in fs if f not in byframe or not byframe[f]['image_available']],'missing sequence witness accounting')
            check(link['all_required_pixels_available_contiguously']==present,'sequence containment')
            check(link['supports_observed_process_positive']==positive and link['binding_value']==(1 if positive else None),'direct positive or unknown only')
            if positive:supporting.append(i)
        native=1 if any(s['visible_chain']=='yes' for s in t['sequence_links']) else None
        check(r['native_direct_process_evidence']==native,'native direct process value')
        check(r['supported_sequence_indices']==supporting,'positive sequence source linkage')
        check(r['observed_process_evidence']==(1 if supporting else None),'no manufactured negative')
        check(r['process_supervision_mask']==bool(supporting),'process unknown mask')
        check(r['training_role_eligible']==(r['role']=='fit') and r['development_training_allowed'] is False,'split isolation')
        check(r['training_executed'] is False and r['inference_labels_allowed'] is False,'no train/runtime target leakage')
        check(r['dense_frame_targets_created']==0 and r['unlisted_point_label'] is None and r['unlisted_motion_label'] is None,'sparse only')
        check(r['wholly_unlabelled_proposal_output_indices']==[z['output_index'] for z in expected_axis if not any(p['output_index']==z['output_index'] for p in r['sparse_relation_targets'])],'unlisted points stay unknown')
        for key in ['whole_word_correctness','learner_error','required_repetitions','exact_event_boundaries','learner_performed_process_absent']:
            check(r[key] is None,'semantic/absence truth forbidden')
        annotation_counts['bound_points']+=len(eligible_points)
        annotation_counts['unbound_points']+=len(outside)
        annotation_counts['whole_intervals']+=len(complete)
        annotation_counts['partial_intervals']+=len(partial)
    for role in ['fit','development']:
        rr=[r for r in b['rows'] if r['role']==role]
        value=dict(queries=len(rr),unique_source_clips=len({r['clip_id'] for r in rr}),
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
        check(value==b['counts'][role],'count reconstruction '+role);process_counts[role]=value
    check(process_counts['fit']['native_positive']==44 and process_counts['fit']['native_unknown']==1,'FIT44+1variant preserved')
    check(process_counts['development']['native_positive']==15,'DEV15 native positive preserved')
    for key in ['new_proposals','new_model_inference','new_DTW','new_training']:check(b[key]==0,'preparation only')
    check(b['scores_read'] is False and b['performance_improvement_claim'] is False,'no score-selection or performance claim')
    report=dict(status='PASS_STORED_FIXED_RUNTIME_DIRECT_SUPERVISION_BINDING',at=datetime.now(timezone.utc).isoformat(),
        counts=process_counts,annotation_counts=dict(annotation_counts),checks=CHECKS,
        binding_sha256=sha(U/'PROCESS_RUNTIME_SUPERVISION_BINDING.json'),code_sha256=sha(__file__),
        source_manifest_sha256=sha(U/'PROCESS_AUTO_NATIVE_MANIFEST.json'),hashed_files=len(HASHES),
        new_model_inference=0,new_training=0,new_DTW=0,scores_read=False,existing_files_changed=False,
        method='Separate saved-value reconstruction from original direct targets and raw frame/time/availability arrays; builder and model modules not imported.',
        caveats=['Same Codex review and verification is not independent human/KSL expert truth.',
                 'Only existing fixed proposal context is supervised; no boundary/required-count or whole-word correctness labels.',
                 'This binding can support a later same-input supervised comparison but is not performance improvement.',
                 'Unknown variant and partial intervals retain unknown labels; unlisted times are not dense negatives.'],
        whole_goal_complete=False)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],counts=process_counts,checks=CHECKS),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'PROCESS_RUNTIME_BINDING_VERIFY_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise

