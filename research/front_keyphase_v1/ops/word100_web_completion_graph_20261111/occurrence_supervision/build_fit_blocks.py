"""FIT-only block/index definitions from admitted sparse RGB labels.

Does not open source videos, models, DEV inputs, or prediction files. Reads the
six existing FIT frame_id arrays only to check source/sequence indexing.
"""
from pathlib import Path
import json, hashlib, datetime, collections
import numpy as np

H=Path(__file__).resolve().parent;U=H.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

files=[U/'reviews/obedience_occurrence_labels.json',U/'reviews/egypt_occurrence_labels.json',H/'DIRECT_ROLE_AUDIT.json',H/'ROOT_ADMISSION.json']
pins={str(p):sha(p) for p in files}
ob,eg,audit,admission=[read(p) for p in files]
assert admission['approved_for_isolated_fit'] and admission['audit_sha256']==pins[str(H/'DIRECT_ROLE_AUDIT.json')]
for p in audit['source_label_pins']:
    if p['path'] in pins:assert pins[p['path']]==p['sha256']
records={r['clip_id']:r for r in audit['clip_summaries']}
sources={r['clip_id']:r for r in ob['clips']+eg['rows']}
assert len(sources)==6 and all(('_REAL01_' in c or '_REAL04_' in c or '_REAL07_' in c) for c in sources)
assert all(r.get('split',r.get('role')) in ['FIT','fit'] for r in sources.values())
assert set(records)==set(sources)
assert all(r['clip_id'] in sources for r in audit['rows'])
accepted_conflicts=set(admission['accepted_conflicts'])

# Each cut is an existing lower end of an uncertainty band, NOT an invented
# dense physiological endpoint. All ambiguous frames remain unlabelled.
cuts={
    'NIA_SL_WORD0739_REAL01_F':[46,60,72,92],
    'NIA_SL_WORD0739_REAL04_F':[66,86,102,130],
    'NIA_SL_WORD0739_REAL07_F':[36,50,64,78],
    'NIA_SL_WORD2523_REAL01_F':[27,39,55],
    'NIA_SL_WORD2523_REAL04_F':[69,87,106],
    'NIA_SL_WORD2523_REAL07_F':[32,45,59],
}
rows=[]
for cid,src in sources.items():
    meta=records[cid];assert src['source_sha256']==meta['source_sha256']
    path=Path(meta['frame_id_array_path']);assert sha(path)==meta['frame_id_array_sha256']
    with np.load(path,allow_pickle=False) as z:ids=z['frame_id'].astype(np.int64).tolist()
    assert ids==list(range(meta['source_frame_count'])),cid
    n=len(ids);fps=float(src['fps']);assert fps==30
    is_ob='WORD0739' in cid
    eps=src['occurrences'] if is_ob else src['episodes']
    main=[e for e in eps if e['role']=='main'];return_ep=next(e for e in eps if e['role'] in ['return','finishing'])
    onset='start_boundary_uncertainty_frames' if is_ob else 'onset_frame_range'
    expected=[e[onset][0] for e in main]+[return_ep[onset][0]]
    assert cuts[cid]==expected,(cid,cuts[cid],expected)
    names=['prefix','A','B','C','suffix'] if is_ob else ['prefix','A','B','suffix']
    boundaries=[0]+cuts[cid]+[n]
    blocks={name:dict(source_span=[lo,hi-1],source_frame_ids=list(range(lo,hi))) for name,lo,hi in zip(names,boundaries,boundaries[1:])}
    order=['prefix','A','C','B','suffix'] if is_ob else ['prefix','B','A','suffix']
    q=[f for name in order for f in blocks[name]['source_frame_ids']];inverse={f:i for i,f in enumerate(q)}
    ts=[round(i*1_000_000/fps) for i in range(n)]
    for name,b in blocks.items():
        b['query_span']=[inverse[b['source_span'][0]],inverse[b['source_span'][1]]]
        b['role_extent_status']='mechanical source chunk, not a dense role label or whole-action correctness assertion'
        if name not in ['prefix','suffix']:
            ep=main[['A','B','C'].index(name)];b['source_occurrence_id']=ep['occurrence_id'];b['source_clause_id']=ep.get('guide_clause',ep.get('clause_id'))
            b['semantic_complete']=None
    bands=meta['boundaries'];excluded_boundary={f for b in bands for f in range(b['range'][0],b['range'][1]+1)}
    # Sensitivity only, never substituted for the parent-confirmed mask.
    strict_boundary={f for b in bands for f in range(max(0,b['range'][0]-4),min(n-1,b['range'][1]+4)+1)}
    joins=[];cursor=0;seam_excluded=set()
    for left_block,right_block in zip(order,order[1:]):
        cursor+=len(blocks[left_block]['source_frame_ids']);ql,qr=cursor-1,cursor;sl,sr=q[ql],q[qr]
        synthetic=sr!=sl+1
        # At30fps the midpoint guard removes four original sample centres on
        # either side. Using both end-frame timestamps would double-count it.
        cut_time_us=(ql+0.5)*1_000_000/fps
        removed=[i for i in range(n) if synthetic and abs(i*1_000_000/fps-cut_time_us)<=133333]
        seam_excluded.update(removed)
        joins.append(dict(left_block=left_block,right_block=right_block,query_left=ql,query_right=qr,source_left=sl,source_right=sr,synthetic_discontinuity=synthetic,seam_midpoint_us=cut_time_us,guard_each_side_us=133333,excluded_query_indices=removed))
    normal=[];eligible=[];excluded=[];strict_eligible=[]
    for r in audit['rows']:
        if r['clip_id']!=cid or r['category']=='uncertain':continue
        f=r['source_frame_id'];assert ids[r['sequence_index']]==f and r['source_sha256']==src['source_sha256']
        if r['category']=='conflict':assert f"{cid}:{r['sequence_index']}" in accepted_conflicts
        family=int(r['proposed_family']);ev=[v for v in r['direct_occurrence_evidence'] if v['family']==family]
        assert ev
        item=dict(source_frame_id=f,source_sequence_index=r['sequence_index'],query_index=inverse[f],family=family,weight=1,occurrence_id=ev[0]['occurrence_id'],source_occurrence_ids=sorted({v['occurrence_id'] for v in ev}),source_audit_category=r['category'],source_audit_subtype=r['subtype'],source_images=r['source_images'],teacher_origin='existing_root_admitted_direct_RGB_sparse_w1_only')
        normal.append(dict(item,query_index=r['sequence_index']))
        reasons=[]
        if family not in range(1,8):reasons.append('synthetic_context_does_not_preserve_preparation_return_or_background_truth')
        if f in excluded_boundary:reasons.append('existing_RGB_uncertainty_band')
        if inverse[f] in seam_excluded:reasons.append('synthetic_splice_plus_minus_133333us')
        if reasons:excluded.append(dict(item,exclusion_reasons=reasons))
        else:
            eligible.append(item)
            if f not in strict_boundary:strict_eligible.append(item)
    count=collections.Counter(str(r['family']) for r in eligible)
    unmasked_main=collections.Counter(str(r['family']) for r in normal if r['family'] in range(1,8))
    row=dict(clip_id=cid,record_id=src.get('person',src.get('record_id')),split='FIT',source=src['source'],source_sha256=src['source_sha256'],fps=fps,frame_origin=0,source_frame_count=n,source_frame_ids=ids,query_to_source_frame_ids=q,query_to_source_sequence_indices=q,query_timestamp_us=ts,blocks=blocks,permutation=order,splices=joins,boundary_uncertainty=bands,boundary_excluded_source_frame_ids=sorted(excluded_boundary),splice_excluded_query_indices=sorted(seam_excluded),normal_control_sampled_labels=normal,eligible_sampled_labels=eligible,excluded_sampled_labels=excluded,teacher_family_counts=dict(sorted(count.items())),source_admitted_main_family_counts=dict(sorted(unmasked_main.items())),all_expected_main_families_have_teacher=all(str(1+i if is_ob else 6+i) in count for i in range(len(main))),status='READY_LOCAL_ROLE_CONTEXT_AUGMENTATION' if eligible else 'DEFERRED_NO_ELIGIBLE_SAMPLED_MAIN_LABELS',source_label_coverage='Only previously sampled/admitted RGB frames carry labels. Boundary brackets define uncertain physical extents, not exact dense boundaries. No new RGB observation or unseen-frame interpolation was made.',boundary_selection='Each new incoming block starts at the lower endpoint of its already reviewed onset uncertainty band. Its uncertain/precondition frames are retained as raw context but never newly labelled. Suffix starts at the lower endpoint of reviewed return-onset band; it may initially retain the final held posture. This keeps global return at the end.',complete_action_extent='Uncertain boundary microframes are not certified complete semantic actions. Reviewed progress and terminal-core context are kept inside ascending source chunks; only local sparse roles are taught. Not a whole-action-normal or whole-sign label.',materialization=dict(raw_permutation_required=True,recompute_frozen_observer_latent=True,shuffle_original_context_latent_forbidden=True,preserve_raw_handshape_and_within_block_time_direction=True,masked_query_indices_have_no_supervised_or_teacher_loss=True,whole_sign_or_order_correctness_loss_allowed=False,word_id_as_model_feature_allowed=False),source_integrity=dict(declared_hash_matches_occurrence_labels=True,declared_hash_matches_prior_RGB_audit=True,source_video_opened_or_modified=False,source_video_bytes_rehashed_this_task=False,consumer_must_rehash_source_before_materialization=True,fit_frame_id_array=str(path),fit_frame_id_array_sha256=meta['frame_id_array_sha256'],frame_id_only_read=True,feature_or_model_arrays_read=False),stricter_boundary_sensitivity=dict(active=False,definition='Additionally expand every natural boundary band by four frames; diagnostic sensitivity only, not active mask.',eligible_sampled_labels=[dict(source_frame_id=r['source_frame_id'],query_index=r['query_index'],family=r['family']) for r in strict_eligible],teacher_family_counts=dict(collections.Counter(str(r['family']) for r in strict_eligible))))
    assert sorted(q)==ids and len(q)==len(set(q))
    assert all(b['source_frame_ids']==list(range(b['source_span'][0],b['source_span'][1]+1)) for b in blocks.values())
    assert all(r['source_frame_id'] not in excluded_boundary and r['query_index'] not in seam_excluded and 1<=r['family']<=7 for r in eligible)
    assert all(len(j['excluded_query_indices'])==8 for j in joins if j['synthetic_discontinuity'])
    rows.append(row)

assert all(sha(p)==v for p,v in pins.items())
out=dict(schema_version=1,status='FIT_LOCAL_ROLE_CONTEXT_BLOCK_DEFINITIONS_READY_WITH_EXPLICIT_DEFERRED_CASES',created_at=datetime.datetime.now().astimezone().isoformat(),author='Codex sequence_target_review',scope='Six already reviewed public FIT recordings01/04/07 of obedience and Egypt; no new training/inference or original RGB read.',source_pins=[dict(path=p,sha256=h) for p,h in pins.items()],mask_contract=dict(existing_boundary_bands='Exclude the original inclusive bands only; do not add133ms around natural boundaries.',synthetic_splices='Exclude sample centres within±133333us of the cut midpoint. At30fps this removes4frames per side of each discontinuous join. Natural source-adjacent joins get no synthetic guard.',source_teacher='Only already admitted direct sampled w1 family1..7. Original0/8/9 remain normal-control metadata but never synthetic teacher. All other frames have no supervision/teacher loss.',correctness='No whole-action correctness, semantic sequence correctness, whole-sign normal/wrong, or duration target is assigned.',context_scope='Ascending raw source chunks preserve their frames and internal direction; temporal encoder latent must be recomputed on the new raw sequence.'),review_disclosure=dict(no_fresh_RGB_review=True,no_DEV_prediction_files_read=True,no_model_outputs_read=True,incidental_historical_DEV_numbers_in_code_search=True,incidental_text_use='Not used to choose boundaries, masks, labels or cases. Not a fully blinded independent study.',independent_human_expert_validation=False),rows=rows,summary=dict(clips=len(rows),ready=sum(r['status'].startswith('READY') for r in rows),deferred=sum(r['status'].startswith('DEFERRED') for r in rows),eligible_sparse_samples=sum(len(r['eligible_sampled_labels']) for r in rows),family_counts=dict(collections.Counter(str(s['family']) for r in rows for s in r['eligible_sampled_labels'])),raw_source_reads=0,model_reads=0,training_runs=0,feature_inference_runs=0,source_files_modified=0))
target=H/'FIT_BLOCKS.json';assert not target.exists();write(target,out)
check=dict(status='PASS_FIT_LABEL_PROVENANCE_AND_INDEX_PRESERVATION',target=str(target),target_sha256=sha(target),source_label_files_unchanged=True,raw_source_preservation='No source videos were opened or written; declared SHA compared to prior audited records, not freshly rehashed. Consumer must enforce raw hash before encoding.',all_source_frames_exactly_once=True,all_blocks_internally_ascending=True,no_new_frame_labels=True,all_teacher_frames_admitted_and_outside_masks=True,all_teachers_main_family_only=True,all_sources_FIT_01_04_07=True,no_DEV_array_or_prediction_read=True,no_models_or_new_inference=True,rows=[dict(clip_id=r['clip_id'],status=r['status'],teacher_family_counts=r['teacher_family_counts'],samples=len(r['eligible_sampled_labels'])) for r in rows])
write(H/'FIT_BLOCKS_CHECK.json',check)
print(json.dumps(dict(summary=out['summary'],rows=check['rows']),ensure_ascii=False))
