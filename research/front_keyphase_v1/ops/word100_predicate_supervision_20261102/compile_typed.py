"""Convert reviewed scoped components into typed, masked learning records."""
from collections import defaultdict,Counter
from compile_supervision import U,read,sha,save

TYPES={
 'obedience_closed_hand':('right_non_thumb_closed','point'),
 'obedience_lateral_arm_then_lower':('right_elbow_lowers_hand_remains_raised','sequence'),
 'well_palm_left_arm':('right_hand_at_left_arm_projection','point'),
 'well_brush':('right_hand_traverses_left_arm_toward_wrist','sequence'),
 'ancestor_thumbs':('both_visible_thumbs_up_with_closed_finger_masses','point'),
 'ancestor_alternation':('bilateral_height_alternation','sequence'),
 'egypt_tip_meeting':('bilateral_upper_ends_converged','point'),
 'egypt_oblique_pose':('bilateral_oblique_long_finger_axes','point'),
 'egypt_lower_shared':('source_variant_specific_lowering','sequence'),
 'egypt_gather_suffix':('each_hand_all_fingertips_gathered','point')}

def main():
    b=read(U/'BINDING_WITNESSES.json');s=read(U/'SUPERVISION.json');store={}
    def add(cid,word,person,typ,kind,frames,times,y,mask,origin,clause=None):
        key=(cid,typ,kind,tuple(frames));r=store.get(key)
        if r is None:
            store[key]=dict(clip_id=cid,word_id=word,person=person,role='fit',target=typ,kind=kind,
                native_frames=frames,native_timestamps_us=times,y=y,mask=mask,provenance=[origin],
                source_clause_ids=[] if clause is None else [clause],whole_clause_truth=None,learner_error=None)
        else:
            # Do not silently resolve conflicts or promote an unknown through
            # aggregate support from a different time scope.
            assert r['y']==y and r['mask']==mask,(key,r,origin)
            assert r['native_timestamps_us']==times and r['word_id']==word and r['person']==person,(key,'source/time mismatch')
            r['provenance'].append(origin)
            if clause and clause not in r['source_clause_ids']:r['source_clause_ids'].append(clause)
    for i,r in enumerate(b['rows']):
        typ,kind=TYPES[r['clause_id']];assert (r['kind']=='point')==(kind=='point')
        known=r['support']!='unknown';value=1 if r['support']=='supported' else 0 if known else None
        origin=dict(path=str(U/'BINDING_WITNESSES.json'),pointer=f'/rows/{i}',type='direct_image_component_review')
        groups=list(zip(r['witness_frames'],r['witness_timestamps_us']))
        if kind=='point':
            for frame,ts in groups:add(r['clip_id'],r['word_id'],r['person'],typ,kind,[frame],[ts],value,known,origin,r['clause_id'])
        else:add(r['clip_id'],r['word_id'],r['person'],typ,kind,r['witness_frames'],r['witness_timestamps_us'],value,known,origin,r['clause_id'])
    join={'R_closed_four':'right_non_thumb_closed','R_hand_at_L_arm':'right_hand_at_left_arm_projection','bilateral_upper_ends_converged':'bilateral_upper_ends_converged'}
    for i,p in enumerate(s['points']):
        if p['role']!='fit':continue
        for old,new in join.items():
            j=s['point_targets'].index(old)
            # Unknown stays in original supervision manifest; not combined with
            # a later direct review of a different image scope.
            if not p['mask'][j]:continue
            origin=dict(path=str(U/'SUPERVISION.json'),pointer=f'/points/{i}',target_index=j,type='existing_same_definition_direct_point')
            add(p['clip_id'],p['word_id'],p['person'],new,'point',[p['frame']],[p['timestamp_us']],int(p['y'][j]),True,origin)
    for i,p in enumerate(s['windows']):
        if p['role']!='fit':continue
        j=s['window_targets'].index('right_hand_toward_left_wrist_along_arm')
        if not p['mask'][j]:continue
        # Interval labels remain interval labels; they are not copied into
        # sparse witness or frame-level labels.
        add(p['clip_id'],p['word_id'],p['person'],'right_hand_traverses_left_arm_toward_wrist','interval',p['frames'],p['timestamps_us'],int(p['y'][j]),True,
            dict(path=str(U/'SUPERVISION.json'),pointer=f'/windows/{i}',target_index=j,type='existing_same_definition_direct_interval'))
    rows=list(store.values());coverage={}
    for r in rows:r['dependency_group']=r['clip_id']
    for typ,kind in TYPES.values():
        rr=[r for r in rows if r['target']==typ]
        groups={k:[r for r in rr if ('unknown' if not r['mask'] else 'positive' if r['y'] else 'negative')==k] for k in ['positive','negative','unknown']}
        coverage[typ]={k:dict(records=len(v),recorded_ids=sorted({r['person'] for r in v}),clips=len({r['clip_id'] for r in v})) for k,v in groups.items()}
        coverage[typ]['has_both_fit_classes']=bool(groups['positive'] and groups['negative'])
        coverage[typ]['does_not_prove_new_person_generalization']=True
    result=dict(rows=rows,types={v[0]:dict(source_clause=k,allowed_kinds=sorted({r['kind'] for r in rows if r['target']==v[0]})) for k,v in TYPES.items()},
        coverage=coverage,pins={str(U/n):sha(U/n) for n in ['BINDING_WITNESSES.json','SUPERVISION.json','DIRECT_BINDING_REVIEW.json']},
        target_is_visible_component_not_whole_phrase=True,point_window_witness_scopes_distinct=True,
        negative_means_component_absent_not_whole_sign_wrong=True,unknowns_retained=True,
        no_word_label_as_truth=True,no_generic_reverse_is_negative=True,
        dependency_policy='Multiple points/witness/interval records in one clip are correlated. Weight by clip/event; do not report record count as independent people/events.',
        training_readiness='Only scoped targets with suitable positive/negative support may enter a fixed comparison. New source-binding DEV labels and controls are still required.',
        whole_goal_complete=False)
    previous=read(U/'TYPED_TARGETS.json')
    assert [{k:v for k,v in r.items() if k!='dependency_group'} for r in rows]==previous['rows']
    assert coverage==previous['coverage']
    save('TYPED_TARGETS_v2.json',result)
    save('TYPED_CORRECTION.json',dict(previous=str(U/'TYPED_TARGETS.json'),previous_sha256=sha(U/'TYPED_TARGETS.json'),
        current=str(U/'TYPED_TARGETS_v2.json'),current_sha256=sha(U/'TYPED_TARGETS_v2.json'),
        reason='Separate agent found single-kind schema could mislead a loader about sequence versus interval. Keep original, add allowed_kinds and explicit clip dependency groups; assert timestamps/identity on deduplication.',
        numeric_labels_masks_times_unchanged=True,records=len(rows),independent_clips=len({r['clip_id'] for r in rows}),
        new_training_or_evaluation=False))
    print('typedrecords',len(rows),'bothclasses',[k for k,v in coverage.items() if v['has_both_fit_classes']])
if __name__=='__main__':main()
