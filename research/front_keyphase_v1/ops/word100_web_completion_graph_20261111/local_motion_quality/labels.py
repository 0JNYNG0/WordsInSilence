"""Public interval annotations -> conservative extent labels; no model scores."""


def event_id(row, index):
    return f"{row['clip_id']}|axis{row['axis']}|{row['moving']}|event{index}"


def assign_extent(a, b, axis, moving, annotations):
    if b <= a:
        return -1, [], 'single_observation_no_temporal_extent'
    rows = [r for r in annotations if r['axis'] == axis and r['moving'] == moving]
    if not rows:
        return -1, [], 'axis_or_side_unannotated'
    nonmotion = [(lo,hi) for r in rows for lo,hi in r['holds'] + r['background']]
    events = [(e,event_id(r,i)) for r in rows for i,e in enumerate(r['events'])]
    complete = [eid for e,eid in events if e['onset'][0] <= a <= e['onset'][1] and e['offset'][0] <= b <= e['offset'][1]]
    inside = [eid for e,eid in events if e['onset'][0] <= a <= b <= e['offset'][1]]
    contained = [eid for e,eid in events if a <= e['onset'][0] and b >= e['offset'][1]]
    hold = any(lo <= a <= b <= hi for lo,hi in nonmotion)
    if hold and (complete or inside):
        return -1, sorted(set(complete+inside)), 'annotation_conflict'
    if hold:
        return 0, [], 'reviewed_hold_or_background'
    if len(complete) == 1:
        return 2, complete, 'reviewed_endpoint_ranges'
    if len(complete) > 1:
        return -1, complete, 'overlapping_complete_annotations'
    if len(inside) == 1:
        return 1, inside, 'within_one_reviewed_event'
    if len(inside) > 1:
        return -1, inside, 'overlapping_event_annotations'
    if len(set(contained)) >= 2:
        return 3, sorted(set(contained)), 'multiple_reviewed_event_identities'
    if contained and any(max(a,lo) <= min(b,hi) for lo,hi in nonmotion):
        return -1, contained, 'movement_plus_nonmotion_mix_not_new_class3_gold'
    return -1, contained, 'unreviewed_or_uncertain_extent'


def direct_candidates(annotations):
    out=[]
    for row in annotations:
        axis,moving=row['axis'],row['moving']
        for i,e in enumerate(row['events']):
            # These are interval-supported labels, not newly reviewed dense RGB.
            for a in sorted(set(e['onset'])):
                for b in sorted(set(e['offset'])):
                    out.append(dict(a=a,b=b,axis=axis,moving=moving,source='reviewed_endpoint_range'))
            a,b=e['onset'][0],e['offset'][1]
            mid=(a+b)//2
            if a < mid < b:
                out += [dict(a=a,b=mid,axis=axis,moving=moving,source='derived_within_event_partial'),dict(a=mid,b=b,axis=axis,moving=moving,source='derived_within_event_partial')]
        for a,b in row['holds']+row['background']:
            out.append(dict(a=a,b=b,axis=axis,moving=moving,source='reviewed_nonmotion'))
        es=sorted(row['events'],key=lambda e:e['onset'][0])
        for left,right in zip(es,es[1:]):
            out.append(dict(a=left['onset'][0],b=right['offset'][1],axis=axis,moving=moving,source='two_reviewed_events'))
    return out


def validate_aux_admission(admission, annotations, frame_count):
    required=('clip_id','axis','moving','event_index','left_source_frame','right_source_frame','observed_removed_frames','rgb_evidence','source_sha256')
    if any(k not in admission for k in required):
        raise ValueError('aux admission missing required provenance')
    if admission.get('admission') != 'approved_source_edit_aux_only':
        raise ValueError('witness review is not an auxiliary admission')
    if admission.get('reviewed_mid_motion') is not True:
        raise ValueError('explicit middle-motion witness required')
    rows=[r for r in annotations if r['clip_id']==admission['clip_id'] and r['axis']==admission['axis'] and r['moving']==admission['moving']]
    if len(rows)!=1:
        raise ValueError('aux relation does not map uniquely to frozen FIT annotations')
    row=rows[0];e=row['events'][int(admission['event_index'])]
    left,right=int(admission['left_source_frame']),int(admission['right_source_frame'])
    if not (0 <= e['onset'][1] <= left < right-1 <= e['offset'][0]-1 < frame_count):
        raise ValueError('delete endpoints must be inside certain reviewed motion interior')
    witness=admission['observed_removed_frames']
    if not witness or not all(left < int(i) < right for i in witness):
        raise ValueError('removed middle witness missing or outside deleted interior')
    if row['source_sha256'] != admission['source_sha256']:
        raise ValueError('aux source hash mismatch')
    return row,e,left,right
