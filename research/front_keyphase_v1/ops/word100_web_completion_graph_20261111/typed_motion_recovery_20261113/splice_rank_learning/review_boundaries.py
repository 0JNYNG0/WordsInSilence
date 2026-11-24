"""Interpret saved J boundaries against existing labels, never tune or relabel."""
import collections
import numpy as np
from prepare import U,H,G,O,W,L,read,write,sha

def main():
    dest=U/'evaluation/BOUNDARY_REVIEW.json';assert not dest.exists()
    result=read(U/'evaluation/COMPARISON.json');pts=read(U/'evaluation/PTS_ORDER_AUDIT.json')
    annotations=G/'reviews/sequence_target/MOTION_EXTENTS.json';dev=next(r for r in read(annotations)['rows']if r['clip_id']=='NIA_SL_WORD0739_REAL11_F')
    rp=O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json'
    replay=next(r for r in read(rp)['rows']if r['clip_id']==dev['clip_id']and r['kind']=='speed125')
    assert sha(replay['path'])==replay['sha256'] and sha(replay['raw_source'])==replay['raw_source_sha256']
    raw=np.load(replay['path']);original=np.load(replay['raw_source']);mapping=np.asarray(replay['source_indices'])
    assert np.array_equal(raw['source_frame_id'],mapping)
    assert np.array_equal(raw['xy_px'],original['xy_px'][mapping]) and np.array_equal(raw['score'],original['score'][mapping])
    record=[];pins={str(p):sha(p)for p in [annotations,rp,U/'evaluation/COMPARISON.json',U/'evaluation/PTS_ORDER_AUDIT.json',__file__]}
    pins.update({replay['path']:replay['sha256'],replay['raw_source']:replay['raw_source_sha256']})
    for row in result['rows']:
        if row['clip_id']!=dev['clip_id']or row['kind']!='speed125':continue
        pr=next(p for p in result['predictions']if(p['seed'],p['clip_id'],p['kind'])==(row['seed'],row['clip_id'],row['kind']))
        query=np.load(pr['source']);assert np.array_equal(query['ts'],raw['timestamp_us'])
        for mode in ['local_full','splice_rank']:
            for c in row['conditions'][mode]['cards']:
                motion=dev['motions']['B'if c['family']==2 else'C'];span=c['chosen']['span'];source_span=mapping[span].tolist()
                onset=motion['onset_uncertainty']['source_span_inclusive'];offset=motion['offset_uncertainty']['source_span_inclusive']
                record.append(dict(seed=row['seed'],condition=mode,card=c['card'],query_span=span,source_span=source_span,
                    onset_band=onset,offset_band=offset,onset_in_band=onset[0]<=source_span[0]<=onset[1],offset_in_band=offset[0]<=source_span[1]<=offset[1],
                    offset_beyond_band_source_frames=max(0,source_span[1]-offset[1]),
                    original_source_witnesses=motion['definite_moving_witness']['source_frame_ids'],
                    labels_scope='Existing exposed DEV uncertainty diagnostic, not exact/KSL gold; never used in training/runtime.'))
    fit=[]
    for path in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',W/'additional_source_review/SOURCE_EPISODES_ADDED.json']:
        pins[str(path)]=sha(path)
        for row in read(path)['rows']:
            if '0739'not in row['clip_id']or row['person']not in range(2,8):continue
            events={r['family']:r for r in row['events']};a,b=events[2],events[3]
            gap=b['onset'][0]-a['offset'][1]
            fit.append(dict(clip_id=row['clip_id'],person=row['person'],A_onset=a['onset'],A_offset=a['offset'],B_onset=b['onset'],B_offset=b['offset'],
                earliest_B_onset_minus_latest_A_offset_frames=gap,
                relation_of_uncertainty_envelopes='shared_boundary_possible'if gap==0 else'positive_gap',
                A_witnesses=a['progression_witness_frames'],B_witnesses=b['progression_witness_frames']))
    assert len(fit)==6 and all(r['earliest_B_onset_minus_latest_A_offset_frames']>=0 for r in fit)
    pmap={(r['seed'],r['clip_id'],r['kind'],r['condition']):r for r in pts['relations']}
    individual=[]
    for row in result['rows']:
        before=row['conditions']['local_full'];after=row['conditions']['splice_rank'];changes=[]
        for a,b in zip(before['cards'],after['cards']):
            sa=a['chosen']['span']if a['chosen']else None;sb=b['chosen']['span']if b['chosen']else None
            changes.append(dict(card=a['card'],family=a['family'],old_span=sa,new_span=sb,span_changed=sa!=sb,
                old_present=a['chosen']is not None,new_present=b['chosen']is not None,
                old_all_witness=a.get('truth',{}).get('all_witness'),new_all_witness=b.get('truth',{}).get('all_witness'),
                old_wrong_points=a.get('truth',{}).get('wrong_points'),new_wrong_points=b.get('truth',{}).get('wrong_points')))
        k=(row['seed'],row['clip_id'],row['kind']);oldorder=pmap.get((*k,'local_full'));neworder=pmap.get((*k,'splice_rank'))
        individual.append(dict(seed=k[0],clip_id=k[1],kind=k[2],cards=changes,
            old_pts_order=oldorder['timestamp_extent_state']if oldorder else None,
            new_pts_order=neworder['timestamp_extent_state']if neworder else None))
    write(dest,dict(status='BOUNDARY_DIAGNOSIS_NO_RETUNING',mapping_checked='Raw xy/score and source_frame_id exactly equal original indexed observations; runtime query PTS equals saved speed replay PTS.',
        speed125_records=record,fit_boundary_pairs=fit,
        conclusion='New raise onset80 and offset88 are inside existing bands80..82/88..90. Existing AND new close ends81, one source frame outside old offset77..80. New33ms overlap combines an unchanged slightly extended close with a permitted earlier raise onset; it is not a synthetic seam or reversed core. It is also not evidence that two gold full extents truly overlap.',
        uncertainty='All6 FIT paired envelopes are separated or touch; 2 permit a common endpoint, none labels positive-duration concurrence. Primary-action dense labels do not establish physical-axis exclusivity. Full argmax extent overlap may reflect boundary uncertainty but cannot simply be declared correct.',
        next_design='Keep raw candidates and semantics unchanged. A conditional distributional order diagnostic may examine alternative boundaries; raw rank softmax is not calibrated action presence. Missing/same-occurrence ambiguity requires abstention, not guide-order forcing. No computation of that new diagnostic in this report.',
        individual246_rows=individual,pins=pins,no_training=True,no_inference=True,no_new_labels=True))
    print('Boundary review complete; raw mapping verified;6 FIT pairs;246 individual rows')

if __name__=='__main__':main()
