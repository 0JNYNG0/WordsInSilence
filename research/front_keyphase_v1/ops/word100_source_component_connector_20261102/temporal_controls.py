"""Constructed temporal checks with explicit semantics, not learner-error gold."""
from copy import deepcopy

DIRECTIONAL={
 'right_elbow_lowers_hand_remains_raised',
 'right_hand_traverses_left_arm_toward_wrist',
}
ALTERNATION='bilateral_height_alternation'

def controls(row):
    """Keep the original source-frame mapping apart from monotone playback time.

    Input is a reviewed positive ordered witness sequence. Missing visual input
    is deliberately unknown. Reversed alternating heights still alternate.
    These checks are correlated with their parent clip, never new people.
    """
    assert row['kind']=='sequence' and row['mask'] and row['y']==1
    target=row['target']; assert target in DIRECTIONAL|{ALTERNATION}
    frames=row['native_frames']; ts=row['native_timestamps_us']
    assert len(frames)>=3 and len(frames)==len(ts)
    assert all(a<b for a,b in zip(frames,frames[1:]))
    assert all(a<b for a,b in zip(ts,ts[1:]))
    origin=[t-ts[0] for t in ts]; end=origin[-1]
    specs=[
      ('speed_075',list(range(len(frames))),[round(t/.75) for t in origin],1,True,True,
       'Same ordered images played more slowly; no frame is removed.'),
      ('speed_125',list(range(len(frames))),[round(t/1.25) for t in origin],1,True,True,
       'Same ordered images played faster; no frame is removed.'),
      ('reverse',list(reversed(range(len(frames)))),[end-t for t in reversed(origin)],
       0 if target in DIRECTIONAL else 1,True,True,
       'Reverses the witnessed directed action; a height alternation remains an alternation.'),
      ('still', [len(frames)//2]*len(frames),origin,0,True,True,
       'Constructed repetition of one still image has no observed movement; not a natural error example.'),
      ('image_gap',list(range(len(frames))),origin,None,False,False,
       'No visual evidence for this span; unknown, not action absent or action correct.'),
    ]
    output=[]
    for name,order,time,y,mask,available,why in specs:
        r=deepcopy(row)
        r.update(variant=name,source_native_frames=[frames[i] for i in order],
          source_native_timestamps_us=[ts[i] for i in order],
          playback_timestamps_us=time,y=y,mask=mask,
          image_available=[available]*len(order),
          synthetic_control=True,natural_learner_error_gold=False,
          educational_error_label=None,control_rationale=why,
          materialization_required=True,
          cached_context_reindex_is_actual_video=False,
          permitted_materialization=[
            'Transform raw RGB frames and recompute all16-frame SignRep/context features for this transformed input',
            'Current-frame geometry-only ordered-feature control, explicitly excluding cached temporal RGB/latent and actual-video performance claims'],
          speed_control_scope='Playback timing contract; actual extractor sampling must be matched before end-to-end latency/accuracy claims',
          dependency_group=row['clip_id'],source_extent_frames=[frames[0],frames[-1]],
          source_extent_timestamps_us=[ts[0],ts[-1]])
        # Native labels remain identifiable as the source witnesses. Do not
        # silently reuse them as if they were the transformed sequence order.
        r.pop('native_frames');r.pop('native_timestamps_us')
        assert all(a<b for a,b in zip(time,time[1:]))
        output.append(r)
    return output

def preflight():
    for target in sorted(DIRECTIONAL|{ALTERNATION}):
        r=dict(clip_id='control-contract-only',target=target,kind='sequence',mask=True,y=1,
               native_frames=[11,15,20,26],native_timestamps_us=[366667,500000,666667,866667])
        v={x['variant']:x for x in controls(r)}
        assert v['reverse']['source_native_frames']==[26,20,15,11]
        assert v['reverse']['y']==(1 if target==ALTERNATION else 0)
        assert v['image_gap']['y'] is None and not v['image_gap']['mask']
        assert v['image_gap']['source_extent_frames']==[11,26]
        assert len(set(v['still']['source_native_frames']))==1
        assert v['speed_075']['source_native_frames']==r['native_frames']
        assert v['speed_125']['source_extent_timestamps_us']==[366667,866667]
        assert all(x['materialization_required'] and not x['cached_context_reindex_is_actual_video'] for x in v.values())
    return dict(passed=True,reversal_semantics_distinct=True,
        source_and_playback_time_separate=True,missing_is_unknown=True,
        no_model_inference_or_training=True)
if __name__=='__main__':print(preflight())
