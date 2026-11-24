"""Read-only research observation interface; no target, word model or DTW verdict."""
from sequence_bundle import FEATURES

def export_window(sequence):
    observations=[]
    for i,frame in enumerate(sequence['frames']):
        contradiction=all(frame['direct'][f]==1 for f in FEATURES)
        observations.append({'frame_id':frame['frame_id'],'pts_us':frame['pts_us'],
          'side':sequence['side'],
          'features':{f:{'source_value':frame['direct'][f],
                         'comparison_value':None if contradiction else frame['direct'][f],
                         'evidence_type':'conflicting_feature_drafts' if contradiction else 'not_observed' if frame['direct'][f] is None else 'image_observation_draft',
                         'is_model_inference':False,'human_verified':False} for f in FEATURES},
          'complete_handshape':None,'semantic_correctness':None})
    return {'side':sequence['side'],'expected_frame_count':len(observations),
            'expected_from_pts_us':observations[0]['pts_us'],'expected_to_pts_us':observations[-1]['pts_us'],
            'observations':observations,'training_ready':False,
            'note':'Existing image-review drafts. This is an interface, not an automatic visual predictor or an executed DTW comparison.'}
