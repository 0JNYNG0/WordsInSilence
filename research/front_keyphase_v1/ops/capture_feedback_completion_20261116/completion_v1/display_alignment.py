"""Playback must not stretch one observed action over the full reference."""
def align_observed(query_features,reference_features,query_events,reference_events,dtw):
 count=len(query_events);used=len(reference_features)
 if 0<count<len(reference_events):used=reference_events[count-1]['end']+1
 path=dtw(query_features,reference_features[:used])['path'].tolist()
 return path,dict(scope='Playback only; no completion or missing-action evidence',reference_frames_used=used,
  partial_reference=count>0 and count<len(reference_events),matched_observed_actions=count)
