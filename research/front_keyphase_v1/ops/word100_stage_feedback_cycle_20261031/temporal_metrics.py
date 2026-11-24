"""Saved-prediction temporal audit. Sparse RGB bounds are not exact event times."""
from learning import *
def main():
 info=read(U/'LEARNING_INPUTS.json');extra=read(U/'review/HELD_TEMPORAL_RGB.json');index={r['alias']:i for i,r in enumerate(info['rows'])};out=[]
 assert extra['aliases']==read(U/'RUNTIME_PROTOCOL.json')['temporal_eval']
 save('TEMPORAL_EVAL_FREEZE.json',dict(labels_sha256=sha(U/'review/HELD_TEMPORAL_RGB.json'),code_sha256=sha(__file__),selection='Predetermined six held clips; no train/threshold changes',threshold=.5,metric='All six observed transition brackets retained; exact event time not known. Adjacent equal RGB sample states tested separately.'))
 for run in read(U/'TRAINING.json')['runs']:
  probs=np.load(U/'runs'/f"{run['condition']}_{run['seed']}"/'probabilities.npy');events=[];clips=[]
  for rr in extra['rows']:
   i=index[rr['alias']];a=np.load(info['rows'][i]['input']);lookup={int(f):j for j,f in enumerate(a['frame_id'])};p=probs[i,:len(a['frame_id'])];pred=p>=.5;correct=known=stable=stable_flips=0
   for fi in extra['features']:
    j=info['targets'].index(fi);frames=rr['frames']
    for z in frames:
     value=z['labels'][fi]
     if value is not None:known+=1;correct+=int(pred[lookup[z['frame_id']],j]==value)
    for before,after in zip(frames[:-1],frames[1:]):
     v1,v2=before['labels'][fi],after['labels'][fi]
     if v1 is not None and v1==v2:
      stable+=1;stable_flips+=int(pred[lookup[before['frame_id']],j]!=pred[lookup[after['frame_id']],j])
   for event in rr['transition_brackets']:
    j=info['targets'].index(event['feature']);lo,hi=[lookup[f] for f in event['frame_bracket']];endpoint=bool(pred[lo,j]==event['from_value'] and pred[hi,j]==event['to_value'])
    changes=np.flatnonzero((pred[:-1,j]==event['from_value'])&(pred[1:,j]==event['to_value']))+1
    within=changes[(changes>lo)&(changes<=hi)];bounds=event['seconds_bracket'];times=a['timestamp_us'][changes]/1e6
    distance=float((np.maximum(bounds[0]-times,0)+np.maximum(times-bounds[1],0)).min()) if len(changes) else None
    events.append(dict(alias=rr['alias'],**event,endpoints_correct=endpoint,predicted_transition_in_bracket=bool(len(within)),both=bool(endpoint and len(within)),nearest_transition_distance_outside_bracket_seconds=distance,predicted_transition_frames=a['frame_id'][within].tolist()))
   clips.append(dict(alias=rr['alias'],known=known,correct=correct,sampled_stable_pairs=stable,predicted_changes_between_equal_observed_samples=stable_flips))
  out.append(dict(condition=run['condition'],seed=run['seed'],events=events,clips=clips,transition_brackets=len(events),both_endpoints_and_bracket=sum(e['both'] for e in events),known_cells=sum(c['known'] for c in clips),correct_cells=sum(c['correct'] for c in clips),sampled_stable_pairs=sum(c['sampled_stable_pairs'] for c in clips),stable_pair_predicted_changes=sum(c['predicted_changes_between_equal_observed_samples'] for c in clips)))
 save('TEMPORAL_SCORES.json',dict(rows=out,truth='Same-Codex sparse RGB physical drafts; not independent KSL semantic stages. No precise change-time labels; no exact timing MAE claim.',negative_only_feature='L_open_four: no positive recall/balanced two-class score'))
 print(json.dumps([{k:r[k] for k in ['condition','seed','transition_brackets','both_endpoints_and_bracket','known_cells','correct_cells','sampled_stable_pairs','stable_pair_predicted_changes']} for r in out]))
if __name__=='__main__':main()
