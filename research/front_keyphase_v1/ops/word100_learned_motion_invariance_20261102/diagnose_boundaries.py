"""Diagnostic intervention using prior reviewed times; never a deployable score."""
import numpy as np
import torch
from evaluate_real_motion import N,M,U,read,exclusive,cases,RealMotionObserver,features,normalize,predict

def main():
 torch.set_num_threads(3)
 labels=[r for r in read(M/'MOTION_TARGETS.json')['rows'] if r['role']=='development' and r['kind']=='directed']
 controls={f"{r['word_id']}_{r['kind']}":r for r in read(U/'VIDEO_CONTROLS.json')['rows']}
 old={(r['seed'],r['case']):r for r in read(N/'COMPARISON.json')['rows'] if r['condition']=='real_variants_motion_inputs'}
 inputs=[]
 for c in cases():
  if c['kind'] not in ['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']:continue
  cid=c['id'] if c['kind']=='native' else f"NIA_SL_{c['base']['word_id']}_REAL12_F"
  mapping=np.asarray(c['a']['frame_id'] if c['kind']=='native' else controls[c['id']]['source_frame_indices'])
  for label in labels:
   if label['clip_id']!=cid:continue
   target=next(j for j in range(5) if label['mask'][j] and label['y'][j])
   ix=np.flatnonzero((mapping>=label['frames'][0])&(mapping<=label['frames'][1]));assert len(ix)>=3
   z,m=features(c['a'],c['shape'],c['relation']);inputs.append((c,label,target,ix,mapping,z,m))
 rows=[]
 for seed in [42,43,44]:
  obs=RealMotionObserver('real_variants_motion_inputs',seed)
  seq=[normalize(z[ix],m[ix],obs.stats) for c,l,t,ix,mp,z,m in inputs];pred=predict(obs.model,seq)
  for (c,l,target,ix,mapping,z,mask),p in zip(inputs,pred):
   before=next(m for m in old[seed,c['id']]['motions'] if m['target']==target)
   available=np.asarray(c['a'].get('image_available',np.ones(len(mapping),bool)))
   cover=float(mask[ix].mean());im=float(available[ix].mean())
   if cover<.5 or im<.8:status='MOTION_OBSERVATION_INSUFFICIENT'
   elif p[5]<.8:status='MOTION_CONTINUITY_UNCONFIRMED'
   elif p[target]>=.8:status='DIRECTED_MOTION_OBSERVED'
   else:status='EXPECTED_MOTION_UNCONFIRMED'
   previous_frames=set(before['frame_ids']);intersection=len(previous_frames.intersection(ix.tolist()))
   rows.append(dict(seed=seed,case=c['id'],kind=c['kind'],target=target,previous_status=before['status'],
    previous_probability=before['probability'],previous_continuity=before['continuity_probability'],
    exact_status=status,exact_probability=float(p[target]),exact_continuity=float(p[5]),
    previous_frames=before['frame_ids'],reviewed_frames=ix.tolist(),source_reviewed_frames=l['frames'],
    reviewed_event_fraction_inside_previous_window=intersection/len(ix),
    previous_window_fraction_inside_reviewed_event=intersection/len(previous_frames)))
 summary={}
 for group,kinds in [('native',['native']),('normal_variants',['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'])]:
  rr=[r for r in rows if r['kind'] in kinds]
  summary[group]=dict(n=len(rr),before_confirmed=sum(r['previous_status']=='DIRECTED_MOTION_OBSERVED' for r in rr),
   exact_confirmed=sum(r['exact_status']=='DIRECTED_MOTION_OBSERVED' for r in rr),
   still_unconfirmed=sum(r['exact_status']!='DIRECTED_MOTION_OBSERVED' for r in rr),
   direction_passing_before=sum(r['previous_probability']>=.8 for r in rr),direction_passing_exact=sum(r['exact_probability']>=.8 for r in rr))
 exclusive(N/'BOUNDARY_INTERVENTION.json',dict(status='DIAGNOSTIC_ONLY',rows=rows,summary=summary,
  manual_reviewed_times_supplied=True,not_deployable_score=True,model_or_threshold_changed=False,
  purpose='Separate an event recognition limitation from learned DTW window placement; no retraining or best-window selection',
  previous_scores_not_replaced=True,not_independent_human_semantics=True))
 print(summary)

if __name__=='__main__':main()
