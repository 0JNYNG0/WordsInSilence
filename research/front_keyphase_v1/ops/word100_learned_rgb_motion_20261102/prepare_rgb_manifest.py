"""Reuse existing public RGB representations; no extraction or evaluation."""
from pathlib import Path
import sys,numpy as np
R=Path(__file__).resolve().parent;O=R.parent;N=O/'word100_learned_motion_invariance_20261102';Q=O/'word100_learned_motion_path_20261102';P=O/'word100_learned_guide_feedback_20261102'
sys.path.insert(0,str(N));from prepare_real_fit import read,sha,exclusive,clock

def main():
 contract=dict(at=clock(),goal='Keep demonstrated relative-motion gain while testing direct existing RGB evidence for remaining motion failures',
  architectural_fact='Current motion head sees geometry plus7shape and3relation probabilities, but not the768-dimensional SignRep video representation directly',
  hypothesis_not_result='Direct image-motion context may resolve normal uncertain motions; could also introduce appearance shortcuts, which must be tested',
  conditions=['no_rgb_replay','rgb_residual'],seeds=[42,43,44],epochs=120,lr=.0003,primary='rgb_residual',deployment_seed=42,
  base=str(Q),base_condition='separated_relative',data=str(N/'DATASET.json'),data_sha256=sha(N/'DATASET.json'),
  design='Preserve two encoders. Add a zero-initialized768-to32 RGB residual before each temporal encoder activation. Identical additional fine-tuning in no-RGB control, with RGB projection frozen at zero. Both initially reproduce Q outputs exactly.',
  known_masks_and_image_availability_preserved=True,normalization='FIT only, unique reviewed native frames within each existing input',
  no_new_model_or_download=True,no_new_backbone_update=True,no_word_or_signer_input=True,unknown_targets_excluded=True,
  temporal_control_limit='Reversing representation order is a training control, not actual re-extraction of reversed RGB; actual reversed-video evaluation remains separate.',
  check_before_training=['initial output identity','image absence cannot create positive evidence','padding and mask correctness','FIT-only normalization','direction/continuity separate gradients'],
  comparison='Reuse frozen baseline results; same288public comparisons and same predeclared promotion gate; all seeds reported. Do not modify threshold, seed, DTW window or labels to pass.',
  heldout='ID14/15 eight public clips still locked; not train, tune or candidate selection. Past TCN exposure disclosed.',
  restrictions='C/T3-F and8804-8806 preserved; user/mentor recordings never opened for training/tuning/selection; no new external download or transfer')
 exclusive(R/'PLAN.json',contract)
 old=read(P/'INPUT_MANIFEST.json')['rows'];new=read(N/'INPUT_MANIFEST.json')['rows'];sources={}
 for r in old:
  if r['role']=='fit':sources[r['clip_id']+'_cached']=dict(input=r['input'],person=r['person'],clip_id=r['clip_id'])
 for r in new:sources[r['clip_id']+'_'+r['kind']]=dict(input=r['input'],person=r['person'],clip_id=r['clip_id'])
 ds=read(N/'DATASET.json');selected={}
 for r in ds['rows']:selected.setdefault(r['source_key'],set()).update(r['frame_ids'])
 assert len(sources)==144 and set(selected)==set(sources)
 total=0;s=np.zeros(768);sq=np.zeros(768);manifest=[]
 for key,row in sources.items():
  a=np.load(row['input']);rgb=a['rgb'];assert rgb.shape==(len(a['frame_id']),768) and np.isfinite(rgb).all()
  ix=np.array(sorted(selected[key]));assert np.array_equal(a['frame_id'][ix],ix)
  available=a['image_available'] if 'image_available' in a else np.ones(len(rgb),bool);q=ix[available[ix]]
  values=rgb[q].astype('float64');total+=len(q);s+=values.sum(0);sq+=(values*values).sum(0)
  manifest.append(dict(key=key,**row,input_sha256=sha(row['input']),reviewed_frames=ix.tolist(),available_reviewed_frames=q.tolist()))
 mean=s/total;std=np.sqrt(np.maximum(sq/total-mean**2,0));std=np.maximum(std,1e-4)
 exclusive(R/'RGB_NORMALIZATION.json',dict(mean=mean.tolist(),std=std.tolist(),fit_only=True,frames=total,zero_observation_policy='missing RGB remains masked, never replaced by a goal-expected feature'))
 exclusive(R/'INPUT_READY.json',dict(status='READY_PUBLIC_CACHED_RGB_ONLY',rows=manifest,files=144,fit_recorded_ids=list(range(1,10)),
  training_events=81,augmented_windows=1044,training_done=False,new_inference_done=False,private_access=False))
 print('Ready144existing FIT feature arrays; no new extraction, training or private access')

if __name__=='__main__':main()
