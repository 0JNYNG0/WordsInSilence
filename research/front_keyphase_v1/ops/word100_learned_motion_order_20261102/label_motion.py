"""RGB-reviewed physical episodes. Endpoints bound an event, not framewise truth."""
from pathlib import Path
import json,hashlib
import numpy as np
M=Path(__file__).resolve().parent;P=M.parent/'word100_learned_guide_feedback_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
 p=M/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Each tuple was read from chronological public RGB contacts, before predictions.
# (directed event start/end, visually stable start/end). Stable permits body jitter.
WELL=[(49,64,64,73),(45,61,61,72),(50,64,64,74),(74,88,88,98),(44,60,60,68),(42,54,54,60),(47,62,62,72),(72,89,89,93),(50,59,59,68),(52,66,66,75),(65,79,79,84),(50,71,71,82)]
APART=[(38,44,44,54),(55,62,66,78),(68,77,77,81),(87,97,97,108),(63,72,72,80),(66,75,75,84),(44,53,53,58),(54,63,63,72),(68,77,77,81),(58,67,67,76),(57,67,67,76),(62,74,74,80)]
DOWN=[(60,70,76,87),(64,71,84,97),(69,81,88,100),(86,100,114,128),(52,65,78,85),(71,78,92,105),(52,63,69,80),(57,69,81,93),(52,64,71,84),(71,78,90,103),(55,68,86,99),(80,93,100,114)]
# right hand relative to left: rising, falling, stable; anatomical right.
ALTERNATE=[(62,67,72,78,83,94),(56,62,68,74,80,92),(47,53,42,47,70,81),(75,81,68,75,100,113),(62,68,52,62,84,90),(61,66,50,61,76,87),(44,56,56,67,67,73),(78,89,67,78,100,106),(66,72,72,82,88,99),(57,67,52,57,67,72),(57,70,51,57,82,88),(53,60,40,47,79,85)]
TARGETS=['right_hand_toward_left_wrist_along_arm','hands_move_apart','right_hand_rises_relative_left','right_hand_descends_face_to_torso','right_hand_falls_relative_left','continuous_path_evidence']

def main():
 manifest=read(P/'INPUT_MANIFEST.json')['rows'];records=[];people=[1,2,3,4,5,6,7,8,9,11,12,13]
 for w,spec,j in [('WORD1133',WELL,0),('WORD2523',APART,1),('WORD0739',DOWN,3),('WORD1567',ALTERNATE,2)]:
  for person,span in zip(people,spec):
   r=next(r for r in manifest if r['word_id']==w and r['person']==person);a=np.load(r['input'])
   events=[('directed',span[:2],j,1)]
   if w=='WORD1567':events+=[('directed',span[2:4],4,1),('stable',span[4:6],None,0)]
   else:events+=[('stable',span[2:4],j,0)]
   for kind,(lo,hi),target,value in events:
    ix=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi));assert len(ix)==hi-lo+1 and hi>lo
    y=[0.]*6;mask=[False]*6;y[5]=1.;mask[5]=True
    if target is not None:y[target]=value;mask[target]=True
    if w=='WORD1567':
     mask[2]=mask[4]=True
     if kind=='directed':y[target]=1.
    rec=dict(id=f'{r["clip_id"]}_{kind}_{lo}_{hi}',clip_id=r['clip_id'],word_id=w,person=person,role=r['role'],kind=kind,
     frames=[lo,hi],timestamps_us=[int(a['timestamp_us'][ix[0]]),int(a['timestamp_us'][ix[-1]])],y=y,mask=mask,
     source_sha256=r['source_sha256'],input_sha256=r['input_sha256'],review='Codex direct chronological RGB; not independent human/KSL expert gold',
     claim='net physical direction over this observed episode; no exact onset, per-frame class, required repetition count, or linguistic error label')
    records.append(rec)
 save('MOTION_TARGETS.json',dict(targets=TARGETS,rows=records,unreviewed_dimensions_masked=True,stable_means_no_target_travel_not_pixel_immobility=True))
 save('TRAINING_CONTRACT.json',dict(conditions=['orderless_pool','ordered_gru'],seeds=[42,43,44],epochs=120,learning_rate=.001,
  amendment_before_predictions='Add opposite vertical direction explicitly; add continuity supervision for FIT-only removed-middle controls. Original PLAN retained.',
  targets=TARGETS,fit_people=list(range(1,10)),development_people=[11,12,13],checkpoint='final epoch only; no DEV model/threshold/seed choice',
  input='current normalized geometry excluding existing velocities + arm-relative projection + frozen7shape/3relation probabilities, each with validity mask; no goal ID, person, stage, absolute time',
  fine_tuning='No backbone update in this unit; existing previously finetuned SignRep and protected observers retained',
  reversal='Reverse ordered feature episodes as directional augmentation/controlled diagnostic only; validate new actual reversed-video pixels separately',
  middle_omission='Remove middle half of each FIT directed episode; mark continuous evidence0, mask directional truth. Not a natural learner error label.',
  speed='Uniform endpoint-preserving temporal subsampling/repetition 0.8/1.25; continuity1. Original frame IDs remain in audit, not model input.',
  feedback=dict(positive=.8,negative=.2,min_observation_coverage=.8,scan_lengths=[7,13,19,25],fixed_runtime_seed=42),
  completion='Report physical event precision/recall by kind, person; real-video speed, cuts, blackout, reverse; protect old outputs. Do not claim complete feedback from event accuracy.',
  source_labels_sha256=sha(M/'MOTION_TARGETS.json')))
 print('Reviewed episodes',len(records),'fit',sum(r['role']=='fit' for r in records),'dev',sum(r['role']=='development' for r in records))
if __name__=='__main__':main()
