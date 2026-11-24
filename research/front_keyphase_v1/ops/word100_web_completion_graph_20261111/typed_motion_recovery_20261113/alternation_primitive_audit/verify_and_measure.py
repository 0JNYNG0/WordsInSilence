"""Saved contracts and physical readouts AFTER manual RGB labels are frozen.
No inference or label correction: observed displacement is not RGB truth.
"""
from pathlib import Path
import hashlib,json,collections
import numpy as np
U=Path(__file__).resolve().parent;G=U.parent.parent;O=G.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 assert not(U/'SAVED_VERIFICATION.json').exists()
 annotation=U/'CANDIDATE_ANNOTATIONS.json';before=sha(annotation);a=read(annotation);m=read(U/'REVIEW_MANIFEST.json');old=read(U/'INPUT_AUDIT.json')
 assert old['candidate_annotations_sha256']==before and not a['training_allowed']and not a['coordinate_signs_used_to_assign_labels']
 allowed={'NIA_SL_WORD1567_REAL02_F','NIA_SL_WORD1567_REAL04_F','NIA_SL_WORD1567_REAL06_F','NIA_SL_WORD0003_REAL01_F','NIA_SL_WORD0003_REAL03_F','NIA_SL_WORD1290_REAL01_F','NIA_SL_WORD1290_REAL03_F'}
 assert {r['clip_id']for r in m['rows']}==allowed
 points={'L':dict(body_wrist=5,hand_root=9,palm=[9,14,18,22,26]),'R':dict(body_wrist=6,hand_root=30,palm=[30,35,39,43,47])}
 sources={r['clip_id']:r for r in m['rows']};cache={};checks=0;measures=[]
 for p,h in a['pins'].items():assert sha(p)==h,p;checks+=1
 for src in m['rows']:
  z=np.load(src['raw']);ch=np.load(src['channel']);x=ch['full'];ts=z['timestamp_us'];fid=z['frame_id'];p=x[:,:102].reshape(-1,51,2);v=x[:,186:237]>.5
  assert np.array_equal(x,z['raw_x'])and np.array_equal(ts,ch['timestamp_us'])and np.array_equal(fid,ch['frame_id']);checks+=3
  assert np.all(np.diff(ts)>0)and np.array_equal(fid,np.arange(len(fid)));checks+=2
  shoulders=v[:,1]&v[:,2];scale=float(np.median(np.linalg.norm(p[:,1]-p[:,2],axis=1)[shoulders]));assert scale>0
  p=(p-(p[:,1]+p[:,2])[:,None]/2)/scale;cache[src['clip_id']]=(p,v&shoulders[:,None],ts)
  pageframes=[f for page in src['pages']for f in page['frames']];assert pageframes==src['review_frames'];checks+=1
 assert len(a['snapshots'])==189 and len(a['rows'])==34
 for r in a['rows']:
  src=sources[r['clip_id']];assert r['split']=='train'and r['person']in[1,2,3,4,6]and r['source_sha256']==src['source_sha256'];checks+=1
  lo,hi=r['frame_bracket'];p,v,ts=cache[r['clip_id']]
  assert lo<hi and[lo,hi]==[r['reviewed_frames'][0],r['reviewed_frames'][-1]];checks+=1
  assert r['timestamp_bracket_us']==[int(ts[lo]),int(ts[hi])];checks+=1
  if'ambi' in r['R_vertical_motion']or'ambi'in r['L_vertical_motion']:assert r['gross_both_hands_counter_motion']is None;checks+=1
  assert r['completion']is None and r['whole_sign_correct']is None and r['required_repetition_count']is None;checks+=1
  measured={}
  for side in ['R','L']:
   values={}
   for name,ids in points[side].items():
    ids=[ids]if isinstance(ids,int)else ids;eligible=bool(v[[lo,hi]][:,ids].all())
    dy=float(-(p[hi,ids,1].mean()-p[lo,ids,1].mean()))if eligible else None
    values[name]=dict(up_positive_displacement_shoulder_units=dy,endpoints_available=eligible)
   manual=r[side+'_vertical_motion'];body=values['body_wrist']['up_positive_displacement_shoulder_units']
   mismatch=manual in ['up','down']and body is not None and ((manual=='up'and body<=0)or(manual=='down'and body>=0))
   measured[side]=dict(manual_rgb_label=manual,points=values,literal_body_wrist_sign_disagrees=bool(mismatch),
                       interpretation='Diagnostic sign comparison only. No magnitude threshold/classification/label change; palm articulation and point differences can matter.')
  measures.append(dict(id=r['id'],clip_id=r['clip_id'],frame_bracket=[lo,hi],sides=measured))
 assert sha(annotation)==before
 write(U/'PHYSICAL_READOUTS.json',dict(status='POST_ANNOTATION_PHYSICAL_READOUT_NO_LABEL_REWRITE',annotation_sha256=before,rows=measures,
       point_contract=points,shoulder_reference='Per-frame shoulder midpoint; per-source median valid shoulder distance, matching motion.body_points. Up is negative image y.',
       no_model_inference=True,no_threshold_tuning=True,annotations_unchanged=True,
       code_pins={str(O/'word100_motion_components_20261109/motion.py'):sha(O/'word100_motion_components_20261109/motion.py'),str(Path(__file__)):sha(__file__)}))
 disagreements=[dict(id=r['id'],side=s,manual=r['sides'][s]['manual_rgb_label'],body_delta=r['sides'][s]['points']['body_wrist']['up_positive_displacement_shoulder_units'])for r in measures for s in ['R','L']if r['sides'][s]['literal_body_wrist_sign_disagrees']]
 result=dict(status='A_STAGE_SAVED_CONTRACTS_PASS_PENDING_INDEPENDENT_RGB_REVIEW',checks=checks,sources=7,source_split='train only; original probe05/08/09 and DEV untouched',
             sheets=sum(len(r['pages'])for r in m['rows']),reviewed_frames=189,primitive_brackets=34,
             kind_counts=dict(collections.Counter(r['kind']for r in a['rows'])),annotation_sha256=before,
             body_sign_disagreements=disagreements,fully_hidden_hand_examples=0,ambiguous_overlap_frames=27,
             root_review_required=True,independent_human_ksl_gold=False,new_training=False,new_model_inference=False,pins_unchanged=True)
 write(U/'SAVED_VERIFICATION.json',result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
