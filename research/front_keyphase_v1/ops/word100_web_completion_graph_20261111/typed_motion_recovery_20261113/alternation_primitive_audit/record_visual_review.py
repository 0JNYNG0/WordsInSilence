"""Serialize manual RGB observations made AFTER viewing the generated sheets.
The literal entries below are visual judgements, never coordinate-sign labels.
Existing observations provide only PTS, provenance and availability audit fields.
"""
from pathlib import Path
import hashlib,json,collections
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
# [start,end,R motion,L motion,relative height change,higher hand before,after,kind,note]
# Movement is shoulder-relative gross wrist/proximal-palm motion in the image plane.
MANUAL={
 '1567_02':[
  [40,46,'up','up','increasing','R','R','preparatory_motion','Both hands rise; right rises farther. Not an alternating exchange in this bracket.'],
  [48,54,'down','up','decreasing','R','L','motion','Both forearms visibly move; hands cross near frame50.'],
  [58,64,'up','down','increasing','L','R','motion','Right rises while left lowers.'],
  [68,74,'down','up','decreasing','R','L','motion','Left becomes upper hand; right lowers toward chest.'],
  [80,96,'hold','hold','stable','L','L','hold','Gross wrist positions maintained in viewed samples; not a required hold duration.']],
 '1567_04':[
  [62,66,'up','up','increasing','R','R','preparatory_motion','Both hands rise at unequal amounts.'],
  [68,74,'down','up','decreasing','R','L','motion','Visible exchange, crossing near70.'],
  [78,86,'up','down','increasing','L','R','motion','Visible opposing motions; right reaches upper position.'],
  [90,98,'down','up','decreasing','R','L','motion','Visible exchange; proximal hand overlap near92/94.'],
  [104,120,'hold','hold','stable','L','L','hold','Small postural adjustments do not establish another gross exchange.']],
 '1567_06':[
  [48,52,'up','up','increasing','R','R','preparatory_motion','Both hands rise; preparation not an exchange.'],
  [54,58,'down','up','decreasing','R','L','motion','Short exchange, overlap at56.'],
  [60,66,'up','down','increasing','L','R','motion','Right rises from lower position.'],
  [68,76,'down','up','decreasing','R','L','motion','Left rises to forehead; full-context frame68 also inspected to check display crop.'],
  [80,92,'hold','hold','stable','L','L','hold','Gross proximal-hand positions remain separated and stable.']],
 '0003_01':[
  [35,39,'down','up','decreasing','R','L','motion','Pointing hands revolve with clear vertical exchange; no depth/contact label.'],
  [40,45,'up','down','increasing','L','R','motion','Second visible counter-motion.'],
  [46,51,'down','up','decreasing','R','L','motion','Third reviewed exchange; finger articulation is not labelled.'],
  [53,59,'hold','hold','stable','L','L','hold','Both gross hand positions held in these samples.']],
 '0003_03':[
  [53,56,'up','up','increasing','R','R','preparatory_motion','Both hands enter the signing space.'],
  [57,62,'down','up','decreasing','R','L','motion','Visible counter-motion and height-order exchange.'],
  [64,69,'up','down','increasing','L','R','motion','Right rises from below.'],
  [71,76,'down','up','decreasing','R','L','motion','Left becomes upper hand; transition at72/73.'],
  [84,90,'hold','hold','stable','R','R','hold','Later held posture differs from prior exchange endpoint; intervening79..83 not treated as this hold.']],
 '1290_01':[
  [46,51,'down','up','decreasing','R','R','motion','Left also rises slightly toward right. Right stays upper; overlap is not contact proof.'],
  [52,56,'up','down','increasing','R','R','motion','Enlarged frames52/56 show small left lowering; do not force reference hand to hold.'],
  [58,61,'down','ambiguous','decreasing','R','R','motion','Right down is clear; small left translation versus palm rotation is uncertain.'],
  [62,66,'up','ambiguous','increasing','R','R','motion','Right up is clear; left small motion is not given a hard direction/hold label.'],
  [67,79,'ambiguous','ambiguous','ambiguous','R','R','uncertain','Only67/73/79 reviewed here; do not infer a dense hold or no small movement.']],
 '1290_03':[
  [57,60,'down','up','decreasing','R','R','motion','Opposing movement closes gap; right remains above left.'],
  [62,66,'up','down','increasing','R','R','motion','Enlarged60/65 images support a small left lowering; independent review requested.'],
  [67,71,'down','ambiguous','decreasing','R','R','motion','Right down visible; lower hand rotation/translation distinction remains uncertain.'],
  [72,76,'up','down','increasing','R','R','motion','Right rises and lower hand moves down slightly, without higher-hand exchange.'],
  [77,79,'ambiguous','ambiguous','ambiguous','R','R','uncertain','Too short and subtle to assert a hold target.']]
}
# Only proximal hand-region identity/vertical order at the sampled crossing is ambiguous.
# Forearms remain visible. These are not fully hidden hands and not zero-motion labels.
AMBIGUOUS={
 '1567_02':[50,60,70], '1567_04':[70,80,92,94], '1567_06':[56,62,72],
 '0003_01':[36,42,47], '0003_03':[59,66,73],
 '1290_01':[50,51,52,60,61], '1290_03':[60,61,62,70,71,72]
}
def main():
 dest=U/'CANDIDATE_ANNOTATIONS.json';assert not dest.exists()
 m=read(U/'REVIEW_MANIFEST.json');rows=[];audits=[];snapshots=[];pins=dict(m['pins'])
 pins[str(U/'REVIEW_MANIFEST.json')]=sha(U/'REVIEW_MANIFEST.json');pins[str(Path(__file__))]=sha(__file__)
 for source in m['rows']:
  cid=source['clip_id'];key=cid.split('WORD')[1][:4]+'_'+f"{source['person']:02d}"
  assert source['split']=='train'and source['person']not in[5,8,9]
  for page in source['pages']:assert sha(page['path'])==page['sha256'];pins[page['path']]=page['sha256']
  z=np.load(source['raw']);ch=np.load(source['channel']);x=ch['full'];ts=z['timestamp_us'];fid=z['frame_id']
  assert np.array_equal(ch['timestamp_us'],ts)and np.array_equal(ch['frame_id'],fid)and np.array_equal(z['raw_x'],x)
  viewed=set(source['review_frames']);assert set(AMBIGUOUS[key])<=viewed
  for f in source['review_frames']:
   ambiguous=f in AMBIGUOUS[key]
   snapshots.append(dict(clip_id=cid,frame_id=f,timestamp_us=int(ts[f]),gross_forearm_visibility=dict(R='visible',L='visible'),
                         proximal_hand_identity='ambiguous_overlap'if ambiguous else'visible',
                         completely_hidden_hands=[],higher_hand_at_overlap='unlabelled'if ambiguous else'not_individually_labelled',
                         basis='Actual displayed RGB frame; not detector mask. Distal fingers/contact/depth unlabelled.'))
  for i,(a,b,rm,lm,rel,first,last,kind,note)in enumerate(MANUAL[key]):
   assert a in viewed and b in viewed and a<b
   evidence=sorted(f for f in viewed if a<=f<=b)
   indices=np.arange(a,b+1);pointmask=x[indices,186:237]>.5;required=pointmask[:,[1,2,5,6]].all(1)
   rid=f'{cid}:primitive_review:{i}'
   rows.append(dict(id=rid,clip_id=cid,person=source['person'],word_id_metadata_only=source['word_id'],split='train',
                    source_sha256=source['source_sha256'],frame_bracket=[a,b],timestamp_bracket_us=[int(ts[a]),int(ts[b])],
                    reviewed_frames=evidence,kind=kind,R_vertical_motion=rm,L_vertical_motion=lm,
                    relative_height_change=rel,higher_hand_start=first,higher_hand_end=last,
                    endpoint_height_order_changed=bool(first!=last),gross_both_hands_counter_motion=None if 'ambiguous'in(rm,lm)else (rm,lm)in[('up','down'),('down','up')],
                    label_status='candidate_pending_independent_review',dense_frame_labels=False,
                    label_basis='Manual RGB review of listed source frames; no coordinate-sign/model-derived labels',
                    note=note,required_repetition_count=None,whole_sign_correct=None,completion=None))
   audits.append(dict(id=rid,source_frames=len(indices),reviewed_rgb_frames=len(evidence),body_shoulders_and_wrists_available=int(required.sum()),
                      body_availability_fraction=float(required.mean()),hand_roots_available=int(pointmask[:,[9,30]].all(1).sum()),
                      rgb_ambiguous_overlap_frames=[f for f in evidence if f in AMBIGUOUS[key]],
                      source_pts_monotonic=bool(np.all(np.diff(ts[indices])>0)),detector_availability_is_not_rgb_visibility=True))
  pins[source['observation_array']]=source['observation_sha256']
 for p,h in pins.items():assert sha(p)==h,p
 write(dest,dict(status='CANDIDATE_RGB_PRIMITIVES_PENDING_ROOT_AND_INDEPENDENT_REVIEW',reviewer='Codex palm_fixture_contract_1113',
                 actual_visual_review='All20 generated sheet pages /189 RGB frames, enlarged1290_03 frames60/65/70 and1290_01 frames52/56;1567_06 full-context frame68.',
                 coordinate_signs_used_to_assign_labels=False,rows=rows,snapshots=snapshots,pins=pins,
                 training_allowed=False,new_inference=False,private_access=False,probe_relabelled=False,
                 annotation_unit='A bracket of actually viewed source samples, not an exact atomic boundary, continuous per-frame truth, required count or KSL correctness.',
                 uncertainty='Hold is a movement judgement, separate from visibility. No fully hidden hand occurs in this selected subset; hidden class coverage is zero, not proven handling. Turning/crossing ambiguity and small uncertain left-hand motion are preserved.',
                 source1290_correction='Earlier provisional one-hand-only interpretation was withdrawn after enlarged RGB review. A reference hand can move slightly. Gap reversal, counter-motion and higher-hand exchange are separate targets.'))
 write(U/'INPUT_AUDIT.json',dict(status='SAVED_OBSERVATION_AND_PTS_AUDIT_COMPLETE_NO_MODEL',rows=audits,
       point_contract=dict(L_shoulder=1,R_shoulder=2,L_wrist=5,R_wrist=6,L_hand_root=9,R_hand_root=30),
       exact_raw_x_channel_full=True,exact_source_pts=True,max_decoded_pts_error_us=max(r['maximum_pts_error_us']for r in m['rows']),
       sources=7,reviewed_rgb_frames=189,primitive_brackets=len(rows),kind_counts=dict(collections.Counter(r['kind']for r in rows)),
       hidden_source_samples=0,ambiguous_overlap_samples=sum(len(v)for v in AMBIGUOUS.values()),
       R_motion_counts=dict(collections.Counter(r['R_vertical_motion']for r in rows)),L_motion_counts=dict(collections.Counter(r['L_vertical_motion']for r in rows)),
       source_pins_unchanged=True,candidate_annotations_sha256=sha(dest),no_training=True,no_model_inference=True))
 print('Manual RGB candidate brackets',len(rows),'source frames',189,'ambiguous overlap',sum(len(v)for v in AMBIGUOUS.values()))
if __name__=='__main__':main()
