from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
f=read(U/'FROZEN.json');e=read(U/'EVIDENCE.json');v=read(U/'VERIFICATION.json');r=read(U/'RESULT.json');assert v['status']=='PASS';assert read(U/'CONTRACT_TESTS.json')['status']=='PASS'
h=[q for q in e if q['source']=='hamer'];new=[q for q in h if q['previous_strict']['supported_side']<0 and q['assessment']['identity_proposal']>=0]
assert len(new)==16
# The human-visible review is performed by the same Codex, NOT an independent
# specialist or exact joint-center annotation. These are supporting hypotheses.
review=dict(status='DIAGNOSTIC_REVIEW_COMPLETE_NOT_ADOPTED',operator='Codex same author, viewed all7figure sheets',new_coarse_support_reviewed=16,
 findings='All16 visible arm/hand regions are consistent with proposed anatomical side at this display scale. This does not certify wrist center or hidden fingers. 15 already have full original observations; only WORD0401 REAL06 f36 lacks the original hand, and its HaMeR estimate was already accepted by previousG/GT. No newly recovered coordinates.',
 controls={'known_conflicts10':'All remain identity-unresolved. f82 before74 region8/12points survives, after88 only5/11 survives and track fails; current contours support different roots for old/new hypotheses so contour alone cannot settle identity.',
 'normal_contact_WORD0429_f59':'Region-only right-hand query follows left region; current arm mismatch blocks proposal. Previous strict two-hand support remains archived and unchanged. New region path must not replace old evidence.',
 'overlap_WORD0920_f53':'Right-side support / left side unresolved, same effective state as prior. Hidden hand shape remains unknown.',
 'cross_WORD0401_f57':'New right-side region evidence visually follows right forearm through crossing; no image-left/right rule.'},
 normality_limit='All original output arrays preserved by construction, not proof of improved automatic assignment on new people.',images=[{'path':x['path'],'sha256':sha(x['path'])} for x in read(U/'VISUAL_ITEMS.json')],
 rendering_note='Local scientific image overlays visually inspected. One extraneous drawing-only finger-chain edge removed before inspection; inference and frozen data unaffected. HTML browser render not verified.')
write(U/'VISUAL_REVIEW.json',review)
details=[];missingnew=[]
for q in h:
 a=q['assessment'];old=q['previous_strict'];z=np.load(f['manifest'][q['clip_index']]['prepared']);t=q['frame'];s=q['side'];absent=not z['full_valid'][t,s].any()
 # Retain both records. Do NOT silently replace a qualified precise-point track
 # with a weaker region result, or turn either result into true finger shape.
 item=dict(clip=q['clip'],frame=t,side=s,original_hand_absent=absent,old_point_identity=old['supported_side'],new_region_identity_proposal=a['identity_proposal'],raw_position_components=a,
           old_evidence_retained=True,extra_region_is_advisory=True,finger_truth=False,eligible_for_automatic_coordinate_replacement=False)
 details.append(item)
 if absent and old['supported_side']<0 and a['identity_proposal']>=0:missingnew.append({'clip':q['clip'],'frame':t,'side':s})
write(U/'SEPARATED_EVIDENCE.json',details)
both=sum(q['previous_strict']['supported_side']>=0 and q['assessment']['identity_proposal']>=0 for q in h)
oldonly=sum(q['previous_strict']['supported_side']>=0 and q['assessment']['identity_proposal']<0 for q in h)
assert both==44 and oldonly==19 and len(missingnew)==1
complete=dict(status='POSITION_SHAPE_EVIDENCE_SPLIT_VERIFIED_NO_NEW_COORDINATE_RECOVERY_NOT_DEPLOYED',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),unit=str(U),report=str(U/'RESULT_KO.html'),implementation='Codex local',verification=v,
 counts=dict(public_primary_clips=7,people=4,expressions=5,primary_frames=812,broad_control_scenes=2,hamer_requests=243,missing_original_hand_slots=57,old_strict_supported=63,both_support=44,old_only_support_retained=19,new_coarse_support=16,additional_missing_hand_support=1,newly_recovered_hands=0,unresolved_identity_frames=10,coordinate_changes=0),
 visual_review=str(U/'VISUAL_REVIEW.json'),separated_evidence=str(U/'SEPARATED_EVIDENCE.json'),new_neural_inference=0,training=0,user_data_access=0,production_change=False,
 limitations=['Image region follows pixels, not exact palm center or finger anatomy.','Same diagnostic TRAIN materials and same-author review; not final independent accuracy.','Uses future frames offline. No end-to-end runtime benchmark.','16 new side proposals are diagnostic support, not16 recovered hands; only1 originally missing and already present as oldHaMeR estimate.'],
 next_single_action='')
# Korean authoritative plan, no auto-training/deployment.
complete['next_single_action']='기존10충돌과정상교차·접촉을고정한공개대조에서,각후보손목이현재RGB의실제팔과손이이어지는부위에맞는지확인할평가기준을먼저정리한다. 팔윤곽만이어지는것과정확한손목위치가맞는것을분리하고,가려져확인할수없는점은범위/불확실로남긴다. 이번영역보조는기존정밀근거를대체하지않고따로보존한다. 정답없는점에문턱을낮춰강제채택하거나같은장면만반복튜닝하지않는다. C·T3-F/운영유지,사용자·멘토촬영사용금지,추가학습·운영DTW연결없음.'
write(U/'COMPLETION.json',complete)
print(json.dumps(complete['counts'],ensure_ascii=False))
