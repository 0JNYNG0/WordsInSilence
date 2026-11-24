"""Preserve earlier imprecise RGB annotations; no new coordinate selection."""
from common import *
from collections import Counter

def main():
 p=O/'word100_mano_missing_hand_accuracy_20261025/REVIEWED_REFERENCE_REGIONS.json';d=read(p)
 items=[]
 for r in d['items']:
  partially='partly' in r['wrist_visibility'];blur=r['motion_blurred']
  category='occlusion_bounded_region' if partially else ('blurred_connection_region' if blur else 'visible_connection_region')
  items.append(dict(id=r['id'],category=category,reference_region=r['wrist'],visible_tip_regions=sum(v is not None for v in r['tips'].values()),exact_skeletal_wrist_ground_truth=False,automatic_coordinate_adoption=False))
 assert len(items)==16 and next(x for x in items if x['id']=='E05')['category']=='occlusion_bounded_region'
 result=dict(status='EVIDENCE_CRITERION_RECORDED_NOT_NEW_ACCURACY_VALIDATION',at=now(),source_pin={str(p):sha(p)},categories=dict(Counter(x['category'] for x in items)),items=items,
 rules=[
 'Record anatomical hand identity separately from image side and from detector slot; hand crossing is valid.',
 'A visible arm-to-hand transition supplies an approximate 2D region, not an exact hidden skeletal joint or 3D angle.',
 'When partly occluded or blurred, retain a bounded region if supported; do not replace it with model-predicted ground truth.',
 'A candidate inside the region is compatible only; this alone does not prove correct identity, finger articulation or depth.',
 'Track region, wrist, each fingertip and each angle as different evidence. Unverifiable coordinates stay uncertain, never zero-error.',
 'Conflicting candidates and original observations are retained. No score-driven threshold relaxation or word-specific rule.',
 'If exact accuracy matters, obtain independently marked visible anatomy from public video, with uncertainty and reviewer disagreement. Do not derive truth from MP/HaMeR agreement alone.'
 ], unresolved='The same10 sponsor frames remain conflicting; E05 frame82 wrist is partly occluded. Existing regions cannot settle exact wrist/identity recovery. No new hand inputs.', inherited_controls='Normal and crossing input preservation is inherited unchanged from the previous verified unit, not a new detector benchmark.', relationship_to_E3='E3 changes output aggregation only; every input coordinate and visibility channel remains the existing fixed C/T3F input.', training=0,new_inference=0,implementer='Codex; metadata reclassification of prior same-Codex visual review, not new independent expert labels')
 write(U/'WRIST_EVIDENCE_CRITERION.json',result)
 write(U/'PERSON_IDENTITY_LIMITS.json',dict(at=now(),correction_to_duplicate_audit='The earlier generic suggestion to inspect representatives does not authorize face matching. No face-based identity comparison was performed or will be used.',known='4exact byte-identical videos under IDs16and11.',unknown='Whether remaining distinct videos depict independent real people.',action='Quarantine recordedID16 beforefreshscores; describeIDs14/15 asrecordedIDs; actualpersonidentityrequiresprovider/recordingmetadata.',face_matching=False))
 print(json.dumps({'status':result['status'],'categories':result['categories'],'new_coordinate_adoptions':0},ensure_ascii=False))
if __name__=='__main__':main()
