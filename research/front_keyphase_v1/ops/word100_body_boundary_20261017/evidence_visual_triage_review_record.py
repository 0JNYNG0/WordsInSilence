"""Record actual source-image review, separating observed facts and inferences."""
import json
import numpy as np
from evidence_visual_triage import U, read, sha, IDS

reviews = {
    'NIA_SL_WORD1000_REAL02_F': {
        'facts': ['At f83–90 (2.767–3.000s), both arms/hands are down beside the body below the fixed line; the source core annotation still includes these frames.',
                  'At f55 (1.833s) and f75 (2.500s), the right RGB hand is visibly above the line but lacks a legacy hand overlay; left hand is down and removed by ROI.',
                  'At sampled f43–74, the raised right hand is above the line. f78 onward begins partial clipping during lowering toward the resting posture.'],
        'inference': 'The longest zero-hand period appears to be ending/rest included in the annotation. The two one-frame zero-hand periods combine inherited right-hand detector loss with intended left-down exclusion, rather than the ROI cutting off the visible right hand.',
        'remaining_uncertainty': 'The linguistic extent of the lowering after f74 is not adjudicated; source-visible right-hand detector misses still need separate consideration.'},
    'NIA_SL_WORD0759_REAL07_F': {
        'facts': ['f47–61 shows both hands raised near the upper chest/chin, well above the line.',
                  'f66–73 shows both hands lowering and crossing the line. At f74–80 (2.467–2.667s), both are down beside the body.',
                  'The core annotation ends at 2.690s, so the final down posture is within its labeled interval.'],
        'inference': 'The reported longest absence appears to represent ending/rest after the raised two-hand action, not loss of the raised hand configuration.',
        'remaining_uncertainty': 'A sparse source review cannot certify every finger configuration between sampled raised frames or linguistically define the exact action endpoint.'},
    'NIA_SL_WORD1133_REAL06_F': {
        'facts': ['f38–61 shows hands interacting above the line at chest height.',
                  'f67–69 begins lowering; f75–77 shows arms down and residual wrist points near the line.',
                  'f78–83 (2.600–2.767s) shows both hands down below the line and falls within the 2.770s annotation endpoint.',
                  'At f35 and f69, a right RGB hand above the line lacks legacy hand overlay, separate from ROI clipping.'],
        'inference': 'The longest zero-hand interval appears to be final lowering/rest included in the annotation; raised two-hand action remains in the reviewed upper-body region.',
        'remaining_uncertainty': 'Inherited hand misses and unsampled contact/shape details are not validated by ROI geometry.'},
    'NIA_SL_WORD0760_REAL07_F': {
        'facts': ['f44–59 shows both hands raised near chest/chin above the line.',
                  'f65–73 shows lowering; f76–80 (2.533–2.667s) has both hands down at the sides.',
                  'The annotation endpoint is 2.673s, so it includes the down posture.',
                  'During f69–73 the right hand is source-visible while legacy hand overlay is absent; this is not caused by removing points below the ROI.'],
        'inference': 'The longest loss appears to occur after the main raised configuration while returning to rest.',
        'remaining_uncertainty': 'Whether every transition contributes to the lexical distinction requires a sign-knowledgeable review; no word-specific boundary change is justified here.'},
    'NIA_SL_WORD1085_REAL06_F': {
        'facts': ['f33 and f38 show RGB hands raised above the line with absent legacy hand overlays; f47 has both raised hands detected.',
                  'f70–85 shows the right hand moving down and then held in front of the waist, remaining fully inside ROI at all reviewed samples (21/21 right points).',
                  'As the right hand resumes lowering at f86–91, retained right points decrease 10,7,5,3,2,1 of 21, respectively.',
                  'f92–96 (3.067–3.200s) shows both arms/hands down at the sides; the core annotation ends at 3.232s.'],
        'inference': 'The low held configuration appears preserved; subsequent partial clipping and complete absence appear to accompany the return to rest. Therefore some moving fingers are geometrically removed, but this is not evidence that the held low configuration was erased.',
        'remaining_uncertainty': 'The exact semantic endpoint of the downward movement remains unconfirmed. Do not infer semantic safety merely from 21/21 points at the held posture.'}
}

items=[]
for cid in IDS:
    p=U/f'evidence_visual_triage_{cid}_v2.json'
    meta=read(p)
    for image_path,expected in meta['images'].items():assert sha(image_path)==expected
    item={'clip_id':cid,'label':meta['label'],'bottom':meta['bottom'],
          'boundary_method':meta['method'],'core_seconds':meta['core_seconds'],
          'source_sha256':meta['source_sha256'],'observation_sha256':meta['observation_sha256'],
          'reviewed_images':meta['images'],'metadata':str(p),'metadata_sha256':sha(p),
          'PTS_max_error_us':meta['PTS_alignment_verified_max_error_us'],
          'all_hand_loss_runs':meta['triage_row']['all_hand_loss_runs'],**reviews[cid]}
    if cid=='NIA_SL_WORD1085_REAL06_F':
        supplement=read(U/'evidence_visual_triage_LOW_TRANSITION.json')
        for image_path,expected in supplement['images'].items():assert sha(image_path)==expected
        item['reviewed_supplement_images']=supplement['images']
        item['supplement_json_sha256']=sha(U/'evidence_visual_triage_LOW_TRANSITION.json')
    items.append(item)

result={'status':'FIVE_PRIORITY_TRAIN_SOURCES_VISUALLY_REVIEWED_NOT_GLOBAL_SEMANTIC_PASS',
        'reviewer':'Codex /root/roi_data_budget', 'scope':'Original RGB and legacy observed/usable landmarks only; no new extraction model execution.',
        'summary_facts':['All five longest zero-hand intervals occur with both hands visibly lowered at the sides near the annotation endpoint.',
                         'Some short zero-hand or missing-side events are inherited detector misses while the RGB hand remains above the boundary.',
                         'The boundary does partially hide hands during raising/lowering; partial loss exists and must not be described as zero geometric loss.',
                         'In the additionally examined low-waist hold of 북구주민지원센터, sampled right hand points remain 21/21 through f85; loss begins during subsequent lowering.'],
        'review_conclusion':'These five reviewed worst total-hand gaps do not demonstrate that the ROI removed a raised/held required sign configuration. This supports distinguishing ending/rest and inherited detector loss from action truncation, but does not certify all 899 TRAIN clips, all 619 flagged clips, or linguistic sufficiency.',
        'limits':['Only these five priority clips were reviewed; they are not a random representative sample.',
                  'Overview is sparse; gap windows include adjacent source frames. Unsampled partial finger changes may remain.',
                  'Continuous sign semantics and exact action endpoints have no expert ground truth in this review.',
                  'New joint-v3 assignment and actual cropped-sensor results can differ from these legacy observations.',
                  'No word-specific condition, threshold, boundary, annotation, training membership or model was changed.'],
        'display_correction':'v1 images and JSON preserved. v2 corrects misleading both-hands-outside caption to no-valid-hand-after-ROI and computes it outside core too. All v2 images listed here were actually viewed.',
        'rows':items,'new_detector_runs':0,'training_runs':0,'classifier_runs':0}
with (U/'evidence_visual_triage_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'reviewed_clips':len(items)},ensure_ascii=False))
