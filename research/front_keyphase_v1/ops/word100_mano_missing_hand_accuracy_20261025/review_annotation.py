"""Preserve initial annotation failure; do not move reference boxes to predictions."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (U/'ANNOTATION_REVIEW.json').exists()
a=json.loads((U/'RGB_REFERENCE_REGIONS.json').read_text('utf-8'))
r=next(r for r in a['items'] if r['id']=='E11');initial=r['tips'].copy()
r['tips']={k:None for k in r['tips']}
r['note_ko']='아래로 모아 편 오른손의 손끝 윤곽은 보이지만 엄지/새끼의 대응 등 손가락 번호를 확정하기 어려움. 앞뒤 원본 28/30/32/34/36프레임 재검토 후 5개 손끝 모두 검증 보류. 손목 범위는 유지.'
a['review']='After scoring revealed a finger-index discrepancy, source-only neighbor images were rechecked. Digit identities remained uncertain. All5 E11 tips are unscorable, not re-labeled or moved to improve scores. This post-score annotation review is disclosed and precludes unbiased accuracy claims.'
save('REVIEWED_REFERENCE_REGIONS.json',a)
save('REVIEWED_REFERENCE_PIN.json',{'path':str(U/'REVIEWED_REFERENCE_REGIONS.json'),'sha256':sha(U/'REVIEWED_REFERENCE_REGIONS.json'),'frozen_before_scoring':False,'wrist_regions':16,'tip_regions':17})
save('ANNOTATION_REVIEW.json',{'at':datetime.datetime.now().astimezone().isoformat(),'status':'INITIAL_FINGER_AGGREGATE_NOT_VALID_AS_ACCURACY','affected':'E11 fingertips4/8/12/16/20 only','initial_regions':initial,'decision':'Unscorable digit identity; all5 held. No candidate selection/gate/coordinates changed. No reference region moved. All16 episode centers retained.',
 'source_context':str(U/'source_views/E11_context.png'),'final_reference':'REVIEWED_REFERENCE_REGIONS.json','final_scores':'REVIEWED_SCORES.json','initial_scores_retained':'SCORES.json','limits':'Same evaluator, after numeric output: biased visual audit risk remains. Report indicative region support, not precision or recognition improvement.',
 'implementation_diagnostic':'First evaluator assertion treated matching NaNs as unequal; all7 saved arrays verified equal_nan=True. Corrected checker only; no input changed.'})
print('Annotation ambiguity preserved and5 points held; wrist and all other regions unchanged')
