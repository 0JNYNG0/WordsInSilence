import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

U=Path(__file__).resolve().parents[1]
rows=[]
def add(case, default, exceptions, note):
    for k in range(1,14):
        vals=exceptions.get(k,default)
        rows.append({'case_id':case,'image_key':f'K{k:02d}','index_only':vals[0],'open_four':vals[1],'note':note})

# These values were transcribed after actually viewing each shuffled contact sheet.
# No time-index maps, ordered sheets, root labels, or A/B/C results were read.
for case in ['T01','T02','T03']:
    add(case,(None,0),{},'Target R hand has clearly folded non-thumb fingers; the index distal portion/extension is obscured by the other hand or foreshortening. Do not infer index-only from the expected expression.')
add('T04',(0,None),{},'At least two non-thumb fingers appear extended in the side view, excluding index-only; all four finger contours cannot be distinguished reliably.')
add('T05',(0,None),{1:(None,0),5:(1,0),12:(1,0)},'K05/K12 show index extended and the other three folded. K01 is a partly curled transition with index-only unclear. Other views exclude index-only but obscure one or more fingers or show intermediate bending.')
add('T06',(0,None),{7:(0,1)},'Side views show more than one extended finger but not every finger contour; K07 exposes four broadly extended fingers sufficiently for a coarse open-four draft.')
for case in ['T07','T08','T09']:
    add(case,(0,None),{},'Target L resting hand shows multiple downward fingers/curvature rather than an isolated extended index. Overlapping side contours leave open-four uncertain. This is not an error or a required-action judgment.')
for case in ['T10','T11','T12']:
    add(case,(0,1),{},'Target R hand shows four broadly extended adjacent fingers in each panel, including panels where the other hand overlaps below. No exact physical contact inferred.')
add('T13',(0,0),{2:(0,1),6:(0,1),7:(0,1),10:(0,None),11:(0,None)},'Directly visible open states K02/K06/K07; partially spread/bent K10/K11 remain unknown for open-four. Other panels have clearly grouped curled fingers. Index-only absent from visible curled/open evidence.')
add('T14',(0,None),{5:(0,0),6:(0,0),9:(0,0),13:(0,0)},'K05/K06/K09/K13 show a grouped folded configuration. The remaining panels show varying partial finger extension/curling, insufficient for a binary open-four assertion. Do not label the window as one static shape.')
add('T15',(0,0),{1:(0,None),2:(0,None),3:(0,1),8:(0,None),13:(0,None)},'K03 shows broadly spread/extended four fingers. K01/K02/K08/K13 are intermediate curved shapes. Other panels show visibly grouped curled fingers.')
add('T16',(0,1),{7:(0,None),9:(0,0)},'Four extended fingers visible in face-adjacent panels; K07 is an oblique closing/intermediate form and K09 is grouped/curled. No movement label is assigned from one frame.')
add('T17',(0,1),{3:(0,0),4:(0,None),8:(0,0),12:(0,None),13:(0,0)},'K03/K08/K13 visibly grouped into a fist-like form; K04/K12 are partially bending. Remaining panels show four broadly extended fingers.')
add('T18',(0,1),{7:(0,None),8:(0,None),13:(0,0)},'K13 visibly grouped/curled; K07/K08 show intermediate bending/oblique contours. Remaining panels show four broadly extended fingers.')

assert len(rows)==234 and len({(r['case_id'],r['image_key']) for r in rows})==234
assert all(not(r['index_only']==1 and r['open_four']==1) for r in rows)
viewed=[]
for n in range(1,19):
    p=U/'visual_inputs'/f'T{n:02d}'/'shuffled_contact.png'
    viewed.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'panels_actually_viewed':13})
rubric=U.parent/'word100_shared_feature_bundle_20261027'/'RUBRIC.json'
o={'at':datetime.now(timezone.utc).isoformat(),'reviewer':'Codex linguistic_eval_review; same-model consistency reviewer, not an independent human/expert',
 'rubric_sha256':hashlib.sha256(rubric.read_bytes()).hexdigest(),
 'scope':'234 shuffled panels, two coarse direct-RGB target-hand features each. K numbers only; no temporal map or ordered/condition results read before this freeze. Existing case familiarity and protocol metadata mean not fully blind.',
 'not_ground_truth':['hidden finger geometry','exact joint coordinates','sign correctness','natural linguistic tolerance','continuous movement phase'],
 'preservation':'Unknown is not absence and is never filled from source gloss, remembered word form, or neighboring-state expectation.',
 'rows':rows,'files_viewed':viewed,'individual_png_views':0,
 'counts':{feature:{'known':sum(r[feature] is not None for r in rows),'unknown':sum(r[feature] is None for r in rows),'positive':sum(r[feature]==1 for r in rows),'negative':sum(r[feature]==0 for r in rows)} for feature in ['index_only','open_four']}}
out=U/'review'/'IMAGE_LABELS.json'
assert not out.exists(), 'First-pass labels already exist; do not overwrite.'
out.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'review'/'IMAGE_LABELS_FIRST_PASS.json').write_bytes(out.read_bytes())
print(json.dumps({'counts':o['counts'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'rows':len(rows)}))
