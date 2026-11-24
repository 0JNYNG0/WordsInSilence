"""Visual labels recorded before viewing candidate outputs; expert truth not claimed."""
import json,hashlib
from pathlib import Path
from datetime import datetime
U=Path(__file__).resolve().parent
selection=json.loads((U/'REVIEW_SELECTION.json').read_text('utf-8'))
outside={6,7,8,16,17,18,37,38,39,40,41,42}
overlap={3,10,11,27,28}
inside={9,21,22,23,26,31,32,33,34}
labels=[]
for item in selection['items']:
    n=int(item['review_id'][1:])
    category=('boundary_or_outside' if n in outside else 'inside_overlap' if n in overlap
              else 'inside_other' if n in inside else 'observed_control')
    assert (category=='observed_control')==(item['type']=='observed_control')
    labels.append({**item,'visual_category':category,'origin':'additional12_blind_review'})
known=[('NIA_SL_WORD0500_REAL06_F',55,68,'inside_overlap'),
       ('NIA_SL_WORD1642_REAL01_F',83,98,'boundary_or_outside'),
       ('NIA_SL_WORD1884_REAL03_F',85,98,'boundary_or_outside'),
       ('NIA_SL_WORD0003_REAL01_F',35,35,'inside_other')]
for cid,a,b,label in known:
    for f in range(a,b+1):labels.append({'review_id':f'K_{cid}_{f}','clip_id':cid,'side':0,'frame':f,'visual_category':label,'origin':'prior_known_gap_review'})
assert not (U/'PREDICTIONS.json').exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'at':datetime.now().astimezone().isoformat(timespec='seconds'),'labels':labels,
 'predictions_seen':False,'operator':'Codex root visual inspection; no expert annotations or second rater',
 'interpretation':'boundary includes hand cut by crop line, not necessarily entire hand outside. overlap describes spatial contact/overlap, not proven sole cause of detection loss. Frames of one gap are correlated.',
 'control_note':'R029 left hand is partly across boundary but existing palm5 usable. Controls test preservation, not physical fully-visible correctness.',
 'pins':{p.name:sha(p) for p in U.glob('review_*.jpg')},
 'prior_review_sha256':sha(U.parent/'word100_overlap_sequence_20261017/VISUAL_REVIEW.json')}
with (U/'VISUAL_LABELS.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('frozen',len(labels),'including additional missing',len(outside|overlap|inside))
