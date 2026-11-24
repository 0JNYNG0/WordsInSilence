from core import *
import copy
p=W/'short_action_review/REVIEW.json'
assert not p.exists()
rows=read(W/'short_action_review/INPUTS.json')['rows']
extra={5:[65,67],6:[71,73],8:[57,59],9:[69,71]}
for r in rows:
 assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256']
 r['viewed']=True;r['additional_progression_frames']=extra[r['person']]
 e=r['original_event'];assert all(e['onset'][1]<f<e['offset'][0] for f in r['additional_progression_frames'])
 r['observation']='Consecutive RGB shows both hands separating and lowering from the joined triangular presentation toward the later held position. Added only strictly interior progression witnesses; no boundary, handshape correctness or required completion relabeling.'
write(p,dict(at=stamp(),status='REVIEWED_SOURCE_PROGRESS_ONLY',reviewer='root Codex visual inspection of four consecutive-frame public RGB sheets; not independent human/KSL expert truth',rows=rows,unchanged_source_annotation=True,semantic_completion=None))
