from pathlib import Path
import json,hashlib,collections
from assemble_evidence import assemble,qualify_path
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
write('ASSEMBLY_FROZEN.json',dict(
 policy='Retain old components, new path, ownership and uncertainty separately. Equal endpoints are not interval maintenance. Cropped-prefix disagreement is not a correction.',
 no_neural_inference=True,no_new_scores=True,no_threshold_change=True,
 pins={str(U/n):sha(U/n)for n in ['assemble_evidence.py','AUTO_INTEGRATION.json','EXTENDED_COMPARISON.json']}))
rows=[assemble(r)for r in read(U/'AUTO_INTEGRATION.json')['rows']]
controls=[dict(clip_id=r['clip_id'],variant=r['variant'],frames=r['source_frames'],evidence=qualify_path(r['evidence']))for r in read(U/'EXTENDED_COMPARISON.json')['rows']]
write('ASSEMBLED_EVIDENCE.json',dict(rows=rows,controls=controls,net_change_not_completion=True,
 no_new_accuracy_claim=True,web_changed=False,remaining='G3 joint semantic completion and false-correction qualification, then sealed evaluation and UI'))
print(json.dumps(dict(rows=len(rows),controls=len(controls),partial_states=[r['evidence']['content_status']for r in controls if r['variant']=='reviewed_partial_prefix'])))
