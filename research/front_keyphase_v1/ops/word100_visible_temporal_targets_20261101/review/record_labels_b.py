"""Serialize manual RGB rows only; never loads words, roles, selection or predictions."""
from pathlib import Path
import json,hashlib,sys
U=Path(__file__).resolve().parents[1]
p=U/'review/LABELS_B_PROGRESS.json'
d=json.loads(p.read_text(encoding='utf8'))
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
for clip in data:
 alias=clip['alias']
 viewed=[]
 for name in ['states.jpg','window_0.jpg','window_1.jpg','window_2.jpg']:
  q=U/'visuals'/alias/name
  viewed.append(dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),viewed_via='view_image'))
 for w,v in enumerate(clip['windows']):
  sid=f'{alias}_W{w}'
  assert sid not in d['completed_samples']
  row=dict(sample_id=sid,center_shape=dict(R=v[0],L=v[1]),center_relation=v[2],motion=dict(R=v[3],L=v[4]),roi_center=dict(R=v[5],L=v[6]),roi_window=dict(R=v[7],L=v[8]),notes=v[9],viewed_images=[viewed[0],viewed[w+1]])
  d['rows'].append(row);d['completed_samples'].append(sid)
 d['viewed_images'].extend(viewed)
d['status']='IN_PROGRESS' if len(d['rows'])<84 else 'COMPLETE_PRIMARY_RGB_DRAFT'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
if len(d['rows'])==84:
 d['rows'].sort(key=lambda x:x['sample_id'])
 (U/'review/LABELS_B.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
print(len(d['rows']),'/84')
