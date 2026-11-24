"""Validate B's manually entered RGB draft and record actually viewed extra originals.
No labels are inferred here; no selection, model, word or person metadata is read.
"""
from pathlib import Path
import hashlib,json
from collections import Counter
U=Path(__file__).resolve().parents[1]
p=U/'review/LABELS_B.json'
d=json.loads(p.read_text(encoding='utf8'))
expected=[f'S{i:03d}_W{w}' for i in range(29,57) for w in range(3)]
rows=d['rows']
assert sorted(r['sample_id'] for r in rows)==expected
assert len(set(r['sample_id'] for r in rows))==84
assert sorted(d['completed_samples'])==expected
assert d['rubric_sha256']==hashlib.sha256((U/'RUBRIC.json').read_bytes()).hexdigest()
for w,frame in enumerate([78,90,102]):
 q=U/'visuals/S050'/f'f{frame:04d}.jpg'
 entry=dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),viewed_via='view_image',purpose='Additional exact center original RGB inspection; no coordinate/model reading')
 row=next(r for r in rows if r['sample_id']==f'S050_W{w}')
 if entry['path'] not in [x['path'] for x in row['viewed_images']]:row['viewed_images'].append(entry)
 if entry['path'] not in [x['path'] for x in d['viewed_images']]:d['viewed_images'].append(entry)
for r in rows:
 for side in ['R','L']:
  a=r['center_shape'][side]
  assert len(a)==3
  assert all(v is None or type(v)==int and v in [0,1] for v in a)
  assert r['motion'][side] in ['up','down','horizontal','still','mixed',None]
  for field in ['roi_center','roi_window']:assert r[field][side] is None or type(r[field][side])==bool
 assert r['center_relation'] in ['R_higher','L_higher','level',None]
 assert r['notes'] and len(r['viewed_images'])>=2
 for e in r['viewed_images']:assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()==e['sha256']
assert len(d['viewed_images'])==115
assert len(set(e['path'] for e in d['viewed_images']))==115
for e in d['viewed_images']:
 assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()==e['sha256']
for i in range(29,57):
 for name in ['states.jpg','window_0.jpg','window_1.jpg','window_2.jpg']:
  q=str(U/'visuals'/f'S{i:03d}'/name)
  assert q in [e['path'] for e in d['viewed_images']]
d['scope_note']='28 anonymous clips, 84 windows of 13 consecutive original frames, and exact centers. Reviewed all 112 supplied sheets plus 3 center originals. Overlapping windows are not dense whole-video ground truth. Center static labels are not propagated to neighbors.'
d['roi_note']='ROI flags are separately recorded from visible physical observations; a visible label outside ROI is not permission to supervise. Partial ambiguity is retained as null.'
d['validation_note']='Structural, coverage and evidence hash checks only; not independent semantic label validation or accuracy.'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(U/'review/LABELS_B_PROGRESS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
static=[v for r in rows for s in ['R','L'] for v in r['center_shape'][s]]
out=dict(status='PASS',scope='Manual draft schema/coverage/hash validation only',rows=84,clips=28,sheets_actually_viewed=112,additional_original_frames_actually_viewed=3,unique_viewed_images=115,shape_cells=dict(total=len(static),known=sum(v is not None for v in static),unknown=sum(v is None for v in static)),motion_counts={s:dict(Counter(str(r['motion'][s]) for r in rows)) for s in ['R','L']},relation_counts=dict(Counter(str(r['center_relation']) for r in rows)),labels_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),truth=d['truth'],excluded_reads=d['not_read'])
(U/'review/LABELS_B_VALIDATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
