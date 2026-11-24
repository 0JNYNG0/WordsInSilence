"""Atomic writer for manually viewed blind secondary RGB annotations only."""
from pathlib import Path
import json,sys,os,hashlib,datetime
U=Path(__file__).resolve().parent.parent
out=U/'review/LABELS_SECONDARY.json'
selection=json.loads((U/'review/SECONDARY_SELECTION.json').read_text(encoding='utf8'))
if out.exists():
    data=json.loads(out.read_text(encoding='utf8'))
else:
    data=dict(status='IN_PROGRESS_BLIND_SECONDARY_RGB',annotator='Codex split_head_contract secondary review',
        exposure='Prior model-contract work read old study interface/target definitions and one old metadata row. During this new secondary task: only RUBRIC, SECONDARY_SELECTION, opaque sample RGB images, and own labels; no new primary labels, roles, word labels, source selection, or model predictions.',
        independent_human_truth=False,rows=[],viewed_image_sha256={})
existing={r['sample_id'] for r in data['rows']}
for group in json.load(sys.stdin):
    alias=group['alias']
    assert alias in selection['aliases'] and len(group['windows'])==3
    for w,compact in enumerate(group['windows']):
        sid=f'{alias}_W{w}'
        assert sid not in existing
        r,l,relation,mr,ml,rc,lc,rw,lw,note=compact
        assert all(v in (None,0,1) for vals in (r,l) for v in vals)
        assert len(r)==len(l)==3 and sum(v==1 for v in r)<=1 and sum(v==1 for v in l)<=1
        assert relation in (None,'R_higher','L_higher','level')
        assert all(m in (None,'up','down','horizontal','still','mixed') for m in (mr,ml))
        data['rows'].append(dict(sample_id=sid,center_shape={'R':r,'L':l},center_relation=relation,
            motion={'R':mr,'L':ml},roi_center={'R':rc,'L':lc},roi_window={'R':rw,'L':lw},notes=note))
        existing.add(sid)
    for name in ('states.jpg','window_0.jpg','window_1.jpg','window_2.jpg'):
        p=U/'visuals'/alias/name
        data['viewed_image_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
assert existing<=set(selection['sample_ids'])
data['rows'].sort(key=lambda r:r['sample_id'])
data['updated_kst']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
data['completed_windows']=len(existing)
data['status']='COMPLETE_BLIND_SECONDARY_RGB' if existing==set(selection['sample_ids']) else 'IN_PROGRESS_BLIND_SECONDARY_RGB'
temp=out.with_suffix('.tmp')
temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
os.replace(temp,out)
print(json.dumps({'status':data['status'],'windows':len(existing)}))
