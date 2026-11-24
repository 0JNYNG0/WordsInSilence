"""Append only manually viewed opaque RGB labels; never inspect task selection."""
import json, sys, hashlib, os
from datetime import datetime, timezone
from pathlib import Path
p=Path(__file__).resolve().parent/'LABELS_C.json'
unit=p.parent.parent
incoming=json.loads(sys.stdin.read().lstrip('\ufeff'))
saved=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status':'IN_PROGRESS','annotator':'Codex core_plan_review primary C','truth':'Direct RGB research draft; no gloss/role/model/other-annotator access. Not independent human KSL truth.','rows':[],'viewed_sources':{}}
ids={r['sample_id'] for r in saved['rows']}
for row in incoming:
    assert row['sample_id'] not in ids, 'Append-only primary labels'
    alias,wi=row['sample_id'].split('_W'); i=int(wi)
    assert 43<=int(alias[1:])<=48 and 0<=i<=2
    ready=unit/'visual_ready'/(alias+'.json')
    meta=json.loads(ready.read_text(encoding='utf-8-sig'))
    win=meta['windows'][i]
    assert win['sample_id']==row['sample_id']
    for side in 'RL':
        assert len(row['center_shape'][side])==3 and all(v in (0,1,None) for v in row['center_shape'][side])
        assert row['motion'][side] in ('up','down','horizontal','still','mixed',None)
        assert row['roi_center'][side] in (True,False,None) and row['roi_window'][side] in (True,False,None)
    assert row['center_relation'] in ('R_higher','L_higher','level',None)
    row['direct_image_review']={'states':meta['states'],'window':win['sheet'],'all_13_frames_viewed':True,'center_frame':win['center_frame'],'center_timestamp_us':win['center_timestamp_us']}
    for path in (str(ready),meta['states'],win['sheet']):
        saved['viewed_sources'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    saved['rows'].append(row); ids.add(row['sample_id'])
saved['rows'].sort(key=lambda r:r['sample_id'])
saved['updated_at_utc']=datetime.now(timezone.utc).isoformat()
saved['completed_clips']=len({r['sample_id'].split('_')[0] for r in saved['rows']})
saved['completed_windows']=len(saved['rows'])
saved['status']='COMPLETE_PRIMARY_RGB_DRAFT' if len(saved['rows'])==18 else 'IN_PROGRESS'
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
print(json.dumps({'status':saved['status'],'windows':len(saved['rows']),'clips':saved['completed_clips']}))
