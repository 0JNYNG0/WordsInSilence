"""Compare frozen image-only wrist intervals with saved public B0/B1 outputs."""
from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent
B=U.parent/'word100_temporal_public_bundle_20261025'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
selection=read(U/'VISIBLE_SAMPLE_SELECTION.json')['cases']
labels=read(U/'VISIBLE_LABELS_FROZEN.json')['labels']
rows={r['clip_id']:r for r in read(B/'MANIFEST.json')}
out=[]
for s,l in zip(selection,labels):
    assert s['sample_id']==l['sample_id']
    r=rows[s['clip_id']];i=s['frame_id'];j=33+s['side']*21
    rec={**s,'image_label':l,'predictions':{}}
    box=l['rectangle'];roi=r['roi_box_source_pixels']
    rec['roi_relation']='unscorable' if box is None else ('outside' if box[1]>=roi[3] or box[3]<roi[1] else ('boundary' if box[3]>=roi[3] else 'inside'))
    with np.load(r['arrays']) as z:
        for cond,pre in [('B0','baseline_roi'),('B1','redetected_roi')]:
            xy=z[pre+'_xy_px'][i,j];use=bool(z[pre+'_usable'][i,j])
            v={'available':use,'xy':xy.tolist() if use else None}
            if use and l['scorable'] and rec['roi_relation']=='inside':
                dx=max(box[0]-xy[0],0,xy[0]-box[2]);dy=max(box[1]-xy[1],0,xy[1]-box[3])
                v['distance_outside_rectangle_px']=float(np.hypot(dx,dy))
                v['consistent_with_coarse_image_box']=bool(dx==0 and dy==0)
            rec['predictions'][cond]=v
    out.append(rec)
summary={c:{'scorable_inside':sum(x['roi_relation']=='inside' for x in out),
    'available':sum(x['roi_relation']=='inside' and x['predictions'][c]['available'] for x in out),
    'box_consistent':sum(x['predictions'][c].get('consistent_with_coarse_image_box',False) for x in out),
    'box_disagree':sum('consistent_with_coarse_image_box' in x['predictions'][c] and not x['predictions'][c]['consistent_with_coarse_image_box'] for x in out)} for c in ['B0','B1']}
with (U/'VISIBLE_BASELINE_CHECK.json').open('x',encoding='utf-8') as f:
    json.dump({'labels_sha256':hashlib.sha256((U/'VISIBLE_LABELS_FROZEN.json').read_bytes()).hexdigest(),
        'summary':summary,'cases':out,'limit':'coarse AI image consistency, not certified joint accuracy'},f,indent=2)
print(json.dumps(summary))
for x in out:print(x['sample_id'],x['roi_relation'],x['predictions'])
