from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;D=U.parent/'word100_positive_direction_20261109'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
cfg=read(D/'FROZEN.json')['direction'];contracts=read(D/'GUIDE_CONTRACTS.json')['contracts'];rows=[]
for p in sorted((D/'predictions').glob('*.npz')):
    if not (D/'final_decisions'/(p.stem+'.json')).exists():continue
    result=read(D/'final_decisions'/(p.stem+'.json'))['after'];z=np.load(p)
    for c in result['cards']:
        if c['id'] not in contracts:continue
        mask=z['phase'][:,np.array(contracts[c['id']]['phase_groups'])+1].sum(1)>=cfg['phase_support']
        ids=np.flatnonzero(mask); splits=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
        spans=[[int(a[0]),int(a[-1])] for a in splits if len(a)]
        gaps=int((~mask[ids[0]:ids[-1]+1]).sum()) if len(ids) else 0
        rows.append(dict(case=p.stem,card_id=c['id'],selected_span=c.get('query_indices'),direction_span=c['motion_evidence'].get('span'),
            direction_state=c['motion_evidence']['state'],phase_islands=spans,unsupported_frames_inside_direction=gaps,
            prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
out=dict(saved_rows=rows,noncontiguous=sum(r['unsupported_frames_inside_direction']>0 for r in rows),
    cause='First-to-last phase support treats unassigned gaps as one movement; endpoints can cancel or include unrelated motion.',
    new_inference=False,source='Previously exposed public saved predictions',not_semantic_truth=True)
with (U/'DIRECTION_SCOPE_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(cards=len(rows),noncontiguous=out['noncontiguous'],cases=[r for r in rows if r['unsupported_frames_inside_direction']]),ensure_ascii=False))
