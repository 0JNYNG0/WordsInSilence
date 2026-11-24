"""Saved input effects only: distinguish unchanged landmarks from new normalization."""
import json,sys,importlib.util
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(O.parent/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation
spec=importlib.util.spec_from_file_location('feature_effect_roi',O/'word100_fullframe_roi_pair_20261016/roi_adapter.py')
roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
def read(p):return json.loads(Path(p).read_text('utf8'))
rows=[]
for r in read(U/'FROZEN.json')['rows']:
    a=roi.build_roi_channels(Observation.load(r['old_masked']))
    b=roi.build_roi_channels(Observation.load(str(U/'outputs'/r['clip_id']/'masked.npz')))
    assert np.array_equal(a['x'][:,:102][a['body_valid'].repeat(2,axis=1).astype(bool)],
                          b['x'][:,:102][a['body_valid'].repeat(2,axis=1).astype(bool)])
    valid=a['hand_valid'].repeat(2,axis=1).astype(bool)
    delta=np.abs(a['x'][:,102:186]-b['x'][:,102:186])
    scales=[]
    for k in a['hand_scales']:
        x,y=float(a['hand_scales'][k]),float(b['hand_scales'][k])
        scales.append(dict(side=k,before=x if np.isfinite(x) else None,after=y if np.isfinite(y) else None,
                           changed=bool(not (x==y or (np.isnan(x) and np.isnan(y))))))
    rows.append(dict(clip_id=r['clip_id'],label=r['label'],hand_scales=scales,
        existing_local_values_changed=int(((delta>0)&valid).sum()),
        max_existing_local_delta=float(delta[valid].max()) if valid.any() else 0))
out=dict(status='SAVED_FEATURE_EFFECT_MEASURED',
    changed_scale_sides=sum(t['changed'] for r in rows for t in r['hand_scales']),
    changed_scale_clips=sum(any(t['changed'] for t in r['hand_scales']) for r in rows),
    existing_local_values_changed=sum(r['existing_local_values_changed'] for r in rows),
    existing_body_relative_values_unchanged=True,
    explanation='Adding valid palms can change the sequence hand-size statistic. Existing source XY/identity stay fixed, but some normalized local values change. This is not a measured recognition regression.',
    rows=rows)
(U/'FEATURE_EFFECT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print({k:v for k,v in out.items() if k!='rows'})
