"""Numeric-only export and post-merge ROI leakage checks. No inference."""
import json,hashlib,sys,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(O.parent/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation
spec=importlib.util.spec_from_file_location('checked_roi',O/'word100_fullframe_roi_pair_20261016/roi_adapter.py')
roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
    if isinstance(x,np.generic):return x.item()
    return x
f=read(U/'FROZEN.json')
if not (U/'EXPORT_FIRST_ERROR.json').exists():
    write(U/'EXPORT_FIRST_ERROR.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        error='ValueError: Object arrays cannot be loaded when allow_pickle=False',
        operation='Post-run inspection of outputs/<clip>/channels.npz all members',
        cause='hand_scales dictionary was serialized as an object array; numeric arrays are intact.',
        preservation='Original frozen outputs and INDEX preserved. No detector/training rerun.',
        recovery='Separate model-ready numeric arrays and JSON metadata into safe_channels.'))
out=U/'safe_channels';out.mkdir(exist_ok=True);pins={};points=0;values=0
for r in f['rows']:
    dst=U/'outputs'/r['clip_id'];b=Observation.load(str(dst/'merged_native.npz'));m=Observation.load(str(dst/'masked.npz'))
    features=roi.build_roi_channels(m)
    with np.load(dst/'channels.npz',allow_pickle=False) as z:
        assert np.array_equal(z['x'],features['x'])
        assert np.array_equal(z['frame_id'],features['frame_id']) and np.array_equal(z['timestamp_us'],features['timestamp_us'])
    box=r['box'];p=b.xy_px
    outside=b.observed&~(np.isfinite(p).all(-1)&(p[...,0]>=box[0])&(p[...,0]<box[2])&(p[...,1]>=box[1])&(p[...,1]<box[3]))
    b.xy_px[outside]=[-1e6,-1e6]
    altered=roi.build_roi_channels(roi.apply_roi(b,box))
    assert np.array_equal(features['x'],altered['x'])
    x=features['x'];assert x.ndim==2 and x.shape[1]==280 and np.isfinite(x).all()
    mask=x[:,186:279];assert np.logical_or(mask==0,mask==1).all()
    assert (x[:,:102].reshape(-1,51,2)[mask[:,:51]==0]==0).all()
    assert (x[:,102:186].reshape(-1,42,2)[mask[:,51:]==0]==0).all()
    assert x[0,279]==0 and (x[1:,279]>0).all()
    numeric={k:v for k,v in features.items() if isinstance(v,np.ndarray) and not v.dtype.hasobject}
    meta={k:safe(v) for k,v in features.items() if k not in numeric}
    npz=out/f"{r['clip_id']}.npz";jp=npz.with_suffix('.json')
    assert not npz.exists() and not jp.exists()
    np.savez_compressed(npz,**numeric);write(jp,meta)
    with np.load(npz,allow_pickle=False) as z:
        for k in z.files:assert np.array_equal(z[k],numeric[k])
    pins[str(npz)]=sha(npz);pins[str(jp)]=sha(jp)
    points+=int(outside.sum());values+=x.size
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
for item in read(U/'INDEX.json'):
    assert sha(item['path'])==item['sha256']
    for p,h in item['outputs'].items():assert sha(p)==h,p
write(U/'EXPORT_VERIFY.json',dict(status='PASS',clips=36,channels=280,
    post_merge_outside_points_perturbed=points,channel_values_unchanged=values,
    all_pickle_free=True,metadata_separate=True,original_outputs_unchanged=True,pins=pins,
    qualification='Perturbation is AFTER whole-source hand association, BEFORE ROI/features. It does not claim detector/association never uses outside-ROI source evidence; that is intentional.',
    training=0,classifier_inference=0,source_redetection_reruns=0))
print('EXPORT PASS',points,'outside points;',values,'channel values')
