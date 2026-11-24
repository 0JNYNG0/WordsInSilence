"""Public-only contracts and fixed cropped-webcam proxy preparation. No inference."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,math,ast
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_multipeople_live_feedback_20261101'
sys.dont_write_bytecode=True
sys.path.insert(0,str(O.parent/'src'))
import adapter
from sensor_boundary import calibrate,diagnostic
from wis_lab.core.observation import Observation
observer=adapter._module('_seated_observer_validation',O/'word100_guide_video_feedback_20261028/motion_observer.py')
guard=adapter._module('_seated_guard_validation',B/'evidence_guard.py')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,value):
    assert bool(value),name
    checks.append(name)
for w,h in [(1280,720),(640,480),(1920,1080),(1920,600)]:
    c=calibrate(SimpleNamespace(validate=lambda:None),w,h)
    check('full actual sensor '+str((w,h)),c['box_source_pixels']==[0,0,w,h] and not c['rest_wrist_required'])
rows=read(B/'PUBLIC_INPUTS.json')['rows'];audit=[];selected=[]
check('public cohort 48',len(rows)==48)
for row in rows:
    path=Path(row['observation']);check(row['clip_id']+' source pin',sha(path)==row['observation_sha256'])
    ob=Observation.load(str(path));w,h=int(ob.sidecar['width']),int(ob.sidecar['height'])
    original={k:getattr(ob,k).copy() for k in ['xy_px','observed','usable','timestamp_us','frame_id']}
    c=calibrate(ob,w,h);a=adapter.roi_arrays(original,c['box_source_pixels'])
    xy=original['xy_px'];inside=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
    check(row['clip_id']+' exact sensor mask',np.array_equal(a['usable'],original['usable']&inside))
    for key in ['xy_px','observed','timestamp_us','frame_id']:
        check(row['clip_id']+' preserve '+key,np.array_equal(a[key],original[key],equal_nan=True))
    baseline=adapter.roi_arrays(original,row['box'])
    check(row['clip_id']+' retained old visible joints',not (baseline['usable']&~a['usable']).any())
    features=observer.extract_features(a);prior=observer.extract_features(baseline)
    common=features['mask']&prior['mask']
    check(row['clip_id']+' exact shared features',np.array_equal(features['x'][common],prior['x'][common]))
    check(row['clip_id']+' retained prior features',not (prior['mask']&~features['mask']).any())
    # One actual sample is neither dropped nor smoothed. A native gap invalidates
    # only dependent features and consecutive velocity, never becomes a match.
    i=len(xy)//2;missing={k:v.copy() for k,v in a.items()};missing['usable'][i,33:54]=False
    mf=observer.extract_features(missing)
    left=[j for j,n in enumerate(features['feature_names']) if n.startswith('L_') and '_angle' in n and 'elbow' not in n]
    right=[j for j,n in enumerate(features['feature_names']) if n.startswith('R_')]
    check(row['clip_id']+' gap cannot invent left finger angles',not mf['mask'][i,left].any())
    check(row['clip_id']+' preserve opposite hand',np.array_equal(mf['mask'][:,right],features['mask'][:,right]))
    check(row['clip_id']+' exact frame count including short sample',len(mf['x'])==len(xy))
    audit.append(dict(clip=row['clip_id'],frames=len(xy),extra_visible_joint_cells=int((a['usable']&~baseline['usable']).sum())))
    if row['person']==12:
        # Fixed public-proxy crop: initial stable shoulder center plus one width.
        sh=xy[:min(30,len(xy)),[11,12]];ok=(a['usable'][:len(sh),[11,12]]).all(1)
        check(row['clip_id']+' available public shoulder crop anchor',ok.any())
        cen=np.median(sh[ok].mean(1),axis=0);scale=np.median(np.linalg.norm(sh[ok,0]-sh[ok,1],axis=1))
        bottom=2*(int(math.floor(float(cen[1]+scale)))//2)
        check(row['clip_id']+' proxy crop within source',0<bottom<h)
        selected.append(dict(**row,crop_bottom=bottom,original_width=w,original_height=h))
check('all four expressions fixed ID12',len(selected)==4 and len({r['word_id'] for r in selected})==4)
# Missing query observations cannot shrink the reference expectation denominator.
links=np.array([[i,i] for i in range(16)]);rm=np.ones((16,3),bool)
good=guard._support(rm,rm,links,[0,15],shape=True)
missing=guard._support(np.zeros_like(rm),rm,links,[0,15],shape=True)
check('reference denominator retained',good['reference_expected']==missing['reference_expected']==48)
check('missing query not similar',missing['coverage']==0 and not guard._eligible(missing))
check('short and observation reasons distinct',diagnostic(['FEWER_THAN16_REAL_FRAMES_NO_PADDING'])!=diagnostic(['NO_SUPPORTED_VISIBLE_FEATURES']))
# Generate actual cropped public pixels, no user input/no resizing, preserve time.
import av
from fractions import Fraction
videos=U/'public_crops';videos.mkdir(exist_ok=True)
prepared=[]
for row in selected:
    src=Path(row['source']);check(row['clip_id']+' video pin',sha(src)==row['source_sha256'])
    dest=videos/(row['clip_id']+'.mp4');assert not dest.exists()
    with av.open(str(src)) as inc, av.open(str(dest),'w') as out:
        vs=inc.streams.video[0];stream=out.add_stream('libx264',rate=vs.average_rate or 30)
        stream.width=row['original_width'];stream.height=row['crop_bottom'];stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'};stream.time_base=Fraction(1,1_000_000)
        stamps=[];origin=None
        for frame in inc.decode(vs):
            if frame.pts is None:continue
            raw=frame.pts*vs.time_base
            if origin is None:origin=raw
            us=int(round(float(raw-origin)*1e6));stamps.append(us)
            image=frame.to_ndarray(format='rgb24')[:row['crop_bottom']]
            vf=av.VideoFrame.from_ndarray(image,format='rgb24');vf.pts=us;vf.time_base=Fraction(1,1_000_000)
            for packet in stream.encode(vf):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    with av.open(str(dest)) as c:
        v=c.streams.video[0];actual=[round(float(f.pts*v.time_base)*1e6) for f in c.decode(v)]
    check(row['clip_id']+' proxy frame count preserved',len(actual)==len(stamps))
    check(row['clip_id']+' proxy timestamps preserved',max(abs(a-b) for a,b in zip(actual,stamps))<=2)
    prepared.append(dict(clip_id=row['clip_id'],word_id=row['word_id'],label=row['label'],source=str(src),
                         source_sha256=row['source_sha256'],video=str(dest),video_sha256=sha(dest),
                         frames=len(actual),crop=[0,0,row['original_width'],row['crop_bottom']],public_only=True))
manifest=read(U/'CODE_MANIFEST.json')
for p,h in manifest['base_pins'].items():check('protected '+p,sha(Path(p))==h)
result=dict(status='PASS',checks=len(checks),check_names=checks,public_observations=audit,
            public_crops=prepared,training=0,private_inputs=0,new_model_inference=0)
with (U/'INPUT_VALIDATION.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status='PASS',checks=len(checks),public_observations=len(audit),public_crops=prepared),ensure_ascii=False),flush=True)
