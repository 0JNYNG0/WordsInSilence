"""Fixed fit-only arithmetic controls. Never calls dtw/compare or a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rp=U/'relational.py'
mp=U.parent/'word100_guide_video_feedback_20261028'/'motion_observer.py'
rel=module('fit_invariance_relational',rp)
motion=module('fit_invariance_motion_observer',mp)
auth=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf-8-sig'))
rows={r['alias']:r for r in auth['rows'] if r['role']=='fit' and r['alias'] in ['P033','P045']}
assert set(rows)=={'P033','P045'} and len(rel.GROUPS)==13
pins={str(p):sha(p) for p in [rp,mp,Path(__file__),U/'AUTHORIZATION.json']}
checks=[];data={};perturbed={};transforms=[]
def group_check(label,a,b,atol):
    table=[]
    for g in rel.GROUPS:
        x,m=a[g];y,n=b[g]
        mask_same=np.array_equal(m,n)
        err=float(np.max(np.abs(x[m]-y[m]))) if m.any() else 0.
        values_same=bool(np.allclose(x[m],y[m],atol=atol,rtol=0))
        table.append({'group':g,'valid_cells':int(m.sum()),'expected_cells':int(m.size),
          'mask_exactly_unchanged':mask_same,'valid_max_abs_difference':err,'valid_values_unchanged':values_same})
        assert mask_same and values_same,(label,g,err)
    checks.append({'case':label,'atol':atol,'rtol':0,'groups':table})

for alias,r in rows.items():
    for p in [r['input'],r['observation']]:pins[p]=sha(p)
    assert pins[r['input']]==r['input_sha256']
    obs=motion.read_observation(r['observation'])
    obs['xy_px']=obs['xy_px'].astype(np.float64)
    xy=obs['xy_px'];x0,y0,x1,y1=r['box']
    inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
    obs['usable']=obs['usable']&inside
    original=motion.extract_features(obs)
    changed={k:v.copy() for k,v in obs.items()}
    changed['xy_px']=obs['xy_px']*1.3+np.array([123.,-55.])
    scaled=motion.extract_features(changed)
    assert np.array_equal(original['mask'],scaled['mask'])
    group_check(alias+'_uniform_scale_translation_after_original_ROI',rel.features(original),rel.features(scaled),1e-11)
    transforms.append({'alias':alias,'native_frames':len(xy),'original_fixed_roi_source_pixels':r['box'],
      'transform':'xy_new = 1.3 * xy_old + [123,-55]','original_roi_mask_held_fixed':True,
      'normalizer_recomputed':True,'valid_69_max_abs_difference':float(np.max(np.abs(original['x'][original['mask']]-scaled['x'][original['mask']])))})
    with np.load(r['input'],allow_pickle=False) as z:
        saved={'x':z['x'].astype(np.float64),'mask':z['mask'].copy()}
    altered={'x':saved['x'].copy(),'mask':saved['mask'].copy()}
    missing=~saved['mask'];n=int(missing.sum())
    altered['x'][missing]=np.where(np.arange(n)%2==0,1.,-1.)*(1e9+np.arange(n)*123.)
    assert np.array_equal(saved['x'][saved['mask']],altered['x'][saved['mask']])
    assert np.array_equal(saved['mask'],altered['mask'])
    data[alias]=rel.features(saved);perturbed[alias]=rel.features(altered)
    group_check(alias+'_masked_storage_large_values',data[alias],perturbed[alias],0)
    transforms[-1]['masked_storage_cells_perturbed']=n

costs=[]
for qa in rows:
    for ra in rows:
        base=rel.alignment_cost(data[qa],data[ra])
        assert np.isfinite(base).all()
        for mode,q,r in [('query_only',perturbed[qa],data[ra]),('reference_only',data[qa],perturbed[ra]),('both',perturbed[qa],perturbed[ra])]:
            z=rel.alignment_cost(q,r);err=float(np.max(np.abs(base-z)))
            assert np.array_equal(base,z),(qa,ra,mode,err)
            costs.append({'query':qa,'reference':ra,'mode':mode,'shape':list(base.shape),
              'entries':int(base.size),'max_abs_difference':err,'exactly_unchanged':True})

assert all(sha(p)==h for p,h in pins.items())
report={'status':'PASS_FIXED_FIT_ARITHMETIC_INVARIANCE_CONTROLS','reviewer':'Codex agent A; same model family, independent control execution from root implementation',
 'scope':{'fit_aliases':['P033','P045'],'people':[1],'groups':rel.GROUPS,'new_model_inference':0,'training':0,'dtw_calls':0,'compare_calls':0,'dev_data_used':0},
 'coordinate_controls':transforms,'group_checks':checks,'alignment_cost_checks':costs,
 'units_and_definition':{
   'shape':'14 projected image-plane interior angles divided by 180 degrees',
   'location':'hand wrist/palm center in contemporaneous shoulder-width units about shoulder center',
   'axis':'wrist to middle MCP 2D unit direction components divided by 2; not a palm normal',
   'arm':'elbow position in shoulder widths; projected elbow angle/180; elbow-to-pose-wrist unit direction/2; forearm length in shoulder widths',
   'hands_relation':'R hand-detector wrist minus L hand-detector wrist, x/y and Euclidean distance in shoulder widths',
   'opposite_forearm':'dot/cross of hand-minus-opposite-elbow with opposite pose forearm, divided by squared forearm length; dimensionless 2D fractions',
   'direction':'consecutive-native-frame normalized hand-wrist motion direction components divided by 2',
   'alignment_cost':'mean over13groups; per-component abs difference clipped to1 with missing-query penalty on expected reference cells; dimensionless'},
 'limits':[
   'Only a common positive uniform scale and translation applied to saved estimates were tested. No real camera viewpoint, perspective, depth, body-proportion or signing-variation invariance is established.',
   'Original ROI membership is fixed before the coordinate transform; this is not a changed-camera ROI calibration test or a new detector extraction.',
   'The two fixed clips are both fit person1. They are arithmetic controls, not evidence of cross-person generalization.',
   'Masked values were changed to large finite +/- values only. Masks remain authoritative. This does not certify arbitrary inf/NaN/overflow robustness.',
   'Availability and numerical invariance do not validate anatomical truth, handshape identity, contact, linguistic sequence or correctness.',
   'Only deterministic feature and alignment_cost functions were called. DTW search, learned inference, thresholds/ranges and old files were not changed.'
 ],'input_pins':pins,'input_pins_unchanged':True}
(U/'review'/'INVARIANCE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'group_cases':len(checks),'group_checks':len(checks)*13,
 'cost_matrices':len(costs),'cost_entries_checked':sum(c['entries'] for c in costs),
 'max_transform_group_error':max(c['valid_max_abs_difference'] for q in checks for c in q['groups']),
 'masked_storage_exact_invariance':True,'dtw_calls':0,'input_pins_unchanged':True}))
