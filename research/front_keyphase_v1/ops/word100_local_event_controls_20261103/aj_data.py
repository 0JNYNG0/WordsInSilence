"""AJ read-only cached input loader. No model/runtime imports or media decode."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

AJ=Path(__file__).resolve().parent
O=AJ.parent
AH=O/'word100_local_event_learning_20261103'
U=O/'word100_learned_feature_retention_20261102'
V=O/'word100_learned_body_relation_20261102'
R=O/'word100_learned_rgb_motion_20261102'
M=O/'word100_learned_motion_order_20261102'
AC=O/'word100_process_contract_alignment_20261102'
WORDS=['WORD0739','WORD1133','WORD1567','WORD2523']
KINDS=['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest','action_cut','main_blackout']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def verify_pins(pins):
    for p,h in pins.items():require(sha(p)==h,'Frozen file changed: '+p)
def fragments(values):
    edge=np.diff(np.r_[False,values,False].astype(np.int8))
    return [[int(a),int(b)-1] for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1))]

def support_module():
    # This previously reviewed helper imports only standard-library arithmetic.
    path=AC/'process_rgb_support.py'
    spec=importlib.util.spec_from_file_location('_aj_exact_rgb_support',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def prepare_arrays(input_path,relation_path,geometry_stats,rgb_stats,geometry_valid):
    """Pure assembly of existing actual-control cache values, in memory only."""
    with np.load(input_path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    with np.load(relation_path,allow_pickle=False) as f:r={k:f[k].copy() for k in f.files}
    ids=a['frame_id'];ts=a['timestamp_us'];n=len(ids)
    require(n>=16 and ids.dtype.kind in 'iu' and ts.dtype.kind in 'iu','Unsupported native axes')
    require(np.array_equal(ids,np.arange(n)) and bool(np.all(np.diff(ts)>0)),'Native index/time order')
    require(all(np.array_equal(a[k],r[k]) for k in ['frame_id','timestamp_us']),'Relation/input axes differ')
    raw=a['x'];mask=a['mask'];shape=a['retained_probability'];relation=r['probability']
    require(raw.shape==mask.shape==(n,69) and mask.dtype.kind=='b','Geometry input contract')
    require(shape.shape==(n,7) and relation.shape==(n,3) and a['rgb'].shape==(n,768),'Cached observer dimensions')
    image=a['image_available'];require(image.shape==(n,) and image.dtype.kind=='b','Explicit image availability required')
    v=raw[:,2:4]-raw[:,4:6];delta=raw[:,36:38]-raw[:,4:6];den=(v*v).sum(1)
    good=mask[:,[2,3,4,5,36,37]].all(1)&np.isfinite(den)&(den>.0025)
    along=(delta*v).sum(1)/np.maximum(den,1e-6)
    side=(delta[:,0]*v[:,1]-delta[:,1]*v[:,0])/np.maximum(den,1e-6)
    z=np.column_stack([raw[:,:61],along,side,shape,relation])
    valid=np.column_stack([mask[:,:61]&np.isfinite(raw[:,:61]),good,good,np.isfinite(shape),np.isfinite(relation)])
    valid &= image[:,None]
    # Match frozen motion_model.features float32 conversion before normalization.
    z=np.where(valid,z,0).astype(np.float32)
    valid &= np.isfinite(z)&image[:,None]
    z=np.where(valid,z,0)
    geo=np.concatenate([np.where(valid,(z-np.asarray(geometry_stats['mean']))/np.asarray(geometry_stats['std']),0),valid],1).astype(np.float32)
    available=image&np.isfinite(a['rgb']).all(1)
    video=np.concatenate([np.where(available[:,None],(a['rgb']-np.asarray(rgb_stats['mean']))/np.asarray(rgb_stats['std']),0),available[:,None]],1).astype(np.float32)
    x=np.concatenate([geo,video],1)
    require(x.shape==(n,915) and bool(np.isfinite(x).all()),'Normalized915 must be finite')
    selected=np.asarray(geometry_valid)[:73].astype(bool)
    observed=available|(valid[:,selected].any(1));fs=fragments(observed);anchor=np.zeros(n,bool)
    for lo,hi in fs:
        if hi>lo:anchor[lo:hi+1]=True
    helper=support_module();support=helper.build_rgb_support(ts,image,ids,rgb_finite=np.isfinite(a['rgb']).all(1))
    return dict(x=x,ids=ids,ts=ts,mask=valid,available=available,image_available=image,
        evidence_available=observed,motion_anchor=anchor,observed_fragments=fs,support=support,
        rgb_lineage='VERIFIED_RETAINED_LIVE_ENCODER_RECIPE_WITHOUT_SAVED_WINDOW_VALUES',
        window_vectors_saved=False,encoder_recomputed=False)

def load_inputs(manifest_path=None):
    """Return case_id -> arrays; never create a model, score, crop or saved array."""
    manifest=read(AJ/'INPUT_FROZEN.json' if manifest_path is None else manifest_path)
    require(manifest['status']=='FROZEN_EXISTING_PIXEL_CONTROL_INPUTS','Unready input contract')
    verify_pins(manifest['pins'])
    require(len(manifest['rows'])==24,'Exactly24 frozen inputs')
    stats=read(M/'NORMALIZATION.json');rgb_stats=read(R/'RGB_NORMALIZATION.json');valid=manifest['geometry_valid'];result={}
    for row in manifest['rows']:
        require(row['word_id'] in WORDS and row['kind'] in KINDS and row['person']==12 and row['role']=='development_controls_not_training','Only fixed public DEV12 controls')
        require(row['case_id']==row['word_id']+'_'+row['kind'] and row['case_id'] not in result,'Unique case binding')
        p=prepare_arrays(row['analysis']['path'],row['relation']['path'],stats,rgb_stats,valid)
        require(p['ids'].tolist()==row['native_frame_ids'] and p['ts'].tolist()==row['timestamp_us'],'Frozen axis changed')
        require(p['evidence_available'].tolist()==row['evidence_available'] and p['observed_fragments']==row['observed_fragments'],'Frozen availability changed')
        # Source/edit maps are diagnostic sidecars. They never enter x or a model.
        p.update(case_id=row['case_id'],word_id=row['word_id'],kind=row['kind'],
            prototype_path=np.asarray(row['prototype_path'],dtype=np.int64),scopes=row['scopes'],
            source_ledger=row['source_ledger'],cache_binding=row['analysis'],video=row['video'])
        result[row['case_id']]=p
    return result
