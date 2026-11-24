"""Local data routing and evidence IO; the model learner is Claude-authored."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/partial_encoder_pair_v1'
sys.path[:0]=[str(H),str(L),str(L/'src'),str(L.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
from pilot.data import build_channels
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(plain(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def policy():
    p=read(H/'comparison_policy.json')
    for path,d in p['pins'].items():assert sha(path)==d,path
    assert sha(p['parent']['path'])==p['parent']['sha256']
    return p
def parent_model(p):
    ck=torch.load(p['parent']['path'],map_location='cpu',weights_only=True)
    m,classes=u.rebuild_from_payload(ck);m.eval();torch.set_num_threads(2)
    return ck,m,classes
def pack(xs):
    offsets=np.cumsum([0]+[len(x) for x in xs],dtype=np.int64)
    return np.concatenate(xs,axis=0),offsets
def unpack(values,offsets):return [values[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])]
def native_input(row,whole=False):
    assert sha(row['path'])==row['sha256'],row['id']
    if whole:
        with np.load(row['path'],allow_pickle=False) as z:return z['x'].copy()
    return u._channels_from_source(row['path'],row['sha256'])['x']
def nikl_input(row):
    path=L/row['npz'];nat,present,ts=u._native_from_nikl(str(path),row['npz_sha256'])
    return build_channels(nat,present,ts)['x']
def train_inputs(p):
    meta=read(p['paths']['metadata']);nikl={r['id']:r for r in read(p['paths']['nikl_manifest'])['nikl']}
    result={}
    for group in ['base','pair_context']:
        xs=[]
        for row in meta[group]:
            assert row['role']=='train'
            if row['id'].startswith('NIKL_'):
                assert group=='base';x=nikl_input(nikl[row['id']])
            else:
                assert row['signer_id'] in range(1,10)
                x=native_input(row,group=='pair_context')
            assert x.dtype==np.float32 and x.ndim==2 and x.shape[1]==280 and np.isfinite(x).all()
            xs.append(x)
        result[group]=xs
    assert [len(result[g]) for g in ['base','pair_context']]==[2122,224]
    return result,meta
def evaluation_inputs(p):
    import tools_visibility_experiment as v
    from wis_lab.core.observation import Observation
    from wis_lab.service import neural
    meta=read(p['paths']['metadata']);groups={}
    for group in ['annotation_dev','whole_dev']:
        assert all(r['signer_id'] in (11,12,13) for r in meta[group])
        groups[group]=[native_input(r,group=='whole_dev') for r in meta[group]]
    provenance=read(p['paths']['visibility_provenance'])
    reference_path=Path(provenance['source_experiment']['directory'])/'inputs.json'
    items=[];eligible={}
    for row in read(reference_path)['nikl']:
        nat,present,ts=u._native_from_nikl(str(L/row['npz']),row['npz_sha256'])
        items.append(dict(clip_id=row['id'],label=row['label'],nat=nat,present=present,ts=ts))
        if row.get('reviewed_non_sign_hand') in ('left','right'):eligible[row['id']]={'side':row['reviewed_non_sign_hand']}
    refs=v.reference_clips(items,eligible)
    source_map={r['id']:r for r in meta['base']}
    ref_map={(r['clip_id'],r['condition']):r['x'] for r in refs}
    groups['public']=[native_input(source_map[r['id']]) if r['condition']=='additional_train' else ref_map[r['id'],r['condition']] for r in meta['public']]
    mp_map={}
    for item in read(p['paths']['mp_plan'])['inputs']:
        assert sha(item['cache'])==item['cache_sha256']
        obs=Observation.load(item['cache'])
        for condition in ['whole_clip','provided_annotation']:
            lo,hi=(0,obs.n_frames-1) if condition=='whole_clip' else item['annotation_rows'];win=slice(lo,hi+1)
            fields={k:getattr(obs,k)[win].copy() for k in ['xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score']}
            cut=Observation(**fields,sidecar=dict(obs.sidecar))
            clip,why,_=neural._channels(cut);assert clip is not None,why
            mp_map[item['id'],condition]=clip['x']
    groups['mp']=[mp_map[r['id'],r['condition']] for r in meta['mp']]
    return groups,meta
