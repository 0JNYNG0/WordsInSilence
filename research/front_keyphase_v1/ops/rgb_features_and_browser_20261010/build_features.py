"""Encode frozen prepared RGB once. No labels, sign inference, optimization or new frames."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from feature_cache import sha,validate

U=Path(__file__).resolve().parent; B=U.parents[1]
sys.path[:0]=[str(B/'ops/rgb_tensor_adapter_20261010'),str(B/'ops/rgb_pose_bridge_20261010')]
from rgb_input import load_cache,pack,scatter_features
from rgb_bridge import FrozenRGBEncoder

def write(path,data):
    with path.open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    plan=json.loads((U/'PLAN.json').read_text(encoding='utf-8'))['inputs']
    source=B/'runs/rgb_tensor_adapter_v1/CACHE_MANIFEST.json'
    index=B/'ops/rgb_comparison_contract_20261010/INPUT_INDEX.json'
    weights=json.loads((B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json').read_text(encoding='utf-8'))
    assert sha(source)==plan['cache_manifest_sha256'] and sha(index)==plan['input_index_sha256']
    assert weights['sha256']==plan['encoder_sha256']
    records=json.loads(source.read_text(encoding='utf-8'))
    metadata={s['sample_id']:s for s in json.loads(index.read_text(encoding='utf-8'))['samples']}
    assert len(records)==len(metadata)==plan['segments']
    out=B/'runs/rgb_features_v1'; out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    encoder=FrozenRGBEncoder(weights['path'],weights['sha256'])
    saved={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    t=time.perf_counter(); manifest=[]; counts=dict(segments=0,time_rows=0,observed_images=0,estimated_images=0,absent_slots=0)
    for r in records:
        a=load_cache(r['cache_path'],r['cache_sha256']); p=pack(a,'C')
        features=scatter_features(encoder(p['images']),p['slots'],len(a['pose_x'])).numpy()
        data={k:a[k] for k in ['pose_x','frame_id','timestamp_us','available','observed_rgb','region_estimated','landmark_observed_count','bbox_xyxy']}
        data['features']=features
        validate(data)
        dest=out/(r['sample_id']+'.npz')
        with dest.open('xb') as f: np.savez_compressed(f,**data)
        m=metadata[r['sample_id']]
        manifest.append({'sample_id':r['sample_id'],'feature_path':str(dest),'feature_sha256':sha(dest),
            'source_cache_path':r['cache_path'],'source_cache_sha256':r['cache_sha256'],
            'encoder_sha256':weights['sha256'],'input_contract_sha256':plan['input_index_sha256'],
            'clip_id_metadata_only':m['clip_id'],'label_metadata_only':m['label_metadata_only'],
            'signer_metadata_only':m['signer_metadata_only'],'split':m['split'],
            'semantic_alignment_verified':m['semantic_alignment_verified'],'frames':len(a['pose_x'])})
        counts['segments']+=1;counts['time_rows']+=len(a['pose_x']);counts['observed_images']+=int(a['observed_rgb'].sum())
        counts['estimated_images']+=int(a['region_estimated'].sum());counts['absent_slots']+=int((~a['available']).sum())
        if len(manifest)%25==0: print('encoded',len(manifest),'/',len(records),flush=True)
    assert all(counts[k]==plan[k] for k in counts)
    assert encoder.forward_images==counts['observed_images']+counts['estimated_images']
    assert all(torch.equal(v,saved[k]) for k,v in encoder.state_dict().items())
    assert sha(weights['path'])==weights['sha256']
    assert sha(source)==plan['cache_manifest_sha256'] and sha(index)==plan['input_index_sha256']
    assert all(sha(r['cache_path'])==r['cache_sha256'] for r in records)
    write(out/'MANIFEST.json',manifest)
    write(U/'BUILD_RESULT.json',{'status':'PASS','counts':counts,'encoded_images':encoder.forward_images,
        'elapsed_seconds':time.perf_counter()-t,'bytes':sum(Path(r['feature_path']).stat().st_size for r in manifest),
        'manifest_path':str(out/'MANIFEST.json'),'manifest_sha256':sha(out/'MANIFEST.json'),
        'source_hashes_and_frozen_encoder_unchanged':True,'training_performed':False,'sign_accuracy_evaluated':False})
    print('Feature cache complete',counts,flush=True)

if __name__=='__main__': main()
