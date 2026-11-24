"""Independent full feature reconstruction plus time/provenance and loader verification."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from feature_cache import load,model_inputs,validate,sha

U=Path(__file__).resolve().parent;B=U.parents[1]

def main():
    torch.set_num_threads(2)
    plan=json.loads((U/'PLAN.json').read_text(encoding='utf-8'))['inputs']
    build=json.loads((U/'BUILD_RESULT.json').read_text(encoding='utf-8'))
    assert sha(build['manifest_path'])==build['manifest_sha256']
    rows=json.loads(Path(build['manifest_path']).read_text(encoding='utf-8'))
    indexpath=B/'ops/rgb_comparison_contract_20261010/INPUT_INDEX.json'
    assert sha(indexpath)==plan['input_index_sha256']
    metadata={r['sample_id']:r for r in json.loads(indexpath.read_text(encoding='utf-8'))['samples']}
    assert len({r['sample_id'] for r in rows})==len(rows)==113
    weights=json.loads((B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json').read_text(encoding='utf-8'))
    assert sha(weights['path'])==plan['encoder_sha256']
    ref=mobilenet_v3_small(weights=None).eval()
    ref.load_state_dict(torch.load(weights['path'],map_location='cpu',weights_only=True),strict=True)
    count={'segments':0,'time_rows':0,'observed_images':0,'estimated_images':0,'absent_slots':0}
    max_error=0.;t=time.perf_counter()
    for r in rows:
        assert sha(r['source_cache_path'])==r['source_cache_sha256']
        with np.load(r['source_cache_path'],allow_pickle=False) as z: src={k:z[k] for k in z.files}
        got=load(r['feature_path'],r['feature_sha256'])
        for key in ['pose_x','frame_id','timestamp_us','available','observed_rgb','region_estimated','landmark_observed_count','bbox_xyxy']:
            assert np.array_equal(src[key],got[key]), (r['sample_id'],key)
        assert (got['features'][~src['available']]==0).all()
        # No producer pack/scatter/encoder calls. Reconstruct pixels and slot ordering directly.
        slots=np.argwhere(src['available'])
        for first in range(0,len(slots),13):
            subset=slots[first:first+13]
            pixels=np.stack([src['rgb'][i,j] for i,j in subset])
            images=torch.tensor(pixels.transpose(0,3,1,2),dtype=torch.float32)/255
            for ch,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                images[:,ch]=(images[:,ch]-mean)/std
            with torch.no_grad(): expected=ref.avgpool(ref.features(images)).flatten(1).numpy()
            actual=np.stack([got['features'][i,j] for i,j in subset])
            error=float(np.max(np.abs(expected-actual)))
            assert error<=1e-5,(r['sample_id'],first,error)
            max_error=max(max_error,error)
        views={a:model_inputs(got,a) for a in 'ABC'}
        assert not views['A']['features'].any() and not views['A']['available'].any()
        assert np.array_equal(views['B']['available'].numpy(),src['observed_rgb'])
        assert np.array_equal(views['C']['available'].numpy(),src['available'])
        assert not views['B']['estimated'].any()
        assert np.array_equal(views['C']['estimated'].numpy(),src['region_estimated'])
        assert torch.equal(views['B']['features'][views['B']['available']],views['C']['features'][views['B']['available']])
        for a in 'ABC':
            assert np.array_equal(views[a]['pose_x'].numpy(),src['pose_x'])
            assert np.array_equal(views[a]['timestamp_us'].numpy(),src['timestamp_us'])
            assert set(views[a])=={'pose_x','pad','features','available','estimated','frame_id','timestamp_us'}
        m=metadata[r['sample_id']]
        for key in ['label_metadata_only','signer_metadata_only','split','semantic_alignment_verified']:
            assert r[key]==m[key]
        count['segments']+=1;count['time_rows']+=len(src['pose_x'])
        count['observed_images']+=int(src['observed_rgb'].sum());count['estimated_images']+=int(src['region_estimated'].sum())
        count['absent_slots']+=int((~src['available']).sum())
        if count['segments']%25==0:print('independently verified',count['segments'],'/ 113',flush=True)
    assert all(count[k]==plan[k] for k in count)
    negative=[]
    def reject(name,fn):
        try:fn()
        except ValueError:negative.append(name)
        else:raise AssertionError(name+' was accepted')
    reject('hash mismatch',lambda:load(rows[0]['feature_path'],'0'*64))
    reject('unknown arm',lambda:model_inputs(got,'unknown'))
    bad={k:v.copy() for k,v in got.items()};bad['label']=np.array([1])
    reject('label enters model cache',lambda:validate(bad))
    bad={k:v.copy() for k,v in got.items()};bad['available'][0,0]=False;bad['observed_rgb'][0,0]=False;bad['region_estimated'][0,0]=False;bad['features'][0,0,0]=1
    reject('nonzero missing feature',lambda:validate(bad))
    bad={k:v.copy() for k,v in got.items()};bad['timestamp_us'][1]=bad['timestamp_us'][0]
    reject('duplicate timestamp',lambda:validate(bad))
    assert sha(weights['path'])==plan['encoder_sha256']
    assert sha(build['manifest_path'])==build['manifest_sha256']
    result={'status':'PASS','counts':count,'independent_images':count['observed_images']+count['estimated_images'],
            'all_metadata_equal':True,'all_abc_views_verified':True,'max_feature_abs_error':max_error,
            'tolerance_frozen_before_run':1e-5,'negative_checks':negative,'elapsed_seconds':time.perf_counter()-t,
            'method':'Direct official torchvision backbone, separate pixel normalization and manual slot indexing; all available images, not a sampled accuracy test',
            'independent_person_accuracy':False,'training_started':False}
    with (U/'FEATURE_CHECK.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('Independent feature check PASS',flush=True)

if __name__=='__main__':main()
