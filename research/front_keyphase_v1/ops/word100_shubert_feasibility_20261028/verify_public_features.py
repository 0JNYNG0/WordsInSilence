"""Recalculate saved feasibility results, never infer or retrain a model."""
from pathlib import Path
import hashlib,json
import numpy as np
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=json.loads((U/'PUBLIC_FEATURE_RESULT.json').read_text(encoding='utf-8'))
    manifest=json.loads((U/'inputs/MANIFEST.json').read_text(encoding='utf-8'))
    assert result['frames']==manifest['frames']==1438
    assert result['clips']==manifest['clips']==12
    assert [r['alias'] for r in result['rows']]==[r['alias'] for r in manifest['rows']]
    assert result['training_runs']==0 and not (U/'FEATURE_RUNNING.json').exists()
    independent=json.loads((U/'ADAPTER_INDEPENDENT.json').read_text(encoding='utf-8'))
    assert independent['status']=='PASS'
    assert independent['adapter_sha256']==result['adapter_sha256']==sha(U/'cpu_shubert_adapter.py')
    counts={'current_hand_crops':0,'carried_hand_crops':0,'provenance_arrays_changed':0}
    rows=[]
    for r in result['rows']:
        alias=r['alias'];f=U/'features'/alias;i=U/'inputs'/alias
        assert r==json.loads((f/'DONE.json').read_text(encoding='utf-8'))
        for k,h in r['output_pins'].items():assert sha(f/k)==h
        y=np.load(f/'last_ffn.npy');streams=np.load(f/'streams.npz');p=np.load(f/'provenance.npz');orig=np.load(i/'provenance.npz')
        assert y.shape==(r['frames'],768) and np.isfinite(y).all()
        assert all(np.array_equal(p[k],orig[k],equal_nan=True) for k in p.files)
        assert set(streams.files)=={'face','left_hand','right_hand','body_posture'}
        assert all(np.isfinite(streams[k]).all() for k in streams.files)
        assert all(streams[k].shape==(r['frames'],14 if k=='body_posture' else 384) for k in streams.files)
        for key,kind in [('current_hand_crops',1),('carried_hand_crops',2)]:
            value=int((p['source_kind'][:,:2]==kind).sum());assert value==r[key];counts[key]+=value
        controls=np.load(f/'controls.npz');kept=controls['omitted_keep_indices']
        lo,hi=r['omitted_indices'];assert np.array_equal(kept,np.r_[0:lo,hi:len(y)])
        reverse=float(np.linalg.norm(y-controls['reverse_last_ffn'][::-1])/np.linalg.norm(y))
        omitted=float(np.linalg.norm(y[kept]-controls['omitted_last_ffn'])/np.linalg.norm(y[kept]))
        assert abs(reverse-r['reverse_context_relative_l2'])<1e-7
        assert abs(omitted-r['omission_context_relative_l2'])<1e-7
        speed=json.loads((f/'speed_timestamp_maps.json').read_text(encoding='utf-8'))
        for factor in (.75,1.25):assert np.array_equal(speed[str(factor)],(p['timestamp_us']-p['timestamp_us'][0])/factor)
        if alias=='V01':assert r['repeat_max_abs']==0
        rows.append({'alias':alias,'frames':len(y),'finite':True,'provenance_equal':True,'reverse_relative_l2':reverse,'omission_relative_l2':omitted})
    times=np.array([r['timing']['primary_feature_seconds'] for r in result['rows']])
    dino=np.array([sum(r['timing'][k+'_dino_seconds'] for k in ['face','left_hand','right_hand']) for r in result['rows']])
    body=np.array([r['timing']['shubert_seconds'] for r in result['rows']])
    verify={'status':'PASS_SAVED_FEATURES_TIMESTAMPS_MASKS_HASHES_AND_ARITHMETIC',
      'scope':'Software and stored numerical consistency, not independent human handshape truth',
      'clips':12,'frames':1438,'counts':counts,'rows':rows,
      'primary_latency_seconds':{'median':float(np.median(times)),'min':float(times.min()),'max':float(times.max()),'sum':float(times.sum())},
      'dino_share_of_primary_seconds':float(dino.sum()/times.sum()),
      'shubert_warm_seconds':{'median':float(np.median(body[1:])),'min':float(body[1:].min()),'max':float(body[1:].max())},
      'latency_note':'Part of first run overlapped verifier; use LATENCY_ISOLATED.json for uncontended explicit 3-clip measurement',
      'prediction_accuracy':None,'training_runs':0,'input_manifest_sha256':sha(U/'inputs/MANIFEST.json'),
      'result_sha256':sha(U/'PUBLIC_FEATURE_RESULT.json'),'verifier_sha256':sha(__file__)}
    (U/'FEATURE_VERIFICATION.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verify.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
