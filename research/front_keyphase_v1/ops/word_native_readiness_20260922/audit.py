"""Independent array-formula audit; uses prepared public candidates only."""
from pathlib import Path
import hashlib,json
import numpy as np

H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def expected_channels(native, present, times):
    order=[0,5,2,6,3,7,4,12,9]+list(range(25,67))
    xy=native[:,order].astype(np.float32); observed=present[:,order]
    shoulder=observed[:,1]&observed[:,2]
    scale=float(np.median(np.sqrt(np.sum((xy[shoulder,1]-xy[shoulder,2])**2,axis=1))))
    body_ok=observed&shoulder[:,None]
    body=np.zeros_like(xy)
    centered=(xy-(xy[:,1:2]+xy[:,2:3])/2)/scale
    body[body_ok]=centered[body_ok]
    hands=np.zeros((len(xy),42,2),dtype=np.float32)
    hand_ok=np.zeros((len(xy),42),dtype=bool)
    for start,out in [(9,0),(30,21)]:
        lengths=[]
        for mcp in [5,9,13,17]:
            good=observed[:,start]&observed[:,start+mcp]
            lengths.extend(np.sqrt(np.sum((xy[good,start+mcp]-xy[good,start])**2,axis=1)).tolist())
        hand_scale=float(np.median(lengths)) if lengths else float('nan')
        if not np.isfinite(hand_scale) or hand_scale<=1e-6:continue
        valid=observed[:,start:start+21]&observed[:,start,None]
        rel=(xy[:,start:start+21]-xy[:,start:start+1])/hand_scale
        hands[:,out:out+21][valid]=rel[valid]
        hand_ok[:,out:out+21]=valid
    dt=np.r_[0,np.diff(times)/1e6].astype(np.float32)
    return np.c_[body.reshape(len(xy),102),hands.reshape(len(xy),84),body_ok.astype(np.float32),hand_ok.astype(np.float32),dt]

def main():
    result=read(H/'PREPARATION.json'); plan=read(H/'PLAN.json')
    assert result['status']=='PREPARED_PENDING_INDEPENDENT_AUDIT' and len(result['rows'])==16
    assert all(sha(p)==d for p,d in plan['source_pins'].items())
    total_frames=0; total_values=0; max_error=0.0
    for row in result['rows']:
        assert row['signer_id']==2 and not row['training_approved'] and row['semantic_target_id'] is None
        assert sha(row['npz'])==row['npz_sha256']
        with np.load(row['npz'],allow_pickle=False) as z:
            native=z['xy_native']; confidence=z['confidence_native']; present=z['coordinate_present']
            frames=z['frame_index']; times=z['timestamp_nominal_us']; take=z['annotation_rows']
            assert np.array_equal(present,np.isfinite(native).all(-1)&np.isfinite(confidence)&~np.all(native==0,axis=-1))
            assert np.array_equal(times,np.rint(frames/30*1e6).astype(np.int64))
            assert np.array_equal(frames[take],np.arange(int(np.ceil(row['annotation_start_s']*30-1e-9)),int(np.ceil(row['annotation_end_s']*30-1e-9))))
            for which,selection in [('full',np.arange(len(frames))),('annotation',take)]:
                actual=z['x_'+which]
                expected=expected_channels(native[selection],present[selection],times[selection])
                error=float(np.max(np.abs(actual-expected))); max_error=max(max_error,error)
                assert np.allclose(actual,expected,atol=1e-6,rtol=1e-6),(row['clip_id'],which,error)
                assert np.array_equal(actual[:,186:279],expected[:,186:279])
                assert actual[0,-1]==0 and np.all(actual[1:,-1]>0)
                total_values+=actual.size
            total_frames+=len(frames)
    assert total_frames==1660
    overlap=read(H/'SOURCE_OVERLAP.json')
    assert all(not v for v in overlap['source_clip_id_intersection_by_group'].values())
    missing=[r['target_name'] for r in overlap['per_target'] if r['existing_dev_rows']==0]
    assert set(missing)=={'너','싫다','자다'}
    out={'status':'PASS_NEW_CANDIDATE_NUMERIC_INPUT','clips':16,'source_frames':total_frames,
        'full_and_annotation_inputs':32,'channel_values_independently_compared':total_values,
        'maximum_absolute_channel_difference':max_error,'all_masks_identical':True,
        'source_code_and_existing_metadata_unchanged':True,
        'existing_development_missing_target_names':missing,
        'semantic_equivalence_confirmed':False,'video_pts_and_time_origin_verified':False,
        'source_coordinate_accuracy_verified':False,'mediapipe_transfer_verified':False,
        'training_runs':0,'inference_runs':0,'user_or_reserved_arrays_read':0,
        'pins':{name:sha(H/name) for name in ['PLAN.json','SOURCE_OVERLAP.json','PREPARATION.json','prepare.py','audit.py']}}
    with (H/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='pins'},ensure_ascii=False))

if __name__=='__main__':main()
