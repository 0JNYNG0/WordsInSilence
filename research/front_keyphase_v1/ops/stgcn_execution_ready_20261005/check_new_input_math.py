"""Independent mathematical verification of the new generic adapter, synthetic only."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REF = HERE.parent / 'stgcn_reference_alignment_20261005'
sys.path.insert(0, str(REF))
from reference_adapter import LAYOUTS, load_reference
from generic_pose_io import prepare_windows, augment_windows

_, tf, _ = load_reference(REF/'reference', 'e9b38fc92391a7e913b1c1c266b65701885bea112c4dc775a026de7b1443e304')
norm = tf.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27', frame_level=False)
normalize = lambda v: norm({'frames':v})['frames']
shear, rotate = tf.ShearTransform(.1), tf.RotatationTransform(.1)
augment = lambda v: rotate(shear({'frames':v}))['frames']
checks = []
for layout, (v, joints) in LAYOUTS.items():
    for t in (1, 37, 59, 60, 61, 123, 203):
        rng = np.random.default_rng(t + v)
        xy = rng.normal(300, 40, (t,v,2)).astype(np.float32)
        mask = rng.random((t,v)) > .2
        left, right = joints[3], joints[4]
        xy[:,left] = [200,180]; xy[:,right] = [320,181]
        mask[:,[left,right]] = True
        xy[~mask] = np.nan
        original_xy, original_mask = xy.copy(), mask.copy()
        got, diag = prepare_windows(xy, mask, joints, [3,4], 60, 10, normalize)
        clean = np.where(mask[:,:,None], xy, 0).astype(np.float64)[:,joints]
        center = ((clean[:,3]+clean[:,4])/2).mean(0)
        scale = np.linalg.norm(clean[:,3]-clean[:,4],axis=1).mean()
        normalized = (clean-center)/scale
        selected = np.arange(t) if t<=60 else np.arange(60)*(t-1)//59
        sequence = np.concatenate((np.zeros((max(0,60-t),27,2)), normalized[selected]))
        expected = sequence.transpose(2,0,1).reshape(2,6,10,27).transpose(1,0,2,3)[None]
        np.testing.assert_allclose(got.numpy(), expected, rtol=2e-5, atol=2e-5)
        assert abs(diag['shoulder_distance']-scale) < 1e-8
        assert np.array_equal(xy,original_xy,equal_nan=True) and np.array_equal(mask,original_mask)
        checks.append({'layout':layout,'T':t,'max_error':float(np.abs(got.numpy()-expected).max())})
        if t in (37,123):
            np.random.seed(20261005)
            packed_aug = augment_windows(got[0],augment)
            raw_tensor = torch.from_numpy(np.ascontiguousarray(np.where(mask[:,:,None],xy,0)[:,joints].transpose(2,0,1)))
            all_norm = normalize(raw_tensor)
            np.random.seed(20261005)
            full_aug = augment(all_norm)
            sampled = full_aug[:,selected]
            padded = torch.cat((full_aug.new_zeros((2,max(0,60-t),27)),sampled),dim=1)
            ordered = padded.reshape(2,6,10,27).permute(1,0,2,3).contiguous()
            torch.testing.assert_close(packed_aug,ordered,rtol=2e-6,atol=2e-6)
            checks.append({'layout':layout,'T':t,'augmentation_commutation':True})
    # No frame has both shoulders visible, but all-T zero-filled distance is positive.
    t=4
    xy=np.ones((t,v,2),dtype=np.float32)*100
    mask=np.ones((t,v),dtype=bool)
    xy[:,left]=[90,80];xy[:,right]=[110,80]
    mask[:2,left]=False;mask[2:,right]=False
    got,diag=prepare_windows(xy,mask,joints,[3,4],60,10,normalize)
    clean=np.where(mask[:,:,None],xy,0)[:,joints]
    scale=np.linalg.norm(clean[:,3].astype(np.float64)-clean[:,4],axis=1).mean()
    assert abs(diag['shoulder_distance']-scale)<1e-8
    assert torch.isfinite(got).all()
    checks.append({'layout':layout,'alternating_missing_shoulders_allT':True})
report={'status':'PASS','synthetic_checks':len(checks),'checks':checks,'actual_data_used':0,
        'math_max_error':max(c.get('max_error',0) for c in checks),'accuracy_claim':False}
(HERE/'NEW_INPUT_MATH_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
