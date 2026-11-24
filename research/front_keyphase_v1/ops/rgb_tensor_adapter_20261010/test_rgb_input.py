import unittest
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from PIL import Image
from rgb_input import crop_original, pack, scatter_features, validate, load_cache, sha

torch.set_num_threads(2)


def fixture():
    available = np.array([[True, True], [False, True], [True, False]])
    estimated = np.array([[False, False], [False, True], [False, False]])
    rgb = np.zeros((3,2,224,224,3), np.uint8)
    rgb[0,0] = (255,0,0); rgb[0,1] = (0,0,255); rgb[1,1] = (0,255,0)
    boxes = np.full((3,2,4), -1, np.int32); boxes[available] = [0,0,224,224]
    return dict(pose_x=np.arange(840,dtype=np.float32).reshape(3,280),
                frame_id=np.array([2,4,8]),timestamp_us=np.array([66667,133333,266667]),
                rgb=rgb,pixel_valid=np.broadcast_to(available[:,:,None,None],(3,2,224,224)).copy(),
                available=available,observed_rgb=available & ~estimated,region_estimated=estimated,
                bbox_xyxy=boxes,landmark_observed_count=((available & ~estimated)*21).astype(np.int32))


class InputTests(unittest.TestCase):
    def test_arm_masks_and_same_pose(self):
        a=fixture(); packs=[pack(a,x) for x in 'ABC']
        self.assertEqual([len(p['images']) for p in packs],[0,3,4])
        for p in packs: np.testing.assert_array_equal(p['pose_x'],a['pose_x'])
        self.assertFalse(packs[1]['region_estimated'].any())
        self.assertTrue(packs[2]['region_estimated'][1,1])

    def test_rgb_order_and_side(self):
        p=pack(fixture()); np.testing.assert_array_equal(p['slots'],[[0,0],[0,1],[1,1],[2,0]])
        expected=(np.array([1.,0.,0.])-[.485,.456,.406])/[.229,.224,.225]
        np.testing.assert_allclose(p['images'][0,:,0,0],expected,atol=1e-6)
        self.assertGreater(p['images'][1,2,0,0],p['images'][1,0,0,0])

    def test_black_is_not_missing(self):
        p=pack(fixture()); self.assertTrue(p['available'][2,0]); self.assertFalse(p['available'][2,1])
        self.assertEqual(len(p['images']),4)
        self.assertTrue((p['images'][-1] < 0).all())

    def test_all_missing(self):
        a=fixture()
        for k in ('available','observed_rgb','region_estimated','pixel_valid','rgb','landmark_observed_count'): a[k][...]=0
        a['bbox_xyxy'][...]=-1
        p=pack(a); self.assertEqual(tuple(p['images'].shape),(0,3,224,224))
        out=scatter_features(torch.empty((0,7)),p['slots'],3)
        self.assertEqual(tuple(out.shape),(3,2,7)); self.assertFalse(out.any())

    def test_scatter_identity_and_gradient(self):
        p=pack(fixture()); f=torch.arange(20,dtype=torch.float32).reshape(4,5).requires_grad_()
        out=scatter_features(f,p['slots'],3)
        self.assertTrue(torch.equal(out[p['available']],f)); self.assertFalse(out[~p['available']].any())
        out.sum().backward(); self.assertTrue(torch.equal(f.grad,torch.ones_like(f)))

    def test_metadata_not_feature_and_nonmutation(self):
        a=fixture(); before=deepcopy(a); p=pack(a)
        a.update(label='WRONG',signer=999,prior_classification='uncertain')
        q=pack(a)
        self.assertEqual(set(q),set(p))
        for k in p: self.assertTrue(torch.equal(p[k],q[k]))
        for k in before: np.testing.assert_array_equal(before[k],a[k])

    def test_wrong_time_rejected(self):
        a=fixture(); a['timestamp_us'][1]=a['timestamp_us'][0]
        with self.assertRaises(ValueError): pack(a)

    def test_bad_arm_and_counts(self):
        with self.assertRaises(ValueError): pack(fixture(),'D')
        a=fixture(); a['landmark_observed_count'][1,1]=21
        with self.assertRaises(ValueError): pack(a)

    def test_missing_payload_rejected(self):
        a=fixture(); a['rgb'][1,0,0,0]=255
        with self.assertRaises(ValueError): pack(a)

    def test_conflicting_provenance(self):
        a=fixture(); a['observed_rgb'][1,1]=True
        with self.assertRaises(ValueError): pack(a)

    def test_bad_scatter(self):
        for slots in (torch.tensor([[0,0],[0,0]]),torch.tensor([[0,0],[0,2]]),torch.tensor([[0,0],[3,0]])):
            with self.assertRaises(ValueError): scatter_features(torch.ones((2,5)),slots,3)
        with self.assertRaises(ValueError): scatter_features(torch.tensor([[float('nan')]]),torch.tensor([[0,0]]),3)

    def test_original_crop_padding_and_edges(self):
        a=np.zeros((240,240,3),np.uint8); a[:,:]=(10,20,30)
        a[:,0]=(255,0,0); a[:,223]=(0,0,255)
        im,mask=crop_original(a,np.array([0,0,224,224]))
        np.testing.assert_array_equal(im,a[:224,:224]); self.assertTrue(mask.all())
        im,mask=crop_original(a,np.array([-4,-4,236,236]))
        reference=np.array(Image.fromarray(a).crop((-4,-4,236,236)).resize((224,224),Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(im,reference); self.assertFalse(mask[0,0]); self.assertTrue(mask[-1,-1])

    def test_crop_invalid_inputs(self):
        a=np.zeros((10,10,3),np.uint8)
        for box in ([-1,0,3,6],[50,50,60,60],[0,0,1,1]):
            with self.assertRaises(ValueError): crop_original(a,np.array(box))
        with self.assertRaises(ValueError): crop_original(a.astype(float),np.array([0,0,5,5]))

    def test_hash_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d)/'fixture.npz'; np.savez_compressed(p,**fixture())
            a=load_cache(p,sha(p)); self.assertEqual(validate(a),3)
            with self.assertRaises(ValueError): load_cache(p,'0'*64)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(InputTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors)}
    dest=Path(__file__).parent/'SYNTHETIC_CHECK.json'
    with dest.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
