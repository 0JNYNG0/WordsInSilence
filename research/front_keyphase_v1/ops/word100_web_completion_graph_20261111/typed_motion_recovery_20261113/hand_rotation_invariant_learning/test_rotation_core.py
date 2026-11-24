"""Synthetic geometry invariants only; no source/model inference or accuracy test."""
import unittest
import numpy as np
from rotation_core import canonicalize_hand63,rotate_hand63,BASIS_EPSILON

class RotationContract(unittest.TestCase):
    def sample(self,n=9):
        p=np.random.default_rng(20261113).normal(size=(n,21,2)).astype(np.float32)
        p[:,0]=0;p[:,9]=np.c_[np.linspace(.3,1.,n),np.linspace(-.5,-1.2,n)]
        return np.c_[p.reshape(n,42),np.ones((n,21),np.float32)].astype(np.float32)

    def test_mcp_basis_up_and_scale_preserved(self):
        x=self.sample();y,v,a=canonicalize_hand63(x);p=y[:,:42].reshape(-1,21,2)
        self.assertTrue(v.all());np.testing.assert_allclose(p[:,9,0],0,atol=2e-7)
        np.testing.assert_allclose(p[:,9,1],-a['basis_length'],atol=2e-7)
        np.testing.assert_array_equal(y[:,42:],x[:,42:])
        np.testing.assert_allclose(np.linalg.norm(p[:,[8,12,16,20]]-p[:,[5,9,13,17]],axis=-1),
            np.linalg.norm(x[:,:42].reshape(-1,21,2)[:,[8,12,16,20]]-x[:,:42].reshape(-1,21,2)[:,[5,9,13,17]],axis=-1),atol=5e-7)

    def test_constant_rigid_rotations_do_not_change_canonical_shape(self):
        x=self.sample();y,v,_=canonicalize_hand63(x)
        for angle in [-90,-60,-30,30,60,90]:
            z,zv,_=canonicalize_hand63(rotate_hand63(x,angle));np.testing.assert_array_equal(v,zv)
            np.testing.assert_allclose(y,z,atol=8e-7)

    def test_framewise_rotation_does_not_become_finger_motion(self):
        x=np.repeat(self.sample(1),19,axis=0);z=rotate_hand63(x,np.linspace(-90,90,19))
        y,v,_=canonicalize_hand63(z);self.assertTrue(v.all())
        np.testing.assert_allclose(y,np.repeat(y[:1],len(y),axis=0),atol=8e-7)

    def test_scale_is_not_collapsed(self):
        x=self.sample();y,_,_=canonicalize_hand63(x);xx=x.copy();xx[:,:42]*=2
        yy,_,_=canonicalize_hand63(xx);np.testing.assert_allclose(yy[:,:42],2*y[:,:42],atol=8e-7)

    def test_missing_basis_and_degenerate_basis_are_unknown(self):
        x=self.sample(4);x[0,42]=0;x[1,51]=0;x[2,18:20]=0;x[3,18:20]=[BASIS_EPSILON/2,0]
        y,v,_=canonicalize_hand63(x);self.assertFalse(v.any());np.testing.assert_array_equal(y[:,:42],0)
        np.testing.assert_array_equal(y[:,42:],x[:,42:])

    def test_unavailable_nonbasis_point_stays_zero(self):
        x=self.sample();x[:,42+8]=0;x[:,16:18]=99
        y,v,_=canonicalize_hand63(x);self.assertTrue(v.all());np.testing.assert_array_equal(y[:,16:18],0)
        np.testing.assert_array_equal(y[:,42:],x[:,42:])

    def test_left_reflection_is_caller_owned_not_applied_twice(self):
        x=self.sample();left=x.copy();left[:,:42:2]*=-1
        caller_canonical=left.copy();caller_canonical[:,:42:2]*=-1
        a,_,_=canonicalize_hand63(x);b,_,audit=canonicalize_hand63(caller_canonical)
        np.testing.assert_allclose(a,b,atol=1e-7);self.assertFalse(audit['anatomical_reflection_applied'])
        unreflected,_,_=canonicalize_hand63(left)
        self.assertGreater(float(np.max(np.abs(a[:,:42]-unreflected[:,:42]))),.1)

    def test_rotation_keeps_missing_mask_and_shape_distances(self):
        x=self.sample();x[:,42+20]=0;x[:,40:42]=0;y=rotate_hand63(x,60)
        np.testing.assert_array_equal(x[:,42:],y[:,42:]);np.testing.assert_array_equal(y[:,40:42],0)
        np.testing.assert_allclose(np.linalg.norm(x[:,:42].reshape(-1,21,2),axis=-1),np.linalg.norm(y[:,:42].reshape(-1,21,2),axis=-1),atol=5e-7)

    def test_malformed_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):canonicalize_hand63(np.zeros((2,62)))
        x=self.sample();x[0,2]=np.nan
        with self.assertRaises(ValueError):canonicalize_hand63(x)

if __name__=='__main__':unittest.main()
