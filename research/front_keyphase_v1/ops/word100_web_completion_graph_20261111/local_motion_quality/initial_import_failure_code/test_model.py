import unittest
from pathlib import Path
import numpy as np,torch
from model import LocalMotionQuality,make_sequence,raw_sequence

O=Path(__file__).resolve().parents[2]

class LocalQualityContractTest(unittest.TestCase):
    def test_no_difference_across_missing_observation(self):
        v=np.arange(20,dtype=np.float32).reshape(5,4);mask=np.array([1,1,0,1,1],bool)
        seq=make_sequence(v,mask,np.arange(5)*33333)
        np.testing.assert_array_equal(seq[2:4,4:8],0)
        np.testing.assert_array_equal(seq[2,:4],0)
        self.assertEqual(seq[3,9],0)
        np.testing.assert_array_equal(seq[4,4:8],v[4]-v[3])
    def test_crop_does_not_inherit_external_step(self):
        v=np.arange(28,dtype=np.float32).reshape(7,4);ts=np.array([0,30000,60000,100000,140000,180000,220000])
        seq=raw_sequence(v,np.ones(7,bool),ts,2,5)
        np.testing.assert_array_equal(seq[0,4:8],0)
        self.assertEqual(seq[0,9],0);self.assertEqual(seq[0,10],0)
        self.assertAlmostEqual(seq[1,10],.04,places=6)
    def test_zero_temporal_extension_preserves_existing_quality(self):
        torch.manual_seed(19);a=LocalMotionQuality('B',42,O).eval();b=LocalMotionQuality('C',42,O).eval()
        summary=torch.randn(2,304);seq=torch.zeros(2,6,11);seq[:,:,8]=1;lengths=torch.tensor([6,4])
        with torch.no_grad():ap,aa=a(summary,seq,lengths);bp,ba=b(summary,seq,lengths)
        torch.testing.assert_close(ap,bp,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(aa,ba,rtol=1e-6,atol=1e-6)
    def test_padding_has_no_effect_even_with_active_time_branch(self):
        torch.manual_seed(19);net=LocalMotionQuality('C',42,O).eval()
        with torch.no_grad():net.summary.weight[:,304:]=.01
        summary=torch.randn(1,304);seq=torch.randn(1,6,11);seq[:,:,8:10]=1;seq[:,:,10]=.033
        padded=torch.cat([seq,torch.full((1,4,11),100.)],1)
        with torch.no_grad():a,_=net(summary,seq,torch.tensor([6]));b,_=net(summary,padded,torch.tensor([6]))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
