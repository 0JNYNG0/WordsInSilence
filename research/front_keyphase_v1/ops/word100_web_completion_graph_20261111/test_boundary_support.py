import unittest
import numpy as np
from boundary_support import attach_boundary_support

class BoundarySupportTest(unittest.TestCase):
    def test_preserves_supported_band_and_point(self):
        p=np.zeros((12,10,3));p[2:5,0,1]=.7;p[7:11,0,2]=.8;p[8,0,2]=.95
        c={'a':3,'b':8,'axis':0,'moving':'R'}
        r=attach_boundary_support([c],p,np.ones((12,10),bool))[0]
        self.assertEqual(r['onset_support_ranges'],[[2,4]])
        self.assertEqual(r['offset_support_ranges'],[[7,10]])
        self.assertEqual((r['a'],r['b']),(3,8));self.assertNotIn('semantic_complete',c)
        self.assertIsNone(r['semantic_complete'])
    def test_missing_sample_splits_band(self):
        p=np.ones((10,10,3));v=np.ones((10,10),bool);v[5,1]=False
        r=attach_boundary_support([{'a':3,'b':7,'axis':0,'moving':'L'}],p,v)[0]
        self.assertEqual(r['onset_support_ranges'],[[0,4]])
        self.assertEqual(r['offset_support_ranges'],[[6,9]])
    def test_unsupported_representative_is_not_filled(self):
        p=np.zeros((10,10,3));r=attach_boundary_support([{'a':2,'b':6,'axis':1,'moving':'R'}],p,np.ones((10,10),bool))[0]
        self.assertEqual(r['onset_support_ranges'],[]);self.assertEqual(r['offset_support_ranges'],[])
    def test_left_and_right_not_merged(self):
        p=np.zeros((10,10,3));p[1:4,1,1]=.8;p[5:8,1,2]=.8
        rr=attach_boundary_support([{'a':2,'b':6,'axis':0,'moving':side} for side in ['L','R']],p,np.ones((10,10),bool))
        self.assertEqual(rr[0]['onset_support_ranges'],[[1,3]])
        self.assertEqual(rr[1]['onset_support_ranges'],[])

if __name__=='__main__':unittest.main()
