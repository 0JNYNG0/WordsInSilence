"""Synthetic checks only; no source videos or devices."""
import unittest
import numpy as np
from PIL import Image
import temporal_roi as r


class Checks(unittest.TestCase):
    def test_weighted_region_not_endpoint_copy(self):
        self.assertEqual(r.propose([0,0,10,10],[20,0,30,10],[0,10000,40000]),[5,0,15,10])

    def test_time_bound(self):
        for ts in ([0,0,1],[1,0,2],[0,1,70000]):
            with self.assertRaises(ValueError):
                r.propose([0,0,10,10],[0,0,10,10],ts)

    def test_box_size_interpolation(self):
        self.assertEqual(r.propose([0,0,10,10],[0,0,20,20],[0,1,2]),[0,0,15,15])

    def test_half_open_containment(self):
        points=np.tile([10,5],(21,1))
        self.assertFalse(r.measures([0,0,10,10],[0,0,10,10],points)['all21'])

    def test_real_pixel_and_padding(self):
        rgb=np.arange(4*4*3,dtype=np.uint8).reshape(4,4,3)
        out,mask=r.crop(rgb,[-1,-1,3,3])
        expected=np.zeros((4,4,3),np.uint8); expected[1:,1:]=rgb[:3,:3]
        expected=np.asarray(Image.fromarray(expected).resize((128,128),Image.Resampling.BILINEAR))
        self.assertTrue(np.array_equal(out,expected))
        self.assertEqual(int(mask.sum()),96*96)
        self.assertTrue(np.array_equal(rgb,np.arange(48,dtype=np.uint8).reshape(4,4,3)))

    def make_observations(self):
        xy=np.full((3,75,2),np.nan); m=np.zeros((3,75),bool)
        for index,hand in enumerate(r.HANDS):
            pts=np.stack([np.arange(21)*.1+20+index*60,np.arange(21)*.1+30],axis=1)
            xy[:,hand]=pts; m[:,hand]=True
            xy[:,15+index]=pts[0]; m[:,15+index]=True
        return xy,m

    def test_hidden_target_cannot_leak(self):
        xy,m=self.make_observations(); original=xy.copy()
        d=r.identity(xy,m,0)
        self.assertTrue(d['accepted'])
        xy[1,r.HANDS[0]]=999999
        self.assertEqual(d,r.identity(xy,m,0))
        self.assertTrue(np.array_equal(m,np.isfinite(original).all(-1)))

    def test_wrong_side_endpoint_rejected(self):
        xy,m=self.make_observations()
        xy[2,r.HANDS[0]]=xy[2,r.HANDS[1]]
        self.assertFalse(r.identity(xy,m,0)['accepted'])


if __name__=='__main__':
    unittest.main()
