import unittest
import numpy as np
from relation_mass import relation_mass,candidate_distribution

class ContractTests(unittest.TestCase):
    def d(self,intervals,weights=None):
        return np.asarray(intervals,np.int64),np.ones(len(intervals))/len(intervals) if weights is None else np.asarray(weights,float)
    def test_shared_boundary(self):
        a=self.d([[0,3]]);b=self.d([[3,5]])
        self.assertEqual(relation_mass(a,b)['forward'],1)
        self.assertEqual(relation_mass(b,a)['reverse'],1)
    def test_positive_overlap(self):
        r=relation_mass(self.d([[0,4]]),self.d([[3,5]]))
        self.assertEqual(r['overlap'],1)
    def test_known_mixture(self):
        r=relation_mass(self.d([[0,2],[4,6],[8,10]],[.2,.3,.5]),self.d([[3,7]]))
        np.testing.assert_allclose([r['forward'],r['reverse'],r['overlap']],[.2,.5,.3])
        self.assertEqual(r['state'],'mixed_candidate_relations')
    def test_identical_ambiguous(self):
        a=self.d([[0,2],[4,6],[8,10]])
        r=relation_mass(a,a)
        np.testing.assert_allclose([r['forward'],r['reverse'],r['overlap']],[1/3]*3)
        self.assertFalse(r['action_presence_established'])
    def test_time_transforms(self):
        a=self.d([[0,5],[2,7]],[.3,.7]);b=self.d([[6,8],[9,12]],[.8,.2]);r=relation_mass(a,b)
        rev=lambda d:(100-d[0][:,::-1],d[1])
        rr=relation_mass(rev(a),rev(b));self.assertAlmostEqual(r['forward'],rr['reverse']);self.assertAlmostEqual(r['reverse'],rr['forward'])
        rr=relation_mass((a[0]*7+91,a[1]),(b[0]*7+91,b[1]));self.assertEqual(r['state'],rr['state']);self.assertAlmostEqual(r['forward'],rr['forward'])
    def test_gate_and_shift(self):
        spans=np.array([[0,2],[1,3]]);t=np.arange(4,dtype=np.int64)*33333
        a=candidate_distribution(spans,[1,2],[0,0],t);b=candidate_distribution(spans,[1001,1002],[0,0],t)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])
        self.assertIsNone(candidate_distribution(spans,[1,2],[-1,-1],t))
        self.assertEqual(relation_mass(a,None)['state'],'no_admitted_candidates')
    def test_duplicates_not_counted(self):
        with self.assertRaises(ValueError):candidate_distribution(np.array([[0,2],[0,2]]),[0,0],[0,0],np.arange(3))
    def test_unobserved_presence_never_inferred(self):
        r=relation_mass(self.d([[0,2]]),self.d([[4,6]]));self.assertEqual(r['state'],'forward');self.assertFalse(r['action_presence_established'])

if __name__=='__main__':unittest.main()
