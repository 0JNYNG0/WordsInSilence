"""New numeric fixtures only. Never call a model or read a filmed source."""
import unittest
import numpy as np
from runtime_contract import transition_evidence,attribute_evidence,prepare_shape_observation

class RuntimeObservationContract(unittest.TestCase):
    def x(self):
        x=np.zeros((5,63),np.float32);x[:,42:]=1;x[:,18:20]=[.5,-1]
        return x
    def test_missing_wrist_blocks_arbitrarily_confident_change(self):
        x=self.x();x[2,42]=0
        self.assertEqual(transition_evidence(x,[0,0,1000])['state'],'observation_unknown')
    def test_missing_mcp_blocks_arbitrarily_confident_stable(self):
        x=self.x();x[0,51]=0
        r=transition_evidence(x,[0,1000,0]);self.assertIsNone(r['class_index'])
        self.assertEqual(r['invalid_basis_frames'],1)
    def test_degenerate_basis_does_not_become_allzero_stable(self):
        x=self.x();x[1,18:20]=0
        self.assertEqual(transition_evidence(x,[0,1000,0])['state'],'observation_unknown')
    def test_missing_fingertip_blocks_shape_even_with_valid_basis(self):
        x=self.x();x[3,42+12]=0
        self.assertEqual(attribute_evidence(x,3,[1000]*4)['state'],'observation_unknown')
        self.assertEqual(transition_evidence(x,[1000,0,0])['state'],'observation_unknown')
    def test_fully_available_fixture_retains_fixed_decision(self):
        r=transition_evidence(self.x(),[0,0,5]);self.assertEqual(r['class_index'],2)
        self.assertIsNone(r['semantic_completion'])
    def test_low_confidence_is_uncertain_not_observation_unknown(self):
        self.assertEqual(transition_evidence(self.x(),[0,0,0])['state'],'uncertain')
    def test_all_missing_cannot_emit_attribute_or_change(self):
        x=self.x();x[:,42:]=0
        self.assertEqual(transition_evidence(x,[1000,0,0])['state'],'observation_unknown')
        self.assertEqual(attribute_evidence(x,2,[1000]*4)['state'],'observation_unknown')
    def test_contract_does_not_mutate_input(self):
        x=self.x();old=x.copy();prepare_shape_observation(x);np.testing.assert_array_equal(old,x)

if __name__=='__main__':unittest.main()
