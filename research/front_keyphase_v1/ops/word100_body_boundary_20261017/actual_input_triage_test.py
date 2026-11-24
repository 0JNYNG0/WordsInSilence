"""Synthetic observation checks; no detector, classifier, source video or training."""
import copy
import unittest
import numpy as np
from actual_input_triage import analyze


def fixture():
    n=15
    xy=np.full((n,75,2),np.nan,dtype=np.float32)
    xy[:,11]=[400,300];xy[:,12]=[600,300]
    xy[:,33:54]=[700,900]  # down hand outside the fixed ROI
    xy[:,54:75]=[450,500]  # raised inside hand
    ok=np.isfinite(xy).all(-1)
    native={'xy_px':xy,'observed':ok.copy(),'usable':ok.copy(),
            'timestamp_us':np.arange(n)*33333,'frame_id':np.arange(n)}
    mask=copy.deepcopy(native)
    inside=ok&(xy[:,:,1]<800)
    mask['observed']=inside.copy();mask['usable']=inside.copy();mask['xy_px'][~inside]=np.nan
    row={'clip_id':'synthetic','canonical_label':'synthetic','signer_id':1,'intervals':[[0,1.0]]}
    rec={'calibration':{'box_source_pixels':[0,0,1920,800],'method':'preserved_stable_rest'},'visual_flags':[]}
    fore={'bottom':800,'method':'preserved_stable_rest'}
    lineage={'existing_source_visual_evidence':[]}
    return row,rec,native,mask,copy.deepcopy(mask),fore,None,lineage


class ActualInputTriageTests(unittest.TestCase):
    def test_down_hand_is_geometry_not_sensor_loss(self):
        args=fixture();r=analyze(*args)
        self.assertEqual(r['hands']['left']['native_to_mask_geometry']['newly_absent_core_frames'],15)
        self.assertEqual(r['hands']['left']['mask_to_sensor_observation']['frames'],0)
        self.assertEqual(r['zero_hand_intervals']['native_has_hand_but_mask_has_none']['frames'],0)
        self.assertEqual(r['semantic_status'],'UNDETERMINED_REVIEW_ONLY')

    def test_sensor_missing_inside_hand_is_separate(self):
        args=fixture();sensor=args[4]
        for key in ('observed','usable'):sensor[key][3:13,54:75]=False
        sensor['xy_px'][3:13,54:75]=np.nan
        r=analyze(*args)
        self.assertIn('sensor_zero_hand_while_mask_has_hand_ge_0_30s',r['tags'])
        self.assertEqual(r['zero_hand_intervals']['mask_has_hand_but_sensor_has_none']['frames'],10)
        self.assertEqual(r['zero_hand_intervals']['native_has_hand_but_mask_has_none']['frames'],0)

    def test_native_missing_hand_is_not_roi_erasure(self):
        args=fixture()
        for obs in args[2:5]:
            for key in ('observed','usable'):obs[key][4:6,54:75]=False
            obs['xy_px'][4:6,54:75]=np.nan
        r=analyze(*args)
        self.assertEqual(r['zero_hand_intervals']['native_has_hand_but_mask_has_none']['frames'],2)
        # Native still has the down left hand; absent right is not listed as newly geometrically removed.
        self.assertEqual(r['hands']['right']['native_to_mask_geometry']['newly_absent_core_frames'],0)
        self.assertEqual(r['hands']['right']['mask_to_sensor_observation']['frames'],0)

    def test_mask_contract_mismatch_is_error(self):
        args=fixture();args[3]['usable'][2,54]=False
        with self.assertRaisesRegex(AssertionError,'masked geometry'):
            analyze(*args)

    def test_frame_alignment_mismatch_is_error(self):
        args=fixture();args[4]['timestamp_us'][2]+=1
        with self.assertRaises(AssertionError):analyze(*args)

    def test_boundary_change_is_review_not_automatic_failure(self):
        args=fixture();args[5]['bottom']=790
        r=analyze(*args)
        self.assertTrue(r['boundary_changed'])
        self.assertEqual(r['boundary_delta_pixels'],10)
        self.assertIn('actual_boundary_changed_review_source_geometry',r['additional_review_reasons'])
        self.assertEqual(r['semantic_status'],'UNDETERMINED_REVIEW_ONLY')


if __name__=='__main__':unittest.main()
