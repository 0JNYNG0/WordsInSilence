"""Pure training-driver contract checks. No optimizer steps, inference, extraction, or inputs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode=True
import numpy as np
from training_core import (SEEDS, ARMS, MODEL, FAILURE_CE, schedule, validate_channels,
                           padded_arrays, prediction, detailed_metrics, paired,
                           aggregate, progression_gate)
import training_driver as driver


def dev_predictions(correct=300, failed=0):
    result=[]
    for label in range(100):
        for signer in (11,12,13):
            row=dict(clip_id=f'fixture_{label:03d}_{signer}',label_index=label,signer_id=signer)
            index=len(result)
            if index>=300-failed:
                result.append(prediction(row,reason='synthetic_missing')); continue
            logits=np.full(100,-3.,dtype=np.float64)
            logits[label if index<correct else (label+1)%100]=3.
            result.append(prediction(row,logits))
    return result


class ContractTests(unittest.TestCase):
    def test_four_epoch_spatial_temporal_factorial(self):
        for cid in ('NIA_SL_WORD0003_REAL01_F','NIA_SL_WORD2732_REAL02_F', 'fixture_123'):
            for start in (1,5,77):
                self.assertEqual({schedule(cid,e,'C') for e in range(start,start+4)},
                                 {(s,t) for s in ('masked','sensor') for t in ('x_whole','x_context')})
            for e in range(1,81):
                parity=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2
                expected='x_context' if ((e+parity)%2==1) else 'x_whole'
                self.assertEqual(schedule(cid,e,'A'),('native',expected))
                self.assertEqual(schedule(cid,e,'B'),('masked',expected))
                self.assertEqual(schedule(cid,e,'C')[1],expected)

    def test_same_permutation_and899_exposure(self):
        for seed in SEEDS:
            for epoch in (1,2,3,4,80):
                orders=[np.random.default_rng(seed+epoch).permutation(899).tolist() for _ in ARMS]
                self.assertEqual(orders[0],orders[1]); self.assertEqual(orders[1],orders[2])
                self.assertEqual(sorted(orders[0]),list(range(899)))
                self.assertEqual(len(range(0,899,32)),29)

    def test_real_missing_frame_not_padding(self):
        a=np.zeros((4,280),np.float32);a[1:,279]=.03
        b=np.zeros((2,280),np.float32);b[1:,279]=.03
        validate_channels(a);validate_channels(b)
        x,p=padded_arrays([a,b])
        np.testing.assert_array_equal(p,[[1,1,1,1],[1,1,0,0]])
        self.assertEqual(x[0,2,186:279].sum(),0)
        self.assertEqual(p[0,2],1)

    def test_masked_coordinate_rejected(self):
        x=np.zeros((1,280),np.float32);x[0,0]=1
        with self.assertRaises(AssertionError): validate_channels(x)
        x[0,0]=0;x[0,102]=1
        with self.assertRaises(AssertionError): validate_channels(x)

    def test_invalid_rows_keep300_denominator_and_ln100(self):
        rows=dev_predictions(correct=299,failed=1)
        m=detailed_metrics(rows)
        self.assertEqual(m['total'],300);self.assertEqual(m['top1_correct'],299)
        self.assertEqual(m['input_failures'],1)
        self.assertEqual(rows[-1]['cross_entropy'],FAILURE_CE)
        self.assertFalse(rows[-1]['correct']);self.assertFalse(rows[-1]['top5_correct'])
        allfailed=dev_predictions(correct=0,failed=300)
        self.assertAlmostEqual(detailed_metrics(allfailed)['mean_ce'],np.log(100),places=12)

    def test_logits_recount_and_top5_tie(self):
        row=dict(clip_id='fixture',label_index=99,signer_id=11)
        p=prediction(row,np.zeros(100))
        self.assertEqual(p['prediction'],0);self.assertEqual(p['top5'],[0,1,2,3,4])
        self.assertAlmostEqual(p['cross_entropy'],FAILURE_CE)
        self.assertFalse(p['correct']);self.assertFalse(p['top5_correct'])

    def test_paired_gains_losses_not_rank_confusion(self):
        a=dev_predictions(290);b=dev_predictions(291)
        x=paired(a,b)
        self.assertEqual(x['net_correct'],1);self.assertEqual(len(x['improved']),1)
        self.assertEqual(len(x['regressed']),0)
        self.assertAlmostEqual(x['percentage_point_difference'],1/3)

    def test_gate_mean_seed_count_and_original_tolerance(self):
        m0={v:detailed_metrics(dev_predictions(290 if v=='native' else 270)) for v in ('native','sensor')}
        arms={a:{} for a in ARMS}
        for a in ARMS:
            for v in ('masked','sensor'):
                n=287 if v=='masked' else (272 if a=='C' else 271)
                arms[a][v]=aggregate([detailed_metrics(dev_predictions(n)) for _ in SEEDS])
        g=progression_gate(arms,m0)
        self.assertTrue(g['pass_to_next_review']);self.assertFalse(g['automatic_adoption'])
        arms['C']['masked']=aggregate([detailed_metrics(dev_predictions(286)) for _ in SEEDS])
        self.assertFalse(progression_gate(arms,m0)['pass_to_next_review'])
        arms['C']['masked']=aggregate([detailed_metrics(dev_predictions(287)) for _ in SEEDS])
        arms['C']['sensor']=aggregate([detailed_metrics(dev_predictions(n)) for n in (277,270,270)])
        g=progression_gate(arms,m0)
        self.assertTrue(g['checks']['C_sensor_mean_exceeds_B'])
        self.assertFalse(g['checks']['C_sensor_better_than_B_at_least_two_seeds'])

    def test_model_initialization_equal_and_parameter_count(self):
        # Construction only: no forward, backward, optimizer step, or real input read.
        for seed in SEEDS:
            hashes=[]
            for _ in ARMS:
                driver.set_seed(seed);m=driver.MaskedSignTCN(**MODEL)
                self.assertEqual(m.parameter_count(),37636)
                hashes.append(driver.state_hash(m.state_dict()))
            self.assertEqual(len(set(hashes)),1)

    def test_driver_has_no_detector_or_source_extractor(self):
        tree=ast.parse(Path(driver.__file__).read_text(encoding='utf-8'))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom): imports.append(node.module or '')
            elif isinstance(node,ast.Import): imports.extend(x.name for x in node.names)
        self.assertFalse(any('mediapipe' in x or 'video_source' in x or 'stable_rest' in x for x in imports))


if __name__=='__main__':
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                failures=[str(x[0]) for x in result.failures],errors=[str(x[0]) for x in result.errors],
                seconds=time.perf_counter()-start,training_steps=0,inference_calls=0,detector_calls=0,
                real_input_arrays_read=0,model_construction_only=True,
                pins={str(p):driver.sha(p) for p in [Path(__file__),Path(driver.__file__),Path(__file__).with_name('training_core.py')]})
    driver.atomic_json(Path(__file__).with_name('training_SELFTEST_RESULT.json'),report)
    sys.exit(0 if result.wasSuccessful() else 1)
