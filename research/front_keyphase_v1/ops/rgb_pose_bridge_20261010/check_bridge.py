"""Connection/gradient checks only; no training, labels, ranking or accuracy evaluation."""
from pathlib import Path
import json
import sys
import copy
import time
import tempfile
import unittest
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

UNIT=Path(__file__).resolve().parent
BASE=UNIT.parents[1]
ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE.parent/'sign_model_pilot_v1/src'),str(BASE/'ops/rgb_tensor_adapter_20261010')]
from pilot.model_masked_v1 import MaskedSignTCN
from rgb_input import load_cache,pack,scatter_features
from rgb_bridge import FrozenRGBEncoder,PoseRGBBridge,file_sha

torch.set_num_threads(2)
torch.manual_seed(640)
PARENT=BASE/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
PARENT_SHA='83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
WEIGHTS=json.loads((UNIT/'WEIGHTS.json').read_text(encoding='utf-8'))


def write(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def parent():
    assert file_sha(PARENT)==PARENT_SHA
    c=torch.load(PARENT,map_location='cpu',weights_only=True)
    cfg=c.get('config',{})
    def get(k): return c[k] if k in c else cfg[k]
    m=MaskedSignTCN(width=get('width'),dilations=tuple(get('dilations')),kernel_size=get('kernel_size'),dropout=get('dropout'),n_classes=len(c['classes']))
    m.load_state_dict(c['state_dict'],strict=True)
    return m.eval()


def clone_state(m): return {k:v.detach().clone() for k,v in m.state_dict().items()}


def state_same(m,s): return all(torch.equal(v,s[k]) for k,v in m.state_dict().items())


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=parent()
        cls.encoder=FrozenRGBEncoder(WEIGHTS['path'],WEIGHTS['sha256'])

    def setUp(self):
        self.bridge=PoseRGBBridge(self.parent)
        self.x=torch.randn(2,7,280)
        self.pad=torch.tensor([[1]*7,[1]*4+[0]*3],dtype=torch.float32)
        self.available=self.pad.bool().unsqueeze(-1).expand(-1,-1,2).clone()
        self.available[0,2,0]=False
        self.estimated=torch.zeros_like(self.available); self.estimated[0,3,1]=True
        self.features=torch.randn(2,7,2,576)

    def test_initial_parity(self):
        with torch.no_grad():
            actual=self.bridge(self.x,self.pad,self.features,self.available,self.estimated)
            expected=self.parent(self.x,self.pad)
        self.assertTrue(torch.equal(actual,expected))

    def test_only_bridge_gradient_and_frozen_statistics(self):
        state=clone_state(self.bridge.base)
        self.bridge.train()
        y=self.bridge(self.x,self.pad,self.features,self.available,self.estimated)
        y.square().mean().backward()
        self.assertGreater(float(self.bridge.rgb_projection.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in self.bridge.base.parameters()))
        self.assertTrue(state_same(self.bridge.base,state)); self.assertFalse(self.bridge.base.training)

    def test_all_missing_even_with_nonzero_projection(self):
        with torch.no_grad(): self.bridge.rgb_projection.weight.fill_(.0001)
        missing=torch.zeros_like(self.available)
        dirty=torch.full_like(self.features,float('nan'))
        with torch.no_grad():
            actual=self.bridge(self.x,self.pad,dirty,missing,missing)
            expected=self.parent(self.x,self.pad)
        self.assertTrue(torch.equal(actual,expected))

    def test_missing_feature_and_padding_ignored(self):
        with torch.no_grad(): self.bridge.rgb_projection.weight.copy_(torch.randn_like(self.bridge.rgb_projection.weight)*.0001)
        changed=self.features.clone(); changed[~self.available]=float('nan')
        with torch.no_grad():
            a=self.bridge(self.x,self.pad,self.features,self.available,self.estimated)
            b=self.bridge(self.x,self.pad,changed,self.available,self.estimated)
        self.assertTrue(torch.equal(a,b))
        # Independently compare the short sequence without padded time.
        with torch.no_grad(): c=self.bridge(self.x[1:2,:4],self.pad[1:2,:4],self.features[1:2,:4],self.available[1:2,:4],self.estimated[1:2,:4])
        torch.testing.assert_close(a[1:2],c,rtol=1e-5,atol=1e-5)

    def test_reference_hook_equivalence(self):
        with torch.no_grad(): self.bridge.rgb_projection.weight.copy_(torch.linspace(-.0001,.0001,self.bridge.rgb_projection.weight.numel()).reshape_as(self.bridge.rgb_projection.weight))
        base=copy.deepcopy(self.parent)
        clean=self.features.clone(); clean[~self.available]=0
        extra=torch.cat([clean.flatten(2),self.available.float(),self.estimated.float()],-1)
        addition=(extra@self.bridge.rgb_projection.weight.T).transpose(1,2)
        handle=base.proj.register_forward_hook(lambda module,args,out:out+addition)
        with torch.no_grad():
            expected=base(self.x,self.pad)
            actual=self.bridge(self.x,self.pad,self.features,self.available,self.estimated)
        handle.remove()
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

    def test_invalid_masks_and_features_fail(self):
        bad=self.estimated.clone(); bad[0,2,0]=True
        with self.assertRaises(ValueError): self.bridge(self.x,self.pad,self.features,self.available,bad)
        bad=self.available.clone(); bad[1,6,0]=True
        with self.assertRaises(ValueError): self.bridge(self.x,self.pad,self.features,bad,self.estimated)
        bad=self.features.clone(); bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError): self.bridge(self.x,self.pad,bad,self.available,self.estimated)

    def test_encoder_empty_and_frozen(self):
        before=clone_state(self.encoder); calls=self.encoder.forward_images
        self.encoder.train()
        empty=self.encoder(torch.empty(0,3,224,224))
        self.assertEqual(tuple(empty.shape),(0,576)); self.assertEqual(calls,self.encoder.forward_images)
        self.assertFalse(self.encoder.training); self.assertTrue(state_same(self.encoder,before))
        self.assertTrue(all(not p.requires_grad for p in self.encoder.parameters()))

    def test_encoder_missing_hash_and_invalid_image_fail(self):
        with self.assertRaises(FileNotFoundError): FrozenRGBEncoder(UNIT/'absent.pt',WEIGHTS['sha256'])
        with self.assertRaises(ValueError): FrozenRGBEncoder(WEIGHTS['path'],'0'*64)
        with self.assertRaises(ValueError): self.encoder(torch.zeros(1,3,128,128))
        with self.assertRaises(ValueError): self.encoder(torch.full((1,3,224,224),float('nan')))


def real_connection_check():
    records=json.loads((BASE/'runs/rgb_tensor_adapter_v1/CACHE_MANIFEST.json').read_text(encoding='utf-8'))
    selected=[records[0]]
    for key,predicate in [('region_estimated',lambda a:bool(a.any())),('available',lambda a:bool((~a).any()))]:
        for r in records:
            with np.load(r['cache_path'],allow_pickle=False) as z: good=predicate(z[key])
            if good:
                if r not in selected: selected.append(r)
                break
    encoder=FrozenRGBEncoder(WEIGHTS['path'],WEIGHTS['sha256'])
    model=PoseRGBBridge(parent())
    parent_state=clone_state(model.base); encoder_state=clone_state(encoder)
    ref=mobilenet_v3_small(weights=None).eval()
    ref.load_state_dict(torch.load(WEIGHTS['path'],map_location='cpu',weights_only=True),strict=True)
    cases=[]; checks=[]; started=time.perf_counter(); feature_variation=False
    for r in selected:
        a=load_cache(r['cache_path'],r['cache_sha256']); p=pack(a,'C')
        features=encoder(p['images'],batch_size=16)
        with torch.no_grad(): reference=ref.avgpool(ref.features(p['images'][:2])).flatten(1)
        reference_error=float((reference-features[:2]).abs().max())
        # Batch-size-dependent CPU roundoff is bounded, never used to select a model.
        assert reference_error<=1e-5
        if len(features)>1: feature_variation |= not torch.equal(features[0],features[1])
        c=scatter_features(features,p['slots'],len(a['pose_x']))
        available=p['available'].unsqueeze(0); estimated=p['region_estimated'].unsqueeze(0)
        x=p['pose_x'].unsqueeze(0); pad=torch.ones(x.shape[:2])
        with torch.no_grad():
            result=model(x,pad,c.unsqueeze(0),available,estimated)
            base=model.base(x,pad)
        assert torch.equal(result,base) and torch.isfinite(result).all()
        # Reuse exact same per-image feature in B, never independently rescore pixels for selection.
        b=c.clone(); b[p['region_estimated']]=0
        with torch.no_grad(): result_b=model(x,pad,b.unsqueeze(0),torch.from_numpy(a['observed_rgb']).unsqueeze(0),torch.zeros_like(estimated))
        assert torch.equal(result_b,base)
        cases.append({'sample_id':r['sample_id'],'frames':len(a['pose_x']),'encoded_images':len(features),
                      'estimated_regions':int(estimated.sum()),'available_feature_shape':list(features.shape),
                      'feature_finite':True,'official_encoder_reference_max_abs_error':reference_error,
                      'initial_full93_logit_max_abs_error':float((result-base).abs().max()),
                      'no_label_ranking_or_accuracy_evaluated':True})
        checks.append({'path':r['cache_path'],'sha256':r['cache_sha256']})
    assert feature_variation and state_same(model.base,parent_state) and state_same(encoder,encoder_state)
    assert file_sha(PARENT)==PARENT_SHA and file_sha(WEIGHTS['path'])==WEIGHTS['sha256']
    assert all(file_sha(c['path'])==c['sha256'] for c in checks)
    write(UNIT/'CONNECTION_CHECK.json',{'status':'PASS','selection':'first fixture, first estimated-region fixture, first missing-image fixture; deduplicated, no label selection',
        'cases':cases,'rgb_feature_dim_per_hand':576,'bridge_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'different_images_have_different_features':True,'source_model_and_cache_hashes_unchanged':True,
        'frozen_encoder_and_parent_buffers_unchanged':True,'elapsed_seconds':time.perf_counter()-started,
        'sign_training_performed':False,'checkpoint_saved':False,'web_registered':False,
        'interpretation':'At zero bridge initialization, output equals pose baseline by design. This is connection validation, not an RGB recognition gain.',
        'verification':'Codex implementation plus separate hook reference/official backbone comparison, not external human review'})


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(UNIT/'SYNTHETIC_CHECK.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures)})
    if not result.wasSuccessful(): raise SystemExit(1)
    real_connection_check()
    print('Connection and frozen-weight checks PASS',flush=True)
