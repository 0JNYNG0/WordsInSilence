"""Synthetic-only tests; no datasets, devices, files, or external calls."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

from geometry_tcn import GeometryTCN, PAIRS, fit_scale, hand_distances, predict, train_pair


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=1)
        self.norm = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(0.8)

    def forward(self, h, mask):
        return (h + self.drop(torch.tanh(self.norm(self.conv(h * mask))))) * mask


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(5, 7, 1)
        self.blocks = nn.ModuleList([Block(7), Block(7)])
        self.head = nn.Linear(7, 3)

    def forward(self, x, pad):
        mask = pad.unsqueeze(1)
        h = self.proj(x.transpose(1, 2) * mask) * mask
        for block in self.blocks:
            h = block(h, mask)
        return self.head((h * mask).sum(2) / mask.sum(2))


def geometry(length, value=1.0):
    return np.concatenate((np.full((length, 18), value, np.float32),
                           np.ones((length, 18), np.float32)), axis=1)


class DistanceTests(unittest.TestCase):
    def test_pairs_345_zero_missing(self):
        self.assertEqual(PAIRS, ((4, 8), (4, 12), (4, 16), (4, 20), (0, 4),
                                 (0, 8), (0, 12), (0, 16), (0, 20)))
        q = np.zeros((2, 2, 21, 2), np.float32)
        q[:, :, 8] = [3, 4]
        present = np.ones((2, 2, 21), bool)
        original = q.copy()
        e = hand_distances(q, present)
        self.assertEqual(e.shape, (2, 36)); self.assertEqual(e.dtype, np.float32)
        np.testing.assert_array_equal(e[:, [0, 5, 9, 14]], 5)
        self.assertEqual(e[0, 1], 0); self.assertEqual(e[0, 19], 1)
        np.testing.assert_array_equal(q, original)
        q[0, 0, 8] = np.nan; present[0, 0, 8] = False
        e = hand_distances(q, present)
        self.assertEqual(e[0, 0], 0); self.assertEqual(e[0, 18], 0)
        self.assertEqual(e[0, 1], 0); self.assertEqual(e[0, 19], 1)
        present[0, 0, 8] = True
        with self.assertRaises(ValueError): hand_distances(q, present)

    def test_rigid_and_scale(self):
        q = np.arange(84, dtype=np.float64).reshape(1, 2, 21, 2) / 31
        p = np.ones(q.shape[:-1], bool)
        a = hand_distances(q, p)
        for changed in (q + [3, -7], q * [-1, 1], q @ np.array([[0., -1.], [1., 0.]])):
            np.testing.assert_allclose(hand_distances(changed, p), a, atol=2e-7)
        scaled = hand_distances(q * 2, p)
        np.testing.assert_allclose(scaled[:, :18], 2*a[:, :18])
        np.testing.assert_array_equal(scaled[:, 18:], a[:, 18:])

    def test_bad_inputs(self):
        q = np.zeros((1, 2, 21, 2), np.float32); p = np.ones(q.shape[:-1], np.float32)
        for badq in (q[0], q[:0], np.zeros(q.shape, np.int32)):
            with self.assertRaises((ValueError, TypeError)): hand_distances(badq, p)
        for badp in (p[:, :, :20], np.full(p.shape, .5), np.full(p.shape, np.nan)):
            with self.assertRaises((ValueError, TypeError)): hand_distances(q, badp)
        q[0, 0, 10] = np.inf
        with self.assertRaises(ValueError): hand_distances(q, p)

    def test_scale_class_balance_and_missing(self):
        a, b, c = geometry(2, 1), geometry(1, 3), geometry(4, 5)
        # Class0 mean of clips (1 + 9)/2 = 5; class1 = 25; class-balanced mean = 15.
        s = fit_scale([a, b, c], np.array([0, 0, 1], np.int64))
        np.testing.assert_allclose(s, np.sqrt(15), rtol=1e-7)
        for e in (a, b, c): e[:, 1] = 0; e[:, 19] = 0; e[:, 2] = 0
        c[:, 0] = 0; c[:, 18] = 0
        s = fit_scale([a, b, c], [0, 0, 1])
        self.assertAlmostEqual(float(s[0]), np.sqrt(5), places=6)
        self.assertEqual(s[1], 1); self.assertEqual(s[2], 1)
        a[0, 1] = 1
        with self.assertRaises(ValueError): fit_scale([a], [0])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.parent = Parent().eval()
        self.xs = [np.linspace(-1, 1, 15, dtype=np.float32).reshape(3, 5),
                   np.linspace(1, -1, 35, dtype=np.float32).reshape(7, 5)]
        self.es = [geometry(3, .7), geometry(7, 1.3)]
        self.scale = np.ones(18, np.float32)

    def test_initial_clone_rng_modes_and_trainables(self):
        before = copy.deepcopy(self.parent.state_dict()); rng = torch.random.get_rng_state().clone()
        model = GeometryTCN(self.parent, self.scale)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(model.extra_weight.shape, (7, 36, 1))
        for name, value in self.parent.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
            self.assertNotEqual(value.data_ptr(), model.base.state_dict()[name].data_ptr())
        names = set(model.trainable_names)
        expected = {'extra_weight'} | {'base.blocks.1.'+n for n,_ in self.parent.blocks[-1].named_parameters()} | {'base.head.'+n for n,_ in self.parent.head.named_parameters()}
        self.assertEqual(names, expected)
        self.assertEqual(model.trainable_count, sum(p.numel() for p in model.parameters() if p.requires_grad))
        model.train(); self.assertTrue(model.training)
        self.assertTrue(all(not m.training for m in model.base.modules()))
        model.eval()
        for x,e in zip(self.xs,self.es):
            xx=torch.from_numpy(x[None]); ee=torch.from_numpy(e[None]); pad=torch.ones((1,len(x)))
            torch.testing.assert_close(model(xx,ee,pad), self.parent(xx,pad), rtol=0, atol=0)

    def test_gradient_padding_and_predict_state(self):
        model=GeometryTCN(self.parent,self.scale)
        x=torch.from_numpy(self.xs[0][None]); e=torch.from_numpy(self.es[0][None]); p=torch.ones((1,3))
        model(x,e,p).square().sum().backward()
        self.assertIsNotNone(model.extra_weight.grad)
        self.assertTrue(bool(model.extra_weight.grad.abs().sum()>0))
        before=copy.deepcopy(model.state_dict()); rng=torch.random.get_rng_state().clone()
        alone=predict(model,self.xs,self.es,batch_size=1)
        together=predict(model,self.xs,self.es,batch_size=2)
        np.testing.assert_allclose(alone,together,rtol=1e-5,atol=1e-7)
        self.assertEqual(together.dtype,np.float64)
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(before[k],v))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        with self.assertRaises(ValueError): predict(model,[],[])
        model.train()
        with self.assertRaises(ValueError): predict(model,self.xs,self.es)

    def test_tiny_pair(self):
        before=copy.deepcopy(self.parent.state_dict()); rng=torch.random.get_rng_state().clone()
        teacher=np.zeros((2,3),np.float64)
        for i,x in enumerate(self.xs):
            with torch.inference_mode(): teacher[i]=self.parent(torch.from_numpy(x[None]),torch.ones((1,len(x)))).numpy()[0]
        schedule=np.array([[0],[1]],np.int64); notices=[]
        result=train_pair(self.parent,self.xs,np.array([0,1],np.int64),self.xs,np.array([2,0],np.int64),teacher,
                          schedule,schedule[::-1].copy(),self.es,self.es,self.scale,
                          dict(lr=.001,weight_decay=.001,anchor_coefficient=.01,output_coefficient=1.),
                          progress=lambda *args:notices.append(args))
        self.assertEqual(set(result),{'control','treatment'})
        for arm in result:
            item=result[arm]; self.assertEqual(item['updates'],2)
            self.assertEqual(item['losses'].shape,(2,4)); self.assertTrue(np.isfinite(item['losses']).all())
            self.assertTrue(item['frozen_parameters_unchanged']); self.assertTrue(item['buffers_unchanged'])
            np.testing.assert_allclose(item['losses'][:,3],item['losses'][:,:3].sum(1),atol=1e-12)
            restored=GeometryTCN(self.parent,self.scale,enabled=arm=='treatment')
            restored.load_state_dict(item['state_dict'])
            self.assertEqual(predict(restored,self.xs,self.es).shape,(2,3))
        self.assertFalse(result['control']['extra_weight_nonzero'])
        self.assertTrue(result['treatment']['extra_weight_nonzero'])
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for k,v in self.parent.state_dict().items(): self.assertTrue(torch.equal(before[k],v))
        self.assertEqual(len(notices),2)

    def test_validation(self):
        with self.assertRaises(ValueError): GeometryTCN(self.parent,np.zeros(18,np.float32))
        with self.assertRaises(ValueError): GeometryTCN(self.parent.double(),self.scale)
        self.parent.float(); model=GeometryTCN(self.parent,self.scale)
        x=torch.zeros((1,3,5)); e=torch.zeros((1,3,36)); p=torch.ones((1,3))
        e[:,:,18:]=.5
        with self.assertRaises(ValueError): model(x,e,p)
        e.zero_()
        with self.assertRaises(ValueError): model(x,e,torch.zeros_like(p))


if __name__ == '__main__':
    unittest.main()
