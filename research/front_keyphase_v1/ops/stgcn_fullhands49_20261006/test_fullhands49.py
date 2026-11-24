"""Synthetic unit tests for fullhands49 (no checkpoints, datasets, files or training).

Run locally: python -m unittest test_fullhands49 -v
"""
import collections
import json
import types
import unittest

import numpy as np
import torch
from torch import nn

import fullhands49 as fh

SPARSE = fh.HAND_SPARSE10
BODY27 = ((2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 17))
# toy sparse hand over local sparse indices 0..9, incl. old wrist->fingertip shortcuts
HAND27 = ((0, 1), (0, 2), (2, 3), (0, 3), (0, 4), (4, 5), (0, 6), (6, 7), (0, 8), (8, 9))
BN_DEFAULTS = {'weight': 1.0, 'bias': 0.0, 'running_mean': 0.0, 'running_var': 1.0}
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


class ToyGraph:
    """Public ST-GCN Graph formulas with a toy mediapipe-27 layout."""

    def __init__(self, layout='mediapipe-27', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout != 'mediapipe-27':
            raise ValueError(layout)
        self.num_node = 27
        hands = [(o + a, o + b) for o in (7, 17) for a, b in HAND27]
        self.edge = [(i, i) for i in range(27)] + list(BODY27) + hands
        self.center = 0

    def get_adjacency(self, strategy):
        if strategy != 'spatial':
            raise ValueError(strategy)
        n, hop_dis, c = self.num_node, self.hop_dis, self.center
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((n, n))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        norm = normalize_digraph(adjacency)
        A = []
        for hop in valid_hop:
            a_root, a_close, a_further = (np.zeros((n, n)) for _ in range(3))
            for i in range(n):
                for j in range(n):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, c] == hop_dis[i, c]:
                            a_root[j, i] = norm[j, i]
                        elif hop_dis[j, c] > hop_dis[i, c]:
                            a_close[j, i] = norm[j, i]
                        else:
                            a_further[j, i] = norm[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        self.A = np.stack(A)


ORIGINAL_GET_EDGE = ToyGraph.get_edge


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = ToyGraph()
        self.register_buffer('A', torch.tensor(self.graph.A, dtype=torch.float32))
        self.data_bn = nn.BatchNorm1d(54, eps=1e-3, momentum=0.05)
        self.gcn = nn.Conv2d(2, 4, 1)  # V-independent stand-ins
        self.tcn = nn.Conv2d(4, 4, (3, 1), padding=(1, 0))
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(3, 27, 27)) for _ in range(10)])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_encoder = FakeEncoder()
        self.agg = nn.GRU(4, 5, batch_first=True)
        self.final_bn = nn.BatchNorm1d(5)
        self.final_fc = nn.Linear(5, 3)


def make_model():
    m = FakeModel()
    with torch.no_grad():
        for n, (key, t) in enumerate(m.state_dict(keep_vars=True).items()):
            if key == 'conv_encoder.A':
                continue
            if key.endswith('num_batches_tracked'):
                t.fill_(17 + n)
                continue
            vals = torch.arange(t.numel(), dtype=t.dtype).reshape(t.shape)
            scale = 0.25 if key.endswith('running_var') else 1.0 / t.numel()
            t.copy_(n + 2 + scale * vals)  # distinct per entry/key, never 0 or 1
    m.train()
    m.agg.eval()
    m.conv_encoder.data_bn.eval()
    m.conv_encoder.tcn.eval()
    m.conv_encoder.edge_importance[3].requires_grad_(False)
    m.conv_encoder.data_bn.bias.requires_grad_(False)
    m.final_fc.bias.requires_grad_(False)
    return m


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class FullHands49Test(unittest.TestCase):
    def setUp(self):
        self.src = make_model()
        self.before = snapshot(self.src)
        self.m49, self.report = fh.expand_model49(self.src)

    def test_point_mapping_and_layouts(self):
        expected = tuple(range(7)) + tuple(7 + s for s in SPARSE) + tuple(28 + s for s in SPARSE)
        self.assertEqual(fh.OLD_TO_NEW, expected)
        self.assertEqual(len(set(fh.OLD_TO_NEW)), 27)
        self.assertEqual(len(self.report['new_only_nodes']), 22)
        for p in self.report['point_mapping']:
            offset = {'body': None, 'left': 7, 'right': 28}[p['part']]
            if offset is None:
                self.assertEqual(p['new'], p['old'])
            else:
                self.assertEqual(p['new'], offset + p['hand_local'])
        for name, body, left, right in (('mp75', (0, 2, 5, 11, 12, 13, 14), 33, 54),
                                        ('native67', (0, 16, 15, 5, 2, 6, 3), 25, 46)):
            with self.subTest(name):
                n, idx = fh.LAYOUTS49[name]
                self.assertEqual(len(idx), 49)
                self.assertEqual(len(set(idx)), 49)
                self.assertLess(max(idx), n)
                self.assertEqual(idx[:7], body)
                for k in range(21):
                    self.assertEqual(idx[7 + k], left + k)
                    self.assertEqual(idx[28 + k], right + k)
                old27 = body + tuple(left + s for s in SPARSE) + tuple(right + s for s in SPARSE)
                self.assertEqual(tuple(idx[j] for j in fh.OLD_TO_NEW), old27)

    def test_anatomical_edges(self):
        und = {frozenset(e) for e in fh.EDGES49}
        self.assertEqual(len(fh.EDGES49), 48)
        self.assertEqual(len(und), 48)
        self.assertTrue(all(len(e) == 2 for e in und))
        body = {frozenset(e) for e in ((2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 28))}
        hand = {frozenset((off + a, off + b)) for off in (7, 28) for chain in CHAINS
                for a, b in zip(chain, chain[1:])}
        self.assertEqual(und, body | hand)
        deg = collections.Counter(v for e in und for v in e)
        for off in (7, 28):
            for tip in (4, 8, 12, 16, 20):
                self.assertEqual(deg[off + tip], 1)
                self.assertNotIn(frozenset((off, off + tip)), und)
            for a, b in ((1, 5), (5, 9), (9, 13), (13, 17), (2, 5)):
                self.assertNotIn(frozenset((off + a, off + b)), und)
            self.assertEqual(deg[off], 6)  # elbow + five finger bases
        self.assertEqual({v for e in und if 5 in e for v in e} - {5}, {3, 7})
        self.assertEqual({v for e in und if 6 in e for v in e} - {6}, {4, 28})
        g = fh.make_graph49(ToyGraph)
        self.assertEqual(set(g.edge), {(i, i) for i in range(49)} | set(fh.EDGES49))
        self.assertEqual(len(g.edge), 97)

    def test_adjacency_support_and_partitions(self):
        g = fh.make_graph49(ToyGraph)
        self.assertIsInstance(g, ToyGraph)
        self.assertEqual({k for k in vars(type(g)) if not k.startswith('__')}, {'get_edge'})
        self.assertEqual((g.num_node, g.center, g.max_hop, g.dilation), (49, 0, 1, 1))
        A = np.asarray(g.A)
        self.assertEqual(A.shape, (3, 49, 49))
        self.assertTrue(np.isfinite(A).all())
        self.assertEqual([int(np.count_nonzero(a)) for a in A], [49, 90, 6])
        expected = np.eye(49, dtype=bool)
        for i, j in fh.EDGES49:
            expected[i, j] = expected[j, i] = True
        np.testing.assert_array_equal((A != 0).any(0), expected)
        further = {tuple(int(x) for x in p) for p in np.argwhere(A[2] != 0)}
        self.assertEqual(further, {(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6)})
        np.testing.assert_allclose(A.sum(axis=(0, 1)), np.ones(49))
        enc = self.m49.conv_encoder
        self.assertIn('A', enc._buffers)
        self.assertEqual(enc.A.dtype, torch.float32)
        self.assertTrue(torch.equal(enc.A, torch.tensor(A, dtype=torch.float32)))
        self.assertIsInstance(enc.graph, ToyGraph)
        self.assertEqual(enc.graph.num_node, 49)
        self.assertEqual(self.report['graph']['partition_nonzero'], [49, 90, 6])

    def test_data_bn_migration(self):
        old, new = self.src.conv_encoder.data_bn, self.m49.conv_encoder.data_bn
        self.assertIsNot(new, old)
        self.assertEqual(new.num_features, 98)
        for attr in ('eps', 'momentum', 'affine', 'track_running_stats', 'training'):
            self.assertEqual(getattr(new, attr), getattr(old, attr), attr)
        for key, default in BN_DEFAULTS.items():
            o, n = getattr(old, key).detach(), getattr(new, key).detach()
            self.assertEqual((n.dtype, n.device), (o.dtype, o.device))
            hit = set()
            for i in range(27):
                for c in range(2):
                    j = 2 * fh.OLD_TO_NEW[i] + c
                    self.assertEqual(n[j].item(), o[2 * i + c].item(), (key, i, c))
                    hit.add(j)
            self.assertEqual(len(hit), 54)
            for j in set(range(98)) - hit:
                self.assertEqual(n[j].item(), default, (key, j))
        self.assertTrue(torch.equal(new.num_batches_tracked, old.num_batches_tracked))
        self.assertTrue(new.weight.requires_grad)
        self.assertFalse(new.bias.requires_grad)

    def test_edge_importance_migration(self):
        idx = list(fh.OLD_TO_NEW)
        fresh = np.ones((49, 49), dtype=bool)
        fresh[np.ix_(idx, idx)] = False
        fresh_t = torch.from_numpy(fresh)
        for k in range(10):
            with self.subTest(k=k):
                old = self.src.conv_encoder.edge_importance[k]
                new = self.m49.conv_encoder.edge_importance[k]
                self.assertIsInstance(new, nn.Parameter)
                self.assertEqual(tuple(new.shape), (3, 49, 49))
                self.assertEqual(new.requires_grad, old.requires_grad)
                n, o = new.detach(), old.detach()
                self.assertTrue(torch.equal(n[:, idx][:, :, idx], o))
                for a, b in ((8, 20), (20, 8), (16, 26), (0, 17), (26, 26), (6, 17)):
                    self.assertTrue(torch.equal(n[:, idx[a], idx[b]], o[:, a, b]))
                self.assertTrue(bool((n[:, fresh_t] == 1).all()))
                self.assertEqual(int((n == 1).sum()), 3 * int(fresh.sum()))
        self.assertFalse(self.m49.conv_encoder.edge_importance[3].requires_grad)

    def test_report_and_other_state_exact(self):
        sd = self.m49.state_dict()
        keys = self.report['keys']
        mapped = {f'conv_encoder.data_bn.{k}' for k in BN_DEFAULTS}
        mapped |= {f'conv_encoder.edge_importance.{k}' for k in range(10)}
        self.assertEqual(set(keys['mapped']), mapped)
        self.assertEqual(keys['recomputed'], ['conv_encoder.A'])
        all_keys = keys['exact'] + keys['mapped'] + keys['recomputed']
        self.assertEqual(len(all_keys), len(set(all_keys)))
        self.assertEqual(set(all_keys), set(sd))
        self.assertIn('conv_encoder.data_bn.num_batches_tracked', keys['exact'])
        self.assertTrue(any(k.startswith('agg.') for k in keys['exact']))
        for k in keys['exact']:
            self.assertEqual(sd[k].dtype, self.before[k].dtype, k)
            self.assertTrue(torch.equal(sd[k], self.before[k]), k)
        self.assertEqual(self.report['shapes']['conv_encoder.data_bn.weight'], {'old': [54], 'new': [98]})
        self.assertEqual(self.report['shapes']['conv_encoder.edge_importance.9'],
                         {'old': [3, 27, 27], 'new': [3, 49, 49]})
        self.assertEqual(self.report['shapes']['conv_encoder.A'], {'old': [3, 27, 27], 'new': [3, 49, 49]})
        self.assertFalse(self.report['output_equivalence'])
        json.dumps(self.report)  # plain data only, no tensors
        self.assertEqual({n: m.training for n, m in self.m49.named_modules()},
                         {n: m.training for n, m in self.src.named_modules()})
        self.assertEqual({n: p.requires_grad for n, p in self.m49.named_parameters()},
                         {n: p.requires_grad for n, p in self.src.named_parameters()})
        self.assertIs(type(self.m49), FakeModel)

    def test_source_unchanged(self):
        sd = self.src.state_dict()
        self.assertEqual(list(sd), list(self.before))
        for k, v in self.before.items():
            self.assertTrue(torch.equal(sd[k], v), k)
        enc = self.src.conv_encoder
        self.assertIs(type(enc.graph), ToyGraph)
        self.assertEqual(enc.graph.num_node, 27)
        self.assertEqual(tuple(enc.A.shape), (3, 27, 27))
        self.assertEqual(enc.data_bn.num_features, 54)
        self.assertEqual(tuple(enc.edge_importance[0].shape), (3, 27, 27))
        self.assertIs(ToyGraph.get_edge, ORIGINAL_GET_EDGE)
        self.assertEqual(ToyGraph().num_node, 27)
        self.assertNotIn('forward', vars(self.src))

    def test_deep_copy_independence(self):
        self.assertIsNot(self.m49, self.src)
        for name in fh.CHILDREN:
            self.assertIsNot(getattr(self.m49, name), getattr(self.src, name))
        self.assertIsNot(self.m49.conv_encoder.graph, self.src.conv_encoder.graph)
        src_ptrs = {t.data_ptr() for t in self.src.state_dict().values()}
        for k, t in self.m49.state_dict().items():
            self.assertNotIn(t.data_ptr(), src_ptrs, k)
        modes = {n: m.training for n, m in self.src.named_modules()}
        with torch.no_grad():
            for t in self.m49.state_dict().values():
                t.add_(1)
        self.m49.eval()
        for k, v in self.src.state_dict().items():
            self.assertTrue(torch.equal(v, self.before[k]), k)
        self.assertEqual({n: m.training for n, m in self.src.named_modules()}, modes)

    def test_repeatable_and_rng_free(self):
        src = make_model()
        torch_state, np_state = torch.get_rng_state(), np.random.get_state()
        a, rep_a = fh.expand_model49(src)
        b, rep_b = fh.expand_model49(src)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        np_after = np.random.get_state()
        self.assertEqual(np_state[0], np_after[0])
        np.testing.assert_array_equal(np_state[1], np_after[1])
        self.assertEqual(np_state[2:], np_after[2:])
        self.assertEqual(rep_a, rep_b)
        sd_b = b.state_dict()
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, sd_b[k]), k)

    def test_rejects_invalid_sources(self):
        def bind(module):
            module.forward = types.MethodType(lambda self, x: x, module)

        def drop_body_edge(m):
            g = m.conv_encoder.graph
            g.edge = [e for e in g.edge if set(e) != {6, 17}]

        enc = lambda m: m.conv_encoder
        cases = {
            'bound_forward': lambda m: bind(m),
            'bound_encoder_forward': lambda m: bind(enc(m)),
            'missing_child': lambda m: delattr(m, 'final_fc'),
            'agg_not_gru': lambda m: setattr(m, 'agg', nn.LSTM(4, 5)),
            'edge_importance_9': lambda m: setattr(
                enc(m), 'edge_importance', nn.ParameterList(list(enc(m).edge_importance)[:9])),
            'edge_importance_shape': lambda m: enc(m).edge_importance.__setitem__(
                0, nn.Parameter(torch.ones(3, 26, 26))),
            'bn_size': lambda m: setattr(enc(m), 'data_bn', nn.BatchNorm1d(52)),
            'bn_no_affine': lambda m: setattr(enc(m), 'data_bn', nn.BatchNorm1d(54, affine=False)),
            'bn_no_stats': lambda m: setattr(enc(m), 'data_bn', nn.BatchNorm1d(54, track_running_stats=False)),
            'nan_param': lambda m: m.final_fc.weight.data[0].fill_(float('nan')),
            'inf_buffer': lambda m: m.final_bn.running_var.fill_(float('inf')),
            'A_shape': lambda m: setattr(enc(m), 'A', torch.zeros(3, 26, 26)),
            'A_mismatch': lambda m: enc(m).A.mul_(2),
            'graph_max_hop': lambda m: setattr(enc(m).graph, 'max_hop', 2),
            'graph_center': lambda m: setattr(enc(m).graph, 'center', 1),
            'graph_body_edge': drop_body_edge,
            'meta_device': lambda m: setattr(m.final_fc, 'weight', nn.Parameter(torch.empty(3, 5, device='meta'))),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                model = make_model()
                mutate(model)
                with self.assertRaises(ValueError):
                    fh.expand_model49(model)

    def test_make_graph49_rejects_broken_graph_class(self):
        class DenseGraph(ToyGraph):
            def get_adjacency(self, strategy):
                self.A = np.ones((3, self.num_node, self.num_node))

        class NanGraph(ToyGraph):
            def get_adjacency(self, strategy):
                self.A = np.full((3, self.num_node, self.num_node), np.nan)

        class FlatGraph(ToyGraph):
            def get_adjacency(self, strategy):
                self.A = np.eye(self.num_node)[None]

        for cls in (DenseGraph, NanGraph, FlatGraph):
            with self.subTest(cls.__name__):
                with self.assertRaises(ValueError):
                    fh.make_graph49(cls)


if __name__ == '__main__':
    unittest.main()
