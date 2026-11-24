"""Expand a verified OpenHands ST-GCN official27 model to a 49-node full-hands graph.

Nodes 0..6 keep body7, 7..27 hold all 21 left-hand landmarks and 28..48 all 21
right-hand landmarks. A deep copy receives a new graph + A buffer, a remapped
98-channel data_bn and remapped edge_importance; every other state entry stays
bit-exact. Outputs are NOT equivalent to the source (graph partitions and node
pooling change). Rebind the variable-V forward on the returned model locally.
"""
import copy

import numpy as np
import torch
from torch import nn

NUM_OLD, NUM_NEW, NUM_BODY = 27, 49, 7
OLD_RIGHT, NEW_LEFT, NEW_RIGHT = 17, 7, 28
HAND_SPARSE10 = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
OLD_TO_NEW = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 19, 20, 23, 24,
              27, 28, 32, 33, 36, 37, 40, 41, 44, 45, 48)
LAYOUTS49 = {
    'mp75': (75, (0, 2, 5, 11, 12, 13, 14) + tuple(range(33, 75))),
    'native67': (67, (0, 16, 15, 5, 2, 6, 3) + tuple(range(25, 67))),
}
SOURCE_BODY_EDGES27 = ((2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 17))
BODY_EDGES49 = ((2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 28))
HAND_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
               (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
HAND_EDGES21 = tuple((a, b) for chain in HAND_CHAINS for a, b in zip(chain, chain[1:]))
EDGES49 = BODY_EDGES49 + tuple((off + a, off + b) for off in (NEW_LEFT, NEW_RIGHT)
                               for a, b in HAND_EDGES21)
PARTITION_NONZERO49 = (49, 90, 6)
NUM_EDGE_IMPORTANCE = 10
CHILDREN = ('conv_encoder', 'agg', 'final_bn', 'final_fc')
BN_KEYS = ('weight', 'bias', 'running_mean', 'running_var')
BN_DEFAULTS = {'weight': 1.0, 'bias': 0.0, 'running_mean': 0.0, 'running_var': 1.0}
# BN_INDEX[2*i+c] is the new data_bn channel of old channel 2*i+c (node-major layout)
BN_INDEX = tuple(2 * j + c for j in OLD_TO_NEW for c in (0, 1))


def _undirected(edges):
    return {frozenset((int(i), int(j))) for i, j in edges if int(i) != int(j)}


def _bits(t):
    return t.detach().reshape(-1).contiguous().view(torch.uint8)


def _same(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
            and torch.equal(_bits(a), _bits(b)))


def _modes(model):
    return {n: m.training for n, m in model.named_modules()}


def _grads(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def _point(i):
    part = 'body' if i < NUM_BODY else 'left' if i < OLD_RIGHT else 'right'
    local = None if part == 'body' else HAND_SPARSE10[(i - NUM_BODY) % 10]
    return {'old': i, 'new': OLD_TO_NEW[i], 'part': part, 'hand_local': local}


def make_graph49(graph_class):
    """Instantiate graph_class with only get_edge overridden for the 49-node layout."""

    class FullHands49Graph(graph_class):
        def get_edge(self, layout):
            self.num_node = NUM_NEW
            self.edge = [(i, i) for i in range(NUM_NEW)] + list(EDGES49)
            self.center = 0

    graph = FullHands49Graph(layout='mediapipe-49', strategy='spatial', max_hop=1, dilation=1)
    A = np.asarray(graph.A, dtype=np.float64)
    if A.shape != (3, NUM_NEW, NUM_NEW) or not np.isfinite(A).all():
        raise ValueError(f'graph49 A must be finite [3,49,49], got shape {A.shape}')
    support = np.eye(NUM_NEW, dtype=bool)
    for i, j in EDGES49:
        support[i, j] = support[j, i] = True
    if not np.array_equal(A.any(0), support):
        raise ValueError('graph49 A support differs from self-loops + anatomical edges')
    counts = tuple(int(np.count_nonzero(a)) for a in A)
    if counts != PARTITION_NONZERO49:
        raise ValueError(f'graph49 partition nonzero counts {counts} != {PARTITION_NONZERO49}')
    return graph


def _validate_source(model):
    if not isinstance(model, nn.Module):
        raise ValueError('model27 must be an nn.Module')
    for name, module in model.named_modules():
        if 'forward' in vars(module):
            where = name or 'root'
            raise ValueError(f'instance-bound forward on {where}; pass the unbound original model')
    children = dict(model.named_children())
    missing = [c for c in CHILDREN if c not in children]
    if missing:
        raise ValueError(f'missing children: {missing}')
    if not isinstance(model.agg, nn.GRU):
        raise ValueError('agg must be nn.GRU')
    for key, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.device.type != 'cpu':
            raise ValueError(f'{key} is on {t.device}; move the model to CPU first')
        if (t.is_floating_point() or t.is_complex()) and not bool(torch.isfinite(t.detach()).all()):
            raise ValueError(f'{key} contains NaN/Inf')

    enc = model.conv_encoder
    graph = getattr(enc, 'graph', None)
    if graph is None or isinstance(graph, (nn.Module, torch.Tensor)):
        raise ValueError('conv_encoder.graph must be an official Graph instance')
    meta = tuple(getattr(graph, k, None) for k in ('num_node', 'center', 'max_hop', 'dilation'))
    if meta != (NUM_OLD, 0, 1, 1):
        raise ValueError(f'source graph num_node/center/max_hop/dilation must be (27, 0, 1, 1), got {meta}')
    gA = np.asarray(getattr(graph, 'A', None), dtype=np.float64)
    if gA.shape != (3, NUM_OLD, NUM_OLD) or not np.isfinite(gA).all():
        raise ValueError(f'source graph A must be finite [3,27,27], got shape {gA.shape}')
    edges = _undirected(getattr(graph, 'edge', ()))
    if not _undirected(SOURCE_BODY_EDGES27) <= edges or any(not 0 <= v < NUM_OLD for e in edges for v in e):
        raise ValueError('source graph edges do not match the official27 layout')

    A = enc._buffers.get('A')
    if A is None or tuple(A.shape) != (3, NUM_OLD, NUM_OLD):
        raise ValueError('conv_encoder.A must be a registered [3,27,27] buffer')
    if not torch.allclose(A.detach().double(), torch.tensor(gA), atol=1e-6, rtol=0):
        raise ValueError('conv_encoder.A does not match conv_encoder.graph.A')

    bn = enc._modules.get('data_bn')
    if not isinstance(bn, nn.BatchNorm1d) or bn.num_features != 2 * NUM_OLD:
        raise ValueError('conv_encoder.data_bn must be BatchNorm1d(54)')
    if not (bn.affine and bn.track_running_stats) or any(
            getattr(bn, k) is None for k in BN_KEYS + ('num_batches_tracked',)):
        raise ValueError('data_bn needs affine weight/bias and running stats')
    ei = enc._modules.get('edge_importance')
    if not isinstance(ei, nn.ParameterList) or len(ei) != NUM_EDGE_IMPORTANCE or any(
            tuple(p.shape) != (3, NUM_OLD, NUM_OLD) for p in ei):
        raise ValueError('conv_encoder.edge_importance must be a ParameterList of 10 [3,27,27]')


def expand_model49(model27):
    """Return (model49, report). model27 must be the unbound, verified official27 model."""
    _validate_source(model27)
    before = {k: v.detach().clone() for k, v in model27.state_dict().items()}
    src_graph = model27.conv_encoder.graph
    src_graph_A = np.array(src_graph.A, copy=True)
    modes, grads = _modes(model27), _grads(model27)

    model49 = copy.deepcopy(model27)
    enc = model49.conv_encoder
    graph49 = make_graph49(src_graph.__class__)
    enc.graph = graph49
    old_A = enc.A
    enc.register_buffer('A', torch.tensor(np.asarray(graph49.A), dtype=old_A.dtype, device=old_A.device),
                        persistent='A' not in enc._non_persistent_buffers_set)

    old_bn = enc.data_bn
    new_bn = nn.BatchNorm1d(2 * NUM_NEW, eps=old_bn.eps, momentum=old_bn.momentum, affine=old_bn.affine,
                            track_running_stats=old_bn.track_running_stats,
                            device=old_bn.weight.device, dtype=old_bn.weight.dtype)
    with torch.no_grad():
        for k in BN_KEYS:
            getattr(new_bn, k)[list(BN_INDEX)] = getattr(old_bn, k)
        new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
    new_bn.weight.requires_grad_(old_bn.weight.requires_grad)
    new_bn.bias.requires_grad_(old_bn.bias.requires_grad)
    new_bn.train(old_bn.training)
    enc.data_bn = new_bn

    node = torch.tensor(OLD_TO_NEW)
    ei = enc.edge_importance
    for k in range(len(ei)):
        old = ei[k]
        new = torch.ones((3, NUM_NEW, NUM_NEW), dtype=old.dtype, device=old.device)
        new[:, node[:, None], node[None, :]] = old.detach()
        ei[k] = nn.Parameter(new, requires_grad=old.requires_grad)

    return model49, _verify_and_report(model27, model49, before, src_graph, src_graph_A, modes, grads)


def _verify_and_report(model27, model49, before, src_graph, src_graph_A, modes, grads):
    after = model49.state_dict()
    if list(after) != list(before):
        raise RuntimeError('state_dict keys changed during expansion')
    bn_keys = {f'conv_encoder.data_bn.{k}': k for k in BN_KEYS}
    ei_keys = [f'conv_encoder.edge_importance.{k}' for k in range(NUM_EDGE_IMPORTANCE)]
    mapped = set(bn_keys) | set(ei_keys)
    recomputed = {'conv_encoder.A'} & set(after)

    bn_new = torch.ones(2 * NUM_NEW, dtype=torch.bool)
    bn_new[list(BN_INDEX)] = False
    for key, k in bn_keys.items():
        new, old = after[key], before[key]
        if not (_same(new[list(BN_INDEX)], old) and bool((new[bn_new] == BN_DEFAULTS[k]).all())):
            raise RuntimeError(f'{key}: data_bn migration check failed')
    node = torch.tensor(OLD_TO_NEW)
    ei_new = torch.ones(NUM_NEW, NUM_NEW, dtype=torch.bool)
    ei_new[node[:, None], node[None, :]] = False
    for key in ei_keys:
        new, old = after[key], before[key]
        if (new.shape != (3, NUM_NEW, NUM_NEW) or not _same(new[:, node[:, None], node[None, :]], old)
                or not bool((new[:, ei_new] == 1).all())):
            raise RuntimeError(f'{key}: edge_importance migration check failed')
    for key in recomputed:
        if after[key].shape != (3, NUM_NEW, NUM_NEW) or not bool(torch.isfinite(after[key]).all()):
            raise RuntimeError(f'{key}: recomputed adjacency is invalid')
    exact = [k for k in after if k not in mapped and k not in recomputed]
    for key in exact:
        if not _same(after[key], before[key]):
            raise RuntimeError(f'{key}: unexpected change (must be bit-exact)')

    now = model27.state_dict()
    if list(now) != list(before) or not all(_same(now[k], before[k]) for k in before):
        raise RuntimeError('source model state changed')
    if model27.conv_encoder.graph is not src_graph or not np.array_equal(np.asarray(src_graph.A), src_graph_A):
        raise RuntimeError('source graph changed')
    if _modes(model27) != modes or _modes(model49) != modes:
        raise RuntimeError('train/eval modes changed')
    if _grads(model27) != grads or _grads(model49) != grads:
        raise RuntimeError('requires_grad flags changed')
    src_ptrs = {t.data_ptr() for t in now.values() if t.numel()}
    if any(t.numel() and t.data_ptr() in src_ptrs for t in after.values()):
        raise RuntimeError('model49 shares storage with the source model')

    graph49 = model49.conv_encoder.graph
    shapes = {k: {'old': list(before[k].shape), 'new': list(after[k].shape)}
              for k in after if k in mapped or k in recomputed}
    return {
        'source_nodes': NUM_OLD,
        'target_nodes': NUM_NEW,
        'old_to_new': list(OLD_TO_NEW),
        'point_mapping': [_point(i) for i in range(NUM_OLD)],
        'new_only_nodes': [j for j in range(NUM_NEW) if j not in OLD_TO_NEW],
        'data_bn_map': 'new[2*OLD_TO_NEW[i]+c] = old[2*i+c]; other channels weight=1 bias=0 mean=0 var=1',
        'edge_importance_map': 'new[:, OLD_TO_NEW[a], OLD_TO_NEW[b]] = old[:, a, b]; other entries 1 (masked by A)',
        'graph': {'class': type(graph49).__name__, 'base': type(src_graph).__name__,
                  'strategy': 'spatial', 'max_hop': 1, 'dilation': 1, 'center': 0,
                  'edges': [list(e) for e in EDGES49],
                  'partition_nonzero': [int(np.count_nonzero(a)) for a in np.asarray(graph49.A)]},
        'keys': {'exact': exact,
                 'mapped': [k for k in after if k in mapped],
                 'recomputed': [k for k in after if k in recomputed]},
        'shapes': shapes,
        'output_equivalence': False,
        'note': 'new graph partitions and node pooling change outputs; rebind the variable-V forward locally',
    }
