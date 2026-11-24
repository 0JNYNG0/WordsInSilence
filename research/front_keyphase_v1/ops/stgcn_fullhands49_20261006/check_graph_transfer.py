"""Independent fullhands49 migration audit; synthetic tensors only.

Read original pinned weights, check the new component against an independent
adjacency oracle, then make one disposable optimizer update. No source clips,
dataset arrays, evaluation scores or checkpoint output are used. Importing this
file performs no audit. Call check(factory49=...) to additionally compare a
runtime factory, or run this file once to write GRAPH_TRANSFER_CHECK.json.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
import time
import traceback

import numpy as np
import torch

U = Path(__file__).resolve().parent
W = U.parent.parent
PREVIOUS = U.parent / 'stgcn_valid_window_pair_20261006'
REFERENCE = U.parent / 'stgcn_reference_alignment_20261005'
CORE = U.parent / 'stgcn_transfer_compare_20261005'
ARTIFACT = U.parent / 'openhands_artifact_20261005'
BASE_PLAN_SHA = 'e4dc2ad90f70bd337de37890d92d614f260d45338dbf97818547fe37b8bc2389'
BASE_INIT_SHA = '087479ad99b7a568158982fb55b9e0b14090e1d491424d0a929a022003e59a5d'
POINT_MAP = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 19, 20, 23, 24, 27,
             28, 32, 33, 36, 37, 40, 41, 44, 45, 48)
EXPECTED_LAYOUTS = {
    'mp75': (75, (0, 2, 5, 11, 12, 13, 14) + tuple(range(33, 75))),
    'native67': (67, (0, 16, 15, 5, 2, 6, 3) + tuple(range(25, 67))),
}


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == Path(path).resolve(), f'Wrong import: {name}')
    return module


def state_digest(state):
    """Independent implementation of the existing gw.state_digest format."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.ascontiguousarray(state[name].detach().cpu().numpy())
        header = json.dumps([name, value.dtype.str, list(value.shape)],
                            sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest.update(header.encode('utf-8'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def modes(model):
    return {name: child.training for name, child in model.named_modules()}


def graph_oracle():
    """Binary edges and closed-form one-hop partition; no production graph calls."""
    edges = [(2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 28)]
    for offset in (7, 28):
        for base in (1, 5, 9, 13, 17):
            chain = (0, base, base + 1, base + 2, base + 3)
            edges.extend((offset + a, offset + b) for a, b in zip(chain, chain[1:]))
    unordered = {tuple(sorted(edge)) for edge in edges}
    need(len(edges) == len(unordered) == 48, 'Independent oracle anatomy count')
    neighbors = [set() for _ in range(49)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    reached, pending = {0}, [0]
    while pending:
        for other in neighbors[pending.pop()]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    need(len(reached) == 49, 'Independent graph must be connected')
    # Retain the official truncated policy: farther-than-one-hop means infinity.
    center_distance = np.full(49, np.inf)
    center_distance[0] = 0
    center_distance[list(neighbors[0])] = 1
    result = np.zeros((3, 49, 49), dtype=np.float64)
    for destination in range(49):
        weight = 1.0 / (len(neighbors[destination]) + 1)
        result[0, destination, destination] = weight
        for source in neighbors[destination]:
            # Official a_root+a_close combines equal and greater source distance.
            partition = 1 if center_distance[source] >= center_distance[destination] else 2
            result[partition, source, destination] = weight
    need([int(np.count_nonzero(p)) for p in result] == [49, 90, 6], 'Oracle partition counts')
    return result, unordered, center_distance


def assert_graph(graph, expected, expected_edges, center_distance):
    need(graph.num_node == 49 and graph.center == 0, 'Graph node count/center')
    need(graph.max_hop == graph.dilation == 1, 'Graph hop/dilation changed')
    actual_edges = [tuple(map(int, edge)) for edge in graph.edge]
    self_edges = [a for a, b in actual_edges if a == b]
    nonself = [tuple(sorted((a, b))) for a, b in actual_edges if a != b]
    need(len(actual_edges) == 97 and sorted(self_edges) == list(range(49)), 'Graph self-loop count/order')
    need(len(nonself) == 48 and set(nonself) == expected_edges, 'Graph anatomy or shortcut mismatch')
    need(np.array_equal(np.asarray(graph.hop_dis)[:, 0], center_distance), 'Truncated center distances changed')
    actual = np.asarray(graph.A)
    need(actual.shape == (3, 49, 49) and np.isfinite(actual).all(), 'Graph shape/nonfinite')
    need(np.allclose(actual, expected, rtol=0, atol=1e-15), 'Independent normalized partition mismatch')


def check(factory49=None, batch_size=16):
    """Run once after application; optional zero-argument runtime factory returns a model."""
    need(isinstance(batch_size, int) and batch_size >= 2, 'Synthetic batch_size must be >=2')
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        need(torch.get_num_interop_threads() == 1, 'Interop threads already configured differently')
    torch.use_deterministic_algorithms(True)
    for directory in (CORE, REFERENCE, PREVIOUS):
        sys.path.insert(0, str(directory))
    import reference_adapter as ra
    import stgcn_core as core
    need(Path(ra.__file__).resolve() == REFERENCE / 'reference_adapter.py', 'Wrong reference adapter')
    need(Path(core.__file__).resolve() == CORE / 'stgcn_core.py', 'Wrong tensor reader')
    old_engine = load_module('_fullhands49_audit_previous_engine', PREVIOUS / 'training_engine.py')
    cached_engine = sys.modules.get('training_engine')
    sys.modules['training_engine'] = old_engine
    try:
        gw = load_module('_fullhands49_audit_previous_workflow', PREVIOUS / 'generic_workflow.py')
    finally:
        if cached_engine is None:
            sys.modules.pop('training_engine', None)
        else:
            sys.modules['training_engine'] = cached_engine
    component = load_module('_fullhands49_audited_component', U / 'fullhands49.py')
    forward_module = load_module('_fullhands49_audit_valid_forward', PREVIOUS / 'valid_window_forward.py')
    pinned = [Path(__file__), U / 'fullhands49.py', U / 'SCOPE.md',
              PREVIOUS / 'FROZEN_EXPERIMENT.json', PREVIOUS / 'CONFIG.json',
              PREVIOUS / 'generic_workflow.py', PREVIOUS / 'training_engine.py',
              PREVIOUS / 'valid_window_forward.py', REFERENCE / 'reference_adapter.py',
              REFERENCE / 'reference/SOURCE_PINS.json', CORE / 'stgcn_core.py',
              ARTIFACT / 'raw_dpc.zip', ARTIFACT / 'TENSOR_METADATA.json',
              W / 'runs/stgcn_full_path_v1/INIT_SELECT.json']
    source_pins = read(REFERENCE / 'reference/SOURCE_PINS.json')
    pinned += [Path(entry['local_path']) for entry in source_pins['files']]
    before_hashes = {str(path): sha(path) for path in pinned}
    need(sha(PREVIOUS / 'FROZEN_EXPERIMENT.json') == BASE_PLAN_SHA, 'Previous plan pin changed')
    plan = read(PREVIOUS / 'FROZEN_EXPERIMENT.json')
    config = read(PREVIOUS / 'CONFIG.json')
    need(config == plan['training_config'], 'Previous config/plan mismatch')
    need(sha(ARTIFACT / 'TENSOR_METADATA.json') == plan['pins']['pretrained_metadata'], 'Metadata pin changed')
    need(sha(REFERENCE / 'reference/SOURCE_PINS.json') == plan['pins']['source_pins'], 'Official-source pin changed')
    need(sha(PREVIOUS / 'valid_window_forward.py') == plan['comparison_source']['source_component_sha256'],
         'Valid-window source changed')
    tensors, _ = core.read_encoder_tensors(ARTIFACT / 'raw_dpc.zip', ARTIFACT / 'TENSOR_METADATA.json',
                                          plan['pins']['pretrained_zip'], encoder_prefix='model.')
    enc = {k[len('conv_encoder.'):]: v for k, v in tensors.items() if k.startswith('conv_encoder.')}
    gru = {k[len('agg.'):]: v for k, v in tensors.items() if k.startswith('agg.')}
    need(len(enc) == 170 and len(gru) == 4, 'Original pretrained state counts')
    dpc, _, _ = ra.load_reference(REFERENCE / 'reference', plan['pins']['source_pins'])

    def base_factory():
        return ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=config['seed'])[0]

    source = base_factory()
    source_state = snapshot(source)
    source_digest = state_digest(source_state)
    need(source_digest == gw.state_digest(source_state) == BASE_INIT_SHA
         == read(W / 'runs/stgcn_full_path_v1/INIT_SELECT.json')['state_digest'], 'Original INIT mismatch')
    source_modes = modes(source)
    source_grad_flags = {name: p.requires_grad for name, p in source.named_parameters()}
    source_graph_class = type(source.conv_encoder.graph)
    source_get_edge = source_graph_class.get_edge
    source_graph_A = np.asarray(source.conv_encoder.graph.A).copy()
    source_graph_edges = [tuple(edge) for edge in source.conv_encoder.graph.edge]
    need(tuple(component.OLD_TO_NEW) == POINT_MAP, 'Component node map mismatch')
    need(component.LAYOUTS49 == EXPECTED_LAYOUTS, 'Component raw input layouts mismatch')
    for name, (raw_count, selected) in EXPECTED_LAYOUTS.items():
        need(len(selected) == len(set(selected)) == 49 and max(selected) < raw_count, f'Invalid layout: {name}')
        need(tuple(selected[i] for i in POINT_MAP) == tuple(ra.LAYOUTS[name][1]), f'Old27 alignment: {name}')
    expected_graph, expected_edges, center_distance = graph_oracle()
    graph = component.make_graph49(source_graph_class)
    assert_graph(graph, expected_graph, expected_edges, center_distance)
    need(type(graph).get_adjacency is source_graph_class.get_adjacency, 'Official partition method replaced')
    need(type(graph).__init__ is source_graph_class.__init__, 'Official graph constructor replaced')

    rng_before = torch.random.get_rng_state().clone()
    model, migration_report = component.expand_model49(source)
    need(torch.equal(rng_before, torch.random.get_rng_state()), 'Expansion consumed global RNG')
    need(model is not source and model.conv_encoder is not source.conv_encoder, 'Expansion did not clone')
    need(state_digest(source.state_dict()) == source_digest and modes(source) == source_modes, 'Expansion mutated source')
    need(source_graph_class.get_edge is source_get_edge, 'Expansion changed original class')
    need(np.array_equal(source.conv_encoder.graph.A, source_graph_A)
         and [tuple(edge) for edge in source.conv_encoder.graph.edge] == source_graph_edges, 'Original graph mutated')
    fresh_graph27 = source_graph_class(layout='mediapipe-27', strategy='spatial', max_hop=1, dilation=1)
    need(np.array_equal(fresh_graph27.A, source_graph_A), 'Original graph construction changed')
    need(modes(model) == source_modes, 'Expansion changed train/eval modes')
    need({name: p.requires_grad for name, p in model.named_parameters()} == source_grad_flags,
         'Expansion changed requires_grad flags')
    need(len(list(model.parameters())) == 108, 'Unexpected official-model parameter count')
    assert_graph(model.conv_encoder.graph, expected_graph, expected_edges, center_distance)
    new_state = model.state_dict()
    need(set(new_state) == set(source_state), 'State keys added, dropped, or renamed')
    point_index = torch.tensor(POINT_MAP, dtype=torch.long)
    channel_index = (point_index[:, None] * 2 + torch.arange(2)).reshape(-1)
    new_channels = torch.tensor(sorted(set(range(98)) - set(channel_index.tolist())), dtype=torch.long)
    new_points = sorted(set(range(49)) - set(POINT_MAP))
    edge_mask = torch.ones((3, 49, 49), dtype=torch.bool)
    edge_mask[:, point_index[:, None], point_index[None, :]] = False
    classifications = {'exact': [], 'mapped': [], 'recomputed': []}
    defaults = {'weight': 1.0, 'bias': 0.0, 'running_mean': 0.0, 'running_var': 1.0}
    for key, old in source_state.items():
        new = new_state[key]
        need(new.dtype == old.dtype and new.device.type == 'cpu', f'Dtype/device changed: {key}')
        need(bool(torch.isfinite(new).all()), f'Nonfinite state: {key}')
        need(new.data_ptr() != source.state_dict()[key].data_ptr(), f'Copied state aliases original: {key}')
        if key == 'conv_encoder.A':
            need(torch.equal(new, torch.as_tensor(expected_graph, dtype=new.dtype)), 'Registered A/oracle mismatch')
            classifications['recomputed'].append(key)
        elif key.startswith('conv_encoder.edge_importance.'):
            need(tuple(new.shape) == (3, 49, 49), f'Edge importance shape: {key}')
            need(torch.equal(new[:, point_index[:, None], point_index[None, :]], old), f'Edge mapping: {key}')
            need(bool((new[edge_mask] == 1).all()), f'New edge default: {key}')
            classifications['mapped'].append(key)
        elif key.startswith('conv_encoder.data_bn.') and key.rsplit('.', 1)[1] in defaults:
            field = key.rsplit('.', 1)[1]
            need(tuple(new.shape) == (98,), f'Input BN shape: {key}')
            need(torch.equal(new[channel_index], old), f'Input BN node-major mapping: {key}')
            need(bool((new[new_channels] == defaults[field]).all()), f'New input BN default: {key}')
            classifications['mapped'].append(key)
        else:
            need(torch.equal(new, old), f'Shared state changed: {key}')
            classifications['exact'].append(key)
    need(len(classifications['mapped']) == 14 and classifications['recomputed'] == ['conv_encoder.A'],
         'Unexpected migration key categories')
    for field in ('eps', 'momentum', 'affine', 'track_running_stats', 'training'):
        need(getattr(model.conv_encoder.data_bn, field) == getattr(source.conv_encoder.data_bn, field),
             f'Input BN setting changed: {field}')
    actual_initial = state_digest(new_state)
    need(actual_initial == gw.state_digest(new_state), 'Canonical digest disagreement')
    repeated_source = base_factory()
    need(state_digest(repeated_source.state_dict()) == source_digest, 'Repeated original factory changed')
    repeated_model, _ = component.expand_model49(repeated_source)
    need(state_digest(repeated_model.state_dict()) == actual_initial, 'Repeated49 factory changed')
    runtime_checked = factory49 is not None
    if runtime_checked:
        runtime_model = factory49()
        need(state_digest(runtime_model.state_dict()) == actual_initial, 'Runtime factory state mismatch')
        del runtime_model
    need(isinstance(migration_report, dict), 'Migration report must be a dict')
    json.dumps(migration_report, allow_nan=False)  # Reject tensor dumps/non-JSON/nonfinite metadata.

    # Exercise the pinned valid-window implementation with a disposable copied49 model.
    generator = torch.Generator(device='cpu').manual_seed(20261006)
    x = torch.randn(batch_size, 6, 2, 10, 49, generator=generator)
    lengths = torch.tensor(([1, 2, 4, 6] * ((batch_size + 3) // 4))[:batch_size])
    mask = torch.arange(6)[None, :] >= (6 - lengths[:, None])
    x[~mask] = 12345.0
    saved_x = x.clone()
    saved_mask = mask.clone()

    def forward(target, values):
        return forward_module.forward_valid_windows(target.conv_encoder, target.agg, target.final_bn,
                                                    target.final_fc, values, mask)

    model.eval()
    with torch.no_grad():
        logits = forward(model, x)
        altered_padding = x.clone()
        altered_padding[~mask] = -54321.0
        need(torch.equal(logits, forward(model, altered_padding)), 'Invalid padding affected eval outputs')
    need(tuple(logits.shape) == (batch_size, 93) and bool(torch.isfinite(logits).all()), 'Eval shape/nonfinite')
    need(state_digest(model.state_dict()) == actual_initial, 'Evaluation mutated initialization')
    model.train()
    x.requires_grad_(True)
    seen = []
    hook = model.conv_encoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], weight_decay=0.0)
    before_update = snapshot(model)
    started = time.perf_counter()
    try:
        # Isolate the disposable dropout stream without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261006)
            optimizer.zero_grad(set_to_none=True)
            train_logits = forward(model, x)
            loss = torch.nn.functional.cross_entropy(train_logits, torch.arange(batch_size) % 93)
            need(bool(torch.isfinite(loss)), 'Nonfinite synthetic loss')
            loss.backward()
            need(x.grad is not None and bool(torch.isfinite(x.grad).all()), 'Input gradient missing/nonfinite')
            need(bool((x.grad[~mask] == 0).all()), 'Invalid windows received gradient')
            new_gradient = x.grad[mask][:, :, :, new_points].abs().sum(dim=(0, 1, 2))
            need(bool((new_gradient > 0).all()), 'One or more added joints lack gradient connection')
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                 'A model parameter lacks a finite gradient')
            optimizer.step()
    finally:
        hook.remove()
    elapsed = time.perf_counter() - started
    need(len(seen) == 1 and torch.equal(seen[0], x.detach()[mask]), 'Encoder did not receive one valid-window batch')
    need(torch.equal(x.detach(), saved_x) and torch.equal(mask, saved_mask), 'Synthetic input mutated')
    after_update = model.state_dict()
    counters = [key for key in before_update if key.endswith('num_batches_tracked')]
    need(all(int(after_update[key]) == int(before_update[key]) + 1 for key in counters), 'BN counters did not advance once')
    need(torch.equal(after_update['conv_encoder.A'], before_update['conv_encoder.A']), 'Optimizer changed graph')
    changed = [key for key, _ in model.named_parameters() if not torch.equal(before_update[key], after_update[key])]
    need(all(any(key.startswith(prefix) for key in changed) for prefix in ('conv_encoder.', 'agg.', 'final_fc.')),
         'Encoder, GRU, or classifier did not update')
    need(all(bool(torch.isfinite(value).all()) for value in after_update.values()), 'Nonfinite updated state')
    need(state_digest(source.state_dict()) == source_digest and modes(source) == source_modes,
         'Disposable training mutated original model')
    need(state_digest(repeated_model.state_dict()) == actual_initial, 'Disposable training aliased repeated factory')
    need(all(sha(Path(path)) == digest for path, digest in before_hashes.items()), 'Protected input/source file changed')
    return {
        'status': 'PASS', 'original_base_initial_state_digest': source_digest,
        'actual_initial_state_digest': actual_initial, 'digest_format': 'generic_workflow.state_digest',
        'original_base_matches_087479ad': True, 'repeated_factories_equal': True,
        'runtime_factory_checked': runtime_checked, 'source_model_unchanged': True,
        'source_files_unchanged': True, 'expansion_rng_unchanged': True,
        'original_pretrained_state_counts': {'encoder': len(enc), 'gru': len(gru)},
        'old_to_new': list(POINT_MAP), 'new_joint_count': len(new_points),
        'graph': {'shape': [3, 49, 49], 'center': 0, 'max_hop': 1, 'dilation': 1,
                  'nonself_undirected_edges': 48, 'self_loops': 49,
                  'partition_nonzero_counts': [49, 90, 6], 'truncated_center_distances_preserved': True},
        'state_classification': classifications,
        'migration_counts': {key: len(value) for key, value in classifications.items()},
        'migration_report': migration_report,
        'official_model_synthetic_integration': {
            'input_shape': list(x.shape), 'input_dtype': str(x.dtype), 'mask_dtype': str(mask.dtype),
            'output_shape': list(logits.shape), 'output_dtype': str(logits.dtype),
            'parameter_tensors': len(list(model.parameters())), 'all_gradients_finite': True,
            'all_22_added_joint_gradients_nonzero': True, 'invalid_window_gradients_zero': True,
            'invalid_padding_eval_invariant': True, 'encoder_calls_per_step': 1,
            'valid_window_count': int(mask.sum()), 'bn_counters_advanced_once': len(counters),
            'changed_parameter_tensors': len(changed), 'optimizer_updates': 1,
            'synthetic_train_step_seconds': elapsed, 'threads': torch.get_num_threads(),
        },
        'real_data_training_or_scoring': 0, 'source_clips_loaded': 0, 'checkpoints_saved': 0,
        'candidate_training_started': False, 'source_plan_sha256': BASE_PLAN_SHA,
        'libraries_and_inputs_sha256': before_hashes,
        'limits': 'One disposable synthetic optimizer step. This checks wiring and migration, not accuracy, '
                  'handshape discrimination, realistic training speed, or output equivalence to27.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    destination = U / 'GRAPH_TRANSFER_CHECK.json'
    need(not destination.exists(), 'Audit result exists; preserve the first result, including failures')
    try:
        result, code = check(batch_size=args.batch_size), 0
    except Exception as exc:
        result = {'status': 'FAIL', 'error_type': type(exc).__name__, 'reason': str(exc),
                  'traceback': traceback.format_exc(), 'candidate_training_started': False}
        code = 1
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['checker_sha256'] = sha(Path(__file__))
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('libraries_and_inputs_sha256', 'migration_report', 'state_classification')},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
