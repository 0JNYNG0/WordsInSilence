"""Synthetic checks by default; explicit two-FIT-row parity preflight separately.

The default does not load real checkpoints, old examples, or old experiment
modules. --real-preflight checks the new candidate's initialization contract;
it is not another evaluation or re-scoring of completed Y results.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
import torch
from torch import nn
import motion_order_model as mm

U = Path(__file__).resolve().parent
ALIASES = [[0], [1], [2], [3], [4], [5, 6, 7, 8, 9], [10], [11, 12]]


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(64, 32, bias=False)
        self.guide = nn.Linear(7, 32)
        self.embedding = nn.Parameter(torch.randn(13, 32))
        self.register_buffer('guide_vectors', torch.randn(13, 7))
        self.rest = nn.Linear(64, 1)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = FakeActor()
        self.feature = nn.Linear(74, 64)
        self.hidden = nn.Linear(74, 48)
        self.rgb = nn.Linear(9, 74)
        self.role_embed = nn.Sequential(nn.Linear(122, 32), nn.GELU())
        self.role_context = nn.GRU(32, 24, batch_first=True, bidirectional=True)
        self.role_output = nn.Linear(48, 3)
        self.flow_bridge = nn.Linear(6, 122, bias=False)
        self.register_buffer('family_guide_keys', torch.randn(8, 32))

    def encode(self, x, rgb, lengths):
        y = x + self.rgb(rgb)
        return self.feature(y), self.hidden(y)


class FakeTied(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = 'tied_readout'
        self.base = FakeBase()
        self.visual_embed = deepcopy(self.base.role_embed)
        self.visual_context = deepcopy(self.base.role_context)
        self.query = nn.Linear(48, 32, bias=False)

    def forward(self, x, rgb, lengths, flow):
        feature, h = self.base.encode(x, rgb, lengths)
        logits = self.base.actor.query(feature) @ (self.base.actor.guide(self.base.actor.guide_vectors) + self.base.actor.embedding).T / (32 ** .5)
        action = torch.cat([torch.stack([torch.logsumexp(logits[:, :, ids], -1) - np.log(len(ids)) for ids in ALIASES], -1), self.base.actor.rest(feature)], -1)
        visual = torch.cat([feature, h, x[:, :, 64:74]], -1)
        def context(embed, gru, value):
            z, _ = gru(nn.utils.rnn.pack_padded_sequence(embed(value), lengths.cpu(), batch_first=True, enforce_sorted=False))
            return nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=x.shape[1])[0]
        role = self.base.role_output(context(self.base.role_embed, self.base.role_context, visual + self.base.flow_bridge(flow)))
        z = context(self.visual_embed, self.visual_context, visual)
        occurrence = self.query(z) @ self.base.family_guide_keys.T / (32 ** .5)
        return action, role, occurrence + (role[:, :, 1]-role[:, :, 0])[:, :, None]


def synthetic():
    torch.manual_seed(42)
    checks = []
    def check(ok, name):
        assert bool(ok), name
        checks.append(name)
    def same(a, b, name, exact=True):
        torch.testing.assert_close(a, b, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-6)
        checks.append(name)
    old = FakeTied().eval()
    models = [mm.MotionOrderNet(c, tied_model=old, aliases=ALIASES).eval() for c in mm.CONDITIONS]
    x, rgb, flow = torch.randn(2, 9, 74), torch.randn(2, 9, 9), torch.randn(2, 9, 6)
    lengths = torch.tensor([9, 5])
    aux = torch.cat([torch.rand(2, 9, 5)*2-1, torch.ones(2, 9, 3)], -1)
    aux[0, 3] = 0
    before = models[0].frozen_state()
    for name, value in models[0].state_dict().items():
        same(value, models[1].state_dict()[name], 'initial_state:'+name)
    with torch.inference_mode():
        outputs = [model(x, rgb, lengths, flow, aux) for model in models]
        original = old(x, rgb, lengths, flow)
    for index in range(3):
        same(outputs[0][index], outputs[1][index], 'matched_initial_output:'+str(index))
        same(outputs[0][index], original[index], 'original_tied_initial_output:'+str(index))
    for model in models:
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        check(names == mm.TRAINABLE_NAMES, model.condition+':only_two_weights')
        check(sum(p.numel() for p in model.parameters() if p.requires_grad) == 1792, model.condition+':count1792')
        model.train()
        check(not model.readout.training and not model.readout.visual_context.training, model.condition+':context_eval')
        a = model.auxiliary_input(aux, lengths)
        same(a[:, :, 5:], models[0].auxiliary_input(aux, lengths)[:, :, 5:], model.condition+':same_availability')
        same(a[1, 5:], torch.zeros_like(a[1, 5:]), model.condition+':padding_zero')
        with torch.inference_mode():
            solo = model(x[1:2, :5], rgb[1:2, :5], lengths[1:2], flow[1:2, :5], aux[1:2, :5])
            for j in range(3):
                same(outputs[models.index(model)][j][1, :5], solo[j][0], model.condition+':valid_padding_invariance:'+str(j), exact=False)
        model.assert_frozen(before)
    same(models[0].auxiliary_input(aux, lengths)[:, :, :5], torch.zeros_like(aux[:, :, :5]), 'control_probability_channels_zero')
    same(models[1].auxiliary_input(aux, lengths)[0, :9], aux[0, :9], 'candidate_preserves_auxiliary_native_rows')
    for channel, value in [(0, float('nan')), (0, 1.1), (5, .5)]:
        bad = aux.clone(); bad[0, 0, channel] = value
        try:
            models[1].auxiliary_input(bad, lengths)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, 'invalid_aux_rejected:'+str(channel)+':'+str(value))
    bad = aux.clone(); bad[0, 3, 0] = -.9
    try:
        models[1].auxiliary_input(bad, lengths)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(rejected, 'unavailable_probability_not_negative')
    for name, values in [('anchor_outside_evidence', [1., 0., 0.]), ('rgb_outside_evidence', [0., 0., 1.])]:
        bad = aux.clone(); bad[0, 3, 5:] = torch.tensor(values)
        try:
            models[1].auxiliary_input(bad, lengths)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, name+'_rejected')
    separate = aux.clone(); separate[0, 3, 5:] = torch.tensor([0., 1., 1.])
    models[1].auxiliary_input(separate, lengths)
    check(True, 'isolated_rgb_evidence_without_motion_anchor_allowed')
    separate[0, 3, 5:] = torch.tensor([1., 1., 0.])
    models[1].auxiliary_input(separate, lengths)
    check(True, 'geometry_anchor_without_rgb_allowed')
    logits = torch.randn(2, 5, 8, requires_grad=True)
    target = torch.full_like(logits, -1); target[0, 1, 2] = 1
    mm.occurrence_group_loss(logits, target, torch.ones(8), ['A', 'B']).backward()
    check(int((logits.grad != 0).sum()) == 1, 'unknown_cells_zero_gradient')
    empty = torch.randn(2, 5, 8, requires_grad=True)
    loss = mm.occurrence_group_loss(empty, torch.full_like(empty, -1), torch.ones(8), ['A', 'B'])
    loss.backward()
    check(float(loss.detach()) == 0 and bool((empty.grad == 0).all()), 'all_unknown_zero_loss')
    check(float(mm.grouped_mean(torch.tensor([1., 3., 100.]), ['A', 'A', 'B'])) == 51, 'variants_mean_then_sources_mean')
    for model in models:
        initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        try:
            model.validate_trained(initial, before)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, model.condition+':no_optimization_rejected')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0005, weight_decay=.01)
        action, role, occurrence = model(x, rgb, lengths, flow, aux)
        target = torch.full_like(occurrence, -1); target[0, 1, 2] = 1; target[1, 3, 4] = 0
        loss = mm.occurrence_group_loss(occurrence, target, torch.ones(8), ['A', 'B'])
        loss.backward()
        check(model.readout.query.weight.grad.abs().sum() > 0, model.condition+':query_gradient')
        check(model.auxiliary.weight.grad.abs().sum() > 0, model.condition+':auxiliary_gradient')
        if model.condition == mm.CONDITIONS[0]:
            check(bool((model.auxiliary.weight.grad[:, :5] == 0).all()), 'control_no_probability_gradient')
        check(all(p.grad is None for n, p in model.named_parameters() if n not in mm.TRAINABLE_NAMES), model.condition+':frozen_gradient_none')
        optimizer.step()
        model.assert_frozen(before)
        changes = model.validate_trained(initial, before)
        check(all(np.isfinite(v) and v > 0 for v in changes.values()), model.condition+':finite_actual_weight_changes')
        with torch.inference_mode():
            new = model(x, rgb, lengths, flow, aux)
        same(new[0], original[0], model.condition+':physical_exact_after_step')
        same(new[1], original[1], model.condition+':role_exact_after_step')
        corrupt = deepcopy(model)
        with torch.no_grad():
            corrupt.auxiliary.weight[0, 0] = float('nan')
        try:
            corrupt.validate_trained(initial, before)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, model.condition+':nonfinite_final_weight_rejected')
    # Exercise the real status writer only under a checked AO-local temporary
    # path; no actual training marker, checkpoint or progress file is touched.
    import train_pair as trainer
    original_unit = trainer.U
    with tempfile.TemporaryDirectory(prefix='synthetic_progress_v2_', dir=U) as temp:
        test_unit = Path(temp).resolve()
        check(test_unit.is_relative_to(U.resolve()), 'temporary_status_path_inside_AO')
        trainer.U = test_unit
        try:
            trainer.assert_not_started()
            trainer.write_new(test_unit/'TRAIN_STARTED.json', {'synthetic_only':True})
            marker = mm.sha(test_unit/'TRAIN_STARTED.json')
            try:
                trainer.assert_not_started()
            except AssertionError:
                rejected = True
            else:
                rejected = False
            check(rejected, 'existing_marker_blocks_duplicate_training')
            trainer.update_progress(status='TRAINING', condition=mm.CONDITIONS[0], epoch=3, optimizer_steps=13, total_optimizer_steps=13)
            state = mm.read(test_unit/'TRAIN_PROGRESS.json')
            check(state['pid'] > 0 and state['epoch'] == 3 and state['optimizer_steps'] == 13, 'progress_records_pid_epoch_optimizerstep')
            trainer.record_failure('synthetic rejected duplicate')
            check(mm.read(test_unit/'TRAIN_PROGRESS.json') == state, 'rejected_invocation_preserves_active_job_status')
            check((test_unit/'TRAIN_REJECTED_INVOCATION_FIRST_FAILURE.txt').exists() and not (test_unit/'TRAIN_FIRST_FAILURE.txt').exists(), 'duplicate_failure_separate_from_original_job')
            trainer.record_failure('second synthetic rejected duplicate')
            check((test_unit/'TRAIN_REJECTED_INVOCATION_FIRST_FAILURE.txt').read_text(encoding='utf-8') == 'synthetic rejected duplicate', 'first_rejected_error_preserved')
            trainer.update_progress(status='FAILED')
            failed = mm.read(test_unit/'TRAIN_PROGRESS.json')
            check(failed['condition'] == mm.CONDITIONS[0] and failed['total_optimizer_steps'] == 13, 'failure_retains_last_known_progress')
            check(mm.sha(test_unit/'TRAIN_STARTED.json') == marker, 'progress_does_not_rewrite_marker')
            check(not (test_unit/'TRAIN_PROGRESS.json.tmp').exists(), 'atomic_status_replace_complete')
        finally:
            trainer.U = original_unit
    return dict(status='PASS', checks=len(checks), items=checks, real_checkpoint_loads=0, real_inputs=0, old_experiment_entrypoints_executed=0, synthetic_optimizer_steps=2, real_training_steps=0)


def real_preflight():
    # The root explicitly invokes this only after authorizing the data/input
    # gate. Fixed first source's two variants, never selected from scores.
    from input_data import load_rows
    from train_pair import source_groups, class_weights
    rows = load_rows('fit')
    groups = source_groups(rows)
    selected = groups[sorted(groups)[0]]
    old = mm.legacy()
    torch.manual_seed(42)
    baseline = mm.MotionOrderNet(mm.CONDITIONS[0]).eval()
    candidate = mm.MotionOrderNet(mm.CONDITIONS[1], tied_model=baseline.readout, aliases=baseline.aliases).eval()
    stats = mm.read(old.D / 'NORMALIZATION.json')
    flow = mm.read(old.FLOW_UNIT / 'FLOW_INPUTS.json')
    x, rgb, lengths, _, _ = old.batch_j(selected, stats)
    fl = old.flow_batch(selected, flow)
    aux = mm.auxiliary_batch(selected)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    fixed = baseline.frozen_state()
    with torch.inference_mode():
        a = baseline(x, rgb, lengths, fl, aux)
        b = candidate(x, rgb, lengths, fl, aux)
        original = baseline.readout(x, rgb, lengths, fl)
        for left, right, previous in zip(a, b, original):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(left, previous, rtol=0, atol=0)
        for model in [baseline, candidate]:
            for i in range(2):
                n = int(lengths[i])
                solo = model(x[i:i+1, :n], rgb[i:i+1, :n], lengths[i:i+1], fl[i:i+1, :n], aux[i:i+1, :n])
                for one, batched in zip(solo, a):
                    torch.testing.assert_close(one[0], batched[i, :n], rtol=1e-5, atol=1e-5)
            model.assert_frozen(fixed)
    unknown = [dict(r, order_queries=[dict(q, target=-1) for q in r['order_queries']]) for r in selected]
    pair, used = mm.pair_group_loss(a[0].softmax(-1), a[2].sigmoid(), unknown)
    assert used == 0 and float(pair) == 0
    pair_module = __import__('pair_contract')
    assert Path(pair_module.__file__).resolve() == (mm.Y/'pair_contract.py').resolve()
    pair_checks = pair_module.behavior()
    expected_modules = {
        'readout_model':mm.Y/'readout_model.py',
        'order_model':mm.OPS/'word100_guided_event_order_20261102/order_model.py',
        'flow_model':mm.OPS/'word100_learned_flow_role_20261102/flow_model.py',
        'episode_model':mm.OPS/'word100_action_episode_supervision_20261102/episode_model.py',
        'evaluate_transfer':mm.OPS/'word100_seated_fit_transfer_20261102/evaluate_transfer.py',
        'pair_contract':mm.Y/'pair_contract.py',
    }
    lineage = {}
    for name, expected in expected_modules.items():
        actual = Path(sys.modules[name].__file__).resolve()
        assert actual == expected.resolve(), 'Generic module collision: '+name
        lineage[name] = dict(path=str(actual), sha256=mm.sha(actual))
    return dict(status='PASS', selected_rows=[dict(clip_id=r['clip_id'], kind=r['kind'], flow_key=r['flow_key'], native_rows=len(r['x'])) for r in selected], initial_y=dict(path=str(mm.INITIAL_Y), sha256=mm.initial_y_sha()), equal_initial_state=True, equal_effective_outputs_exact=True, original_tied_physical_role_occurrence_exact=True, frozen_state_exact=True, padding_invariance=True, unknown_pair_zero_loss=True, original_pair_contract=pair_checks, imported_module_lineage=lineage, trainable_parameters_each=1792, positive_weights=class_weights().tolist(), training_steps=0, real_input_rows=2, purpose='New candidate initialization and padding contract only; no completed-result re-evaluation or accuracy score')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    dest = U / ('REAL_PREFLIGHT.json' if args.real_preflight else 'SYNTHETIC_CHECKS_v3.json')
    assert not dest.exists(), 'Completed check output is immutable'
    result = real_preflight() if args.real_preflight else synthetic()
    result['code_pins'] = {str(U/n):mm.sha(U/n) for n in ['motion_order_model.py', 'train_pair.py', 'check_model.py']}
    with dest.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'items','code_pins'}}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = U / 'MODEL_CHECK_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
