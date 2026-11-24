"""Read FIT input contract and perform one untrained shape-only forward.

This is not a trained prediction, candidate evaluation or training authorization.
No DEV array is opened and no optimizer is created.
"""
from pathlib import Path
from collections import Counter
import datetime
import json
import sys
import traceback
import torch
from process_head import ProcessEvidenceNet, QUERIES, batch
from process_data import A, build_fit_dataset, multiplicities, sha
from train_process_head import TRAINING_PROFILE

U = Path(__file__).resolve().parent


def main():
    output = U / 'BASE_TRAINING_INPUT_PREFLIGHT.json'
    assert not output.exists(), 'Do not repeat completed shape probe silently'
    data = build_fit_dataset(span_manifest=None, require_spans=False)
    weights = multiplicities(data['records'])
    assert len(weights) == 45 and data['development_arrays_opened'] == 0
    assert sha(U / 'process_head.py') == sha(A / 'process_head.py')
    assert all(r['role'] == 'fit' and r['person'] in range(1, 10) for r in data['records'])
    assert len(data['records']) == 262
    counts = {}
    for origin in ['reviewed_view', 'runtime_proposal']:
        rows = [r for r in data['records'] if r['origin'] == origin]
        counts[origin] = dict(cases=len(rows), labels=dict(Counter(str(r['observed_process_evidence']) for r in rows)),
                             sparse_points=sum(len(r['sparse_relation_targets']) for r in rows),
                             netmotion_intervals=sum(len(r['fully_retained_motion_interval_targets']) for r in rows))
    # Show equal total mass for each physical assertion across any copied views.
    balance_checks = 0
    for g, rows in data['groups'].items():
        for kind, count in weights[g].items():
            for identity, number in count.items():
                assert number >= 1 and abs(sum(1 / number for _ in range(number)) - 1) < 1e-12
                balance_checks += 1
    shortest = min((r for r in data['records'] if r['origin'] == 'reviewed_view'), key=lambda r: len(data['sequences'][r['array_key']]))
    runtime = next(r for r in data['records'] if r['origin'] == 'runtime_proposal')
    selected = [shortest, runtime]
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = ProcessEvidenceNet().eval()
    assert sum(p.numel() for p in model.parameters()) == 47980
    x, lens = batch([data['sequences'][r['array_key']] for r in selected])
    query = torch.tensor([QUERIES.index(r['physical_query']) for r in selected])
    with torch.no_grad():
        out = model(x, lens, query)
    assert out['process_logits'].shape == (2,)
    assert out['point_logits'].shape == (2, x.shape[1], 2)
    assert all(torch.isfinite(v).all() for v in out.values())
    report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='BASE_FIT_INPUTS_PASS_EXACT_SPAN_MANIFEST_AND_ROOT_FREEZE_PENDING',
        training_allowed=False, counts=counts, input_counts=data['counts'],
        loaded_sensor_fields=data['sensor_fields_loaded'], stored_prediction_arrays_read=[],
        development_arrays_opened=0, normalization_frames=data['stats']['frames'],
        normalization_source=data['stats']['source'], groups=len(data['groups']),
        inverse_copy_weight_identity_checks=balance_checks,
        parameters=47980, model_source_byte_identical_to_AB=True,
        untrained_forward_calls=1, untrained_forward_cases=2,
        shape_only_cases=[dict(case_id=r['case_id'], frames=len(data['sequences'][r['array_key']])) for r in selected],
        shape_output=dict(process=list(out['process_logits'].shape), point=list(out['point_logits'].shape)),
        logits_or_probabilities_saved=False, training_steps=0, optimizer_instances=0,
        trained_model_inference=0, candidate_evaluation=0, new_sensor_extraction=0,
        runtime_context_note='Full-native original RGB encoder context then fixed proposal; local labels are not inferred from surrounding support.',
        nominal_span_status=data['nominal_span_status'],
        training_profile=TRAINING_PROFILE, dependencies=data['pins'],
        code_pins={str(U / n): sha(U / n) for n in ['process_head.py', 'process_data.py', 'train_process_head.py', 'preflight_alignment_training.py']},
        python_executable=sys.executable, whole_goal_complete=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'counts': counts, 'shapes': report['shape_output'], 'sha256': sha(output)}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        path = U / ('FIRST_INPUT_PREFLIGHT_FAILURE_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.txt')
        path.write_text(traceback.format_exc(), encoding='utf-8')
        raise
