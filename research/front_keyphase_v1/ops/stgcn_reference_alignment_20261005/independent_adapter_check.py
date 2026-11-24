"""Codex independent verification only; no training/evaluation or deployment API."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
sys.path.insert(0, str(P.parent / 'stgcn_transfer_compare_20261005'))
import reference_adapter as adapter
import stgcn_core

PIN_SHA = 'e9b38fc92391a7e913b1c1c266b65701885bea112c4dc775a026de7b1443e304'
MP = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
NATIVE = [0, 16, 15, 5, 2, 6, 3, 25, 29, 30, 33, 34, 37, 38, 41, 42, 45, 46, 50, 51, 54, 55, 58, 59, 62, 63, 66]


def reference_tensor(xy, mask, joints, width, height, transforms):
    # Independent math oracle, followed by explicit window stacking.
    raw = np.where(mask[..., None], xy, 0).astype(np.float32)
    raw = raw / np.array([width, height], dtype=np.float32)
    points = torch.from_numpy(raw[:, joints].copy())
    center = ((points[:, 3] + points[:, 4]) / 2).mean(0)
    scale = torch.sqrt(((points[:, 3] - points[:, 4]) ** 2).sum(-1)).mean()
    manual = ((points - center) * (1 / scale)).permute(2, 0, 1)
    actual = transforms.CenterAndScaleNormalize(reference_points_preset='shoulder_mediapipe_holistic_minimal_27', frame_level=False)({'frames': points.permute(2, 0, 1).clone()})['frames']
    torch.testing.assert_close(manual, actual, rtol=1e-6, atol=1e-6)
    if actual.shape[1] < 60:
        actual = torch.nn.functional.pad(actual, (0, 0, 60 - actual.shape[1], 0))
    return torch.stack([actual[:, i:i+10] for i in range(0, 60, 10)]).unsqueeze(0)


def main():
    torch.set_num_threads(2)
    model_module, transforms, sources = adapter.load_reference(P / 'reference', PIN_SHA)
    checks = []
    for layout, size, joints in [('mp75', 75, MP), ('native67', 67, NATIVE)]:
        for count in [1, 59, 60, 61, 123]:
            rng = np.random.default_rng(20261005 + count)
            xy = rng.uniform(10, 800, size=(count, size, 2)).astype(np.float32)
            present = rng.random((count, size)) > .2
            present[:, [joints[3], joints[4]]] = True
            xy[~present] = np.nan
            original = xy.copy()
            output, diagnostics = adapter.prepare_windows(xy, present, layout, 1920, 1080, transforms)
            expected = reference_tensor(xy, present, joints, 1920, 1080, transforms)
            torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
            assert np.array_equal(xy, original, equal_nan=True)
            checks.append({'layout': layout, 'frames': count, 'max_error': float((output-expected).abs().max())})

    # Only public training references; no user/evaluation rows and no model scoring.
    rows = json.loads((P.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json').read_text(encoding='utf-8'))['rows']
    rows = [r for r in rows if r['role'] == 'train' and r['layout'] == 'mp75' and r['id'].startswith('NIKL_')]
    assert len(rows) == 18
    real_checks = []
    for row in rows:
        path = Path(row['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        metadata = json.loads(path.with_suffix('.sidecar.json').read_text(encoding='utf-8'))
        assert metadata['observation_sha256'] == row['sha256']
        assert metadata['mirrored'] is False and metadata['rotation_deg'] == 0
        with np.load(path, allow_pickle=False) as data:
            interval = slice(row['start'], row['end_exclusive'])
            xy, present = data['xy_px'][interval], data['observed'][interval]
        out, diag = adapter.prepare_windows(xy, present, 'mp75', metadata['width'], metadata['height'], transforms)
        expected = reference_tensor(xy, present, MP, metadata['width'], metadata['height'], transforms)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
        real_checks.append({'id': row['id'], 'max_error': float((out-expected).abs().max()), 'frames': len(xy), 'present_semantics': 'observed', 'source_sha256': row['sha256']})

    artifact = P.parent / 'openhands_artifact_20261005'
    states, _ = stgcn_core.read_encoder_tensors(artifact / 'raw_dpc.zip', artifact / 'TENSOR_METADATA.json', '9cee49383343e85c387f84dff5aee07132796acbbeaf6c0e3fbbd739403d53d4', encoder_prefix='model.')
    encoder = {k.removeprefix('conv_encoder.'): v for k,v in states.items() if k.startswith('conv_encoder.')}
    gru = {k.removeprefix('agg.'): v for k,v in states.items() if k.startswith('agg.')}
    model, model_audit = adapter.make_reference_model(model_module, encoder, gru)
    assert len(encoder) == 170 and len(gru) == 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261005)
        direct = model_module.DPC_RNN_Finetuner(num_class=93)
    direct.conv_encoder.load_state_dict(encoder, strict=True)
    direct.agg.load_state_dict(gru, strict=True)
    direct.eval()
    assert all(torch.equal(v, direct.state_dict()[k]) for k,v in model.state_dict().items())
    # Synthetic inference only; new random 93-class head is not an accuracy result.
    synthetic = torch.linspace(-1, 1, 2*6*2*10*27).reshape(2, 6, 2, 10, 27)
    with torch.no_grad():
        output, expected = model(synthetic), direct(synthetic)
    assert torch.equal(output, expected) and torch.isfinite(output).all()
    record = {'status': 'PASS', 'role': 'Codex independent verification', 'source_pins_sha256': PIN_SHA, 'adapter_sha256': hashlib.sha256((P/'reference_adapter.py').read_bytes()).hexdigest(), 'source_imports': sources, 'synthetic_preprocessing': checks, 'public_train_preprocessing': real_checks, 'public_train_model_inference_count': 0, 'model': model_audit, 'actual_vs_direct_full_state_equal': True, 'synthetic_logits_max_error': float((output-expected).abs().max()), 'training_updates': 0, 'dataset_accuracy_evaluation': 0, 'limits': ['Nondegenerate shoulders only', 'No train augmentation in verification adapter', 'Native67 aspect ratio unresolved', 'Pose/Hand Tasks do not reproduce Holistic observations']}
    (P / 'INDEPENDENT_ADAPTER_CHECK.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'synthetic_input_cases': len(checks), 'public_train_input_cases': len(real_checks), 'model_state_equal': True, 'synthetic_logits_max_error': record['synthetic_logits_max_error'], 'training_updates': 0, 'accuracy_evaluation': 0}))


if __name__ == '__main__':
    main()
