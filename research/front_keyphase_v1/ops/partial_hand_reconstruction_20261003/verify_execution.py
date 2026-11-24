"""Independent execution audit. Never edits candidate, data, or model."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from verify_inputs import ARMS, load_environment, prepare_row, sha
from verify_reference import array_hash

ROOT = Path(__file__).resolve().parent


def write_report(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def inputs(cfg):
    R, api, groups = load_environment(cfg)
    sys.path.insert(0, str(ROOT / 'implementation'))
    HG = importlib.import_module('handgap')
    records = []
    for group, rows in groups.items():
        counts = {'rows': 0, 'raw_arrays_equal': 0, 'built_inputs_equal': 0, 'provenance_equal': 0}
        for i, row in enumerate(rows):
            p = prepare_row(R, api, group, row, cfg['tolerance'])
            base = p['arms']['original']
            args = [base[k] for k in ('xy', 'present', 'ts')]
            masked = HG.mask_one_hand(*args, row['id'])
            actual = {'original': base, 'original_repair': HG.restore(*args), 'masked': masked,
                      'masked_repair': HG.restore(*[masked[k] for k in ('xy', 'present', 'ts')])}
            for arm in ARMS:
                ref = p['arms'][arm]
                for key in ('xy', 'present', 'ts'):
                    a, b = actual[arm][key], ref[key]
                    if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                        raise AssertionError((group, row['id'], arm, key, 'reference mismatch'))
                    counts['raw_arrays_equal'] += 1
                for key in ('observed', 'estimated', 'injected'):
                    if key in ref:
                        assert np.array_equal(actual[arm][key], ref[key]), (group, i, arm, key)
                        counts['provenance_equal'] += 1
                x = api['build_channels'](*[actual[arm][k] for k in ('xy', 'present', 'ts')])['x']
                assert array_hash(x) == array_hash(p['features'][arm]['x']), (group, i, arm, 'x')
                counts['built_inputs_equal'] += 1
            counts['rows'] += 1
        records.append({'group': group, **counts})
    write_report('INDEPENDENT_CANDIDATE_INPUT_CHECK.json', {'status': 'PASS', 'groups': records,
                 'candidate_sha256': sha(HG.__file__), 'model_calls': 0,
                 'comparison': 'exact dtype, array equality incl NaN; exact normalized input hashes',
                 'config_sha256': sha(ROOT / 'CONFIG_LOCAL_ONLY.json')})
    print(json.dumps(records))


def forward(cfg):
    R, api, groups = load_environment(cfg)
    sys.path.insert(0, cfg['lab_root'])
    u = importlib.import_module('tools_prep_union93_candidate')
    from pilot.data import collate
    import torch
    torch.set_num_threads(2)
    model, classes = u.rebuild_from_payload(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True))
    model.eval()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    records = []
    for group, indices in cfg['repro_indices'].items():
        with np.load(ROOT / 'result' / f'SCORES_{group}.npz', allow_pickle=False) as f:
            scores = f['logits'].copy()
            assert f['classes'].tolist() == list(classes)
            assert f['arms'].tolist() == list(ARMS)
            ids = f['ids'].tolist()
        for i in indices:
            row = groups[group][i]
            assert ids[i] == row['id']
            p = prepare_row(R, api, group, row, cfg['tolerance'])
            for j, arm in enumerate(ARMS):
                x = p['features'][arm]['x']
                bx, pad, _ = collate([{'x': x, 'frames': len(x), 'label': 0}])
                with torch.inference_mode():
                    logits = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
                delta = float(np.abs(logits - scores[i, j]).max())
                assert delta <= cfg['tolerance'] and logits.argmax() == scores[i, j].argmax(), (group, i, arm, delta)
                records.append({'group': group, 'index': i, 'id': row['id'], 'arm': arm, 'max_abs_diff': delta,
                                'top1': str(classes[int(logits.argmax())])})
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert not any(m.training for m in model.modules())
    for path, expected in cfg['pins'].items():
        assert sha(path) == expected
    write_report('INDEPENDENT_FORWARD_CHECK.json', {'status': 'PASS', 'model_calls': len(records),
                 'model_sha256': sha(cfg['checkpoint']), 'model_unchanged': True,
                 'reference_input_not_candidate': True, 'records': records,
                 'max_abs_diff': max(r['max_abs_diff'] for r in records)})
    print(json.dumps({'status': 'PASS', 'model_calls': len(records), 'max_abs_diff': max(r['max_abs_diff'] for r in records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inputs', 'forward'))
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'CONFIG_LOCAL_ONLY.json').read_text(encoding='utf-8-sig'))
    globals()[args.mode](cfg)
