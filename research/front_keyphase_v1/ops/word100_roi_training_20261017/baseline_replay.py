"""Read-only M0/development replay; writes only baseline_ evidence in this unit.

No detector/video/camera/training is invoked. Existing source observations are
reconstructed through the current production _channels adapter and must match
the frozen original x_whole arrays exactly before the frozen TCN is evaluated.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
LAB = UNIT.parents[1]
ROOT = LAB.parents[1]
OLD = LAB / 'ops/word100_training_20261013'
RUN = LAB / 'runs/word100_tcn_v1/whole_context'
DATA = Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
EXPECTED_MODEL_SHA = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
for p in (LAB / 'src', ROOT / 'research/sign_model_pilot_v1/src'):
    sys.path.insert(0, str(p))

import numpy as np
import torch
from pilot.model_masked_v1 import MaskedSignTCN
from wis_lab.core.observation import Observation
from wis_lab.service.neural import _channels


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    assert path.parent == UNIT and path.name.startswith('baseline_')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    start = time.perf_counter()
    result_path = UNIT / 'baseline_RESULT.json'
    if result_path.exists():
        old_result = read(result_path)
        assert old_result['status'] == 'PASS'
        assert sha(RUN / 'best.pt') == old_result['checkpoint_sha256'] == EXPECTED_MODEL_SHA
        assert sha(UNIT / 'baseline_PREDICTIONS.json') == old_result['new_prediction_sha256']
        print(json.dumps({'status': 'REUSED_EXISTING_BASELINE', 'result': str(result_path)}, ensure_ascii=False))
        return
    assert not (UNIT / 'baseline_PREDICTIONS.json').exists(), 'Preserve incomplete replay; review first error before recovery'
    assert not (UNIT / 'baseline_FIRST_ERROR.json').exists(), 'Review preserved first failure before recovery'
    protected = {}
    current_clip = None
    model = None
    before_state = None
    stage = 'pin_existing_evidence'
    try:
        frozen = read(OLD / 'FROZEN_EXPERIMENT.json')
        checked = read(OLD / 'INPUT_CHECK.json')
        metrics_check = read(OLD / 'METRICS_CHECK.json')
        old_result = read(RUN / 'RESULT.json')
        old_scores = read(RUN / 'PREDICTIONS.json')
        assert checked['status'] == metrics_check['status'] == 'PASS'
        for path, expected in metrics_check['artifacts'].items():
            assert sha(path) == expected, f'Previous metric artifact changed: {path}'
        manifest_path = OLD / 'MANIFEST.json'
        assert sha(manifest_path) == frozen['source_sha256'][str(manifest_path)]
        manifest = [r for r in read(manifest_path) if r['role'] == 'development_only']
        assert len(manifest) == len({r['clip_id'] for r in manifest}) == 300
        assert Counter(r['signer_id'] for r in manifest) == {11: 100, 12: 100, 13: 100}
        assert Counter(r['label_index'] for r in manifest) == {i: 3 for i in range(100)}
        saved = {r['clip_id']: r for r in old_scores}
        assert len(saved) == 300 and set(saved) == {r['clip_id'] for r in manifest}
        assert sha(RUN / 'best.pt') == old_result['checkpoint_sha256'] == EXPECTED_MODEL_SHA
        code_files = [ROOT / ('research/sign_model_pilot_v1/src/pilot/' + f)
                      for f in ('data.py', 'web_input.py', 'model_masked_v1.py')]
        for path in code_files:
            assert sha(path) == frozen['source_sha256'][str(path)], f'Frozen feature/model code changed: {path}'
        pinned = [manifest_path, OLD / 'FROZEN_EXPERIMENT.json', OLD / 'INPUT_CHECK.json',
                  OLD / 'METRICS_CHECK.json', RUN / 'best.pt', RUN / 'RESULT.json', RUN / 'PREDICTIONS.json',
                  LAB / 'src/wis_lab/service/neural.py', LAB / 'src/wis_lab/core/observation.py',
                  Path(__file__), *code_files]
        protected.update({str(path): sha(path) for path in pinned})
        clips = []
        tensor_values = 0
        stage = 'reconstruct_dev_only_current_adapter'
        for row in manifest:
            current_clip = row['clip_id']
            rp = DATA / 'records' / f'{current_clip}.json'
            assert sha(rp) == checked['record_hashes'][current_clip]
            rec = read(rp)
            assert rec['role'] == 'development_only' and rec['signer_id'] == row['signer_id']
            assert rec['label_index'] == row['label_index'] and rec['channel_file'] is not None
            op = Path(rec['observation'])
            cp = Path(rec['channel_file'])
            assert sha(op) == rec['observation_sha256']
            assert sha(cp) == rec['channel_sha256']
            protected.update({str(rp): sha(rp), str(op): rec['observation_sha256'], str(cp): rec['channel_sha256']})
            obs = Observation.load(str(op))
            feat, reason, support = _channels(obs)
            assert reason is None and feat is not None, f'Original input gate changed: {reason}'
            with np.load(cp, allow_pickle=False) as z:
                assert 'x_context' not in z.files, 'Development cache unexpectedly contains oracle context'
                np.testing.assert_array_equal(feat['x'], z['x_whole'])
                x = feat['x'].copy()
            assert x.ndim == 2 and x.shape[1] == 280 and np.isfinite(x).all()
            tensor_values += x.size
            clips.append(dict(clip_id=current_clip, y=row['label_index'], signer=row['signer_id'], x=x))
        stage = 'frozen_checkpoint_eval_no_grad'
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        ck = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True)
        assert ck['classes'] == frozen['labels'] and ck['word_codes'] == frozen['classes']
        cfg = dict(frozen['model'])
        cfg.pop('type')
        model = MaskedSignTCN(**cfg)
        model.load_state_dict(ck['state_dict'], strict=True)
        model.eval()
        assert model.parameter_count() == 37636
        before_state = state_sha(model)
        out = []
        max_diff = 0.0
        infer_start = time.perf_counter()
        with torch.inference_mode():
            for pos in range(0, len(clips), 32):
                part = clips[pos:pos + 32]
                length = max(len(c['x']) for c in part)
                x = np.zeros((len(part), length, 280), dtype=np.float32)
                pad = np.zeros((len(part), length), dtype=np.float32)
                for i, c in enumerate(part):
                    x[i, :len(c['x'])] = c['x']
                    pad[i, :len(c['x'])] = 1
                logits = model(torch.from_numpy(x), torch.from_numpy(pad)).cpu().numpy()
                assert logits.shape == (len(part), 100) and np.isfinite(logits).all()
                for c, scores in zip(part, logits):
                    previous = saved[c['clip_id']]
                    assert previous['label_index'] == c['y'] and previous['signer_id'] == c['signer']
                    assert previous['status'] == 'OK'
                    max_diff = max(max_diff, float(np.max(np.abs(scores.astype(np.float64) - np.asarray(previous['logits'])))))
                    rank = np.argsort(-scores, kind='stable')
                    pred = int(rank[0])
                    s64 = scores.astype(np.float64)
                    ce = float(s64.max() + np.log(np.exp(s64 - s64.max()).sum()) - s64[c['y']])
                    out.append(dict(clip_id=c['clip_id'], label_index=c['y'], signer_id=c['signer'],
                                    prediction=pred, correct=pred == c['y'], top5=rank[:5].tolist(),
                                    top5_correct=c['y'] in rank[:5], cross_entropy=ce,
                                    logits=scores.tolist(), previous_prediction=previous['prediction'], status='OK'))
        infer_elapsed = time.perf_counter() - infer_start
        out.sort(key=lambda r: r['clip_id'])
        write_new(UNIT / 'baseline_PREDICTIONS.json', out)
        stage = 'compare_and_immutability_check'
        assert all(r['prediction'] == r['previous_prediction'] for r in out), 'Baseline predictions changed'
        assert max_diff <= 1e-5, f'Baseline logit deviation {max_diff}'
        correct = sum(r['correct'] for r in out)
        mean_ce = float(np.mean([r['cross_entropy'] for r in out]))
        assert correct == old_result['metrics']['correct'] == 290
        assert abs(mean_ce - old_result['selected_development_ce']) < 2e-6
        assert before_state == state_sha(model), 'Evaluation changed in-memory model state'
        assert all(sha(path) == digest for path, digest in protected.items()), 'Protected baseline artifacts changed'
        write_new(UNIT / 'baseline_PROVENANCE.json', dict(source_pins=protected,
                    code_sha256=sha(__file__), original_checkpoint_sha256=EXPECTED_MODEL_SHA,
                    current_adapter='wis_lab.service.neural._channels',
                    allowed_roles=['development_only'], extraction=False, training=False, camera=False, user_data=False))
        report = dict(status='PASS', completed_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Existing300 development observations -> current production adapter exact cached-input equality -> frozenM0 replay',
                      correct=correct, total=300, accuracy_percent=correct / 3,
                      top5_correct=sum(r['top5_correct'] for r in out), input_failures=0,
                      by_signer={str(s): dict(correct=sum(r['correct'] for r in out if r['signer_id'] == s), total=100) for s in (11, 12, 13)},
                      mean_ce=mean_ce, max_abs_saved_logit_difference=max_diff,
                      matched_feature_values=tensor_values, checkpoint_sha256=EXPECTED_MODEL_SHA,
                      new_prediction_sha256=sha(UNIT / 'baseline_PREDICTIONS.json'),
                      inference_seconds=infer_elapsed, elapsed_seconds=time.perf_counter() - start,
                      protected_file_count=len(protected), all_protected_files_unchanged=True,
                      model_state_unchanged=True, gradient_steps=0, extraction=0, user_data=0,
                      note='Development-only replay; not a new independent performance estimate or candidate selection')
        write_new(result_path, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        error = dict(status='FAILED', stage=stage, clip_id=current_clip, type=type(exc).__name__,
                     message=str(exc), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start,
                     checkpoint_sha256=sha(RUN / 'best.pt'), checkpoint_unchanged=sha(RUN / 'best.pt') == EXPECTED_MODEL_SHA,
                     changed_protected_files=[p for p, h in protected.items() if not Path(p).exists() or sha(p) != h],
                     gradient_steps=0, extraction=0, user_data=0)
        path = UNIT / 'baseline_FIRST_ERROR.json'
        if not path.exists():
            write_new(path, error)
        raise


if __name__ == '__main__':
    main()
