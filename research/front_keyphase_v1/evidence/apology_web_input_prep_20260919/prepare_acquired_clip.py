"""Operational input preparation: existing extractor/adapters, one acquired train-side clip."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

EVIDENCE = Path(__file__).resolve().parent
WEB = EVIDENCE.parents[1]
ROOT = WEB.parents[1]
OUT = WEB / 'runs/apology_web_input_prep_v1'
sys.path[:0] = [str(WEB / 'src'), str(ROOT / 'research/sign_model_pilot_v1/src')]

import numpy as np
from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.service.neural import _channels

spec = importlib.util.spec_from_file_location(
    'existing_input_audit', WEB / 'evidence/isolated_word_input_prep_20260918/prepare_and_verify.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    started = time.perf_counter()
    acquisition_path = WEB / 'evidence/sen_source_access_20260919/archive03_inspection.json'
    acquisition = json.loads(acquisition_path.read_text(encoding='utf-8'))
    assert acquisition['status'] == 'COMPLETE_TARGETS_FOUND'
    rows = [r for r in acquisition['selected'] if r['source']['filename'] == 'NIA_SL_SEN0356_REAL02_F.mp4']
    assert len(rows) == 1
    row = rows[0]
    assert row['source']['signer_id'] == 2 and row['source']['split_role'] == 'train_candidate'
    source_path = Path(row['path']).resolve()
    assert source_path.parent == (WEB / 'runs/apology_source_review_v1/raw').resolve()
    assert source_path.name == 'NIA_SL_SEN0356_REAL02_F.mp4'
    assert source_path.stat().st_size == row['bytes']
    contract_path = WEB / 'config/contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    pin_paths = [Path(__file__), acquisition_path, contract_path, Path(audit.__file__)] + [
        WEB / 'src/wis_lab' / p for p in (
            'capture/landmarker.py', 'capture/video_source.py', 'core/observation.py', 'service/neural.py')]
    pins = {str(p): audit.sha(p) for p in pin_paths}
    assert not (EVIDENCE / 'result.json').exists(), 'preserve first result'
    OUT.mkdir(parents=True, exist_ok=False)
    source = dict(row['source'])
    source.update(clip_id=source_path.stem, local_video=str(source_path), video_sha256=row['sha256'],
                  raw_label='미안합니다', source_dataset=103, archive_member=row['archive_member'],
                  zip_membership='CONFIRMED', semantic_family_mapping='USER_CONFIRMED_FROM_REAL01',
                  this_signer_visual_review='NOT_DONE', training_registered=False)
    result = {'status': 'RUNNING', 'started_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
              'purpose': 'Prepare newly acquired REAL02 for later same-source web input use; not a classifier experiment',
              'source': source, 'source_pins_before': pins, 'training_runs': 0, 'classification_inferences': 0,
              'heldout_or_development_media_opened': 0, 'masking_or_augmentation': False}
    try:
        with Landmarker(contract) as lm:
            obs = extract(str(source_path), lm, source_path.stem)
        assert obs.sidecar['source_sha256'] == row['sha256']
        obs.sidecar.update(provenance=source, diagnostic_only=True, training_registered=False)
        obs_path = OUT / (source_path.stem + '.npz')
        obs.save(str(obs_path))
        channels, why, support = _channels(obs)
        assert channels is not None, why
        idx = np.flatnonzero(obs.usable[:, 33:].any(axis=1))
        span = slice(int(idx[0]), int(idx[-1]) + 1)
        channel_path = OUT / (source_path.stem + '.tcn_input.npz')
        np.savez_compressed(channel_path, x=channels['x'], body_valid=channels['body_valid'],
                            hand_valid=channels['hand_valid'], timestamp_us=obs.timestamp_us[span],
                            frame_id=obs.frame_id[span])
        checked = audit.verify(obs_path, channel_path, source, contract)
        metadata = {'source': source, 'observation_sha256': audit.sha(obs_path),
                    'tcn_input_sha256': audit.sha(channel_path), 'extractor_id': obs.sidecar['extractor_id'],
                    'temporal_policy': 'existing web first-to-last usable hand span; full observation retained',
                    'annotation_policy': 'metadata only; not verified complete action boundaries',
                    'provided_keypoint_frame_alignment': 'UNVERIFIED', 'training_registered': False,
                    'support': support, 'verification': checked}
        audit.write_json(channel_path.with_suffix('.json'), metadata)
        assert checked['raw_frames'] == row['decoded_frames']
        assert {str(p): audit.sha(p) for p in pin_paths} == pins
        result.update(status='PASS_NEW_INPUT_PREPARATION_ONLY', observation_path=str(obs_path),
                      tcn_input_path=str(channel_path), verification=checked, source_pins_unchanged=True,
                      elapsed_seconds=round(time.perf_counter() - started, 3),
                      output_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    except Exception as exc:
        result.update(status='FAIL_FIRST_RESULT_PRESERVED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['finished_kst'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        audit.write_json(EVIDENCE / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds', 'output_bytes', 'verification')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
