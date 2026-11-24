"""Codex independent transport/unchanged-feedback verification; camera blocked."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
sys.path[:0] = [str(LAB / 'src'), str(LAB.parent / 'sign_model_pilot_v1/src')]
BLOCKED = []


def no_device(*args, **kwargs):
    BLOCKED.append('device_access')
    raise AssertionError('camera/device access forbidden in this audit')


import cv2
cv2.VideoCapture = no_device
from wis_lab.service import camera
camera.CameraWorker.start = no_device
camera.list_devices = no_device
from wis_lab.service import app as service
from fastapi.testclient import TestClient
import torch
torch.set_num_threads(2)

IDS = ('NIKL_GAMSAHAMNIDA_01', 'NIKL_MEOKDA_01', 'NIKL_NA_01')
RAW = 'tcn_control_83212868'
NEW = 'tcn_control_83212868_handgap'


def stable(x):
    if isinstance(x, dict):
        return {k: stable(v) for k, v in x.items()
                if k not in ('elapsed_ms', 'server_predict_ms', 'result_id', 'generated_at', 'created_at')}
    if isinstance(x, list):
        return [stable(v) for v in x]
    return x


def capture(client, key):
    out = {}
    for cid in IDS:
        body = {'observation_id': cid, 'compare_to_word_id': cid.split('_')[1]}
        if key:
            body['research_model_key'] = key
        resp = client.post('/api/v1/replays', json=body)
        assert resp.status_code == 200, (cid, resp.status_code, resp.text[:200])
        out[cid] = stable(resp.json())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('before', 'after'))
    phase = parser.parse_args().phase
    before_path = ROOT / 'API_BEFORE.json'
    client = TestClient(service.app)
    records = {'off': capture(client, None), 'raw': capture(client, RAW)}
    if phase == 'before':
        with before_path.open('x', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print('Saved public replay baseline, device calls 0')
    else:
        old = json.loads(before_path.read_text(encoding='utf-8'))
        assert old == records, 'Opt-out or raw TCN replay result changed (excluding wallclock metadata)'
        records['repaired'] = capture(client, NEW)
        for cid, result in records['repaired'].items():
            base = records['off'][cid]
            for key in base:
                if key in ('research_model', 'research_comparison', 'lexical_identification'):
                    continue
                assert result.get(key) == base[key], (cid, key, 'feedback/default changed')
            assert result['research_comparison']['model_key'] == NEW
            assert result['research_model']['model_key'] == NEW
            assert result['accepted'] is None and result['execution_assessment'] == 'UNCONFIGURED'
        records['status'] = 'PASS'
        records['checks'] = ['opt_out_exact', 'old_raw_tcn_exact', 'default_dtw_and_finger_feedback_exact',
                             'candidate_identity_separate', 'no_device_requests']
        with (ROOT / 'API_AFTER.json').open('x', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print('API paired paths and unchanged feedback PASS')
    assert not BLOCKED, BLOCKED
    assert service.S().camera is None and not service.S().sessions


if __name__ == '__main__':
    main()
