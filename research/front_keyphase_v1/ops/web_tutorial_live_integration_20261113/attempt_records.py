"""Local inference diagnostics only; never a training/candidate manifest.

Keep each attempt beside its existing local capture. Do not duplicate dense
landmarks, encode session tokens, or expose these records through a new route.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re


def save_once(directory, filename, attempt_id, payload):
    if not re.fullmatch(r'[a-f0-9]{24}', attempt_id):
        raise ValueError('Invalid diagnostic attempt identity')
    if filename not in ('inference_result.json', 'browser_ack.json'):
        raise ValueError('Invalid diagnostic record name')
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('Capture directory unavailable')
    path = root / filename
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['attempt_id'] != attempt_id:
            raise ValueError('Capture record belongs to a different attempt')
        return previous
    data = dict(payload, attempt_id=attempt_id, training_use=False,
                tuning_use=False, candidate_selection_use=False,
                recorded_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return data


def save_result(directory, attempt, result, model_admissions):
    # Dense observations already live in rtmw/observations.npz. The results and
    # native-time spans are sufficient to reconstruct what the UI actually said.
    clean = {k: v for k, v in result.items() if k not in ('query_xy', '_stop_clock', '_ready_clock')}
    return save_once(directory, 'inference_result.json', attempt['id'],
                     dict(attempt=dict(attempt), result=clean,
                          model_admissions=model_admissions, scope='Local diagnostic inference only'))
