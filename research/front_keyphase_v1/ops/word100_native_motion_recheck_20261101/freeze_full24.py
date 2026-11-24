from pathlib import Path
import json,hashlib,datetime
N=Path(__file__).resolve().parent; F=N/'full24_native'; W=N.parent/'word100_interval_motion_20261101'
paths=[N/'audit_full24.py',N/'export_remaining.py',F/'PLAN_FROZEN.json',F/'MANIFEST.json',F/'BLIND_NATIVE.json',
    N/'MANIFEST.json',N/'review/NATIVE_A.json',N/'review/NATIVE_B.json',N/'NATIVE_DIAGNOSIS.json',
    N/'support_inventory/SUPPORT_RESULT.json',W/'PILOT_MANIFEST.json',W/'PILOT_GATES.json',W/'RUBRIC.json',
    W/'audit_presence.py',W/'PRESENCE_PILOT_AUDIT.json']
assert not (F/'REVIEW_A.json').exists() and not (F/'REVIEW_B.json').exists()
out=dict(status='FROZEN_BEFORE_REMAINING15_LABELS',at=datetime.datetime.now().astimezone().isoformat(),
    unchanged_gates=True,pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (F/'AUDIT_FROZEN.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(out['status'])
