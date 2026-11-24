"""Load current neural consumer with W: reads denied. No camera or inference."""
import json
from pathlib import Path
import sys
import time
import traceback

U = Path(__file__).resolve().parent
OPS = U.parent
seen = set()
blocked = []
sys.dont_write_bytecode = True


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
        return
    raw = args[0].decode(errors='replace') if isinstance(args[0], bytes) else args[0]
    normalized = raw.replace('\\', '/').lower()
    if normalized.startswith('w:/') or normalized.startswith('//?/w:/'):
        blocked.append(raw)
        raise PermissionError('Portability audit denied W: read: '+raw)
    seen.add(raw)


sys.addaudithook(audit)
sys.path.insert(0, str(OPS/'word100_action_feedback_demo_20261108'))
t = time.perf_counter()
result = dict(scope='current FeedbackConsumer import + weight/config load only',
              camera_opened=False, inference_calls=0, source_video_reads=0,
              actual_disk_disconnected=False,
              limitation='Python open audit denies W:. Native-engine/filesystem/complete UI not tested.')
try:
    from consumer import FeedbackConsumer
    c = FeedbackConsumer()
    result['status'] = 'LOAD_PASS_WITH_W_OPEN_DENIED'
except Exception:
    result['status'] = 'LOAD_FAILED'
    result['traceback'] = traceback.format_exc()
result['seconds'] = time.perf_counter()-t
result['blocked_W_reads'] = blocked
result['opened_project_files'] = sorted(x for x in seen if 'WordsInSilence' in x and Path(x).suffix in ('.pt','.json','.onnx','.npz'))
(U/'RUNTIME_LOAD_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
