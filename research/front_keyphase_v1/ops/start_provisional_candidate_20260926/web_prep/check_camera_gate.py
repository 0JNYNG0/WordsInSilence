"""Mock-device check of the real-camera gate in tests/test_r1c_camera.py. Touches NO real device:
a recording fake replaces cv2.VideoCapture BEFORE the test module is imported, in every case."""
import io, json, os, pathlib, subprocess, sys
H = pathlib.Path(__file__).resolve().parent
L = H.parents[2]

CHILD = r'''
import contextlib, io, json, os, sys
import cv2
FAKE_CALLS = []
class Fake:
    def __init__(self, *a, **k): FAKE_CALLS.append(list(a))
    def isOpened(self): return False
    def read(self): return False, None
    def release(self): pass
    def set(self, *a): return False
cv2.VideoCapture = Fake                                  # mock device, installed first
sys.path[:0] = [os.path.join(sys.argv[1], "tests"), os.path.join(sys.argv[1], "src")]
runs_before = set(os.listdir(os.path.join(sys.argv[1], "runs")))
import test_r1c_camera as t
from wis_lab.service import camera
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = t.main()
listing = None
try:
    listing = camera.list_devices()
except RuntimeError as e:
    listing = "RAISED: " + str(e)
print(json.dumps({"allowed_flag": t.REAL_CAMERA_ALLOWED, "main_return": rc, "main_output": buf.getvalue().strip()[:200],
                  "videocapture_is_blocker": cv2.VideoCapture is t._NoRealDevice,
                  "mock_device_calls": len(FAKE_CALLS), "blocked_calls": len(t.DEVICE_CALLS_BLOCKED),
                  "list_devices": listing, "checks_defined": len(t.CHECKS),
                  "new_run_dirs": sorted(set(os.listdir(os.path.join(sys.argv[1], "runs"))) - runs_before)}))
'''

def run(env_value, call_main):
    env = {k: v for k, v in os.environ.items() if k not in ("WIS_ALLOW_REAL_CAMERA", "WIS_SKIP_CAMERA")}
    if env_value is not None:
        env["WIS_ALLOW_REAL_CAMERA"] = env_value
    code = CHILD if call_main else CHILD.replace("rc = t.main()", "rc = 'not called'")
    p = subprocess.run([sys.executable, "-c", code, str(L)], env=env, capture_output=True, text=True, encoding="utf-8")
    line = [x for x in p.stdout.splitlines() if x.startswith("{")]
    return json.loads(line[-1]) if line else {"error": p.stderr[-800:]}

res = {
    "default_unset": run(None, True),
    "wrong_value_0": run("0", True),
    "wrong_value_true": run("true", True),
    # allowed: only the device layer is exercised (main() would open sessions); the MOCK must receive the probe
    "allowed_1_device_layer_only": run("1", False),
}
ok = (all(r.get("videocapture_is_blocker") and r.get("mock_device_calls") == 0 and r.get("main_return") == 0
          and str(r.get("list_devices", "")).startswith("RAISED") and not r.get("new_run_dirs")
          for k, r in res.items() if k != "allowed_1_device_layer_only")
      and res["allowed_1_device_layer_only"].get("videocapture_is_blocker") is False
      and res["allowed_1_device_layer_only"].get("mock_device_calls") == 5
      and not res["allowed_1_device_layer_only"].get("new_run_dirs"))
res["verdict"] = "PASS" if ok else "FAIL"
res["real_device_touched"] = False
(H / "CAMERA_GATE_CHECK.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
