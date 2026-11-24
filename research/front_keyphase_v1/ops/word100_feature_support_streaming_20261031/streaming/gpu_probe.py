"""Real construction attempt only, preserving each failure; never processes input."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def main():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    attempts = []
    for kind, model in [('hand', 'hand_landmarker.task'), ('pose', 'pose_landmarker_lite.task')]:
        path = ROOT / 'Tools' / 'models' / model
        started = time.perf_counter()
        record = dict(kind=kind, model=str(path), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), requested_delegate='GPU', running_mode='VIDEO', input_frames=0)
        task = None
        try:
            base = python.BaseOptions(model_asset_path=str(path), delegate=python.BaseOptions.Delegate.GPU)
            if kind == 'hand':
                task = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=2))
            else:
                task = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1))
            record.update(status='CONSTRUCTION_SUCCEEDED_NOT_INFERENCE_VALIDATED')
        except BaseException as error:
            record.update(status='CONSTRUCTION_FAILED', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            failure = HERE / f'GPU_{kind.upper()}_FIRST_FAILURE.json'
            if not failure.exists():
                with failure.open('x', encoding='utf8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
        finally:
            if task is not None:
                task.close()
        record['seconds'] = time.perf_counter()-started
        attempts.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result = dict(status='GPU_CONSTRUCTION_UNAVAILABLE_CPU_FALLBACK' if any(r['status']=='CONSTRUCTION_FAILED' for r in attempts) else 'GPU_CONSTRUCTION_ONLY_SUCCEEDED_NEEDS_SEPARATE_VALIDATION', platform=platform.platform(), python=sys.version, mediapipe=mp.__version__, attempts=attempts, benchmark_delegate='original CPU default', no_input_inference=True, downloads=0, official_reference='https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions')
    with (HERE/'GPU_PROBE.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        path=HERE/'GPU_PROBE_FIRST_FAILURE.txt'
        if not path.exists():
            path.write_text(traceback.format_exc(), encoding='utf8')
        raise
