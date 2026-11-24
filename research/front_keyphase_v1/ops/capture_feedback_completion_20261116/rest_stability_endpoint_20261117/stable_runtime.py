"""Admit only the publicly checked automatic-stop change; keep feedback frozen."""
from pathlib import Path
import sys, types
D = Path(__file__).resolve().parent
P = D.parent / 'seated_stop_short_cycle_20261117'
sys.path.insert(0, str(P))
from upgraded_runtime import CaptureService as PreviousCapture, FeedbackConsumer, ConsumerProxy, EvidenceCapture, bind_dependency, admitted
import upgraded_capture
sys.path.insert(0, str(D))
from stable_rest import StableRest

VERSION = 'bounded_weak_rest_pause_v2'
env = dict(upgraded_capture.build_loop.__globals__)
env['ReturnToRest'] = StableRest
CAMERA_LOOP = types.FunctionType(upgraded_capture.build_loop.__code__, env)()

class CaptureService(PreviousCapture):
    _camera_loop = bind_dependency(EvidenceCapture.camera_loop, '_paced_camera_loop', CAMERA_LOOP)

    def prepare(self):
        self.rest_stability_admitted = False
        super().prepare()
        if self.state != 'ready':
            return
        self.state = 'preparing'
        try:
            self.rest_stability_admitted = admitted(D / 'web/ADMISSION.json')
            if not self.rest_stability_admitted:
                raise RuntimeError('자동 종료 안정화 검증 파일이 일치하지 않습니다.')
            self.state = 'ready'
        except Exception as exc:
            self.error = str(exc)
            self.state = 'error'

    def status(self):
        q = super().status()
        q['rest_stability_active'] = getattr(self, 'rest_stability_admitted', False)
        q['rest_stability_version'] = VERSION
        return q

    def camera_loop(self):
        super().camera_loop()
        if self.state == 'done' and self.result is not None:
            self.result['rest_stability_version'] = VERSION
