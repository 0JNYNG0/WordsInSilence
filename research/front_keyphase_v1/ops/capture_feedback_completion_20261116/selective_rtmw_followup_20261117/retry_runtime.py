"""Optional post-capture recovery; frozen recognizer and capture loop retained."""
from pathlib import Path
import sys,json,time,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'public_frontal_evidence_repair_20261117'
sys.path.insert(0,str(P))
from frontal_repair_runtime import CaptureService as PreviousCapture
from selective_retry import retry,VideoFrames,VERSION


class ConsumerProxy:
    def __init__(self,inner,service):self.inner=inner;self.service=service
    def __getattr__(self,name):return getattr(self.inner,name)
    def raw(self,raw,word):
        initial=self.inner.raw(raw,word)
        s=self.service
        if not s.retry_admitted or s.state!='analysing' or s.directory is None:
            return initial
        # Session.finish released this lock before the analysis phase. Never
        # compete with an active camera stream or another GPU recovery request.
        if not s.engine.session_lock.acquire(False):
            out=dict(initial);out['selective_retry']=dict(version=VERSION,accepted=False,pose_calls=0,reason='engine_busy',seconds=0.)
            return out
        began=time.perf_counter()
        try:
            out,fixed=retry(raw,initial,VideoFrames(s.directory/'recorded.mp4',raw),s.engine,self.inner.raw,word)
            out['compute_seconds']=initial['compute_seconds']+out['selective_retry']['seconds']
            out['timing']=dict(out.get('timing',{}),selective_reobservation_seconds=out['selective_retry']['seconds'])
            out['observation_guard']=dict(out.get('observation_guard',{}),raw_preserved=True,selective_reobserved_points=out['selective_retry'].get('replaced_points',0)if out['selective_retry']['accepted']else 0,interpolated_points=False)
            # Original video and RTMW arrays remain untouched. Save only recovery
            # metadata and the frame/point identities used for this inference.
            try:
                (s.directory/'selective_retry.json').write_text(json.dumps(out['selective_retry'],ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as exc:
                out['selective_retry']['diagnostic_write_error']=type(exc).__name__
            return out
        except Exception as exc:
            # An optional retry or diagnostic-write failure cannot break capture.
            out=dict(initial);out['selective_retry']=dict(version=VERSION,accepted=False,pose_calls=None,reason='optional_retry_failure',error=type(exc).__name__,seconds=time.perf_counter()-began)
            return out
        finally:s.engine.session_lock.release()


class CaptureService(PreviousCapture):
    def __init__(self):
        super().__init__();self.retry_admitted=False
    def prepare(self):
        began=time.perf_counter()
        super().prepare()
        if self.state!='ready':return
        # Do not expose ready while the additional admission hashes are still
        # being checked; the base prepare has only finished its own components.
        self.state='preparing'
        try:
            p=D/'web/ADMISSION.json'
            if p.is_file():
                a=json.loads(p.read_text(encoding='utf-8'))
                self.retry_admitted=a['admitted']is True and all(Path(f).is_file()and hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in a['pins'].items())
            if not self.retry_admitted:raise RuntimeError('재관측 기능의 검증 파일이 일치하지 않습니다.')
            self.consumer=ConsumerProxy(self.consumer,self)
            self.prepare_seconds=time.perf_counter()-began;self.state='ready'
        except Exception as exc:
            self.error=str(exc);self.state='error'
    def camera_loop(self):
        super().camera_loop()
        if self.state=='done'and self.result is not None:
            self.result['second_pass_detector_calls']=self.result.get('selective_retry',{}).get('pose_calls',0)
            self.result['whole_video_redetection']=False
    def status(self):
        q=super().status();q['selective_retry_active']=self.retry_admitted;q['selective_retry_version']=VERSION;return q
