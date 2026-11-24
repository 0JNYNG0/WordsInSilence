"""Guard the existing recording loop before camera frames become an attempt.
Keeps original pacing, encoder, RTMW, and return-to-rest thresholds.
"""
import inspect,textwrap
import capture
from auto_end import ReturnToRest as PreviousRest
from capture_pacing import paced_loop

class ReadyRest(PreviousRest):
    @property
    def ready(self):return len(self.samples)>=6

def guarded_loop():
    original=capture.CaptureService.camera_loop
    source=textwrap.dedent(inspect.getsource(original))
    marker="    if time.perf_counter()>=countdown_end:\n"
    assert source.count(marker)==1
    source=source.replace(marker,marker+"     if not rest.ready:\n      self.state='preview';self.countdown=0;self.diagnostics['auto_end_ready']=False;self.diagnostics['preparation_message']='양팔을 편하게 내리고 잠시 기다려 주세요. 준비되면 시작할 수 있어요.';continue\n")
    marker="observation=self.engine.infer(bgr);rest.calibrate(observation['xy'],w,h)"
    assert source.count(marker)==1
    source=source.replace(marker,marker+";self.diagnostics['auto_end_ready']=rest.ready;self.diagnostics['rest_samples']=len(rest.samples);self.diagnostics['preparation_message']='준비됐어요. 촬영을 시작해 주세요.' if rest.ready else '양팔을 편하게 내리고 잠시 기다려 주세요.'")
    namespace=dict(original.__globals__);namespace['ReturnToRest']=ReadyRest
    # Preserve the established bounded acquisition pacing exactly.
    # paced_loop inspects the provided function, so add readiness to its input
    # source through a local inspect proxy instead of mutating the frozen module.
    class SourceInspect:
        @staticmethod
        def getsource(_):return source
    ns=dict(paced_loop.__globals__);ns['inspect']=SourceInspect()
    import types
    factory=types.FunctionType(paced_loop.__code__,ns)
    stub=types.FunctionType(original.__code__,namespace,original.__name__)
    return factory(stub)

READY_CAMERA_LOOP=guarded_loop()
