"""Preserve readiness/pacing, substitute capture-only rest observation policy."""
import inspect,textwrap,types
import capture
from capture_pacing import paced_loop
from return_rest import ReturnToRest

def build_loop():
 original=capture.CaptureService.camera_loop
 source=textwrap.dedent(inspect.getsource(original))
 marker="    if time.perf_counter()>=countdown_end:\n"
 assert source.count(marker)==1
 source=source.replace(marker,marker+"     if not rest.ready:\n      self.state='preview';self.countdown=0;self.diagnostics['auto_end_ready']=False;self.diagnostics['preparation_message']='양팔을 편하게 내리고 잠시 기다려 주세요.';continue\n")
 marker="observation=self.engine.infer(bgr);rest.calibrate(observation['xy'],w,h)"
 assert source.count(marker)==1
 source=source.replace(marker,marker+";self.diagnostics['auto_end_ready']=rest.ready;self.diagnostics['rest_samples']=len(rest.samples);self.diagnostics['preparation_message']='준비됐어요. 촬영을 시작해 주세요.' if rest.ready else '양팔을 편하게 내리고 잠시 기다려 주세요.'")
 marker="  self.result=self.consumer.raw(arrays,self.word);"
 assert source.count(marker)==1
 source=source.replace(marker,"  self.auto_end_diagnostic=rest.diagnostic()\n"+marker)
 ns=dict(original.__globals__);ns['ReturnToRest']=ReturnToRest
 class SourceInspect:
  @staticmethod
  def getsource(_):return source
 env=dict(paced_loop.__globals__);env['inspect']=SourceInspect()
 return types.FunctionType(paced_loop.__code__,env)(types.FunctionType(original.__code__,ns,original.__name__))
CAMERA_LOOP=build_loop()
