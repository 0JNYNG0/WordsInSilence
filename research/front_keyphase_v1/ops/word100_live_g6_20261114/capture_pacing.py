"""Bound camera acquisition to two outstanding RTMW observations.
No accepted frame is dropped or interpolated. Camera sampling rate can fall;
timestamps and measured gaps remain truthful and are reported for acceptance.
Original capture source and models stay untouched.
"""
import inspect,textwrap,types,functools
def paced_loop(original):
 source=textwrap.dedent(inspect.getsource(original))
 marker='   ok,bgr=cap.read();captured=time.perf_counter()'
 assert source.count(marker)==1
 source=source.replace(marker,'''   if self.state=='recording' and stream is not None:
    waited=time.perf_counter()
    while submitted-len(stream.results)>=2 and not self.stop_event.is_set():
     if stream.errors or not stream.thread.is_alive():raise RuntimeError('관절 관측이 중단됐어요. 다시 촬영해 주세요.')
     if time.perf_counter()-waited>2:raise RuntimeError('관절 관측이 너무 늦어졌어요. 현재 동작을 확정하지 않고 촬영을 중단해요.')
     time.sleep(.001)
    self.diagnostics['acquisition_backpressure_seconds']=self.diagnostics.get('acquisition_backpressure_seconds',0)+(time.perf_counter()-waited)
    if self.stop_event.is_set():break
'''+marker)
 namespace=dict(original.__globals__);exec(compile(source,'<g6-paced-camera-loop>','exec'),namespace)
 result=namespace[original.__name__]
 return functools.update_wrapper(result,original)
