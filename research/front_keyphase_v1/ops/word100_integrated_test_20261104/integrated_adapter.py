"""UI integration, unchanged frozen learners and blind C/T3-F inference."""
from pathlib import Path
import sys,json,time,threading
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(U));sys.path.insert(0,str(O.parent/'src'))
import learner_adapter as core
from wis_lab.core.observation import Observation
from wis_lab.service import word100_candidate as C, word100_t3f as T
WORDS=core.WORDS
_preview=None;_preview_lock=threading.Lock()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def warmup():
 out=core.warmup();out['tcn']=T.identity()
 if out['tcn']['model_status']!='READY':raise RuntimeError(str(out['tcn']))
 return out
def classify(directory,dimensions):
 ob=Observation.load(str(Path(directory)/'observations.npz'))
 finalized=C.finalize_observation(ob,dimensions['width'],dimensions['height'])
 t=time.perf_counter();t3,c=T.recognise_pair(finalized)
 return dict(T3F=t3,C=c,seconds=time.perf_counter()-t,requested_word_used=False,
     input_observation_sha256=core.sha(Path(directory)/'observations.npz'))
def enrich_saved(result,directory):
 cached=read(U/'public_tcn.json')
 assert cached['source_sha256']==result['source_sha256']
 result['tcn']=cached['tcn'];result['ui_scope']='saved public example, unchanged feedback'
 return result
def analyze(video,directory,word,condition='learned_motion',progress=None,*,provenance):
 start=time.perf_counter()
 out=core.analyze(video,directory,word,condition,progress,provenance=provenance)
 if progress:progress(dict(stage='recognizing_word'))
 # Reuse exactly the full-frame native observation. No second detector pass.
 out['tcn']=classify(directory,out['overlay'])
 out['integrated_analysis_seconds']=time.perf_counter()-start
 out['ui_scope']='TCN recognition plus existing learned physical-feature feedback; not educational correctness'
 core.write(Path(directory)/'INTEGRATED_FEEDBACK.json',out)
 return out
def preview(body):
 global _preview
 # Only low-resolution in-memory frames, never labels/learning or saved images.
 with _preview_lock:
  if _preview is None:
   import importlib.util
   sp=importlib.util.spec_from_file_location('_integrated_preview',U/'preview_runtime.py')
   m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);_preview=m.Preview()
  return _preview.process(body)
def close():
 core.close()
 with _preview_lock:
  if _preview is not None:_preview.close()
