"""Isolated source-linked demo using the unchanged learned neural pipeline."""
from pathlib import Path
import sys,importlib.util,time
import numpy as np
from source_bridge import SourceBridge
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_learned_feedback_integration_20261102'
sys.path.insert(0,str(G))
spec=importlib.util.spec_from_file_location('_source_demo_existing_learner',G/'learner_adapter.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
read=old.read;write=old.write;WORDS=old.WORDS
bridge=SourceBridge()

def analyze(source,directory,word,condition='learned_motion',progress=None):
 old.U=U  # Only new output destination changes; all weights stay frozen.
 started=time.perf_counter();base=old.analyze(source,directory,word,condition,progress)
 a=dict(np.load(Path(directory)/'analysis.npz',allow_pickle=False));result=bridge.attach(base,a)
 result['latency']['source_bridge_seconds']=result['source_component_contract']['bridge_seconds']
 result['latency']['total_analysis_seconds']=time.perf_counter()-started
 result['base_learned_feedback_path']=str(Path(directory)/'FEEDBACK.json')
 write(Path(directory)/'SOURCE_FEEDBACK.json',result)
 return result

def warmup():
 old.U=U
 return old.warmup()
