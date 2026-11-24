from pathlib import Path
N=Path(__file__).resolve().parent;G=N.parent
src=(G/'public_visibility_roles_20261116/check_capture.py').read_text(encoding='utf-8')
src=src.replace('from check_web import *',"from check_runtime import *\nD=N;A=G/'apple_v1';V=G/'completion_v1'")
src=src.replace('from visibility_web_adapter import CaptureService','from frontal_web_adapter import CaptureService')
src=src.replace("[('before',PriorConsumer),('after',FeedbackConsumer)]","[('after',FeedbackConsumer)]")
src=src.replace("a,b=[r for r in rows if r['word']==word]","a=next(r for r in read(G/'public_cycle_observability_20261116/web/CAPTURE_CHECK.json')['rows']if r['word']==word and r['phase']=='after');b=next(r for r in rows if r['word']==word)")
src=src.replace('Not physical webcam or browser-paint latency.','Not physical webcam or browser-paint latency. Prior capture baseline reused; one current run per word, not a controlled latency benchmark.')
(N/'check_capture.py').write_text(src,encoding='utf-8')
