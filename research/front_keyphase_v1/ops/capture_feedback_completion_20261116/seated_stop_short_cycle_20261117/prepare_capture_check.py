from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent/'selective_rtmw_anchor_repair_20261117'
s=(R/'check_capture.py').read_text('utf-8')
s=s.replace('from probe import *','from check_runtime import *\nimport torch,cv2\nfrom stream_runtime import Engine')
s=s.replace('from retry_runtime import CaptureService,ConsumerProxy','from upgraded_runtime import CaptureService,ConsumerProxy')
s=s.replace("for kind in ['normal','short_public_observation_dropout']:","for kind in ['normal','short_public_observation_dropout']:")
s=s.replace("q['source']='public_capture_fixture';", "assert (service.directory/'auto_end_diagnostic.json').is_file();q['source']='public_capture_fixture';")
(D/'check_capture.py').write_text(s,encoding='utf-8')
# Preserve all serving routes/assets and change only verified computation service.
s=(R/'serve.py').read_text('utf-8').replace('from retry_runtime import CaptureService','from upgraded_runtime import CaptureService')
(D/'serve.py').write_text(s,encoding='utf-8')
