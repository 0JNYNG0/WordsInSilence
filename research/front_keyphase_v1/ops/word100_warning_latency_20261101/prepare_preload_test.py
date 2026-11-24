from pathlib import Path
import hashlib,json
U=Path(__file__).resolve().parent
t=(U/'check_candidate_browser.py').read_text(encoding='utf8')
assert "PREFIX='CANDIDATE_'" in t
t=t.replace("PREFIX='CANDIDATE_'","PREFIX='PRELOAD_'")
with (U/'check_preload_browser.py').open('x',encoding='utf8') as f:f.write(t)
print('prepared same fixed public clip, different once-only upload intent')
