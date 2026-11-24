"""Parse returned files for review; do not yet expose a camera-capable page."""
from pathlib import Path
import hashlib
import json

OPS=Path(__file__).resolve().parent
response=json.loads((OPS/'claude_response.json').read_text(encoding='utf-8'))
assert response['type']=='result' and response['subtype']=='success' and response['is_error'] is False
raw=response['result'].strip()
if raw.startswith('```json\n') and raw.endswith('\n```'):raw=raw[8:-4]
files=json.loads(raw)
assert set(files)=={'frozen-capture.html','frozen-capture.js'}
target=OPS/'claude_generated'
target.mkdir()
for name,code in files.items():
    assert isinstance(code,str) and code.strip()
    with (target/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code)
record={'status':'RETURNED_AND_STAGED_PENDING_REVIEW','authorship':'Claude Code','response_sha256':hashlib.sha256((OPS/'claude_response.json').read_bytes()).hexdigest(),
        'files':{name:{'bytes':(target/name).stat().st_size,'sha256':hashlib.sha256((target/name).read_bytes()).hexdigest()} for name in files},
        'production_page_changed':False,'camera_opened':False}
with (OPS/'code_staging.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
