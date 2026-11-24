from pathlib import Path
import json
import hashlib
OPS=Path(__file__).resolve().parent
target=OPS/'claude_generated_v2'
target.mkdir()
files={}
for kind in ('html','js'):
    response=json.loads((OPS/f'claude_{kind}_v2_response.json').read_text(encoding='utf-8'))
    assert response['subtype']=='success' and response['is_error'] is False
    source=response['result'].strip()+'\n'
    assert source.startswith('<!doctype html>') if kind=='html' else source.startswith("'use strict';")
    assert source.rstrip().endswith('</html>') if kind=='html' else source.rstrip().endswith('})();')
    p=target/f'frozen-capture.{kind}'
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with (OPS/'code_staging_v2.json').open('x',encoding='utf-8') as f:
    json.dump({'authorship':'Claude Code','applied_verbatim':True,'status':'COMPLETE_FILES_STAGED_PENDING_REVIEW','files':files},f,indent=2)
print(json.dumps(files))
