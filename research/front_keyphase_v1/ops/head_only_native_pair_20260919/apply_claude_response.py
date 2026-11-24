"""Apply the reviewed Claude response verbatim, removing presentation fences only."""
from pathlib import Path
import ast
import hashlib
import json

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
response=json.loads((OPS/'claude_response.json').read_text(encoding='utf-8'))
assert response['type']=='result' and response['subtype']=='success' and response['is_error'] is False
code=response['result'].strip()
assert code.startswith('```python\n') and code.endswith('\n```')
code=code[len('```python\n'):-len('\n```')]+'\n'
ast.parse(code)
target=WEB/'tools_head_only_native_pair.py'
with target.open('x',encoding='utf-8',newline='\n') as f:f.write(code)
record={'authorship':'Claude Code','applied_verbatim_except_markdown_fences':True,
        'response_sha256':hashlib.sha256((OPS/'claude_response.json').read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'review':'Head-only optimizer, encoder eval, all persistent non-head state compared without restoration, no new file/network operations, host hooks restored in finally. No project data sent.'}
with (OPS/'code_application.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
