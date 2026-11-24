"""Apply completed Claude source verbatim into a new staging directory, syntax-only check."""
from pathlib import Path
import ast,hashlib,json
here=Path(__file__).resolve().parent
stage=here/'generated_v1'
stage.mkdir(exist_ok=True)
records=[]
for part,name in [('core','frozen_cohort.py'),('driver','frozen_replay.py')]:
    completion=json.loads((here/f'{part}_completion.json').read_text(encoding='utf-8'))
    if completion['exit_code']!=0:raise RuntimeError(f'{part}: nonzero author exit')
    response=json.loads((here/f'{part}_response.json').read_text(encoding='utf-8'))
    if response.get('is_error'):raise RuntimeError(f'{part}: API response error')
    code=response['result'].strip()
    if code.startswith('```python\n') and code.endswith('\n```'):code=code[len('```python\n'):-len('\n```')]
    elif code.startswith('```\n') and code.endswith('\n```'):code=code[len('```\n'):-len('\n```')]
    ast.parse(code,filename=name)
    with (stage/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code+'\n')
    records.append({'part':part,'file':str(stage/name),'sha256':hashlib.sha256((stage/name).read_bytes()).hexdigest(),'lines':len(code.splitlines())})
record={'author':'Claude Code','application':'exact source; optional markdown wrapper removed','syntax':'PASS','behavior':'UNVERIFIED','files':records}
(here/'staging_v1.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
