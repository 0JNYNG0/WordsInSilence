"""Mechanically apply reviewed Claude replacement functions to the isolated staged UI."""
from pathlib import Path
import hashlib
import json
import re

OPS=Path(__file__).resolve().parent
base=OPS/'claude_generated_v2'
target=OPS/'claude_generated_v3'
response=json.loads((OPS/'claude_fixes_v3_response.json').read_text(encoding='utf-8'))
assert response['subtype']=='success' and response['is_error'] is False
raw=response['result'].strip()
if raw.startswith('```json\n') and raw.endswith('\n```'):raw=raw[8:-4]
patch=json.loads(raw)
functions=['render','trialAction','onRetry','restore','onOpen','recoverOwnOpeningFailure']
assert set(patch)==set(functions+['pagehide','html_insertion','literal_edits'])
js=(base/'frozen-capture.js').read_text(encoding='utf-8')
html=(base/'frozen-capture.html').read_text(encoding='utf-8')
def pattern(name):return re.compile(r'^  function '+re.escape(name)+r'\b.*?^  }\n',re.M|re.S)
def indent(code):return '\n'.join('  '+line if line else '' for line in code.strip().splitlines())+'\n'
for name in functions:
    code=patch[name]
    assert isinstance(code,str) and code.strip().startswith('function '+name+'(')
    if name=='recoverOwnOpeningFailure':
        js=js.replace('  /* ---------- trials ---------- */',indent(code)+'\n  /* ---------- trials ---------- */',1)
    else:
        regex=pattern(name)
        assert len(regex.findall(js))==1,name
        js=regex.sub(lambda m:indent(code),js,count=1)
regex=re.compile(r"^  window\.addEventListener\('pagehide', function \(\) \{.*?^  \}\);",re.M|re.S)
assert len(regex.findall(js))==1
assert patch['pagehide'].strip().startswith("window.addEventListener('pagehide'")
js=regex.sub(lambda m:indent(patch['pagehide']).rstrip(),js,count=1)
assert len(patch['literal_edits'])==3
for edit in patch['literal_edits']:
    assert set(edit)=={'function','old','new'}
    assert edit['function'] in ('validatePlan','onSkip','doClose')
    regex=pattern(edit['function']); matches=list(regex.finditer(js));assert len(matches)==1
    old_function=matches[0].group(0)
    assert old_function.count(edit['old'])==1,edit['function']
    new_function=old_function.replace(edit['old'],edit['new'],1)
    js=js[:matches[0].start()]+new_function+js[matches[0].end():]
for name in ('next','skip'):
    regex=re.compile(r'<button\b[^>]*\bid="'+name+r'"[^>]*>.*?</button>',re.S)
    assert len(regex.findall(html))==1
    html=regex.sub('',html,count=1)
assert html.count('<footer class="muted">')==1
for name in ('next','skip'):assert patch['html_insertion'].count('id="'+name+'"')==1
html=html.replace('<footer class="muted">',patch['html_insertion'].strip()+'\n\n<footer class="muted">',1)
target.mkdir()
files={}
for suffix,source in [('js',js),('html',html)]:
    p=target/f'frozen-capture.{suffix}'
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    files[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with (OPS/'code_application_v3.json').open('x',encoding='utf-8') as f:
    json.dump({'authorship':'Claude Code','application':'Exact returned function replacements and scoped literal edits; move returned navigation button block outside hidden review section.',
               'response_sha256':hashlib.sha256((OPS/'claude_fixes_v3_response.json').read_bytes()).hexdigest(),'files':files,'production_exposed':False},f,indent=2)
print(json.dumps(files))
