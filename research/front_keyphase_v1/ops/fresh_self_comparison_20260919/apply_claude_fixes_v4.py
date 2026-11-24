from pathlib import Path
import json,re,hashlib
OPS=Path(__file__).resolve().parent
response=json.loads((OPS/'claude_fixes_v4_response.json').read_text(encoding='utf-8'))
assert response['subtype']=='success' and response['is_error'] is False
patch=json.loads(response['result'])
assert set(patch)=={'edits'} and len(patch['edits'])==6
source=(OPS/'claude_generated_v3/frozen-capture.js').read_text(encoding='utf-8')
for edit in patch['edits']:
    assert set(edit)=={'function','old','new'}
    assert edit['function'] in ('render','restore','recoverOwnOpeningFailure')
    matches=list(re.finditer(r'^  function '+re.escape(edit['function'])+r'\b.*?^  }\n',source,re.M|re.S))
    assert len(matches)==1
    match=matches[0]
    assert match.group(0).count(edit['old'])==1
    source=source[:match.start()]+match.group(0).replace(edit['old'],edit['new'],1)+source[match.end():]
target=OPS/'claude_generated_v4'
target.mkdir()
files={'frozen-capture.js':source,'frozen-capture.html':(OPS/'claude_generated_v3/frozen-capture.html').read_text(encoding='utf-8')}
for name,code in files.items():
    with (target/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code)
record={'authorship':'Claude Code','exact_returned_literal_edits_applied':6,'files':{name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in files},'production_exposed':False}
with (OPS/'code_application_v4.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
