"""Mechanical application of reviewed Claude-authored replacements, preserving v1."""
from pathlib import Path
import ast,hashlib,json
here=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def result(stem):
    if read(here/(stem+'_completion.json'))['exit_code']!=0:raise RuntimeError(stem)
    response=read(here/(stem+'_response.json'))
    if response.get('is_error'):raise RuntimeError(stem)
    raw=response['result'].strip()
    if raw.startswith('```json\n') and raw.endswith('\n```'):raw=raw[8:-4]
    return json.loads(raw)
edits=result('fixes')['edits'];small=result('last_fix')
bad='if outcome not in ("COMPLETED", "SKIPPED", "ABORTED", "TIMEOUT", "ERROR"):'
guard_response=read(here/'outcome_guard_response.json')
if guard_response.get('is_error'):raise ValueError('Author guard error')
guard=json.loads(guard_response['result'])['outcome_condition'].strip()
if guard!='if outcome not in FAILURE_OUTCOMES:':raise ValueError('Unexpected outcome correction')
source={name:(here/'generated_v1'/name).read_text(encoding='utf-8') for name in ['frozen_cohort.py','frozen_replay.py']}
for edit in edits:
    filename=edit['file'];old,new=edit['old'],edit['new']
    if filename not in source or source[filename].count(old)!=1:raise ValueError('Non-unique literal replacement: '+filename)
    source[filename]=source[filename].replace(old,new,1)
if source['frozen_cohort.py'].count(bad)!=1:raise ValueError('Expected draft condition absent')
source['frozen_cohort.py']=source['frozen_cohort.py'].replace(bad,guard,1)
stage=here/'generated_v2';stage.mkdir(exist_ok=False)
for name,code in source.items():
    ast.parse(code,filename=name)
    (stage/name).write_text(code,encoding='utf-8')
record={'status':'APPLIED_PENDING_VERIFICATION','author':'Claude Code','literal_edits':len(edits)+1,'files':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in source},'preserved_failed_v1':True,'discarded_draft_outcome_condition':bad}
(here/'staging_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
