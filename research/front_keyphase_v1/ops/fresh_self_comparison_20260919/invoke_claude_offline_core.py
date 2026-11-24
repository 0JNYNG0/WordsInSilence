from pathlib import Path
from datetime import datetime,timezone,timedelta
import subprocess,json,hashlib,time
ops=Path(__file__).resolve().parent
prompt=ops/'claude_offline_core_contract.txt'
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--output-format','json']
started=time.time()
with (ops/'claude_offline_core_response.json').open('xb') as out,(ops/'claude_offline_core_stderr.txt').open('xb') as err:
    child=subprocess.Popen(argv,cwd=ops,stdin=subprocess.PIPE,stdout=out,stderr=err)
    record={'pid':child.pid,'started_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'private_source_data_sent':False}
    (ops/'claude_offline_core_worker.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    child.communicate(prompt.read_bytes())
record={'exit_code':child.returncode,'seconds':round(time.time()-started,3)}
(ops/'claude_offline_core_completion.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
