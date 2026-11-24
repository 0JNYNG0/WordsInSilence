from pathlib import Path
from datetime import datetime,timezone,timedelta
import subprocess,json,hashlib,time
OPS=Path(__file__).resolve().parent
prompt=OPS/'claude_fixes_v3.txt'
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--output-format','json']
started=time.time()
with (OPS/'claude_fixes_v3_response.json').open('xb') as out,(OPS/'claude_fixes_v3_stderr.txt').open('xb') as err:
    p=subprocess.Popen(argv,cwd=OPS,stdin=subprocess.PIPE,stdout=out,stderr=err)
    with (OPS/'claude_fixes_v3_worker.json').open('x',encoding='utf-8') as f:json.dump({'pid':p.pid,'started_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'private_source_data_sent':False},f,indent=2)
    print(json.dumps({'pid':p.pid,'status':'STARTED'}),flush=True)
    p.communicate(prompt.read_bytes())
record={'exit_code':p.returncode,'seconds':round(time.time()-started,3),'response_bytes':(OPS/'claude_fixes_v3_response.json').stat().st_size}
with (OPS/'claude_fixes_v3_completion.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
