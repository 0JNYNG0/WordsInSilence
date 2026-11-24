"""Send only the generic interface contract to the authorized implementation author."""
import hashlib,json,subprocess,time
from datetime import datetime
from pathlib import Path
here=Path(__file__).resolve().parent
prompt=(here/'claude_generic_contract.txt').read_bytes()
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
started=time.time()
with (here/'claude_response.json').open('xb') as output,(here/'claude_stderr.txt').open('xb') as error:
    child=subprocess.Popen(argv,cwd=here,stdin=subprocess.PIPE,stdout=output,stderr=error)
    worker={'pid':child.pid,'started_at':datetime.now().isoformat(),'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'private_source_or_data_sent':False}
    (here/'claude_worker.json').write_text(json.dumps(worker,indent=2),encoding='utf-8')
    print(json.dumps(worker),flush=True)
    child.communicate(prompt)
record={'exit_code':child.returncode,'elapsed_seconds':round(time.time()-started,3),'response_bytes':(here/'claude_response.json').stat().st_size}
(here/'claude_completion.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
