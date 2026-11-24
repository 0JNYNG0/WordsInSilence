from pathlib import Path
import json,subprocess,time
here=Path(__file__).resolve().parent
prompt=(here/'last_contract_fix.txt').read_bytes()
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
start=time.time()
with (here/'last_fix_response.json').open('xb') as out,(here/'last_fix_stderr.txt').open('xb') as err:
    p=subprocess.Popen(argv,cwd=here,stdin=subprocess.PIPE,stdout=out,stderr=err)
    (here/'last_fix_worker.json').write_text(json.dumps({'pid':p.pid}),encoding='utf-8')
    print(json.dumps({'pid':p.pid}),flush=True)
    p.communicate(prompt)
r={'exit_code':p.returncode,'seconds':round(time.time()-start,3)}
(here/'last_fix_completion.json').write_text(json.dumps(r),encoding='utf-8');print(json.dumps(r),flush=True)
