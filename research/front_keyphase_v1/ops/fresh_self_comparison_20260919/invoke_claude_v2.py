"""Two small independent generic-code requests, with durable PID and no artificial wait cutoff."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import subprocess
import time

OPS=Path(__file__).resolve().parent
CLI='C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe'
args=[CLI,'-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--output-format','json']
jobs=[]
for part in ('html','js'):
    prompt=OPS/f'claude_{part}_v2.txt'
    out=(OPS/f'claude_{part}_v2_response.json').open('xb')
    err=(OPS/f'claude_{part}_v2_stderr.txt').open('xb')
    proc=subprocess.Popen(args,cwd=OPS,stdin=subprocess.PIPE,stdout=out,stderr=err)
    proc.stdin.write(prompt.read_bytes());proc.stdin.close()
    jobs.append((part,proc,out,err,time.time()))
    with (OPS/f'claude_{part}_v2_worker.json').open('x',encoding='utf-8') as f:
        json.dump({'pid':proc.pid,'started_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),
                   'generic_prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'actual_source_or_data_sent':False},f,indent=2)
    print(json.dumps({'part':part,'pid':proc.pid,'status':'STARTED'}),flush=True)
for part,proc,out,err,started in jobs:
    code=proc.wait()
    out.close();err.close()
    record={'part':part,'pid':proc.pid,'exit_code':code,'elapsed_seconds':round(time.time()-started,3),
            'response_bytes':(OPS/f'claude_{part}_v2_response.json').stat().st_size}
    with (OPS/f'claude_{part}_v2_completion.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)
