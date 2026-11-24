"""Dispatch only the existing generic specification; preserve earlier cancelled requests."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, hashlib, json, subprocess, time

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('part',choices=['core','driver'])
args=parser.parse_args()
part=args.part
prompt=(here.parent/f'claude_offline_{part}_contract.txt').read_text(encoding='utf-8')
prompt=prompt.replace('Return ONLY JSON {"code":"complete Python source"}; no markdown.', 'Return complete plain Python source only; no JSON or markdown wrappers.')
prompt=prompt.replace('Output only JSON {"code":"complete source"}.', 'Return complete plain Python source only, no JSON or markdown wrappers.')
prompt+='\nPrefer compact readable code, within about 400 lines. Implement the specified validation and reporting fully. Do not narrate plans or include tests.\n'
with (here/f'{part}_prompt.txt').open('x',encoding='utf-8') as f:f.write(prompt)
argv=['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe','-p','--safe-mode','--tools','','--strict-mcp-config','--disable-slash-commands','--no-session-persistence','--effort','low','--output-format','json']
started=time.time()
with (here/f'{part}_response.json').open('xb') as out,(here/f'{part}_stderr.txt').open('xb') as err:
    child=subprocess.Popen(argv,cwd=here,stdin=subprocess.PIPE,stdout=out,stderr=err)
    record={'pid':child.pid,'part':part,'started_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'private_source_data_sent':False}
    (here/f'{part}_worker.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    child.communicate(prompt.encode())
record={'exit_code':child.returncode,'seconds':round(time.time()-started,3),'response_bytes':(here/f'{part}_response.json').stat().st_size}
(here/f'{part}_completion.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
