from pathlib import Path
from datetime import datetime, timezone, timedelta
import subprocess, json, hashlib, time
ops = Path(__file__).resolve().parent
prompt = ops / 'claude_preview_css_v5.txt'
argv = ['C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe', '-p', '--safe-mode', '--tools', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence', '--output-format', 'json']
started = time.time()
with (ops/'claude_preview_v5_response.json').open('xb') as out, (ops/'claude_preview_v5_stderr.txt').open('xb') as err:
    process = subprocess.Popen(argv, cwd=ops, stdin=subprocess.PIPE, stdout=out, stderr=err)
    record = {'pid':process.pid, 'started_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(), 'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(), 'private_source_data_sent':False}
    (ops/'claude_preview_v5_worker.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    process.communicate(prompt.read_bytes())
result = {'exit_code':process.returncode, 'seconds':round(time.time()-started,3)}
(ops/'claude_preview_v5_completion.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
