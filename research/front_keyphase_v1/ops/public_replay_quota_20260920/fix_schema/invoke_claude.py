"""Send only the checked generic interface specification; no project file tools."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

OPS = Path(__file__).resolve().parent
PROMPT = OPS / 'claude_generic_contract.txt'
CLI = Path('C:/Users/local-user/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe')
argv = [str(CLI), '-p', '--safe-mode', '--tools', '', '--strict-mcp-config',
        '--disable-slash-commands', '--no-session-persistence', '--output-format', 'json']
started = time.time()
with (OPS / 'claude_response.json').open('xb') as out, (OPS / 'claude_stderr.txt').open('xb') as err:
    proc = subprocess.Popen(argv, cwd=OPS, stdin=subprocess.PIPE, stdout=out, stderr=err)
    proc.communicate(PROMPT.read_bytes(), timeout=600)
record = {'exit_code': proc.returncode, 'seconds': round(time.time()-started, 3),
          'generic_prompt_sha256': hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
          'response_bytes': (OPS / 'claude_response.json').stat().st_size,
          'source_media_arrays_models_user_records_sent': False,
          'builtin_tools_disabled': True, 'safe_mode': True}
with (OPS / 'claude_invocation.json').open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
sys.exit(proc.returncode)
