"""Operational launch gate: verify physical-byte pins before invoking the fixed CLI."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OPS = Path(__file__).resolve().parent
WEB = OPS.parent.parent
ready = json.loads((OPS / 'execution_ready.json').read_text(encoding='utf-8'))
assert ready['status'] == 'FIXED_PARENT_ANCHOR_PAIR_READY'
assert not Path(ready['out']).exists(), 'Output already exists; do not rerun automatically'
for name, expected in ready['frozen_execution_inputs'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
started = time.time()
argv = [sys.executable, '-u', '-B', '-X', 'utf8', str(WEB / 'tools_parent_anchor_pair.py'),
        '--plan', str(OPS / 'plan.json'), '--out', ready['out'], '--execute']
with (OPS / 'training_console.txt').open('x', encoding='utf-8') as log:
    process = subprocess.Popen(argv, cwd=WEB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='strict')
    with (OPS / 'training_started.json').open('x', encoding='utf-8') as f:
        json.dump({'pid': process.pid, 'argv': argv,
                   'at': datetime.datetime.now().astimezone().isoformat(),
                   'ready_sha256': hashlib.sha256((OPS / 'execution_ready.json').read_bytes()).hexdigest(),
                   'all_pins_verified_before_Popen': True}, f, indent=2)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
    code = process.wait()
with (OPS / 'training_exit.json').open('x', encoding='utf-8') as f:
    json.dump({'exit_code': code, 'seconds': time.time() - started,
               'pid': process.pid, 'out': ready['out']}, f, indent=2)
raise SystemExit(code)
