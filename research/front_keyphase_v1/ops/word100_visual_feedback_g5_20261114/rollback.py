from pathlib import Path
import json,hashlib,shutil
U=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in json.loads((U/'CONNECTION.json').read_text('utf-8'))['files']:
    if sha(r['path'])!=r['after_sha256']:raise RuntimeError('Subsequent edits detected; refusing overwrite: '+r['path'])
    if sha(r['backup'])!=r['before_sha256']:raise RuntimeError('Backup mismatch')
for r in json.loads((U/'CONNECTION.json').read_text('utf-8'))['files']:shutil.copy2(r['backup'],r['path'])
print('Prior route/page restored. Restart existing server to activate rollback.')
