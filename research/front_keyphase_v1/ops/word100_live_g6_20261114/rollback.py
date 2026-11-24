"""Restore the five web files only if no later edits have intervened. Does not restart."""
from pathlib import Path
import json,hashlib,shutil,argparse
U=Path(__file__).resolve().parent;W=U.parent/'web_tutorial_live_integration_20261113'
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
manifest=json.loads((U/'CONNECTION.json').read_text('utf-8'))['files']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,hashes in manifest.items():
 assert sha(U/'before'/name)==hashes['before'],name
 assert sha(W/name)==hashes['after'],f'Later change present; do not overwrite {name}'
if a.apply:
 for name in manifest:shutil.copy2(U/'before'/name,W/name)
print('Restored; restart server separately with camera off.' if a.apply else 'Rollback preflight passed; no files changed.')
