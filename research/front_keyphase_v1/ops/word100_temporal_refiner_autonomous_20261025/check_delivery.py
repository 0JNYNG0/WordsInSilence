from pathlib import Path
import json,hashlib,re
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(U/'FINAL_AUDIT.json')
refs=re.findall(r'(?:href|src)="([^"]+)"',(U/'RESULT_KO.html').read_text(encoding='utf-8'))
assert all((U/r).is_file() for r in refs)
assert all(sha(p)==v for p,v in read(U/'FROZEN_PROTOCOL.json')['source_pins'].items())
assert all(sha(p)==v for p,v in a['root_sync'].items())
assert all(sha(U/p)==v for p,v in a['reports'].items())
assert not (U/'RUNNING.json').exists()
assert read(U/'downstream/RESULT.json')['observed_residual_preservation_checks']==660
assert read(U/'FINAL_RESULT.json')['production_changed'] is False
print(json.dumps(dict(status='PASS',local_artifact_links=len(refs),frozen_pins=True,root_sync=True,report_pins=True,active_lock=False)))
