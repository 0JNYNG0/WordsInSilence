from pathlib import Path
import sys,json,hashlib,time
import numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;P=G/'seated_stop_short_cycle_20261117';N=G/'frontal_scope_recovery_20261117'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():
 rows=read(N/'INPUTS.json')['rows']
 for r in rows:assert r['id'].endswith('_F') and 'Temp' not in r['raw'] and 'user_' not in r['raw']
 return rows
