from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
BASE=O/'word100_body_boundary_20261017';PQ=O/'word100_assignment_training_20261018';FACT=O/'word100_assignment_scale_factorial_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x):
 p=Path(p);assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def resultpath(a,s):return BASE/f'training_C_{s}_RESULT.json' if a=='C' else PQ/f'Q_{s}_RESULT.json' if a=='Q' else FACT/f'{a}_{s}_RESULT.json'
def historypath(a,s):return BASE/f'training_C_{s}_history.json' if a=='C' else PQ/f'Q_{s}_history.json' if a=='Q' else FACT/f'{a}_{s}_history.json'
