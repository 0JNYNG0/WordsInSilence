from pathlib import Path
import sys,json,hashlib,datetime,importlib.util
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];V=O/'word100_visible_hand_recheck_20261026';I=O/'word100_wrist_only_inputs_20261026';B=O/'word100_body_boundary_20261017'
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
C=mod('frozen_visible_schedule',V/'candidate.py')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
