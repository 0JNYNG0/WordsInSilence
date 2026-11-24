import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
Y=O/'word100_yolo_selective_wrist_20261025'
H=O/'word100_mano_position_shape_split_20261026'
D=O/'word100_point_sequence_split_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def same(a,b):return np.array_equal(a,b,equal_nan=True)
