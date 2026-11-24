"""Prepare a pinned, existing-raw-only comparison against joint v3. No replay."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
J=U.with_name('word100_hand_assignment_joint_20261016')
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

protected=read(J/'PROTECTED_FILES.json')
for p,h in protected.items():assert sha(Path(p))==h,p
for p in J.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
result=read(J/'REPLAY_RESULT.json')
rows=[]
for r in result['rows']:
    d=Path(r['raw_dir']);prev=J/'replay'/r['clip_id']/r['mode']/'assignment.npz'
    assert prev.exists()
    pins={str(p):sha(p) for p in [d/'raw.npz',d/'baseline.npz',d/'baseline.sidecar.json',prev]}
    protected.update(pins)
    rows.append({k:r[k] for k in ['clip_id','label','mode','frames','raw_dir','cohort']}|{'previous_assignment':str(prev),'pins':pins})
assert len(rows)==40 and sum(r['frames'] for r in rows)==5072
new(U/'PROTECTED_FILES.json',protected)
new(U/'INPUT_MANIFEST.json',dict(status='FROZEN_INPUTS_BEFORE_NEW_CANDIDATE_REPLAY',previous_candidate='joint_v3',
    previous_result_sha256=sha(J/'REPLAY_RESULT.json'),conditions=40,frames=5072,raw_hand_detections=6720,
    dataset_role='public TRAIN engineering diagnostic fixtures; no user/dev/test records',rows=rows))

s=(J/'replay.py').read_text('utf-8')
a=s.index("V2=U.with_name(");b=s.index("sys.stdout.reconfigure",a)
s=s[:a]+"J=U.with_name('word100_hand_assignment_joint_20261016')\n\n"+s[b:]
a=s.index("    capture=read(P/'RAW_CAPTURE_RESULT.json')");b=s.index("    frozen=read(U/'CANDIDATE_FREEZE.json')",a)
s=s[:a]+"    capture=read(U/'INPUT_MANIFEST.json')\n    ctx=read(P/'RAW_CAPTURE_CONTEXT.json')\n    broadctx=read(J/'BROADENED_CAPTURE_CONTEXT.json')\n    assert len(capture['rows'])==40\n"+s[b:]
a=s.index("    files=[U/'BROADENED_TRAIN_PLAN.json'");b=s.index("    pins={str(p):sha256_file(p) for p in files}",a)
s=s[:a]+"    files=[U/x for x in ['INPUT_MANIFEST.json','CANDIDATE_FREEZE.json','SCOPE.json','assignment.py','replay.py','POINT_CONTRACT.json','PROTECTED_FILES.json']]\n    for r in capture['rows']:\n        for p,h in r['pins'].items():assert sha256_file(p)==h,p\n        files.extend(Path(p) for p in r['pins'])\n"+s[b:]
s=s.replace("previous=v2module.HandAssigner();prev_array=np.full((old.n_frames,2),-1,np.int8)","prev_array=np.full((old.n_frames,2),-1,np.int8)")
a=s.index("        cached_v2=V2/");b=s.index("        base=Landmarker",a)
s=s[:a]+"        with np.load(rec['previous_assignment'],allow_pickle=False) as z:cached_previous=z['sides']\n        assert cached_previous.shape==prev_array.shape\n\n"+s[b:]
a=s.index("            if cached_previous is not None:");b=s.index("            for i in range(n):",a)
s=s[:a]+"            prev_array[f]=cached_previous[f]\n\n"+s[b:]
s=s.replace("extractor_id='handassoc4_'","extractor_id='handassoc_armchain_'")
s=s.replace("'research_causal_bounded_hand_association'","'research_armchain_association'")
s=s.replace("'status':'COMPUTED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW'","'status':'COMPUTED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW','previous_candidate':'joint_v3'")
compile(s,str(U/'replay.py'),'exec')
with (U/'replay.py').open('x',encoding='utf-8') as f:f.write(s)
print(json.dumps({'prepared_conditions':len(rows),'protected_files':len(protected),'candidate_executed':False}))
