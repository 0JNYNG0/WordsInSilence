"""Independent result arithmetic; no model imports, reruns or changes to saved results."""
from pathlib import Path
import json, hashlib, math, statistics
import numpy as np
root=Path(__file__).resolve().parent
def read(n):return json.loads((root/n).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
checks=[]
def check(ok,label):
    checks.append(dict(check=label,pass_=bool(ok)))
    if not ok:raise AssertionError(label)
b=read('BENCHMARK.json'); frozen=read('FROZEN.json'); inp=read('INPUTS.json'); acquired=read('ACQUISITION.json')
check(digest(root/'assets/ckpt.pt')==frozen['checkpoint_sha256'],'checkpoint hash')
check(digest(root/'FROZEN.json')==b['freeze_sha256']==inp['freeze_sha256'],'frozen comparison unchanged')
for row in acquired['vendor_files']:check(digest(root/'vendor/SignRep'/row['file'])==row['sha256'],'official source '+row['file'])
check(len(b['rows'])==24,'4 conditions x3 clips x2 repeats')
seen=set(); recalculated=[]
for r in b['rows']:
    key=(r['condition'],r['alias'],r['repeat']);check(key not in seen,'unique result '+str(key));seen.add(key)
    row=next(x for x in inp['rows'] if x['alias']==r['alias']); n=row['frames']
    ss=[s for s in range(n-15) if s%2==0]
    if ss[-1]!=n-16:ss.append(n-16)
    check(ss==row['window_starts'],'window index '+str(key))
    check(set(range(n))=={t for s in ss for t in range(s,s+16)},'all real frames covered '+str(key))
    file=root/'features'/f"{r['condition']}_{r['alias']}_r{r['repeat']}.npy"
    a=np.load(root/'features'/f"torch_cpu_fp32_b1_{r['alias']}_r0.npy",allow_pickle=False).astype(np.float64)
    p=np.load(file,allow_pickle=False).astype(np.float64)
    check(p.shape==(len(ss),768),'output shape '+str(key));check(np.isfinite(p).all(),'finite '+str(key))
    rels=[]; cosines=[]
    for ref,pred in zip(a,p):
        norm_a=math.sqrt(float(np.dot(ref,ref))); norm_b=math.sqrt(float(np.dot(pred,pred)))
        delta=ref-pred
        rels.append(math.sqrt(float(np.dot(delta,delta)))/max(norm_a,1e-12))
        cosines.append(float(np.dot(ref,pred))/max(norm_a*norm_b,1e-12))
    original=next(x for x in b['parity'] if (x['condition'],x['alias'],x['repeat'])==key)
    check(abs(max(rels)-original['max_relative_l2'])<1e-10 and abs(min(cosines)-original['min_cosine'])<1e-10,'separate numerical arithmetic '+str(key))
    recalculated.append(dict(condition=r['condition'],alias=r['alias'],repeat=r['repeat'],max_relative_l2=max(rels),min_cosine=min(cosines),pass_=max(rels)<=.01 and min(cosines)>=.999,file_sha256=digest(file)))
for row in inp['rows']:
    check(digest(row['source'])==row['source_sha256'],'public original unchanged '+row['alias'])
    check(digest(root/'inputs'/row['alias']/'rgb224.npy')==row['input_sha256'],'input unchanged '+row['alias'])
for r in read('COMPILES.json')['rows']:check(all(x.startswith('GPU') for x in r['devices']),'no CPU fallback '+r['condition'])
summary=[]
for name in dict.fromkeys(r['condition'] for r in b['rows']):
    rows=[r for r in b['rows'] if r['condition']==name]
    p=[r for r in recalculated if r['condition']==name]
    sums=[r['seconds']+next(x['seconds'] for x in b['preprocessing'] if x['alias']==r['alias'] and x['repeat']==r['repeat']) for r in rows]
    summary.append(dict(condition=name,inference_median_seconds=statistics.median(r['seconds'] for r in rows),inference_range_seconds=[min(r['seconds'] for r in rows),max(r['seconds'] for r in rows)],separate_component_sum_median_seconds=statistics.median(sums),separate_component_sum_range_seconds=[min(sums),max(sums)],numerical_pass=all(x['pass_'] for x in p),max_relative_l2=max(x['max_relative_l2'] for x in p),min_cosine=min(x['min_cosine'] for x in p),latency_prerequisite=statistics.median(sums)<=5))
result=dict(status='PASS',checks=len(checks),conditions=summary,parity=recalculated,meaning='Separate same-Codex arithmetic, not independent human semantic assessment',timing_note='Sum of separately timed decoding/normalization and inference. Not measured end-to-end feedback.',training_performed=False,private_access=False)
p=root/'VERIFICATION.json'
if p.exists():raise FileExistsError('Do not overwrite verification')
p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','checks','conditions']},ensure_ascii=False,indent=2))
