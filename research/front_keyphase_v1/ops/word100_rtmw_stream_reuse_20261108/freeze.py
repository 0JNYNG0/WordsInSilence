from pathlib import Path
import json, hashlib, sys
from datetime import datetime, timezone
U=Path(__file__).resolve().parent
B=U.parent/'word100_rtmw_mediapipe_compare_20261108'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def main():
    old=json.loads((B/'FROZEN.json').read_text(encoding='utf-8'))
    ids=['NIA_SL_WORD0739_REAL02_F','NIA_SL_WORD1133_REAL02_F','NIA_SL_WORD1567_REAL02_F',
         'NIA_SL_WORD0003_REAL02_F','NIA_SL_WORD0185_REAL05_F','NIA_SL_WORD0004_REAL02_F']
    rows=[]
    for cid in ids:
        r=dict(next(r for r in old['rows'] if r['clip_id']==cid))
        assert sha(r['source'])==r['source_sha256']
        p=B/'runs'/cid/'rtmw_gpu/rtmw.npz'
        r.update(saved_baseline=str(p),saved_baseline_sha256=sha(p))
        rows.append(r)
    pins=dict(old['protected'])
    for p,h in old['models'].items():pins[str(B/p)]=h
    for p in [B/'FROZEN.json',B/'SPEED_SUMMARY.json',B/'RESULT_KO.txt',U/'PLAN_KO.txt']:
        pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    obj=dict(created=datetime.now(timezone.utc).isoformat(),rows=rows,protected=pins,
             runtime_python=str(U.parent/'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'),
             device='gpu',queue_capacity=8,lag_p95_budget_ms=250,post_capture_budget_ms=1000,
             repetitions=2,training=0,source_role='public_development_replay',
             scope='paced full-resolution BGR input; predecoded before timer; no browser/camera/render overhead',
             authority='User requested immediate streaming capture/reuse work after concise overall plan')
    with (U/'FROZEN.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    print('Frozen',len(rows),'public clips')
if __name__=='__main__':main()
