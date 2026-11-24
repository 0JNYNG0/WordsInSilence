"""Read public source caches only; no camera, training or production mutation."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OPS=HERE.parent
BASIC=OPS/'basic_demo_feedback_20261115'
MOTION=BASIC/'acquisition_motion_followup'
sys.path[:0]=[str(MOTION),str(BASIC),str(OPS/'word100_rtmw_shared_input_20261108')]
from motion_model import Observer, feature
from check_motion_bridge import bridge
from basic_runtime import RelationObserver
from core import Features
import torch,adapter

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():
    rows=read(BASIC/'source_corrected/CONFIRMATION_INPUTS.json')['rows']
    rows=[dict(r,raw_path=r['raw'],channel_path=r['channels']) for r in rows]
    rows+=read(BASIC/'local_shape/fresh/EXTRACTED.json')['rows']
    rows+=[r for r in read(MOTION/'confirmation/EXTRACTED.json')['rows'] if r['word']=='WORD1290']
    return rows

def main():
    torch.set_num_threads(2)
    ff=Features(); base=RelationObserver(); ob=Observer('mlp32'); out=[]
    for r in sources():
        p=r['person'];ch=dict(np.load(r['channel_path']))
        end={14:54,15:58,16:51,17:90,18:60}[p]
        for variant in ['normal','first_only']:
            c={k:(v[:end+1].copy() if v.ndim and len(v)==len(ch['frame_id']) else v.copy()) for k,v in ch.items()} if variant=='first_only' else ch
            f,q,_=ff(c);s,old=base(f,q,c['timestamp_us']);m=ob.sequence(c);events=bridge(s,m)
            # Explicitly record only actually existing windows: no clipped/padded future.
            windows=[dict(a=a,b=b,label=ob.predict(feature(c,a,b))) for b in range(max(1,end-10),min(len(c['frame_id']),end+4)) for a in [b-6,b-4,b-3] if a>=0]
            out.append(dict(person=p,variant=variant,n=len(s),events=events,static=s.tolist(),motion=m.tolist(),terminal_windows=windows))
    write(HERE/'TERMINAL_DIAGNOSIS.json',dict(rows=out,whole_word_correct=None,training_runs=0))
    for r in out:
        e={14:54,15:58,16:51,17:90,18:60}[r['person']]
        print(r['person'],r['variant'],'count',len(r['events']),'endstates',list(enumerate(zip(r['static'],r['motion'])))[e-10:e+4])
if __name__=='__main__':main()
