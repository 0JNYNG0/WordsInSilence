from pathlib import Path
import sys,json,time
import numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;P=G/'public_frontal_evidence_repair_20261117';N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(P))
from check_runtime import FeedbackConsumer,close,read,write,require_front
from stream_runtime import Engine
from selective_retry import retry,requests,VideoFrames
import torch,cv2

def main():
    tag=sys.argv[1]if len(sys.argv)>1 else 'probe'
    torch.set_num_threads(2);cv2.setNumThreads(2)
    net=FeedbackConsumer();engine=Engine();rows=[]
    try:
        for r in read(N/'INPUTS.json')['rows']:
            if r['person']not in [1,2]:continue
            require_front(r);z=dict(np.load(r['raw']));normal=net.raw(z,r['word'])
            guide=next(c for c in normal['cards']if c['id']==('right_shape'if r['word']=='WORD1290'else'index_shape'))
            a,b=guide['query_spans'][0];mid=(a+b)//2;lo=mid-2;hi=mid+2
            for kind in ['normal','missing_one_finger','missing_hand']:
                raw={k:v.copy()for k,v in z.items()}
                if kind!='normal':
                    joint=slice(125,129)if kind=='missing_one_finger'else slice(112,133)
                    raw['xy_px'][lo:hi+1,joint]=np.nan;raw['score'][lo:hi+1,joint]=0
                initial=net.raw(raw,r['word']);reader=VideoFrames(r['video'],raw)
                q,fixed=retry(raw,initial,reader,engine,net.raw,r['word'])
                statuses=lambda x:{c['id']:c['status']for c in x['lesson_acceptance']['criteria']}
                row=dict(id=r['id'],kind=kind,span=[lo,hi],before=statuses(initial),after=statuses(q),retry=q['selective_retry'])
                rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
                write(D/tag/f"{r['id']}_{kind}.json",q)
        write(D/(tag.upper()+'.json'),dict(rows=rows,devices=engine.devices,private_data=False,scope='FIT engineering, known public F clips with injected missing observations; image pixels unmodified'))
    finally:close(net)
if __name__=='__main__':main()
