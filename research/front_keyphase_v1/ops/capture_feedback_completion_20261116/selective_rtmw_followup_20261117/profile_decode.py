from pathlib import Path
import json,time,hashlib
import av
import numpy as np
D=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
records=read(D.parent/'frontal_scope_recovery_20261117/INPUTS.json')['rows']
rows=[]
for scene in read(D/'DIAGNOSIS.json')['rows']:
    r=next(r for r in records if r['id']==scene['id']);z=dict(np.load(r['raw']))
    a,b=scene['span'];wanted={int(z['timestamp_us'][i]):i for i in range(a-1,b+2)}
    for mode in ['default','bounded_parallel']:
        start=time.perf_counter();hashes={};decoded=0
        with av.open(scene['source']) as c:
            s=c.streams.video[0]
            if mode=='bounded_parallel':s.thread_type='AUTO';s.codec_context.thread_count=2
            c.seek(int(min(wanted)/1e6/float(s.time_base)),stream=s,backward=True)
            for f in c.decode(s):
                decoded+=1
                pts=round(float(f.pts*f.time_base)*1e6)
                hits=[k for k in wanted if abs(k-pts)<=2]
                if hits:hashes[wanted[hits[0]]]=hashlib.sha256(f.to_ndarray(format='bgr24').tobytes()).hexdigest()
                if len(hashes)==len(wanted):break
        row=dict(id=r['id'],mode=mode,seconds=time.perf_counter()-start,decoded=decoded,hashes=hashes)
        rows.append(row);print(row['id'],mode,round(row['seconds'],3),flush=True)
    assert rows[-1]['hashes']==rows[-2]['hashes']
(D/'DECODE_PROFILE.json').write_text(json.dumps(dict(rows=rows,pixels_equal=True,one_run_per_condition=True),indent=2),encoding='utf-8')
