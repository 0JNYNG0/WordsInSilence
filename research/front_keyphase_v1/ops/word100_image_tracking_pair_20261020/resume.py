"""Storage-only recovery: preserve first failure and22 complete results.
Nonfinite diagnostic floats -> JSON null; tracking formulas/parameters unchanged.
"""
import json,math,inspect,os,time
from pathlib import Path
import run as R
U=R.U
def safe(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    return v
def write(path,v):
    Path(path).write_text(json.dumps(safe(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

if __name__=='__main__':
    assert (U/'FIRST_ERROR.json').exists()
    rec=U/'STORAGE_RECOVERY.json'
    if not rec.exists():
        old={}
        for p in (U/'records').glob('*.json'):
            x=R.read(p);assert len(x['frames'])>0
            old[str(p)]=R.sha(p)
        assert len(old)==22
        write(rec,dict(at=R.now(),reason='Nonfinite diagnostic optical-flow values cannot serialize as strict JSON; map to null only.',
             exit_observed=1,completed_preserved=old,first_error_sha256=R.sha(U/'FIRST_ERROR.json'),
             original_run_sha256=R.sha(U/'run.py'),recovery_sha256=R.sha(__file__),
             algorithm_changed=False,parameters_changed=False))
        os.rename(U/'RUNNING.json',U/'FAILED_RUNNING.json')
    recovery=R.read(rec);R.check(recovery['completed_preserved'])
    R.write=write
    source=inspect.getsource(R.run)
    needle="        cid=r['clip_id'];z=load(r['views']['sensor']['raw']);obs=load(r['views']['sensor']['observation'])"
    replacement="""        cid=r['clip_id']
        saved=U/'records'/f'{cid}.json'
        if str(saved) in recovery['completed_preserved']:
            assert sha(saved)==recovery['completed_preserved'][str(saved)]
            x=read(saved);assert x['clip_id']==cid
            results.append(dict(clip_id=cid,label=r['label'],frames=len(x['frames']),blocks=x['blocks'],
                                path=str(saved),sha256=sha(saved),elapsed_tracker_s=x['elapsed_tracker_s']))
            counts['frames']+=len(x['frames'])
            print('REUSE',ri+1,cid,flush=True)
            continue
        z=load(r['views']['sensor']['raw']);obs=load(r['views']['sensor']['observation'])"""
    assert source.count(needle)==1
    source=source.replace(needle,replacement)
    R.__dict__['recovery']=recovery
    R.cv2.setNumThreads(1)
    exec(compile(source,str(U/'run.py')+':storage_resume','exec'),R.__dict__)
    try:R.run()
    except BaseException:
        import traceback
        write(U/'RECOVERY_ERROR.json',dict(at=R.now(),error=traceback.format_exc()))
        raise
