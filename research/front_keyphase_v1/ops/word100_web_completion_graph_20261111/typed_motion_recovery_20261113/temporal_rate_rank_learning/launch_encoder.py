"""One-shot import-path correction for the unchanged, pinned preparation code."""
from pathlib import Path
import hashlib,json,os,runpy,sys,time,traceback
import psutil
K=Path(__file__).resolve().parent;J=K.parent/'splice_rank_learning'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    correction=json.loads((K/'LAUNCH_CORRECTION.json').read_text(encoding='utf-8'))
    for p,h in correction['pins'].items():assert sha(p)==h,p
    for name in ['canonical','bags','PREPARATION_INPUTS_COMPLETE.json','LAUNCH_STARTED.json']:
        assert not(K/name).exists(),f'Existing output/launch requires review: {name}'
    matches=[];unreadable=[]
    for process in psutil.process_iter(['pid','name']):
        if process.pid==os.getpid()or(process.info['name']or'').lower()not in ['python.exe','pythonw.exe']:continue
        try:args=process.cmdline()
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied:unreadable.append(process.pid);continue
        if any('temporal_rate_rank_learning'in arg and ('prepare_encoder.py'in arg or'launch_encoder.py'in arg)for arg in args):matches.append(process.pid)
    assert not matches and not unreadable,dict(matches=matches,unreadable_python=unreadable)
    # Exclusive creation is a second guard against simultaneous launches.
    with (K/'LAUNCH_STARTED.json').open('x',encoding='utf-8')as f:
        json.dump(dict(pid=os.getpid(),at=time.time(),active_matching_jobs_before=[],original_code_sha256=sha(K/'prepare_encoder.py'),
            correction_sha256=sha(K/'LAUNCH_CORRECTION.json'),command='runpy unchanged prepare_encoder.py run-reviewed'),f,indent=2)
    sys.dont_write_bytecode=True
    # Keep K local imports and provide J's sibling splice_core dependency.
    sys.path[:0]=[str(K),str(J)]
    sys.argv=[str(K/'prepare_encoder.py'),'run-reviewed']
    try:
        runpy.run_path(str(K/'prepare_encoder.py'),run_name='__main__')
        for p,h in correction['pins'].items():assert sha(p)==h,p
        assert(K/'PREPARATION_INPUTS_COMPLETE.json').exists()
        write(K/'LAUNCH_FINISHED.json',dict(pid=os.getpid(),at=time.time(),status='COMPLETED',
            input_manifest_sha256=sha(K/'PREPARATION_INPUTS_COMPLETE.json'),original_code_unchanged=True,no_training=True))
    except BaseException:
        write(K/'LAUNCH_FINISHED.json',dict(pid=os.getpid(),at=time.time(),status='FAILED_REQUIRES_REVIEW',
            original_code_sha256=sha(K/'prepare_encoder.py'),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise
if __name__=='__main__':main()
