"""Retain the initial execution evidence and update only a display-code pin."""
from pathlib import Path
import hashlib,json,shutil
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=U/'EXECUTION_FROZEN.json';backup=U/'EXECUTION_FROZEN_INITIAL.json'
    assert not backup.exists(), 'Amendment must be accepted only once'
    a=read(U/'UI_DISPLAY_AMENDMENT.json');f=read(p)
    assert sha(p)==a['execution_frozen']['sha256']
    old=a['previous_demo_js'];new=a['current_demo_js']
    assert sha(old['path'])==old['sha256'] and sha(new['path'])==new['sha256']
    assert f['pins'][new['path']]==old['sha256']
    for path,digest in f['pins'].items():
        assert sha(path)==(new['sha256'] if path==new['path'] else digest), path
    shutil.copy2(p,backup)
    f['pins'][new['path']]=new['sha256']
    f['display_only_amendment']=dict(initial_freeze=str(backup),initial_sha256=sha(backup),
        amendment=str(U/'UI_DISPLAY_AMENDMENT.json'),amendment_sha256=sha(U/'UI_DISPLAY_AMENDMENT.json'),
        new_inference=0,live_backend_process_unchanged=True,
        fresh_job_executed_under_initial_freeze='dff00320fc0b497db638b0408244db20')
    p.write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert all(sha(q)==h for q,h in read(p)['pins'].items())
    print('Accepted display-only amendment; preserved initial freeze; verified',len(f['pins']),'pins')
if __name__=='__main__':main()
