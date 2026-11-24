"""Copy the single already-owned WORD archive to W: and independently verify it."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

UNIT = Path(__file__).resolve().parent
STATE = UNIT/'LOCAL_COPY_PROGRESS.json'

def report(**values):
    values.update(at_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(), pid=os.getpid())
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,STATE)

def main():
    if (UNIT/'LOCAL_COPY_RESULT.json').exists():
        raise RuntimeError('Completed copy exists; reuse its result')
    plan=json.loads((UNIT/'ACQUISITION_PLAN.json').read_text(encoding='utf-8'))
    row=next(r for r in plan['archives'] if r['action']=='copy_existing_no_download')
    src=Path(row['existing_source']); dst=Path(row['destination'])
    dst.parent.mkdir(parents=True,exist_ok=True)
    part=dst.with_name(dst.name+'.copying')
    if dst.exists() or part.exists():
        raise RuntimeError('Destination already exists; preserve it and inspect before recovery')
    assert src.stat().st_size==row['fileSize']
    source_stat=src.stat()
    h=hashlib.sha256(); count=0; last=0
    with src.open('rb') as a, part.open('xb') as b:
        while chunk:=a.read(8*1024*1024):
            b.write(chunk); h.update(chunk); count+=len(chunk)
            if time.monotonic()-last>5:
                report(status='COPYING_EXISTING_NOT_NETWORK',bytes=count,total=row['fileSize'])
                last=time.monotonic()
        b.flush(); os.fsync(b.fileno())
    assert count==row['fileSize'] and src.stat().st_mtime_ns==source_stat.st_mtime_ns
    check=hashlib.sha256(); count=0; last=0
    with part.open('rb') as f:
        while chunk:=f.read(8*1024*1024):
            check.update(chunk); count+=len(chunk)
            if time.monotonic()-last>5:
                report(status='VERIFY_COPY_SHA256',bytes=count,total=row['fileSize'])
                last=time.monotonic()
    assert check.digest()==h.digest()
    report(status='VERIFY_ZIP_CRC',bytes=count,total=row['fileSize'])
    with zipfile.ZipFile(part) as z:
        assert z.testzip() is None
        members=sum(not r.is_dir() for r in z.infolist())
    part.rename(dst)
    result={'status':'PASS','source':str(src),'destination':str(dst),'bytes':row['fileSize'],
      'sha256':h.hexdigest(),'copied_hash_matches':True,'all_zip_member_crc_pass':True,
      'members':members,'new_download_bytes':0,'source_preserved':True}
    with (UNIT/'LOCAL_COPY_RESULT.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    report(**result)

if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        report(status='STOPPED_PRESERVE_PARTIAL',error_type=type(e).__name__,detail=str(e))
        raise
