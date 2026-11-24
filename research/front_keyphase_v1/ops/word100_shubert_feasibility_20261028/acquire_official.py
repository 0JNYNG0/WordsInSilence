"""Acquire the three approved public SHuBERT artifacts and pinned official code."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
import time
import zipfile
import requests

U=Path(__file__).resolve().parent
W=U/'weights'; W.mkdir(exist_ok=True)
S=U/'vendor'; S.mkdir(exist_ok=True)
log=U/'ACQUISITION_EVENTS.jsonl'
def event(**kw):
    obj={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),**kw}
    with log.open('a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fetch(url,dest,expected=None):
    if dest.exists():
        if expected is not None and dest.stat().st_size!=expected: raise RuntimeError('Existing file size mismatch')
        event(status='reuse',name=dest.name,bytes=dest.stat().st_size,sha256=sha(dest));return
    partial=dest.with_name(dest.name+'.part')
    if partial.exists(): raise RuntimeError('Partial file preserved; explicit verified resume required: '+str(partial))
    event(status='start',name=dest.name,url=url,expected_bytes=expected)
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        content_type=r.headers.get('Content-Type','')
        if 'text/html' in content_type:
            (U/(dest.name+'.response.html')).write_bytes(r.content)
            raise RuntimeError('Received HTML instead of artifact; preserved response')
        with partial.open('wb') as f:
            total=0; announced=0
            for chunk in r.iter_content(1024*1024):
                if not chunk:continue
                total+=len(chunk)
                if expected and total>expected:raise RuntimeError('Larger than official size')
                if total>2_000_000_000:raise RuntimeError('Bounded artifact size exceeded')
                f.write(chunk)
                if total-announced>128*1024*1024:
                    event(status='progress',name=dest.name,bytes=total);announced=total
    if expected and partial.stat().st_size!=expected:raise RuntimeError('Incomplete official artifact')
    digest=sha(partial)
    partial.rename(dest)
    event(status='complete',name=dest.name,bytes=dest.stat().st_size,sha256=digest)
def unpack(archive,label):
    out=S/label
    if out.exists():event(status='source_exists',name=label);return
    out.mkdir()
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>200_000_000:raise RuntimeError('Source archive too large')
        for member in z.infolist():
            relative=Path(*Path(member.filename).parts[1:])
            if not relative.parts:continue
            target=(out/relative).resolve()
            if not target.is_relative_to(out.resolve()):raise RuntimeError('Unsafe archive path')
            if member.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    event(status='source_unpacked',name=label)

if __name__=='__main__':
    try:
        auth=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf-8'))
        assert auth['candidate']=='SHuBERT'
        manifest=json.loads((U/'OFFICIAL_ARTIFACTS.json').read_text())
        required=sum(x['bytes'] for x in manifest['files'] if not (W/x['name']).exists())
        if shutil.disk_usage(U).free<required+3_000_000_000:raise RuntimeError('Insufficient free space reserve')
        commit=json.loads((U/'OFFICIAL_MAIN_COMMIT.json').read_text())['sha']
        code=S/'shubert_source.zip'
        fetch('https://codeload.github.com/ShesterG/SHuBERT/zip/'+commit,code)
        unpack(code,'shubert')
        # Read-only version metadata precedes the necessary DINO implementation source.
        r=requests.get('https://api.github.com/repos/facebookresearch/dinov2/commits/main',timeout=30);r.raise_for_status()
        (U/'DINO_MAIN_COMMIT.json').write_text(r.text,encoding='utf-8')
        dino_commit=r.json()['sha']
        code=S/'dinov2_source.zip'
        fetch('https://codeload.github.com/facebookresearch/dinov2/zip/'+dino_commit,code)
        unpack(code,'dinov2')
        for row in manifest['files']:
            fetch('https://drive.usercontent.google.com/download?id='+row['id']+'&export=download&confirm=t',W/row['name'],row['bytes'])
        event(status='all_acquired',weight_files=3,weight_bytes=manifest['total_bytes'])
    except Exception as exc:
        event(status='failed',error=repr(exc));raise
