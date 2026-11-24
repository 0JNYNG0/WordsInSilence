"""Acquire only user-approved official SignRep assets; no downloaded code execution."""
from pathlib import Path, PurePosixPath
import urllib.request
import json, hashlib, zipfile, shutil, traceback
from datetime import datetime, timezone

U=Path(__file__).resolve().parent
EXPECTED='f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(name,obj):
    (U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'WordsInSilence-public-research'}),timeout=90)
def download(url,path,expected=None):
    if path.exists():
        if expected: assert sha(path)==expected
        return
    if path.with_suffix(path.suffix+'.part').exists():raise RuntimeError('Partial download preserved; inspect before retry')
    tmp=path.with_suffix(path.suffix+'.part')
    with request(url) as response, tmp.open('xb') as out:
        total=0
        while True:
            b=response.read(1048576)
            if not b:break
            total+=len(b);out.write(b)
            if total%(32*1048576)==0:print(json.dumps({'asset':path.name,'MB':round(total/1048576)}),flush=True)
        final_url=response.url
    digest=sha(tmp)
    if expected:assert digest==expected,(digest,expected)
    tmp.rename(path)
    with (U/'ACQUISITION_EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'url':url,'final_url':final_url,'file':str(path),'bytes':total,'sha256':digest})+'\n')
def main():
    (U/'assets').mkdir(exist_ok=True)
    meta=U/'OFFICIAL_COMMIT.json'
    if not meta.exists():
        with request('https://api.github.com/repos/ryanwongsa/SignRep/commits/main') as r:write(meta.name,json.load(r))
    commit=json.loads(meta.read_text())['sha']
    archive=U/'assets'/f'SignRep-{commit}.zip'
    download(f'https://codeload.github.com/ryanwongsa/SignRep/zip/{commit}',archive)
    vendor=U/'vendor'/'SignRep'; vendor.mkdir(parents=True,exist_ok=True)
    files=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts=PurePosixPath(info.filename).parts
            if info.is_dir():continue
            assert len(parts)>1 and '..' not in parts and info.file_size<10_000_000
            rel=Path(*parts[1:]);target=vendor/rel
            assert target.resolve().is_relative_to(vendor.resolve())
            data=z.read(info)
            if target.exists():assert target.read_bytes()==data
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            files.append({'file':str(rel),'bytes':len(data),'sha256':sha(target)})
    checkpoint=U/'assets'/'ckpt.pt'
    download('https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt',checkpoint,EXPECTED)
    write('ACQUISITION.json',{'status':'PASS','commit':commit,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size,'vendor_files':files,'downloaded_code_executed':False,'disk_free_after':shutil.disk_usage(U).free})
    print(json.dumps({'status':'ACQUIRED_VERIFIED','commit':commit,'bytes':checkpoint.stat().st_size}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        name='ACQUISITION_FIRST_FAILURE.json'
        if not (U/name).exists():write(name,{'at':datetime.now(timezone.utc).isoformat(),'error':repr(e),'traceback':traceback.format_exc()})
        raise
