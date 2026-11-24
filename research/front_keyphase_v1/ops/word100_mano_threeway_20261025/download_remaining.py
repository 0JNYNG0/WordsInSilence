"""Resume the official archive using verified disjoint HTTP ranges."""
from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time

ROOT = Path(__file__).resolve().parent
TOTAL = 6037554929
URL = 'https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz'
ETAG = '"167ddc6f1-60c69e4dd5f93"'
prefix = ROOT / 'downloads/hamer_demo_data.tar.gz.part'
size = prefix.stat().st_size
assert 0 < size < TOTAL
out = ROOT / 'downloads/hamer_demo_data.tar.gz'
assert not out.exists()
jobs = []
for i in range(4):
    start = size + (TOTAL-size)*i//4
    end = size + (TOTAL-size)*(i+1)//4 - 1
    jobs.append((i,start,end))
manifest = {'url': URL, 'etag': ETAG, 'expected_bytes': TOTAL,
            'prefix_bytes': size, 'prefix_sha256': hashlib.file_digest(prefix.open('rb'),'sha256').hexdigest(),
            'reason': 'Initial curl session 7729 deliberately interrupted after clean write; reuse all bytes and resume four disjoint ranges.',
            'ranges': jobs, 'status': 'running'}
(ROOT/'DOWNLOAD_RANGES.json').write_text(json.dumps(manifest,indent=2))

def fetch(job):
    i,start,end=job
    part=ROOT/f'downloads/range_{i}.part'
    assert not part.exists()
    headers=ROOT/f'logs/range_{i}.headers'
    with (ROOT/f'logs/range_{i}.txt').open('w') as log:
        subprocess.run(['curl.exe','--fail','--location','--retry','2','--connect-timeout','30',
                        '--header','If-Range: '+ETAG,'--range',f'{start}-{end}',
                        '--dump-header',str(headers),'--output',str(part),URL],
                       stdout=log,stderr=log,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    assert part.stat().st_size==end-start+1, (i,part.stat().st_size)
    header=headers.read_text().lower()
    assert f'content-range: bytes {start}-{end}/{TOTAL}' in header, header
    print('range complete',i,flush=True)
    return part

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(fetch,jobs))
    assert prefix.stat().st_size==size
    h=hashlib.sha256()
    with out.open('xb') as dest:
        for part in [prefix]+parts:
            with part.open('rb') as src:
                while block:=src.read(8*1024*1024):
                    dest.write(block);h.update(block)
    assert out.stat().st_size==TOTAL
    manifest.update(status='complete',sha256=h.hexdigest(),output=str(out))
except Exception as exc:
    manifest.update(status='failed',error=repr(exc))
    raise
finally:
    (ROOT/'DOWNLOAD_RANGES.json').write_text(json.dumps(manifest,indent=2))
