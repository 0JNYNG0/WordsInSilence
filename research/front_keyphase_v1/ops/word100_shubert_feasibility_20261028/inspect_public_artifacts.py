"""Fetch official public metadata only. Never uploads local data or fetches weights."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import re
import requests

U = Path(__file__).resolve().parent
SOURCES = {
    'PUBLIC_FOLDER_PAGE.html': 'https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV?usp=sharing',
    'OFFICIAL_REPO_METADATA.json': 'https://api.github.com/repos/ShesterG/SHuBERT',
    'OFFICIAL_MAIN_COMMIT.json': 'https://api.github.com/repos/ShesterG/SHuBERT/commits/main',
}
out = []
for name, url in SOURCES.items():
    r = requests.get(url, timeout=40)
    (U/name).write_bytes(r.content)
    out.append({'name': name, 'url': url, 'status': r.status_code, 'bytes': len(r.content)})
    r.raise_for_status()
page = (U/'PUBLIC_FOLDER_PAGE.html').read_text(encoding='utf-8')
print(json.dumps({'files':out, 'title':re.findall(r'<title>(.*?)</title>',page)[:1],
    'drive_ivd':'_DRIVE_ivd' in page,
    'weight_names':sorted(set(re.findall(r'[A-Za-z0-9_.-]+\.(?:pt|pth|ckpt|pkl|zip)',page)))[:30]},ensure_ascii=False))
(U/'ACQUISITION_INSPECTION.json').write_text(json.dumps({'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
    'scope':'public metadata only','weight_downloads':0,'sources':out,
    'first_failure':'Default sandbox requests DNS failed for drive.google.com; same public read retried with network permission.'},ensure_ascii=False,indent=2),encoding='utf-8')
