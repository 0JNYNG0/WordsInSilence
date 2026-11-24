"""Fetch one public NIKL reference PDF, capped at25MiB; no archive acquisition."""
from pathlib import Path
import hashlib,json,urllib.request

H=Path(__file__).resolve().parent
P=H/'nikl_handshape_appendix2_2017.pdf'
URL='https://korean.go.kr/common/download.do?c_file_name=c4e6e38d-407f-46a6-b743-381e092c0a41_2.pdf&file_path=reportData&o_file_name=%EC%88%98%ED%98%95+%EA%B8%B0%EB%B0%98+%ED%95%9C%EA%B5%AD%EC%88%98%EC%96%B4%EC%82%AC%EC%A0%84+%EA%B5%AC%EC%B6%95-%EC%B5%9C%EC%A2%85%EB%B3%B4%EA%B3%A0%EC%84%9C-%EB%B6%80%EB%A1%9D+2.pdf'
assert not P.exists()
req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=35) as response:
    assert int(response.headers.get('Content-Length',0))<=25*1024*1024
    body=response.read(25*1024*1024+1)
assert len(body)<=25*1024*1024 and body.startswith(b'%PDF-')
with P.open('xb') as f:f.write(body)
record={'url':URL,'path':str(P),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),
    'purpose':'Read-only lexical/form cross-reference; not training data',
    'initial_sandbox_attempt':'WinError10013; body0bytes saved0files',
    'AIHub_archives_downloaded':0}
with (H/'DICTIONARY_SOURCE.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='url'},ensure_ascii=False))
