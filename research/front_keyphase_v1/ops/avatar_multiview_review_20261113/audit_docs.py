from pathlib import Path
import json,re,hashlib
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
BASE=Path('W:/WordsInSilence/data/AIHub_103/docs')
result=[]
terms=['카메라','촬영','동시','동기','보정','캘리','calib','intrinsic','extrinsic','렌즈','아바타','3D','3차원','좌표','keypoint','형식']
for p in BASE.glob('*.pdf'):
    r=PdfReader(p); pages=[]
    for i,page in enumerate(r.pages):
        t=page.extract_text() or ''
        pages.append({'page':i+1,'text':t})
    (HERE/f'{p.stem}_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    matches=[]
    for rec in pages:
        ls=rec['text'].splitlines()
        hits=[j for j,line in enumerate(ls) if any(w.lower() in line.lower() for w in terms)]
        if hits: matches.append({'page':rec['page'],'matches':['\n'.join(ls[max(0,j-1):min(len(ls),j+2)]) for j in hits]})
    result.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':len(pages),'matches':matches})
(HERE/'DOCUMENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
