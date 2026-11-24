from common_split import *
import subprocess,re
ids=set(r['clip_id']for r in read(U/'PRIOR_LOSS_AUDIT.json')['rows'])
ids|={f'NIA_SL_{w}_REAL{p:02d}_F'for w in ['WORD2700','WORD2718','WORD1290','WORD2748']for p in [2,5,7,10,16]}
r=subprocess.run(['rg','--files',str(O),'-g','*.mp4'],capture_output=True,text=True,encoding='utf-8');paths={}
for line in r.stdout.splitlines():
 m=re.search(r'NIA_SL_WORD\d+_REAL\d+_F',line)
 if m and m[0]in ids:paths.setdefault(m[0],[]).append(line)
found={}
for cid in sorted(ids):
 rec=J/'records'/f'{cid}.json';expected=read(rec)['source_sha256']if rec.exists()else None
 matches=[p for p in paths.get(cid,[])if expected and sha(p)==expected]
 if matches:found[cid]=dict(path=matches[0],sha256=expected,record=str(rec))
write(U/'SOURCE_CATALOG.json',dict(found=found,missing=sorted(ids-found.keys()),search_warning=r.stderr.strip(),note='Matched original source hashes; reencoded/reversed derivatives never substituted.'))
print('found',len(found),'missing',len(ids-found.keys()))
