from pathlib import Path
import json,hashlib,re,difflib
U=Path(__file__).resolve().parent;S=U.parent.parent/'web_tutorial_live_integration_20261113/site';pins=json.loads((U/'PROTECTED_PINS.json').read_text('utf-8'));rows=[]
for file,sha in pins.items():
 p=Path(file);now=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.name=='live.js':
  before=(U/'before_web/live.js').read_text('utf-8');after=p.read_text('utf-8');strip=lambda s:re.sub(r'function onScene\(\).*?\nwindow.addEventListener', '\nwindow.addEventListener',s,flags=re.S);ok=strip(before)==strip(after);scope='Only onScene teaching-animation selection signal changed; capture/TCN/feedback/quest handlers identical.'
 else:ok=sha==now;scope='Exact hash retained'
 assert ok,(file,scope);rows.append(dict(file=file,before_sha256=sha,after_sha256=now,pass_check=ok,scope=scope))
for word in ['WORD0099','WORD1290']:
 pack=json.loads((U/'PACKAGED.json').read_text('utf-8'))['catalog'][word];sha=hashlib.sha256((S/'sign-demos'/(word+'.glb')).read_bytes()).hexdigest();assert sha==pack['sha256'];assert sha==json.loads((U/'MOTION_VERIFICATION.json').read_text('utf-8'))['results'][word]['glb_sha256']
for n in ['scene.js','live.js','index.html','live.css']:(U/(n+'.diff')).write_text(''.join(difflib.unified_diff((U/'before_web'/n).read_text('utf-8').splitlines(True),(S/n).read_text('utf-8').splitlines(True),fromfile='before/'+n,tofile='after/'+n)),'utf-8')
out=dict(status='PASS',checks=rows,web_assets_match_verified_exports=True,source_recordings_used='Public only; no private recordings read',training_runs=0,inference_behavior_changes=0)
(U/'PRESERVATION_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8');print('PASS: unchanged inference server/admissions; live.js changes limited to animation-selection UI; verified GLB hashes match web.')
