"""User-authorized official model download; no training or local media upload."""
from pathlib import Path
import urllib.request, json, hashlib, zipfile, time, traceback
U=Path(__file__).resolve().parent
URLS={
 'rtmw_l_384':'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',
 'person_yolox_m':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
}
def fetch(url,p):
 if p.exists(): return
 part=p.with_suffix(p.suffix+'.part')
 if part.exists(): raise RuntimeError('Inspect preserved partial before retry: '+str(part))
 req=urllib.request.Request(url,headers={'User-Agent':'WordsInSilence-public-model-comparison/1.0'})
 with urllib.request.urlopen(req,timeout=90) as r,part.open('xb') as f:
  expected=r.headers.get('Content-Length');n=0;t=time.monotonic()
  while True:
   b=r.read(1024*1024)
   if not b:break
   f.write(b);n+=len(b)
   if time.monotonic()-t>15:print(p.name,n,flush=True);t=time.monotonic()
  if expected:assert n==int(expected)
 part.rename(p)
def extract(p,dest):
 dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(p) as z:
  for i in z.infolist():
   target=(dest/i.filename).resolve()
   assert target.is_relative_to(dest.resolve())
  z.extractall(dest)
def main():
 m=U/'models';m.mkdir(exist_ok=True);records=[]
 for name,url in URLS.items():
  p=m/(name+'.zip');fetch(url,p);extract(p,m/name)
  records.append(dict(name=name,url=url,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  print(name,'downloaded',p.stat().st_size,flush=True)
 info=json.load(urllib.request.urlopen('https://pypi.org/pypi/rtmlib/json',timeout=30))
 wheel=next(x for x in info['urls'] if x['filename'].endswith('.whl'))
 p=U/wheel['filename'];fetch(wheel['url'],p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==wheel['digests']['sha256']
 extract(p,U/'vendor')
 records.append(dict(name='rtmlib',version=info['info']['version'],url=wheel['url'],sha256=wheel['digests']['sha256']))
 (U/'DOWNLOADS.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/('DOWNLOAD_FAILURE_'+str(time.time_ns())+'.txt');p.write_text(traceback.format_exc(),encoding='utf-8');raise
