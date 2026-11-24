from pathlib import Path
import subprocess,time,json,urllib.request,websocket,psutil,uuid
U=Path(__file__).resolve().parent;p=U/('codec_probe_'+uuid.uuid4().hex[:6]);p.mkdir()
proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(p),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
w=None
try:
 for _ in range(100):
  if (p/'DevToolsActivePort').exists():break
  time.sleep(.1)
 port=(p/'DevToolsActivePort').read_text().splitlines()[0];tab=next(t for t in json.load(urllib.request.urlopen('http://127.0.0.1:'+port+'/json/list')) if t['type']=='page')
 w=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=20)
 # about:blank lacks secure-context codecs. A local file is a secure context.
 w.send(json.dumps({'id':1,'method':'Page.navigate','params':{'url':(U/'test.html').as_uri()}}))
 while json.loads(w.recv()).get('id')!=1:pass
 time.sleep(.5)
 code="""(async()=>{let out=[];for(let codec of ['vp8','vp09.00.10.08','avc1.42002a','avc1.64002a'])for(let hardwareAcceleration of ['prefer-hardware','no-preference']){let config={codec,width:1920,height:1080,framerate:30,bitrate:8000000,latencyMode:'realtime',hardwareAcceleration};if(codec.startsWith('avc'))config.avc={format:'annexb'};try{out.push({config,supported:(await VideoEncoder.isConfigSupported(config)).supported})}catch(e){out.push({config,error:String(e)})}}return out})()"""
 w.send(json.dumps({'id':2,'method':'Runtime.evaluate','params':{'expression':code,'awaitPromise':True,'returnByValue':True}}))
 while True:
  r=json.loads(w.recv())
  if r.get('id')==2:break
 (U/'CODEC_SUPPORT.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
finally:
 if w:w.close()
 for child in psutil.Process(proc.pid).children(recursive=True):
  try:child.terminate()
  except psutil.NoSuchProcess:pass
 proc.terminate();proc.wait(timeout=10)
