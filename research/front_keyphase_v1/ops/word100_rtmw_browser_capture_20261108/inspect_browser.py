from pathlib import Path
import json,urllib.request,websocket
u=Path(__file__).resolve().parent
p=max(u.glob('browser_*/profile/DevToolsActivePort'),key=lambda p:p.stat().st_mtime)
port=p.read_text().splitlines()[0]
tab=next(t for t in json.load(urllib.request.urlopen('http://127.0.0.1:'+port+'/json/list')) if t['type']=='page')
w=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=4)
w.send(json.dumps({'id':1,'method':'Runtime.evaluate','params':{'expression':'''JSON.stringify({url:location.href,ready:window.testReady,status:document.querySelector('#status')?.textContent,v:document.querySelector('video')?.readyState,run:window.activeCapture?.frames,body:document.body.innerText.slice(-1000)})''','returnByValue':True}}))
print(w.recv());w.close()
