"""Actual HTTP retention/security check, with public video and a fake camera only."""
from pathlib import Path
import json, threading, urllib.request, urllib.error, hashlib
import serve as app

D=Path(__file__).resolve().parent
s=app.server
public=D/'public_capture_artifacts/WORD0099_normal'
assert (public/'recorded.mp4').is_file()

class PublicService:
    state='done';directory=public
    def open_camera(self):
        self.state='preview';self.directory=None

def main():
    s.SERVICE=PublicService();s.OWNER=None;s.RESULT=None
    app.RECORDINGS.clear()
    http=s.LocalServer(('127.0.0.1',0),app.Handler)
    s.PORT=http.server_address[1]
    threading.Thread(target=http.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{s.PORT}';checks=[]
    def request(path,data=None,headers=None):
        h={'X-Session-Token':s.TOKEN,**(headers or {})}
        if data is not None:h['Content-Type']='application/json'
        req=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=h)
        try:
            with urllib.request.urlopen(req,timeout=10) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)));assert passed,name
    key='a'*24;url=f'/session-recording/{key}.mp4'
    try:
        s.ATTEMPT={'id':key}
        check('no token denied',request(url)[0]==403)
        check('wrong token denied',request(url+'?token=wrong')[0]==403)
        status,body=request(url+'?token='+s.TOKEN)
        check('finished public video identical',status==200 and hashlib.sha256(body).digest()==hashlib.sha256((public/'recorded.mp4').read_bytes()).digest())
        check('cross host denied',request(url+'?token='+s.TOKEN,headers={'Host':'example.invalid'})[0]==403)
        check('nested route denied',request('/session-recording/nested/'+key+'.mp4?token='+s.TOKEN)[0]==404)
        check('wrong suffix denied',request('/session-recording/'+key+'.json?token='+s.TOKEN)[0]==404)
        check('unknown session denied',request('/session-recording/'+'f'*24+'.mp4?token='+s.TOKEN)[0]==404)
        # Exercise the real /api/open handler; PublicService cannot access a device.
        code,_=request('/api/open',{'client':'public-http-check-0001'})
        check('preview reopens and clears current attempt',code==200 and s.ATTEMPT is None and s.SERVICE.state=='preview')
        check('recording remains after preview reopen',request(url+'?token='+s.TOKEN)[0]==200)
        code,part=request(url+'?token='+s.TOKEN,headers={'Range':'bytes=0-31'})
        check('seek range preserved',code==206 and part==(public/'recorded.mp4').read_bytes()[:32])
        s.ATTEMPT={'id':'b'*24};s.SERVICE.directory=public
        app.remember()
        check('unfinished session not retained','b'*24 not in app.RECORDINGS)
        for digit in '12345':
            s.SERVICE.state='done';s.ATTEMPT={'id':digit*24};app.remember()
        check('bounded last four recordings',len(app.RECORDINGS)==4 and key not in app.RECORDINGS and '1'*24 not in app.RECORDINGS)
        check('expired session denied',request(url+'?token='+s.TOKEN)[0]==404)
        out=dict(passed=True,checks=checks,physical_camera=False,private_data_read=False,fixture=str(public),scope='Actual HTTP routing with fake camera service; public recording only')
        (D/'web/HTTP_RECORDING_CHECK.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        print('HTTP RECORDING PASS',len(checks))
    finally:http.shutdown();http.server_close()

if __name__=='__main__':main()
