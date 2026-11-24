"""Temporary browser UI fixture on8876. No camera imports, no model, no human sessions."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import copy
import json
import mimetypes
from urllib.parse import urlsplit

U=Path(__file__).resolve().parent;BASE=U.parents[1];WEB=BASE/'web'
cfg=json.loads((BASE/'config/word100_practice.json').read_text('utf-8'))
result=json.loads((U/'MOCK_RESULT.json').read_text('utf-8'))
events=[];request={};closed=False

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def send(self,data,kind='application/json',code=200):
        if not isinstance(data,bytes): data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/v1/practice100/catalog':
            return self.send(dict(model=dict(model_status='READY',model_key=cfg['model_key'],checkpoint_sha256=cfg['checkpoint_sha256']),items=cfg['items'],practice_count=12))
        if '/practice100/references/' in path:
            word=path.split('/')[-2];x=next(x for x in cfg['items'] if x['word_id']==word)
            return self.send(Path(x['video_path']).read_bytes(),'video/mp4')
        if path.endswith('/preview.jpg'):
            svg='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#eaf0ed"/><text x="320" y="240" text-anchor="middle" font-size="24" fill="#203d45">UI TEST - NO CAMERA</text></svg>'
            return self.send(svg.encode(),'image/svg+xml')
        if path.startswith('/api/v1/trials/'):
            return self.send(dict(state='FEEDBACK',result=current(),error=None))
        if path=='/api/v1/fixture-events': return self.send(events)
        name=path.lstrip('/') or 'practice.html';p=(WEB/name).resolve()
        if p.parent!=WEB.resolve() or not p.is_file():return self.send({},code=404)
        data=p.read_bytes()
        if name=='practice.html':data=data.replace(b'<title>',b'<title>QA - ')
        return self.send(data,mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
    def do_POST(self):
        global request,closed
        path=urlsplit(self.path).path;n=int(self.headers.get('Content-Length',0));body=json.loads(self.rfile.read(n) or b'{}')
        events.append(dict(path=path,body=body));(U/'UI_EVENTS.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),'utf-8')
        if path=='/api/v1/sessions':return self.send(dict(session_id='mock_session',camera=True,heartbeat_hint_s=5))
        if path.endswith('/trials'):
            request=body;return self.send(dict(trial_id='mock_trial',state='RECORDING'))
        if path.endswith('/stop'):return self.send(dict(state='FEEDBACK',result=current(),error=None))
        if path.endswith('/review'):return self.send(dict(status='STORED',stored=True))
        return self.send(dict(status='OK'))

def current():
    r=copy.deepcopy(result);r.update(session_id='mock_session',trial_id='mock_trial',result_id='mock_result',
        requested_word_id=request['requested_word_id'],word_label_match=request['requested_word_id']=='WORD0003')
    r['diagnostic_capture']={'status':'DISABLED'}
    return r

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8876),Handler).serve_forever()
