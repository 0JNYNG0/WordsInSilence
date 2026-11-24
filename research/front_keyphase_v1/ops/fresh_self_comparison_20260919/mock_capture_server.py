"""Isolated browser test surface: no real camera, neural inference, or human session writes."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import json
import mimetypes
import urllib.parse

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]/'web'
plan=json.loads((WEB/'frozen-self-plan.json').read_text(encoding='utf-8'))
plan['protocol_id']+='__SYNTHETIC_BROWSER_TEST'
events=[]
session={'session_id':'synthetic_capture_session','client_id':None,'camera':False,'state':'CLOSED'}
trials={}
request_ids={}
flags={'review_fail_once':False,'create_response_fail_once':False,'terminal':'FEEDBACK'}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,data):
        raw=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        route=urllib.parse.urlparse(self.path).path
        if route=='/__test__/events':
            return self.send(200,{'events':events,'session':session,'trials':trials,'flags':flags,
                                  'hardware_camera_calls':0,'human_observation_files_written':0})
        if route=='/frozen-self-plan.json':return self.send(200,plan)
        if route=='/api/v1/references':return self.send(200,{'set_id':plan['reference_set_id'],'items':plan['words']})
        if route.startswith('/api/v1/trials/'):
            tid=route.rsplit('/',1)[-1]
            if tid not in trials:return self.send(404,{'error':'NO_SUCH_TRIAL'})
            t=trials[tid]
            if t['state']=='RECORDING':t['state']=flags['terminal']
            events.append({'method':'GET','route':route,'state':t['state']})
            return self.send(200,{**t,'result':{'top1_word_id':'SENTINEL_PREDICTION_MUST_STAY_HIDDEN','ranking':[{'score':0.9999}],
                                            'result_id':'synthetic_result','origin':'camera'} if t['state']=='FEEDBACK' else None})
        if route=='/api/v1/sessions/'+session['session_id']:return self.send(200,session)
        if route.endswith('/preview.jpg'):
            # Empty 204 is sufficient for a mock; never open hardware for test images.
            self.send_response(204);self.end_headers();return
        if route.startswith('/api/v1/references/'):
            self.send_response(204);self.end_headers();return
        files={'/frozen-capture.html':'frozen-capture.html','/frozen-capture.js':'frozen-capture.js'}
        if route in files:
            p=WEB/files[route]
            raw=p.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(str(p))[0]+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw);return
        return self.send(404,{'error':'TEST_ROUTE_NOT_SUPPORTED'})
    def do_POST(self):
        route=urllib.parse.urlparse(self.path).path
        raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
        body=json.loads(raw) if raw else {}
        events.append({'method':'POST','route':route,'body':body})
        if route=='/__test__/control':
            assert set(body)<=set(flags)
            flags.update(body)
            return self.send(200,flags)
        if route=='/api/v1/sessions':
            session.update(client_id=body.get('client_id'),camera=True,state='OPEN')
            return self.send(200,{**session,'heartbeat_hint_s':5,'lease_ttl_s':20,'camera_error':None})
        if route.endswith('/heartbeat'):return self.send(200,{'state':session['state']})
        if route.endswith('/close'):
            session.update(camera=False,state='CLOSED')
            for t in trials.values():
                if t['state']=='RECORDING':t['state']='CANCELLED'
            return self.send(200,{**session,'camera_released':True})
        if route.endswith('/trials'):
            rid=body['request_id']
            dedup=rid in request_ids
            if not dedup:
                tid='synthetic_trial_%02d'%(len(trials)+1)
                request_ids[rid]=tid
                trials[tid]={'trial_id':tid,'state':'RECORDING','requested_word_id':body['requested_word_id'],'error':None}
            tid=request_ids[rid]
            if flags['create_response_fail_once']:
                flags['create_response_fail_once']=False
                return self.send(503,{'error':'SYNTHETIC_RESPONSE_LOST_AFTER_CREATE'})
            return self.send(200,{'trial_id':tid,'state':trials[tid]['state'],'deduplicated':dedup})
        if route.endswith('/review'):
            if flags['review_fail_once']:
                flags['review_fail_once']=False
                return self.send(503,{'error':'SYNTHETIC_REVIEW_WRITE_FAILURE'})
            return self.send(200,{'stored':True,'status':'STORED','revision':1})
        if route.endswith('/stop') or route.endswith('/cancel'):
            tid=route.split('/')[-2]
            trials[tid]['state']='CANCELLED' if route.endswith('/cancel') else 'FEEDBACK'
            return self.send(200,trials[tid])
        return self.send(404,{'error':'TEST_ROUTE_NOT_SUPPORTED'})

print('Synthetic browser test server http://127.0.0.1:8766; hardware camera and human records unavailable.',flush=True)
ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
