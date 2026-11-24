"""Public stream through real loopback HTTP. No browser camera is opened."""
import json,time,threading,http.client,traceback
from pathlib import Path
import av,cv2,numpy as np
from stream_runtime import Engine,U
from stream_http import Application,make_server
from freeze import sha
def request(c,token,route,body,headers=None):
    h={'X-Session-Token':token,'Content-Type':'application/octet-stream'};h.update(headers or {})
    c.request('POST',route,body=body,headers=h);r=c.getresponse();data=r.read();return r.status,json.loads(data)
def main():
    scope=U/'HTTP_SCOPE.json'
    if (U/'HTTP_COMPLETE.json').exists():raise RuntimeError('Do not repeat a completed HTTP run')
    if not scope.exists():
        scope.write_text(json.dumps(dict(clips=['NIA_SL_WORD0739_REAL02_F','NIA_SL_WORD1567_REAL02_F','NIA_SL_WORD0003_REAL02_F'],
            repetitions=1,pixel_format='full resolution RGBA bytes',max_post_ms=1000,
            scope='HTTP transfer and server color conversion included; source RGBA frames predecoded; browser ImageData/readback and game rendering excluded',
            deliberate_control_tests=['wrong token','wrong session','nonmonotone frame','invalid size','duplicate finish']),indent=2),encoding='utf-8')
    cfg=json.loads(scope.read_text(encoding='utf-8'));plan=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
    for p,h in plan['protected'].items():assert sha(p)==h,p
    e=Engine();app=Application(e,U/'http_runs');server=make_server(app);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    results=[];controls=[]
    try:
        for cid in cfg['clips']:
            row=next(r for r in plan['rows'] if r['clip_id']==cid);assert sha(row['source'])==row['source_sha256']
            ts=[];rgba=[]
            with av.open(row['source']) as container:
                st=container.streams.video[0];base=None
                for f in container.decode(st):
                    if base is None:base=f.pts
                    ts.append(round(float((f.pts-base)*st.time_base)*1e6))
                    # Same decoded BGR pixels as baseline; rearrange to browser-style RGBA.
                    rgba.append(cv2.cvtColor(f.to_ndarray(format='bgr24'),cv2.COLOR_BGR2RGBA).tobytes())
            with np.load(row['saved_baseline']) as z:
                width,height=int(z['source_width']),int(z['source_height']);old={k:z[k] for k in z.files}
            c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=15)
            status,start=request(c,app.token,'/start',json.dumps(dict(width=width,height=height)).encode());assert status==200
            sid=start['session_id'];source_start=time.perf_counter();rtts=[];slips=[]
            for i,(t,pixels) in enumerate(zip(ts,rgba)):
                due=source_start+t/1e6;wait=due-time.perf_counter()
                if wait>0:time.sleep(wait)
                sent=time.perf_counter();slips.append(1000*(sent-due))
                code,r=request(c,app.token,'/frame',pixels,{'X-Capture-ID':sid,'X-Frame-ID':str(i),'X-Timestamp-US':str(t)})
                rtts.append((time.perf_counter()-sent)*1000);assert code in [200,429],r
            last_capture=source_start+ts[-1]/1e6
            code,r=request(c,app.token,'/finish',json.dumps(dict(session_id=sid,expected_frames=len(ts))).encode());assert code==200
            r['last_capture_to_input_ready_ms']=(time.perf_counter()-last_capture)*1000
            r['request_rtt_p95_ms']=float(np.percentile(rtts,95));r['source_send_slip_max_ms']=max(slips);r['clip_id']=cid
            with np.load(Path(r['artifact'])/'observations.npz') as z:
                r['same_pixels']=bool(np.array_equal(z['frame_sha256'],old['frame_sha256']))
                r['same_coordinates']=bool(np.array_equal(z['xy_px'],old['xy_px'],equal_nan=True))
                r['same_times']=bool(np.array_equal(z['timestamp_us'],old['timestamp_us']))
            assert r['same_pixels'] and r['same_coordinates'] and r['same_times']
            r['engineering_targets_pass']=r['input_complete'] and r['last_capture_to_input_ready_ms']<=cfg['max_post_ms']
            results.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
            code,_=request(c,app.token,'/finish',json.dumps(dict(session_id=sid,expected_frames=len(ts))).encode());assert code==409;controls.append('duplicate_finish_rejected')
            c.close()
        # Non-camera synthetic protocol tests; separate engine avoids extra real model calls.
        from test_stream import Fake
        a=Application(Fake(),U/'http_controls');s=make_server(a);t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
        try:
            c=http.client.HTTPConnection('127.0.0.1',s.server_port,timeout=5)
            code,_=request(c,'wrong','/start',b'{"width":10,"height":10}');assert code==403;controls.append('token_rejected');c.close()
            c=http.client.HTTPConnection('127.0.0.1',s.server_port,timeout=5)
            _,v=request(c,a.token,'/start',b'{"width":10,"height":10}');sid=v['session_id'];b=bytes(400)
            headers={'X-Capture-ID':sid,'X-Frame-ID':'0','X-Timestamp-US':'0'}
            code,_=request(c,a.token,'/frame',b,{**headers,'X-Capture-ID':'wrong'});assert code==409;controls.append('session_rejected')
            code,_=request(c,a.token,'/frame',b[:-1],headers);assert code==400;controls.append('size_rejected')
            code,_=request(c,a.token,'/frame',b,headers);assert code==200
            code,_=request(c,a.token,'/frame',b,headers);assert code==400;controls.append('nonmonotone_rejected')
            code,r=request(c,a.token,'/finish',json.dumps(dict(session_id=sid,expected_frames=2)).encode());assert code==200 and not r['input_complete'];controls.append('missing_tail_not_success');c.close()
        finally:s.shutdown();s.server_close();t.join()
        for p,h in plan['protected'].items():assert sha(p)==h,p
        result=dict(status='COMPLETE',runs=results,controls=controls,camera_used=False,training=0,
            limits=cfg['scope'],old_web_modified=False)
        with (U/'HTTP_COMPLETE.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':
    try:main()
    except BaseException:
        (U/('HTTP_FIRST_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
