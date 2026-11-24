"""Local report integration test in an isolated, installed headless Edge.
No model inference, existing browser profile, downloads or camera access.
"""
from pathlib import Path
import json, time, subprocess, urllib.request, urllib.error, base64, socket
import websocket

U=Path(__file__).resolve().parent; Q=U/'qa'; Q.mkdir(exist_ok=True)
URL='http://127.0.0.1:8800'
results=[]; events=[]; serial=0; ws=None; proc=None
def check(name,ok,detail=None):
    results.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok: raise AssertionError((name,detail))
def get(path):
    with urllib.request.urlopen(URL+path,timeout=8) as r:return r.read()
def rpc(method,params=None):
    global serial
    serial+=1; mid=serial
    ws.send(json.dumps(dict(id=mid,method=method,params=params or {})))
    while True:
        d=json.loads(ws.recv())
        if d.get('id')==mid:
            if 'error' in d:raise RuntimeError(d['error'])
            return d.get('result',{})
        if d.get('method')=='Runtime.exceptionThrown':events.append(d)
def js(code):
    r=rpc('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True))
    if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
    return r.get('result',{}).get('value')
def until(code,timeout=20):
    stop=time.monotonic()+timeout
    while time.monotonic()<stop:
        if js(code):return
        time.sleep(.15)
    raise TimeoutError(code)
def select(id,value):
    js(f"document.getElementById({json.dumps(id)}).value={json.dumps(value)}; document.getElementById({json.dumps(id)}).dispatchEvent(new Event('change',{{bubbles:true}}));")
def ready():until("document.getElementById('load-status').textContent.includes('원본 결과 한 건')")
def shot(name):
    r=rpc('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
    (Q/name).write_bytes(base64.b64decode(r['data']))

try:
    data=json.loads(get('/VIEW_DATA.json'))
    check('index_denominator',len(data['records'])==2835,len(data['records']))
    check('all_video_headers',all(urllib.request.urlopen(urllib.request.Request(URL+'/video/'+a,method='HEAD'),timeout=8).status==200 for a in data['clips']))
    with urllib.request.urlopen(urllib.request.Request(URL+'/video/P048',headers={'Range':'bytes=0-1023'}),timeout=8) as r:
        b=r.read();check('video_range',r.status==206 and len(b)==1024,dict(status=r.status,size=len(b)))
    try:get('/data/not-an-allowed-id')
    except urllib.error.HTTPError as e:check('unlisted_result_blocked',e.code==404)
    profile=Q/('edge_profile_'+str(int(time.time())))
    log=(Q/'edge.log').open('ab')
    proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
       '--headless=new','--disable-gpu','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',
       '--user-data-dir='+str(profile),'--no-first-run','--no-default-browser-check',
       '--disable-background-networking','--disable-extensions','--disable-sync','about:blank'],
       stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    portfile=profile/'DevToolsActivePort';deadline=time.monotonic()+20
    while not portfile.exists() and time.monotonic()<deadline:time.sleep(.2)
    port=int(portfile.read_text().splitlines()[0])
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
    target=next(t for t in targets if t['type']=='page')
    ws=websocket.create_connection(target['webSocketDebuggerUrl'],timeout=20,suppress_origin=True)
    rpc('Page.enable');rpc('Runtime.enable')
    rpc('Emulation.setEmulatedMedia',dict(features=[dict(name='prefers-reduced-motion',value='reduce')]))
    rpc('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
    rpc('Page.navigate',dict(url=URL+'/'))
    until("!!document.getElementById('load-status') && document.getElementById('load-status').textContent.includes('원본 결과 한 건')")
    until("[...document.querySelectorAll('video')].every(v=>v.readyState>=2 && !v.error)")
    initial=js("({title:document.getElementById('record-title').textContent,id:document.getElementById('record-id').textContent,videos:[...document.querySelectorAll('video')].map(v=>({id:v.id,time:v.currentTime,duration:v.duration,w:v.videoWidth,h:v.videoHeight})),width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})")
    check('initial_pair', 'pair_P048_P045_P045_s02__R_wrist' in initial['id'],initial)
    check('desktop_no_overflow',initial['scroll']<=initial['width']+1)
    shot('desktop_summary.png')
    js("document.getElementById('record-panel').scrollIntoView({block:'start'})")
    shot('desktop_comparison.png')
    js("document.getElementById('mode-candidate').click();var s=document.getElementById('path-slider');s.value=s.max;s.dispatchEvent(new Event('input',{bubbles:true}));")
    until("[...document.querySelectorAll('video')].every(v=>!v.seeking)")
    chosen=json.loads(get('/data/pair_P048_P045_P045_s02__R_wrist'))
    p=chosen['comparison']['candidate']['path'][-1]
    expect=[chosen['comparison'][axis]['timestamp_us'][i]/1e6 for axis,i in zip(['query','reference'],p)]
    actual=js("[document.getElementById('query-video').currentTime,document.getElementById('reference-video').currentTime]")
    check('candidate_slider_exact_pts',all(abs(a-b)<.001 for a,b in zip(expect,actual)),dict(expected=expect,actual=actual))
    check('ambiguity_visible','경로가 여러 개' in js("document.getElementById('candidate-note').textContent"))
    shot('candidate_endpoint.png')
    js("document.getElementById('xy-plot').scrollIntoView({block:'start'})")
    check('both_graphs_drawn',js("document.getElementById('xy-plot').children.length>5 && document.getElementById('time-plot').children.length>5"))
    shot('trajectory_graphs.png')
    gap=next(r for r in data['records'] if r['kind']=='natural' and r['query']=='P048' and r['reference']=='P045' and r['parent_id']=='P045_s02' and not r['complete'])
    select('component-select',gap['component']);ready()
    check('missing_candidate_disabled',js("document.getElementById('mode-candidate').disabled && document.getElementById('new-distance').textContent==='관측 보류'"))
    shot('missing_observation.png')
    js("document.getElementById('tab-control').click()");ready()
    select('alias-select','P048');ready();select('control-select','swap_adjacent_core_blocks')
    until("document.getElementById('load-status').textContent.includes('저장된 기술 통제 자료가 없는')")
    check('missing_control_retained',js("document.getElementById('record-panel').hidden"))
    select('alias-select','P033');ready();select('control-select','speed_125');ready()
    check('synthetic_not_fake_video',js("document.getElementById('query-video-wrap').hidden && !document.getElementById('control-detail').hidden"))
    check('speed_sampling_warning','표본 삭제' in js("document.getElementById('candidate-note').textContent"))
    until("document.getElementById('reference-video').readyState>=2 && !document.getElementById('reference-video').seeking")
    js("document.getElementById('reference-video').scrollIntoView({block:'start'})")
    shot('technical_speed.png')
    rpc('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
    js('window.scrollTo(0,0)');time.sleep(.3)
    mobile=js('({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})')
    check('mobile_no_document_overflow',mobile['scroll']<=mobile['width']+1,mobile);shot('mobile_summary.png')
    check('no_javascript_errors',not events,events)
except Exception as e:
    results.append(dict(name='exception',passed=False,detail=repr(e)))
    print(repr(e),flush=True)
finally:
    if ws:
        try:rpc('Browser.close')
        except Exception:pass
        ws.close()
    if proc:
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:proc.terminate()
    output=dict(status='PASS' if all(r['passed'] for r in results) else 'FAIL',checks=results,
        browser='installed Edge isolated headless',external_download=False,model_inference=0,
        ui_tool_fallback='cua_repl and node_repl failed kernel assets path; no runtime permission bypass',
        screenshots=[p.name for p in Q.glob('*.png')])
    name='BROWSER_QA.json'; attempt=1
    while (Q/name).exists():
        attempt+=1; name=f'BROWSER_QA_{attempt:02d}.json'
    (Q/name).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(output['status'],len(results),flush=True)
    if output['status']!='PASS':raise SystemExit(1)
