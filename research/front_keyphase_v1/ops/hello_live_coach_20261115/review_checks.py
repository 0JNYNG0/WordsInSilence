"""Independent black-box coaching checks. Public observations only, no camera.

Synthetic coordinate edits are safety tests, not human error accuracy evidence.
"""
from pathlib import Path
import json, copy, hashlib, time
import numpy as np

U=Path(__file__).resolve().parent

def run():
    from coach import Coach
    code_sha=hashlib.sha256((U/'coach.py').read_bytes()).hexdigest()
    p=U/'source/reference.npz';before=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p) as z: raw={k:z[k].copy()for k in z.files}
    xy=raw['xy_px'];sc=raw['score'];ts=raw['timestamp_us'];n=len(ts)
    w=int(raw['source_width']);h=int(raw['source_height'])
    cases=[]
    def add(name,x=xy,s=sc,t=ts,expect_no_full=False):
        cases.append(dict(name=name,xy=x.copy(),score=s.copy(),ts=t.copy(),expect_no_full=expect_no_full))
    add('reference')
    add('static_sweep_start',np.repeat(xy[38:39],n,axis=0),np.repeat(sc[38:39],n,axis=0),expect_no_full=True)
    add('static_fists',np.repeat(xy[60:61],n,axis=0),np.repeat(sc[60:61],n,axis=0),expect_no_full=True)
    add('reverse_sequence',xy[::-1],sc[::-1],expect_no_full=True)
    add('first_action_only',xy[:46],sc[:46],ts[:46],expect_no_full=True)
    add('second_action_only',xy[48:81],sc[48:81],ts[48:81]-ts[48],expect_no_full=True)
    blank=xy.copy();blank[:,91:]=np.nan
    add('all_hands_missing',blank,expect_no_full=True)
    blank=xy.copy();blank[48:,91:112]=np.nan
    add('second_left_hand_missing',blank,expect_no_full=True)
    blank=xy.copy();blank[:]=np.nan
    add('all_points_missing',blank,expect_no_full=True)
    # Change current right fingers while retaining each frame's wrist position.
    for name,source,a,b in [('closed_during_sweep',60,30,48),('open_during_repeat',38,48,83)]:
        q=xy.copy();hand=xy[source,112:133]-xy[source,112]
        q[a:b,112:133]=hand[None]+q[a:b,112:113]
        add(name,q,expect_no_full=True)
    add('timestamp_2x_duration',t=ts*2)
    add('timestamp_half_duration',t=ts//2)
    c=Coach();out=[]
    for a in cases:
        c.reset();snap=[];t0=time.perf_counter()
        for x,s,t in zip(a['xy'],a['score'],a['ts']):
            r=c.update(x,s,int(t),w,h)
            json.dumps(r,allow_nan=False)
            snap.append(copy.deepcopy(r))
        last=snap[-1]
        completed=last.get('completed',[])
        full=bool(len(completed)==2 and all(completed))
        scope=all(r.get('whole_word_correct') is None for r in snap)
        unknown_states=sorted({ch.get('state') for r in snap for ch in r.get('checks',[]) if ch.get('state') not in ['ok','wrong','unknown','pending']})
        transitions=[];prior=None
        for i,r in enumerate(snap):
            q=(r.get('stage_index'),tuple(r.get('completed',[])))
            if q!=prior:transitions.append(dict(input_frame=i,stage=q[0],completed=list(q[1]),message=r.get('message')));prior=q
        out.append(dict(case=a['name'],full_core_observed=full,expected_no_full=a['expect_no_full'],
            false_completion_blocked=(not full) if a['expect_no_full']else None,
            whole_word_unclaimed=scope,invalid_check_states=unknown_states,transitions=transitions,
            elapsed_seconds=time.perf_counter()-t0,final=last))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==before
    # Controller-only fixtures isolate movement checks from the learned observer.
    # Known shape states are injected here ONLY, never into the product camera path.
    geometry=[]
    opened={hand:dict(state=0,fingers=[1]*4)for hand in ['L','R']}
    closed={hand:dict(state=1,fingers=[0]*4)for hand in ['L','R']}
    for mode in ['both_move','right_only','left_only']:
        c.reset()
        for i in range(46):c.consume_observation(xy[i].copy(),int(ts[i]),w,h,opened)
        assert c.stage_index==1, 'Fixture must reach stage 2 through observed stage 1 geometry'
        sca=float(np.linalg.norm(xy[60,5]-xy[60,6]))
        for i,d in enumerate([0,.08,.16,.08,0,.08,.16]):
            q=xy[60].copy()
            if mode in ['both_move','left_only']:
                q[91:112,1]+=d*sca;q[9,1]+=d*sca
            if mode in ['both_move','right_only']:
                q[112:133,1]+=d*sca;q[10,1]+=d*sca
            r=c.consume_observation(q,2000000+i*33333,w,h,closed)
        geometry.append(dict(mode=mode,completed=r['completed'],expected_both=(mode=='both_move'),passed=(all(r['completed'])==(mode=='both_move'))))
    report=dict(kind='Independent black-box public replay and synthetic safety checks',
        camera_opened=False,training=False,private_data=False,raw_sha256=before,raw_unchanged=True,
        code_sha256=code_sha,code_unchanged_during_test=hashlib.sha256((U/'coach.py').read_bytes()).hexdigest()==code_sha,
        whole_sign_accuracy=None,independent_person_test=False,rows=out,controller_fixtures=geometry,
        reference_core_observed=out[0]['full_core_observed'],
        safety_ok=all(r['false_completion_blocked']is not False and r['whole_word_unclaimed'] and not r['invalid_check_states']for r in out) and all(r['passed']for r in geometry))
    (U/'REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['reference_core_observed','safety_ok']},ensure_ascii=False))
    for r in out:print(r['case'],r['full_core_observed'],r['false_completion_blocked'],r['transitions'])

def run_http():
    import urllib.request,urllib.error
    base='http://127.0.0.1:8822';checks=[]
    def req(path,body=None,token=None,extra=None):
        headers={}if token is None else {'X-Session-Token':token}
        if extra:headers.update(extra)
        data=None
        if body is not None:headers['Content-Type']='application/json';data=json.dumps(body).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=10)as r:
                raw=r.read();return r.status,json.loads(raw)if 'json'in r.headers.get('Content-Type','')else raw
        except urllib.error.HTTPError as e:return e.code,e.read()
    status,meta=req('/api/meta');assert status==200;token=meta['token']
    assert req('/api/status')[0]==403
    checks.append(dict(name='status_token_required',passed=True))
    assert req('/api/start',{'mode':'camera'})[0]==403
    checks.append(dict(name='camera_without_token_rejected_no_device_access',passed=True))
    assert req('/api/start',{'mode':'public_replay'},token,{'Origin':'https://example.invalid'})[0]==403
    checks.append(dict(name='foreign_origin_rejected',passed=True))
    for route in ['/','/app.js','/style.css','/media/stage1.jpg','/media/stage2.jpg']:
        status,data=req(route);assert status==200 and len(data)>50
    status,data=req('/media/reference.mp4',extra={'Range':'bytes=0-99'});assert status==206 and len(data)==100
    checks.append(dict(name='public_assets_and_video_range',passed=True))
    status,state=req('/api/status',token=token)
    assert state['state']in ['ready','done'],state['state']
    assert req('/api/start',{'mode':'public_replay'},token)[0]==200
    time.sleep(.6)
    _,before=req('/api/status',token=token);gen=before['generation']
    assert before['source']=='public_replay'
    assert req('/api/start',{'mode':'public_replay'},token)[0]==409
    status,reset=req('/api/reset',{},token);assert status==200 and reset['generation']>gen and reset['observation']is None
    observations=[];deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        _,state=req('/api/status',token=token)
        assert state['generation']==reset['generation']
        assert state['source']=='public_replay'
        assert state['state']!='error',state.get('error')
        assert state['metrics'].get('camera_opened') is not True
        observations.append(dict(state=state['state'],seq=state['seq'],stage=(state.get('observation')or{}).get('stage_index')))
        if state['state']=='done':break
        time.sleep(.15)
    assert state['state']=='done' and state['observation']['completed']==[True,True]
    assert state['observation']['whole_word_correct']is None
    checks.append(dict(name='public_replay_reset_then_core_completion',passed=True,generation=state['generation'],observations=observations))
    assert req('/api/start',{'mode':'public_replay'},token)[0]==200
    time.sleep(.25);assert req('/api/stop',{},token)[0]==200
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        _,state=req('/api/status',token=token)
        if state['state']=='ready':break
        time.sleep(.05)
    assert state['state']=='ready'and state['observation']is None
    checks.append(dict(name='public_stop_clears_observation',passed=True))
    p=U/'REVIEW.json';report=json.loads(p.read_text(encoding='utf-8'))
    report['http']=dict(passed=True,camera_opened=False,private_data=False,checks=checks,
        server_sha256=hashlib.sha256((U/'server.py').read_bytes()).hexdigest())
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'http_passed':True,'checks':[c['name']for c in checks]},ensure_ascii=False))

if __name__=='__main__':
    import sys
    run_http()if '--http'in sys.argv else run()
