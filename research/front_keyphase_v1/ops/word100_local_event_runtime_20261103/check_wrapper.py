"""Synthetic wrapper checks: no old model imports, decoder or neural forward."""
from pathlib import Path
import ast,json,time,hashlib,types
import numpy as np
U=Path(__file__).resolve().parent

def function(path,name,env):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]

def main():
    checks=[]
    def check(name,condition):
        assert bool(condition),name
        checks.append(name)
    for p in U.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
        checks.append('syntax:'+p.name)
    work=U/'structural_test_artifacts';work.mkdir(exist_ok=False)
    ids=np.arange(32,dtype=np.int64)*2
    ts=np.arange(32,dtype=np.int64)*33127
    starts=np.array([0,16]);centers=(ts[starts]+ts[starts+15])/2
    expected_features=np.stack([np.arange(768),np.arange(768)+3]).astype(np.float32)
    expected_pre=np.stack([np.arange(768)+11,np.arange(768)+17]).astype(np.float32)
    counter=[]
    def compiled(inputs):
        i=len(counter);counter.append(i)
        return {0:expected_features[i:i+1],1:expected_pre[i:i+1]}
    def encode(video,box,given_ids,given_ts,tap,progress):
        check('full-frame extent',box==[0,0,640,480])
        check('native axis untouched at encoder',np.array_equal(given_ids,ids) and np.array_equal(given_ts,ts))
        values=np.concatenate([tap(None)[0],tap(None)[0]])
        rgb=np.stack([np.interp(ts,centers,values[:,j]) for j in range(768)],1).astype(np.float32)
        return rgb,values,starts,centers
    extract=function(U/'learner_adapter.py','extract_with_support',dict(np=np,time=time))
    shared=types.SimpleNamespace(base=types.SimpleNamespace(_encode=encode))
    rgb,pre,seconds=extract(work,shared)('synthetic',ids,ts,640,480,compiled,None)
    expected_rgb=np.stack([np.interp(ts,centers,expected_features[:,j]) for j in range(768)],1).astype(np.float32)
    expected_aligned_pre=np.stack([np.interp(ts,centers,expected_pre[:,j]) for j in range(768)],1).astype(np.float32)
    check('exact old feature interpolation',np.array_equal(rgb,expected_rgb))
    check('exact old prelinear interpolation',np.array_equal(pre,expected_aligned_pre))
    check('no duplicate backbone calls',len(counter)==2)
    saved=dict(np.load(work/'RGB_NATIVE_SUPPORT.npz',allow_pickle=False))
    for name,value in dict(features=expected_features,prelinear_windows=expected_pre,frame_id=ids,
        timestamp_us=ts,starts=starts,center_timestamp_us=centers).items():
        check('saved:'+name,np.array_equal(saved[name],value))
    allowed=function(U/'order_event_worker.py','allowed',dict(Path=Path,U=U))
    good=U/'public_runtime_runs'/('a'*32)/'analysis'
    check('public job accepted',allowed(good)==good.resolve())
    for name,path in [('private',U/'private_sessions'/('a'*32)/'analysis'),
                      ('outside',U.parent/'analysis'),('bad_id',U/'public_runtime_runs'/'not-an-id'/'analysis'),
                      ('different_leaf',U/'public_runtime_runs'/('a'*32)/'input')]:
        rejected=False
        try:allowed(path)
        except (AssertionError,ValueError):rejected=True
        check('reject:'+name,rejected)
    value_sha=function(U/'order_event_worker.py','value_sha',dict(json=json,hashlib=hashlib))
    check('canonical value hash',value_sha({'b':1,'a':'손'})==hashlib.sha256(
        json.dumps({'a':'손','b':1},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    report=dict(status='PASS_SYNTHETIC_WRAPPER_CHECKS',checks_count=len(checks),checks=checks,
        model_loads=0,neural_forwards=0,video_decodes=0,training=0,whole_goal_complete=False)
    with (U/'WRAPPER_CHECKS.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(report['status'],len(checks))

if __name__=='__main__':main()
