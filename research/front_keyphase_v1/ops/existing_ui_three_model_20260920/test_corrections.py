"""Only failed cases and affected integration paths; unchanged24 checks are reused."""
import ast,importlib.util,json,sys
from pathlib import Path
from synthetic_support import make_case,read,write,fake_modules,fixture,sha
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('corrected_replayer',here/'session_replay_v2.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
checks=[]
def check(name,value):checks.append({'name':name,'passed':bool(value)})
def rejects(name,fn):
    try:fn()
    except (RuntimeError,ValueError):check(name,True)
    else:check(name,False)

def functions(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a=functions(here/'session_replay.py');b=functions(here/'session_replay_v2.py')
check('only two reviewed functions changed',set(a)==set(b) and {n for n in a if a[n]!=b[n]}=={'_collect_reviews','execute'})
case,config=make_case('v2_old_revision_conflict')
with (case/'web/runs/ses_synthetic/trial_reviews.jsonl').open('a',encoding='utf-8') as f:
    f.write(json.dumps(dict(fixture()[3][0],actual_word_id='W17'))+'\n')
rejects('conflicting superseded revision rejected',lambda:runner.prepare(str(config),str(case/'out')))

for name,corrupt_hash in [('v2_normal',False),('v2_contract_hash',True)]:
    root,config=make_case(name)
    manifest=runner.prepare(str(config),str(root/'out'))
    modules,neural,calls,loads=fake_modules(read(config))
    if corrupt_hash:
        real=neural.recognise
        def wrong(obs,key):
            result=real(obs,key);result['checkpoint_sha256']='0'*64;return result
        neural.recognise=wrong
    saved={n:sys.modules.get(n) for n in modules};sys.modules.update(modules)
    try:
        if corrupt_hash:
            rejects('normalization hash mismatch still fatal',lambda:runner.execute(manifest))
            check('fatal contract mismatch leaves no final result',not (root/'out/result_LOCAL_ONLY.json').exists() and (root/'out/execution_started.json').is_file())
        else:
            result=read(runner.execute(manifest))
            check('runtime failure retained and all later observations attempted',len(calls)==result['inference_calls']==9 and len(loads)==4 and len(result['load_failures'])==1)
            check('new output matches independently calculated hit counts',result['summary']['confirmed_n']==5 and [result['summary']['scopes']['project18']['per_role'][r]['correct'] for r in ('parent','control','treatment')]==[2,2,1])
            check('failed arm and observation kept',result['predictions']['tri_03']['treatment']['status']=='ERROR' and all(p['status']=='ERROR' for p in result['predictions']['tri_04'].values()))
            before=len(calls)
            rejects('completed replay refuses rerun',lambda:runner.execute(manifest))
            check('no calls after duplicate request',len(calls)==before)
        check(name+' restores registry and default',neural.MODELS=={'base':{'kept':True}} and neural.DEFAULT_MODEL=='base')
    finally:
        for n,v in saved.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'reused_unaffected_v1_checks':24,'source_sha256':sha(here/'session_replay_v2.py'),'real_inference_calls':0,'user_records_read':0,'camera_calls':0}
write(here/'verification_v2.json',report)
print(json.dumps({k:report[k] for k in ('status','passed','total','reused_unaffected_v1_checks')}))
print(json.dumps({'failed':[c['name'] for c in checks if not c['passed']]}))
sys.exit(0 if report['status']=='PASS' else 1)
