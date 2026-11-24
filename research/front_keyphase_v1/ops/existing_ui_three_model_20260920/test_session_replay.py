"""Independent verification of the variable-length existing-UI offline path."""
import copy,importlib.util,json,math,sys
from pathlib import Path
from synthetic_support import make_case,read,write,fake_modules,fixture,predictions,sha

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_replayer',here/'session_replay.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
checks=[]
def check(name,value):checks.append({'name':name,'passed':bool(value)})
def rejects(name,fn):
    try:fn()
    except (ValueError,RuntimeError):check(name,True)
    else:check(name,False)

root,config=make_case('v1_normal')
session=root/'web/runs/ses_synthetic'
s=read(session/'session.json');s['purpose']='development';write(session/'session.json',s)
review_path=session/'trial_reviews.jsonl'
reviews=[json.loads(line) for line in review_path.read_text().splitlines()]
for r in reviews:r['note']='ordinary existing UI self report'
review_path.write_text(''.join(json.dumps(r)+'\n' for r in reviews),encoding='utf-8')
mpath=runner.prepare(str(config),str(root/'out'))
m=read(mpath);rows=m['rows']
labels={w:d['label'] for w,d in m['config']['words'].items()}
check('existing UI8 attempts retained without fabricated54 slots or journal',len(rows)==8 and [r['trial_id'] for r in rows]==[f'tri_{i:02}' for i in range(1,9)])
check('all known truth5 and eligible4 retained',m['counts']['confirmed_n']==5 and m['counts']['eligible_n']==4 and m['counts']['unconfirmed_n']==3)
check('original newest review rather than requested word',rows[0]['review_revision']==2 and rows[0]['actual_word_id']=='W00' and rows[2]['actual_differs_from_requested'])
check('missing artifact known labels stay',rows[5]['truth_known'] and not rows[5]['eligible'] and rows[6]['truth_known'] and not rows[6]['eligible'])
check('cancelled and unverified terminal distinct',rows[4]['terminal']=='CANCELLED' and rows[7]['terminal']=='UNVERIFIED_TERMINAL')
pred=predictions()
for roles in pred.values():
    for p in roles.values():
        if p['status']!='READY':p.update(project18=None,full93=None)
summary=runner.summarize(rows,pred,labels)
for scope in ('project18','full93'):
    arms=summary['scopes'][scope]['per_role']
    check(scope+' independent hits2/2/1 denominator5',[(arms[a]['correct'],arms[a]['confirmed_n']) for a in ('parent','control','treatment')]==[(2,5),(2,5),(1,5)])
    check(scope+' support4 macroF1 independently calculated',all(arms[a]['macro_f1_class_count']==4 and math.isclose(arms[a]['macro_f1'],v) for a,v in [('parent',5/12),('control',1/3),('treatment',1/8)]))
    pair=summary['scopes'][scope]['paired']['treatment_vs_control']
    check(scope+' gains/regressions preserve failed answer',pair['gains']==['tri_01'] and pair['regressions']==['tri_02','tri_03'] and pair['both_answered']==2)
    check(scope+' all18 classes including14 zero-support',len(arms['parent']['per_class'])==18 and sum(x['support']==0 for x in arms['parent']['per_class'].values())==14)
check('visible development label and all18 coverage retained','NOT_BLINDED' in summary['study_labels'] and len(summary['coverage_by_actual'])==18)
zero=copy.deepcopy(rows)
for row in zero:row.update(actual_word_id=None,truth_known=False,actual_label=None)
z=runner.summarize(zero,{},labels)
check('zero labels do not claim accuracy',z['confirmed_n']==0 and z['scopes']['project18']['per_role']['parent']['accuracy'] is None)
bad=copy.deepcopy(pred);bad['tri_01']['parent']['elapsed_ms']=float('nan')
rejects('nonfinite timing rejected',lambda:runner.summarize(rows,bad,labels))
bad=copy.deepcopy(pred);bad['tri_05']=bad['tri_01']
rejects('ineligible trial prediction rejected',lambda:runner.summarize(rows,bad,labels))
bad=copy.deepcopy(pred);bad['tri_01']['parent']['status']='ERROR'
rejects('error result with apparently correct text cannot count',lambda:runner.summarize(rows,bad,labels))

for name,change in [
 ('open',lambda r,c:write(r/'web/runs/ses_synthetic/session.json',dict(read(r/'web/runs/ses_synthetic/session.json'),state='OPEN'))),
 ('unaccounted',lambda r,c:(r/'web/runs/ses_synthetic/trials/tri_unknown.npz').write_bytes(b'SYNTHETIC')),
 ('malformed',lambda r,c:(r/'web/runs/ses_synthetic/trial_reviews.jsonl').write_text('{broken',encoding='utf-8')),
 ('old_revision_conflict',lambda r,c:(r/'web/runs/ses_synthetic/trial_reviews.jsonl').open('a',encoding='utf-8').write(json.dumps(dict(fixture()[3][0],actual_word_id='W17'))+'\n'))]:
    case,cp=make_case('v1_'+name);change(case,cp)
    rejects('prepare rejects '+name,lambda:runner.prepare(str(cp),str(case/'out')))

modules,neural,calls,loads=fake_modules(read(config))
saved={n:sys.modules.get(n) for n in modules};sys.modules.update(modules)
runtime_error=None
try:
    try:result_path=runner.execute(mpath)
    except Exception as exc:runtime_error={'type':type(exc).__name__,'message':str(exc)}
    check('recogniser runtime failure retained rather than aborting',runtime_error is None)
    check('registry/default restored after execution',neural.MODELS=={'base':{'kept':True}} and neural.DEFAULT_MODEL=='base')
    if runtime_error is None:
        result=read(result_path)
        check('actual9 recognise calls; same3 inputs and4 load attempts',result['inference_calls']==len(calls)==9 and len(loads)==4)
        check('load failure kept and not three fake inference calls',len(result['load_failures'])==1 and result['summary']['confirmed_n']==5)
    before=len(calls)
    rejects('repeat execution refused',lambda:runner.execute(mpath))
    check('duplicate request did not infer',len(calls)==before)
finally:
    for name,value in saved.items():
        if value is None:sys.modules.pop(name,None)
        else:sys.modules[name]=value

report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'code_sha256':sha(here/'session_replay.py'),'runtime_error':runtime_error,'real_inference_calls':0,'user_records_read':0,'camera_calls':0}
write(here/'verification_v1.json',report)
print(json.dumps({k:report[k] for k in ('status','passed','total','runtime_error')}))
print(json.dumps({'failed':[c['name'] for c in checks if not c['passed']]}))
sys.exit(0 if report['status']=='PASS' else 1)
