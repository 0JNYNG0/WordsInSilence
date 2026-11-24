"""Independent synthetic verification; never imports the real neural adapter or human data."""
import argparse,copy,hashlib,importlib.util,io,json,math,sys,types
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from synthetic_fixture import fixture,predictions

sys.dont_write_bytecode=True
here=Path(__file__).resolve().parent
version=sys.argv[1]
stage=here/('generated_'+version)
sys.path.insert(0,str(stage))
import frozen_cohort as core
import frozen_replay as driver
checks=[]
def check(name,condition):checks.append({'name':name,'passed':bool(condition)})
def rejects(name,fn):
    try:fn()
    except (ValueError,RuntimeError):check(name,True)
    else:check(name,False)
def changed(index,mutate):
    data=fixture();mutate(data[index]);return core.build_cohort(*data)

cohort=core.build_cohort(*fixture())
check('all54 slots retained including46 unrecorded',len(cohort['rows'])==54 and cohort['counts']['not_captured']==46)
check('known labels retain missing and corrupt artifacts',cohort['counts']['confirmed']==5 and cohort['counts']['eligible']==4)
check('authoritative newest review wins over false client saved flag',cohort['rows'][0]['actual_word_id']=='W00' and cohort['rows'][0]['review_revision']==2)
check('requested label never replaces UNKNOWN',cohort['rows'][3]['actual_word_id'] is None)
check('actual mismatch retained',cohort['rows'][2]['actual_differs_from_requested'] and cohort['rows'][2]['actual_word_id']=='W00')
check('cancel and unverified terminal distinct',cohort['rows'][4]['terminal_state']=='CANCELLED' and cohort['rows'][7]['terminal_state']=='UNVERIFIED_TERMINAL')
check('missing NPZ and hash mismatch remain rows',cohort['rows'][5]['artifact_status']=='MISSING_NPZ' and cohort['rows'][6]['artifact_status']=='HASH_MISMATCH')
label_map={f'W{i:02}':f'L{i:02}' for i in range(18)}
summary=core.summarize(cohort,predictions(),label_map)
for scope in ('project18','full93'):
    arms=summary['scopes'][scope]['arms']
    check(scope+' independent correct counts and full confirmed denominator',[(arms[r]['correct'],arms[r]['confirmed_n']) for r in ('parent','control','treatment')]==[(2,5),(2,5),(1,5)])
    check(scope+' F1 includes failed predictions and actual4-class support',all(math.isclose(arms[r]['macro_f1'],v) and arms[r]['macro_f1_actual_class_count']==4 for r,v in [('parent',5/12),('control',1/3),('treatment',1/8)]))
    check(scope+' errors with apparently correct strings cannot count correct',[arms[r]['inference_failures_on_confirmed'] for r in ('parent','control','treatment')]==[2,2,3])
    pair=summary['scopes'][scope]['pairs']['treatment_vs_control']
    check(scope+' paired gains and losses preserve inference failure',pair['gains']==[1] and pair['regressions']==[2,3] and pair['both_answered']==2)
    check(scope+' all18 per-class entries and explicit54 coverage',len(arms['parent']['per_class'])==18 and arms['parent']['planned']==54 and math.isclose(arms['parent']['correct_over_planned_rate'],2/54))
check('unknown captured and unrecorded listed separately',summary['unconfirmed_slots']==[4,5,8] and summary['not_captured_slots']==list(range(9,55)))
check('offline timing excludes stored server timing',summary['offline_elapsed_ms_by_arm']==dict.fromkeys(('parent','control','treatment'),10.0))
rejects('duplicate trial rejected',lambda:changed(1,lambda j:j['slots'][1].update(trial_id='tri_01')))
rejects('unmatched server attempt rejected',lambda:changed(2,lambda s:s['trial_ids'].append('tri_extra')))
rejects('old development session rejected',lambda:changed(2,lambda s:s.update(created_epoch=0)))
rejects('open server session rejected',lambda:changed(2,lambda s:s.update(state='OPEN')))
rejects('wrong stable request id rejected',lambda:changed(1,lambda j:j['slots'][0].update(request_id='wrong')))
rejects('review wrong slot rejected',lambda:changed(3,lambda r:r[1].update(note='protocol=synthetic_v1;slot=2;')))
rejects('conflicting same revision rejected',lambda:changed(3,lambda r:r.append(dict(r[1],actual_word_id='W01'))))
rejects('result plus cancelled event rejected',lambda:changed(4,lambda e:e.append({'event':'trial_closed','session_id':'ses_synthetic','trial_id':'tri_01','outcome':'CANCELLED'})))
rejects('unknown event outcome rejected',lambda:changed(4,lambda e:e[0].update(outcome='MADE_UP')))
rejects('missing clip binding rejected',lambda:changed(5,lambda r:r['tri_01']['reference_binding'].pop('clip_id')))
rejects('journal shuffled order rejected',lambda:changed(1,lambda j:j['slots'].reverse()))
rejects('plan wrong temporal round rejected',lambda:changed(0,lambda p:[s.update(round=4-s['round']) for s in p['slots']]))
rejects('journal noninteger slot rejected',lambda:changed(1,lambda j:j.update(current_slot=1.25)))
bad=predictions();bad['tri_01']['control']['elapsed_ms']=float('nan')
rejects('nonfinite timing rejected',lambda:core.summarize(cohort,bad,label_map))
bad=predictions();bad['tri_50']=bad['tri_01']
rejects('unrecorded prediction rejected',lambda:core.summarize(cohort,bad,label_map))
empty=fixture();empty[2]['trial_ids']=[];empty[1]['status']='STOPPED'
for row in empty[1]['slots']:row.update(trial_id=None,request_id=None,terminal_state=None)
zero=core.summarize(core.build_cohort(*empty[:3],[],[],{},{}),{},label_map)
check('zero-label cohort reports unknown accuracy and F1',zero['confirmed_n']==0 and zero['scopes']['project18']['arms']['parent']['accuracy_over_confirmed'] is None and zero['scopes']['project18']['arms']['parent']['macro_f1'] is None)

# Driver checks use toy bytes, synthetic JSON, and explicitly injected fake modules.
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')
def make_files(name):
    root=here/f'fixtures_{version}'/name
    if root.exists():raise FileExistsError(root)
    p,j,s,reviews,events,res,arts=fixture()
    j['status']='COMPLETED'
    web=root/'web';pilot=root/'pilot';pilot.mkdir(parents=True)
    session=web/'runs'/s['session_id'];trials=session/'trials';trials.mkdir(parents=True)
    for tid,a in arts.items():
        if a['npz_exists']:(trials/(tid+'.npz')).write_bytes(('SYNTHETIC '+tid).encode())
        if a['result_exists']:
            r=copy.deepcopy(res[tid]);r['reference_binding'].pop('clip_id')
            digest=sha(trials/(tid+'.npz')) if a['npz_exists'] else 'a'*64
            r['observation_sha256']=digest
            write(trials/(tid+'.result.json'),r)
            write(trials/(tid+'.sidecar.json'),{'observation_sha256':'f'*64 if tid=='tri_07' else digest})
    write(session/'session.json',s)
    (session/'trial_reviews.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in reviews),encoding='utf-8')
    (session/'timing_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events),encoding='utf-8')
    p['models']={}
    for role in ('parent','control','treatment'):
        path=web/'runs'/('model_'+role)/'toy.pt';path.parent.mkdir();path.write_bytes(role.encode())
        p['models'][role]={'path':str(path),'sha256':sha(path)}
    guard=root/'guard.txt';guard.write_text('SYNTHETIC PROTECTED FILE',encoding='utf-8')
    p['protected_sha256']={str(guard):sha(guard)}
    write(root/'protocol.json',p);write(root/'journal.json',j)
    args=argparse.Namespace(protocol=str(root/'protocol.json'),journal=str(root/'journal.json'),session_dir=str(session),web_root=str(web),pilot_root=str(pilot),out_dir=str(root/'out'))
    return root,args

capture=io.StringIO()
root,args=make_files('normal')
with redirect_stdout(capture),redirect_stderr(capture):
    try:prepared=driver.cmd_prepare(args)==0
    except Exception as exc:prepared=False;capture.write(type(exc).__name__+': '+str(exc)+'\n')
check('driver accepts actual colon request IDs and completed partial journal',prepared)
if prepared:
    manifest=root/'out'/'manifest_LOCAL_ONLY.json'
    m=json.loads(manifest.read_text(encoding='utf-8'))
    check('prepare performs no inference and retains54 rows',m['inference_calls']==0 and len(m['cohort']['rows'])==54)
    calls=[];loads=[]
    fake=types.ModuleType('wis_lab.service.neural');fake.MODELS={'base':{'kept':True}};fake.DEFAULT_MODEL='base';fake.PROJECT_18=label_map
    def load(key):
        role=next(r for r in ('parent','control','treatment') if '/model_'+r+'/' in fake.MODELS[key]['path'].replace('\\','/'))
        return {'status':'READY','checkpoint_sha256':fake.MODELS[key]['sha256'],'checkpoint_step':220 if role=='parent' else 660,'classes':list(label_map.values())+[f'X{i}' for i in range(75)]}
    def recognise(obs,key):
        role=next(r for r in ('parent','control','treatment') if '/model_'+r+'/' in fake.MODELS[key]['path'].replace('\\','/'))
        calls.append((obs,role))
        if obs=='tri_03' and role=='treatment':raise ValueError('synthetic per-model runtime failure')
        p=predictions()[obs][role]
        return {'model_status':'READY','checkpoint_sha256':fake.MODELS[key]['sha256'],'accepted':None,'execution_assessment':'UNCONFIGURED','elapsed_ms':2.0,'scopes':{'project18':{'top1':p['project18']},'full':{'top1':p['full93']}}}
    fake._load=load;fake.recognise=recognise
    observation=types.ModuleType('wis_lab.core.observation')
    class Observation:
        @staticmethod
        def load(path):
            tid=Path(path).stem;loads.append(tid)
            if tid=='tri_04':raise ValueError('synthetic undecodable observation')
            return tid
    observation.Observation=Observation
    modules={n:types.ModuleType(n) for n in ('torch','wis_lab','wis_lab.service','wis_lab.core')}
    modules.update({'wis_lab.service.neural':fake,'wis_lab.core.observation':observation})
    saved={n:sys.modules.get(n) for n in modules};sys.modules.update(modules)
    try:
        with redirect_stdout(capture),redirect_stderr(capture):rc=driver.cmd_execute(argparse.Namespace(manifest=str(manifest)))
        check('execute tolerates observation and model runtime failures',rc==0)
        if rc==0:
            result=json.loads((root/'out'/'result_LOCAL_ONLY.json').read_text(encoding='utf-8'))
            check('only actual recognise calls counted; same3 inputs across3 models',result['inference_calls']==len(calls)==9 and len(loads)==4 and all(sum(o==tid for o,r in calls)==3 for tid in ('tri_01','tri_02','tri_03')))
            check('observation failure retained separately',len(result['observation_load_failures'])==1 and result['summary']['confirmed_n']==5)
            check('driver metrics agree with independent synthetic expected counts',[result['summary']['scopes']['project18']['arms'][r]['correct'] for r in ('parent','control','treatment')]==[2,2,1])
            check('registry and default restored',fake.MODELS=={'base':{'kept':True}} and fake.DEFAULT_MODEL=='base')
            rejects('second execute refused without new calls',lambda:driver.cmd_execute(argparse.Namespace(manifest=str(manifest))))
            check('no extra calls after duplicate execution request',len(calls)==9)
    finally:
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value

# Deliberately malformed server store must fail rather than act like no confirmations.
for name,change in [
 ('bad_jsonl',lambda root,args:(Path(args.session_dir)/'trial_reviews.jsonl').write_text('{broken',encoding='utf-8')),
 ('missing_protected',lambda root,args:write(Path(args.protocol),dict(json.loads(Path(args.protocol).read_text()),protected_sha256={str(root/'nonexistent_guard'):'a'*64}))),
 ('path_escape',lambda root,args:setattr(args,'session_dir',str(root)))] :
    case,case_args=make_files(name);change(case,case_args)
    with redirect_stdout(capture),redirect_stderr(capture):rejects('driver rejects '+name,lambda:driver.cmd_prepare(case_args))

report={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','version':version,'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks),
        'real_model_loads':0,'human_records_used':0,'camera_calls':0,'code_sha256':{p.name:sha(p) for p in stage.glob('*.py')}}
(here/f'verification_{version}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(here/f'verification_{version}_output.txt').write_text(capture.getvalue(),encoding='utf-8')
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[x['name'] for x in checks if not x['passed']]},indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
