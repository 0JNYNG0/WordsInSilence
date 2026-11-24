"""Independent synthetic fixtures and adapter double; no human data or real model access."""
import copy,hashlib,json,shutil,sys,types
from pathlib import Path
here=Path(__file__).resolve().parent
old=here.parent/'fresh_self_comparison_20260919/resume_20260920'
sys.path.insert(0,str(old))
from synthetic_fixture import fixture,predictions

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):Path(p).write_text(json.dumps(obj),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_case(name):
    root=here/'synthetic_cases'/name
    root.mkdir(parents=True,exist_ok=False)
    source=old/'fixtures_v2/normal/web'
    shutil.copytree(source,root/'web')
    (root/'pilot').mkdir()
    guard=root/'guard.txt';guard.write_text('SYNTHETIC ONLY',encoding='utf-8')
    plan=fixture()[0]
    config={'web_root':str(root/'web'),'pilot_root':str(root/'pilot'),'session_id':'ses_synthetic','participant_id':'self',
            'minimum_created_epoch':1767225600,'reference_set_id':'synthetic_refs',
            'words':{w['word_id']:{'label':'L'+w['word_id'][1:],'clip_id':w['clip_id']} for w in plan['words']},
            'models':{},'protected_sha256':{str(guard):sha(guard)},'data_role':'VISIBLE_FEEDBACK_SINGLE_PERSON_DEVELOPMENT; NOT_BLINDED; NOT_EXECUTION_CORRECTNESS'}
    for role in ('parent','control','treatment'):
        path=root/'web/runs'/('model_'+role)/'toy.pt'
        config['models'][role]={'path':str(path),'sha256':sha(path),'step':220 if role=='parent' else 660}
    cpath=root/'config.json';write(cpath,config)
    return root,cpath

def fake_modules(config):
    calls=[];loads=[]
    neural=types.ModuleType('wis_lab.service.neural')
    neural.MODELS={'base':{'kept':True}};neural.DEFAULT_MODEL='base'
    neural.PROJECT_18={w:s['label'] for w,s in config['words'].items()}
    def role_of(key):
        path=neural.MODELS[key]['path'].replace('\\','/')
        return next(r for r in ('parent','control','treatment') if '/model_'+r+'/' in path)
    def load(key):
        role=role_of(key)
        return {'status':'READY','checkpoint_sha256':config['models'][role]['sha256'],'checkpoint_step':220 if role=='parent' else 660,
                'classes':list(neural.PROJECT_18.values())+[f'X{i}' for i in range(75)]}
    def recognise(obs,key):
        role=role_of(key);calls.append((obs,role))
        if obs=='tri_03' and role=='treatment':raise ValueError('synthetic model runtime failure')
        pred=predictions()[obs][role]
        return {'model_status':pred['status'],'checkpoint_sha256':neural.MODELS[key]['sha256'],'accepted':None,'execution_assessment':'UNCONFIGURED',
                'scopes':{'project18':{'top1':pred['project18']},'full':{'top1':pred['full93']}},'elapsed_ms':2.0}
    neural._load=load;neural.recognise=recognise
    observation=types.ModuleType('wis_lab.core.observation')
    class Observation:
        @staticmethod
        def load(path):
            tid=Path(path).stem;loads.append(tid)
            if tid=='tri_04':raise ValueError('synthetic observation load failure')
            return tid
    observation.Observation=Observation
    modules={n:types.ModuleType(n) for n in ('torch','wis_lab','wis_lab.service','wis_lab.core')}
    modules.update({'wis_lab.service.neural':neural,'wis_lab.core.observation':observation})
    return modules,neural,calls,loads
