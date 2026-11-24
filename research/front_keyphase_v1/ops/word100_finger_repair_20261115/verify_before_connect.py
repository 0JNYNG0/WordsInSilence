from common import *
import torch,ast,collections,copy
from finger_model import features
from finger_runtime import FingerObserver,refine
from projection_extension import project_fingers
import importlib.util
spec=importlib.util.spec_from_file_location('g5_project_review',O/'word100_visual_feedback_g5_20261114/adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2);checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
points=read(U/'SUPPLEMENT_POINT_CHECK.json');replay=read(U/'REPLAY_RESULTS.json');cfg=read(U/'TRAIN_FROZEN.json')
for p,h in read(U/'BEFORE_PINS.json').items():ck('pin:'+p,sha(p)==h)
for p,h in read(U/'INPUTS.json')['pins'].items():ck('source:'+p,sha(p)==h)
for p,h in cfg['pins'].items():ck('training_contract:'+p,sha(p)==h)
for r in read(U/'TRAINED.json')['runs']:ck('model:'+r['kind']+str(r['seed']),sha(r['model'])==r['sha256'])
ck('point_old_normal_not_wrong',points['metrics']['old']['old_correct_to_wrong']==0)
ck('additional_combination',points['metrics']['ring_little']['wrong']==0 and points['metrics']['ring_little']['correct']>points['metrics']['ring_little']['baseline_correct'])
ck('unrestricted_finger_output_disabled',points['finger_template_output_enabled']==False)
ck('normal_replay',replay['summary']['normal_cases']==64 and replay['summary']['new_normal_finger_warnings']==0)
ck('full_replay_count',replay['summary']['cases']==122)
ck('actual_detail_not_all_unknown',replay['summary']['detail_cases']==5 and replay['summary']['details']==20)
guides=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
for r in replay['rows']:
 d=read(r['output']);q=d['after'];p=project_fingers(m.project,q,[s for s in guides[r['clip_id'].split('_')[2]]if s['required']],480)
 ck('projection_keeps_unknown:'+r['clip_id']+r['kind'],all(not c['cues']for c in p if c['raw_state']in ['evidence_conflict','action_link_uncertain','component_link_uncertain']))
 if r['normal']:ck('no_normal_finger:'+r['clip_id']+r['kind'],not any(e['kind']=='finger'for c in p for e in c['cues']))
 for e in r['details']:ck('finger_real_span',e['span'][1]>e['span'][0]and e['coarse_evidence']['state']=='difference')
 # Preserve preexisting order/direction cues after adding detail.
 before=m.project(q,[s for s in guides[r['clip_id'].split('_')[2]]if s['required']],480)
 ck('other_cues_preserved',[[e for e in c['cues']if e['kind']in ['direction','order']]for c in p]==[[e for e in c['cues']if e['kind']in ['direction','order']]for c in before])
r=read(P/'INTEGRATED_RESULTS.json')['rows'][0];zz=np.load(r['source']);x=zz['x'];obs=FingerObserver();f,k=features(x,'R');trans=x.copy();trans[:,:102]=(trans[:,:102].reshape(-1,51,2)*1.4+np.array([.2,-.4])).reshape(-1,102);ft,kt=features(trans,'R');ck('relative_translation_scale',np.allclose(f,ft,atol=2e-4)and np.array_equal(k,kt))
mirror=x.copy();pp=mirror[:,:102].reshape(-1,51,2);pp[:,9:30]=pp[:,30:51].copy();pp[:,:,0]*=-1;mirror[:,195:216]=mirror[:,216:237].copy();fl,kl=features(mirror,'L');ck('shared_hands_reflection',np.allclose(f,fl,atol=2e-4)and np.array_equal(k,kl))
missing=x.copy();missing[:,18:102]=0;missing[:,195:237]=0;mo=obs.infer(missing,'R');ck('unobserved_unknown',np.all(mo['states']==-1))
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8-sig'))
write(U/'CONNECT_GATE.json',dict(passed=all(c['passed']for c in checks),checks=checks,admitted='Direction supplement plus conditional individual-finger refinement of existing handshape difference only',not_admitted='Unrestricted framewise finger correction or standalone replacement observer',known_residual='1/895olddevelopment finger observation remains wrong; no baseline-correct-to-wrong; conditional cues tested separately',whole_sign_accuracy=False,independent_expert=False))
print(json.dumps(dict(passed=True,checks=len(checks),scope='conditional_refinement_only')))
