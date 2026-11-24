from common import *
import shutil
from collections import Counter
assert not (U/'PREP_CONTEXT.json').exists(), 'Reuse existing frozen inputs, never overwrite'
plan=read(PLAN/'PLAN.json');checkpins(plan['source_pins']);assert sha(PLAN/'PLAN_KO.txt')==plan['plan_sha256']
auth={'approved':True,'at':now(),'user_request':'그래 그 계획대로 진행해줘. 먼저 입력 자료를 준비하고 검증해서 문제없으면 학습과 기존 C 비교까지 이어서 진행해줘. 내 촬영은 학습이나 튜닝에 쓰지 말고, 중요한 문제가 생기면 바로 알려줘. 완료되면 개선된 점과 남은 문제를 정리해서 보고해줘.','plan':str(PLAN/'PLAN.json'),'plan_sha256':sha(PLAN/'PLAN.json'),'conditional_training':True,'production_replacement':False,'user_mentor_data_use':False}
if not (U/'AUTHORIZATION.json').exists():write(U/'AUTHORIZATION.json',auth,new=True)
else:auth=read(U/'AUTHORIZATION.json')
before=U/'before';before.mkdir(exist_ok=True)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 if not (before/p.name).exists():shutil.copy2(p,before/p.name)
c=read(BASE/'training_CONTEXT.json');manifest=read(O/'word100_training_20261013/MANIFEST.json'); mm={r['clip_id']:r for r in manifest}
old=read(CURRENT/'SCOPE.json');known={r['clip_id'] for r in old['records']}
pins={str(PLAN/'PLAN.json'):sha(PLAN/'PLAN.json'),str(PLAN/'PLAN_KO.txt'):sha(PLAN/'PLAN_KO.txt'),str(BASE/'training_CONTEXT.json'):sha(BASE/'training_CONTEXT.json'),str(CURRENT/'OUTPUT.json'):sha(CURRENT/'OUTPUT.json'),**plan['source_pins']}
for seed in (20261013,20261014,20261015):
 for suffix in ('RESULT.json','best.pt'):
  p=BASE/f'training_C_{seed}_{suffix}';pins[str(p)]=sha(p)
records=[]; missing=[]
for row in c['ordered_rows']:
 cid=row['clip_id'];p=BASE/'data/records'/f'{cid}.json';r=read(p);pins[str(p)]=sha(p)
 assert r['signer_id'] in list(range(1,10))+[11,12,13]
 entry={k:row[k] for k in ('clip_id','role','signer_id','label_index','word_code')};entry.update(label=r['canonical_label'],record=str(p),source=r['source'],box=r['calibration']['box_source_pixels'],intervals=mm[cid]['intervals'],views={})
 for v in ('masked','sensor'):
  cap=r['native_capture' if v=='masked' else 'sensor_capture'];ass=Path(cap['candidate']).with_name('assignment.npz')
  paths={'observation':r['views'][v]['observation'],'channels':r['views'][v]['channel_file'],'raw':cap['raw'],'assignment':str(ass)}
  for kind,name in paths.items():
   if name is None or not Path(name).exists():missing.append([cid,v,kind,name]);continue
   h=sha(name);expected=r['outputs'].get(name,cap['pins'].get(name))
   if expected is not None:assert h==expected,(cid,v,kind,'original pin mismatch')
   pins[name]=h
  entry['views'][v]=paths
 records.append(entry)
 if len(records)%200==0:print('Inventory',len(records),flush=True)
if missing:
 write(U/'FIRST_INPUT_ERROR.json',{'at':now(),'missing':missing,'training':0},new=True);raise RuntimeError('Required cache unavailable; preserve and report')
assert len(records)==1199 and Counter(r['role'] for r in records)=={'train_candidate':899,'development_only':300}
review=[];already_complete=[]
for word in sorted({r['word_code'] for r in records}):
 pool=[r for r in records if r['role']=='train_candidate' and r['word_code']==word and r['clip_id'] not in known]
 if not pool:
  previous=[r for r in records if r['role']=='train_candidate' and r['word_code']==word]
  assert previous and all(r['clip_id'] in known for r in previous)
  already_complete.append({'word_code':word,'clips':[r['clip_id'] for r in previous]});continue
 review.append(min(pool,key=lambda r:(r['signer_id'],r['clip_id']))['clip_id'])
context={'status':'FROZEN_INPUTS_READY_FOR_ADAPTER','at':now(),'authorization':auth,'pins':pins,'records':records,'known_regression_clips':sorted(known),'additional_review_clips':review,'all_train_clips_already_reviewed_words':already_complete,'plan':plan,'labels':c['labels'],'word_codes':c['word_codes'],'missingness_groups':c['missingness_groups'],'source_order_preserved':True,'training_allowed':False}
write(U/'INVENTORY_RECOVERY.json',{'at':now(),'first_error':'FIRST_INVENTORY_ERROR.json','change':'All TRAIN clips already reviewed for4words: reuse existing complete9signer coverage;96other words get new review. No dataset/model/threshold change.','review_word_coverage':len(review)+len(already_complete),'training':0},new=True)
write(U/'PREP_CONTEXT.json',context,new=True)
write(U/'STATUS.json',{'at':now(),'stage':'INPUT_INVENTORY_FROZEN','records':1199,'review_clips':len(review),'pins':len(pins),'training':0})
print(json.dumps({'records':1199,'pins':len(pins),'review_clips':len(review),'prep_sha256':sha(U/'PREP_CONTEXT.json')}))
