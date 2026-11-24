"""Audit exposure and freeze the complete official Validation subset before extraction."""
from common import *
from collections import Counter
import re,subprocess,shutil,ast

def main():
 assert not (U/'EVAL_CONTEXT.json').exists()
 for f in ['common.py','prepare_inputs.py','evaluate.py']:ast.parse((U/f).read_text(encoding='utf8'))
 old=read(B/'training_CONTEXT.json');trained=read(T/'TRAIN_CONTEXT.json');assert trained['labels']==old['labels'] and trained['word_codes']==old['word_codes']
 labels=old['labels'];codes=old['word_codes'];prior=old['ordered_rows'];assert len(prior)==1199
 assert Counter(r['signer_id'] for r in prior)=={**{i:100 for i in range(1,9)},9:99,11:100,12:100,13:100}
 assert set(trained['ordered_train_ids'])=={r['clip_id'] for r in prior if r['role']=='train_candidate'}
 join=O/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl';allrows=[json.loads(l) for l in join.read_text(encoding='utf8').splitlines()]
 candidates=[r for r in allrows if r['word_code'] in codes and 14<=int(re.search('_REAL(\d+)',r['clip_id'])[1])<=18]
 selected=[r for r in candidates if int(re.search('_REAL(\d+)',r['clip_id'])[1]) in (17,18)]
 assert len(selected)==200 and len({r['clip_id'] for r in selected})==200
 assert {r['word_code'] for r in selected}==set(codes)
 rows=[];archive_stat={}
 for r in sorted(selected,key=lambda x:(x['word_code'],x['clip_id'])):
  idx=codes.index(r['word_code']);sid=int(re.search('_REAL(\d+)',r['clip_id'])[1])
  assert r['role']=='official_validation_metadata_only' and r['metadata_video_join_eligible'] and r['labels']==[labels[idx]]
  assert r['split']=='2.Validation' and not r['problems'] and len(r['video_entries'])==1
  e=r['video_entries'][0];assert e['filename_signer_id']==sid==e['folder_id'] and e['extra'] is None
  st=Path(e['archive']).stat();archive_stat[e['archive']]=[st.st_size,st.st_mtime_ns]
  rows.append({**r,'original_role':r['role'],'role':'heldout_evaluation_only','signer_id':sid,'label_index':idx,'canonical_label':labels[idx]})
 assert Counter(r['signer_id'] for r in rows)=={17:100,18:100}
 newids={r['clip_id'] for r in rows};assert newids.isdisjoint(r['clip_id'] for r in prior)
 prior_entries={(r['video_entries'][0]['bytes'],r['video_entries'][0]['crc32']) for r in read(O/'word100_training_20261013/MANIFEST.json')}
 assert not any((r['video_entries'][0]['bytes'],r['video_entries'][0]['crc32']) in prior_entries for r in rows)
 # Read prior source hashes only, not any user's recordings.
 source_hashes=[]
 for r in prior:source_hashes.append(read(B/'data/records'/f"{r['clip_id']}.json")['source']['sha256'])
 # Search public experiment artifacts for exact held-out IDs. Metadata-only inventories are allowed and excluded explicitly.
 args=['rg','-l','NIA_SL_WORD[0-9]{4}_REAL(17|18)_F','research/front_keyphase_v1/ops','research/front_keyphase_v1/runs','research/front_keyphase_v1/cache','research/sign_model_pilot_v1','-g','*.json','-g','*.jsonl','-g','*.py','-g','*.txt']
 exclude=['**/before/**','**/synthetic*/**','**/word_acquisition_inventory_20261011/**','**/word_dataset_plan_20261013/**','**/word_vocabulary_manifest_20261013/**','**/word_distribution_20261011/**','**/word_keypoint_archive_proposal_20260922/**','**/stgcn_execution_ready_20261005/**','**/*user*/**','**/*self*/**','**/sessions/**','**/data/raw/**','**/word100_t3f_heldout_20261019/**']
 for g in exclude:args+=['-g','!'+g]
 proc=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf8');assert proc.returncode in [0,1],proc.stderr
 hits=proc.stdout.splitlines();assert not hits,('Possible prior exposure requires inspection',hits)
 # Filename inventory only: do not read private artifacts. Native extracted public cache names would be a contrary exposure signal.
 named=[]
 for root in [ROOT/'research/front_keyphase_v1/cache',ROOT/'research/sign_model_pilot_v1/evidence',Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013'),Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')]:
  for p in root.rglob('*'):
   if p.is_file() and p.suffix.lower() in ['.npz','.mp4','.png','.jpg'] and any(cid in p.name for cid in newids):named.append(str(p))
 assert not named,('Prior extracted arrays/images require exposure review',named)
 models={a:{} for a in ['C','T3F']}
 for seed in SEEDS:
  for a,folder,stem in [('C',B,f'training_C_{seed}'),('T3F',T,f'T3F_{seed}')]:
   result=read(folder/(stem+'_RESULT.json'));cp=folder/(stem+'_best.pt');assert sha(cp)==result['checkpoint_sha256']
   models[a][str(seed)]={'path':str(cp),'sha256':sha(cp),'selected_epoch':result['selected_epoch'],'selected_using':'original DEV11-13 before heldout evaluation'}
 auth={'status':'APPROVED','approved':True,'at':now(),'user_request':'T3-F를 고정하고 학습이나 모델 선택에 쓰지 않은 공개 자료로 기존 C와 비교. 기존 C 유지, 사용자 촬영 학습/튜닝 금지.','scope':'Exposure audit; exact frozen pipeline inputs for official Validation17/18 full100word set; one evaluation each3paired checkpoints and2views; independent verification/report. No training or reselection.'}
 write(U/'AUTHORIZATION.json',auth,new=True)
 protocol={'status':'FROZEN_BEFORE_HELDOUT_EXTRACTION_OR_INFERENCE','at':now(),'cohort':'All200 current100-word officialValidation front videos REAL17/18','cohort_chosen_without_scores':True,'why200':'Complete predefined officialValidation split; keep reserved14-16 untouched for later independently planned evaluations. No easy-word/duration/quality/score filtering.','reserved_not_opened':{'eligible14_16':299,'identity_unresolved':1,'unresolved_id':'NIA_SL_WORD0500_REAL14_U_F'},'models':models,'seed_policy':'All3 paired existing checkpoints; no ensemble, no bestseed choice. Predefined deployment seed20261013 additionally reported, no adoption.','input':'Exact C joint_v3 detector/assignment; original stable-rest/body fallback1.477334276 coefficient, wrist filterOFF; native->masked and fresh croppedRGB->sensor; real missingframes/PTS preserved, unchanged280channels active_span. No label/core times in input span.','calibration_failure':'Retain failed clip in full200 denominator as incorrect for both models/views; no new boundary fallback or culling. Unexpected source/hash/code failure stops execution.','primary':'sensor top1 average across3 paired models, each200 clips; report all counts,range,perseed%p,top5,signer,observationgroup,class,gain/loss. masked secondary.','decision':'Descriptive improvement maintained if sensor mean higher and >=2/3paired seeds improve; report masked and subgroup regression explicitly. Stronger engineering consistency additionally requires masked average/top5/signer/group means nondecreasing. Not proof of population generalization or automatic adoption.','uncertainty':'Two heldout signer IDs from same corpus; 3seeds are not3subjects. Repeated evaluations of same200 videos not600independent people. Once scored, this cohort is consumed for future blindness. No result-directed tuning or additional search.','verification':'Separate reconstruction of input masks/channels, score arithmetic,model/source hashes. SameCodex,notindependentperson.','prohibited':['training','model/seed/checkpoint selection on heldout','user/mentor recording access or tuning','download','camera','external transfer','productionC/web/DTW change','Git writes']}
 write(U/'FROZEN_PROTOCOL.json',protocol,new=True)
 exposure={'status':'PASS_C_T3F_LINEAGE_NO_HELDOUT_EXPOSURE_FOUND','at':now(),'train_signers':list(range(1,10)),'development_signers':[11,12,13],'heldout_signers':[17,18],'train_ids':899,'dev_ids':300,'heldout_ids':200,'exact_clip_id_overlap':0,'prior_source_crc_size_matches':0,'original_roles_metadata_only':True,'labels_aligned':200,'prior_public_artifact_search_command':args,'search_hits':hits,'search_errors':proc.stderr,'prior_named_extracted_arrays':named,'metadata_only_inventory_seen_previously':True,'not_claimed':'No claim of externally certified identities or unknowable history outside repository. Prior metadata inventory is not training/extraction/performance exposure.','source_hash_overlap_checked_during_extraction':True}
 write(U/'EXPOSURE_AUDIT.json',exposure,new=True)
 # Pin the transitive exact C preparation code/model files against its original freeze.
 pins={}
 for p,h in read(B/'PREP_CONTEXT.json')['pins'].items():
  if Path(p).suffix in ['.py','.task'] or p.endswith('contract.json') or p.endswith('design_data_COEFFICIENT.json'):
   assert sha(p)==h;pins[p]=h
 fixed=[join,O/'word100_training_20261013/MANIFEST.json',B/'training_CONTEXT.json',B/'training_core.py',T/'TRAIN_CONTEXT.json',T/'frozen_temporal.py',T/'temporal.py',T/'COMPARISON_INDEPENDENT.json',T/'FINAL_AUDIT.json']
 fixed+=[U/f for f in ['common.py','freeze.py','prepare_inputs.py','evaluate.py','AUTHORIZATION.json','FROZEN_PROTOCOL.json','EXPOSURE_AUDIT.json']]
 for p in fixed:pins[str(p)]=sha(p)
 for arm in models.values():
  for info in arm.values():pins[info['path']]=info['sha256']
 write(U/'EVAL_CONTEXT.json',{'status':'FROZEN','at':now(),'rows':rows,'labels':labels,'word_codes':codes,'models':models,'prior_source_sha256':sorted(set(source_hashes)),'archive_stat':archive_stat,'pins':pins,'requested':200,'untouched_reserved14_16':299},new=True)
 before=U/'before';before.mkdir()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,before/p.name)
 action='T3-F/C 각3체크포인트 고정, 미사용공개 officialValidation ID17/18의100단어200영상 입력준비→독립검증→1회평가→저장점수검산·개선회귀보고. 보류14~16 299건은미개봉, 사용자촬영미사용, 학습/운영교체금지.'
 s=read(O/'STATE.json');s.update(current_stage='WORD100_T3F_HELDOUT_PREPARATION',next_single_action=action,updated_date_kst=now()[:10]);s[U.name]={'status':'AUTHORIZED_FROZEN_PREPARATION','at':now(),'cohort':200,'signer_ids':[17,18],'models_frozen':6,'training':0,'production_changed':False,'user_data_used':False,'next_single_action':action};write(O/'STATE.json',s)
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8');text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-10-19 T3-F 미사용 공개 평가 승인·동결\nops/'+U.name+'. '+action,1);text=text.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-10-19 T3-F 고정 후 공개 보류평가\n'+action,1);p.write_text(text,encoding='utf8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now()+'\n\n'+action+'\n\n단위: research/front_keyphase_v1/ops/'+U.name+'\nEXPOSURE_AUDIT/EVAL_CONTEXT/FROZEN_PROTOCOL 확인. 학습0, 추가모델선택금지. 현단계 새미사용200평가 준비. PREP_RUNNING/STATUS/실제exec확인후이어받고중복실행하지않음. 완료C/T3-F이전DEV평가재실행금지.\n',encoding='utf8')
 event(stage='COHORT_AND_MODELS_FROZEN',clips=200,models=6,source_bytes=sum(r['video_entries'][0]['bytes'] for r in rows),exposure='PASS')

if __name__=='__main__':main()
