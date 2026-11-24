from common import *
import shutil
contract={'at':now(),'authorization':'공개 자료의 반복 혼동 및 학습 후반 흔들림 원인 분석과 공통 보완 계획 보고. 추가 학습은 결과 확인 후. 기존 C 유지, 사용자 촬영 학습·튜닝 제외.',
 'scope':'Stored C/Q/R10/R01 x three seeds, same public TRAIN899/DEV300. Read-only saved predictions, histories, observations, selected public source videos. No training, normalization updates, model inference, detector execution, deployment or user/mentor data.',
 'fixed_analysis':{'late_epochs':[61,80],'preceding_epochs':[41,60],'arms':['C','Q','R10','R01'],'seeds':[20261013,20261014,20261015],
 'error_ranking':'All 300 sensor DEV clips, number of wrong predictions over 12 stored runs; tie by clip_id. Review first five clips; compare all three DEV signers and all nine existing TRAIN records of their labels. Visual reference is smallest available TRAIN signer, never best scoring.',
 'visual':'Review full source RGB with fixed ROI and saved sensor observation overlay at uniformly spaced annotated-motion times plus start/end; compare frequent wrong-label TRAIN examples. No re-extraction or coordinate edits.',
 'history':'Describe counts/losses over fixed late window, adjacent changes, 4-epoch schedule phases, selection loss versus accuracy. Do not reselect checkpoint or infer missing per-epoch predictions.',
 'hypotheses':'Input loss versus actual public variant versus close decision boundary; optimizer LR, running statistics, schedule. Distinguish confirmed records/code from causal hypotheses.'},
 'outputs':['SAVED_ANALYSIS.json','INPUT_QUALITY.json','VISUAL_REVIEW.json','RESULT_KO.html','VERIFICATION.json','NEXT_PLAN.json'],
 'limits':'Repeated DEV is diagnostic, not blind performance evidence. RGB inspection does not certify linguistic correctness. No guaranteed improvement.',
 'implementation':'Codex local, separate arithmetic verification by same operator; no independent human/agent review.',
 'protected_pins':{p:h for p,h in read(FACT/'FINAL_AUDIT.json')['models_checked'].items()},
 'prior_result_pins':{str(FACT/n):sha(FACT/n) for n in ['FINAL_AUDIT.json','COMPARISON_INDEPENDENT.json','PREP_CONTEXT.json','TRAIN_CONTEXT.json']}}
write(U/'CONTRACT.json',contract)
(U/'before').mkdir()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
p=O/'STATE.json';s=read(p);s.update(updated_at=now(),current_stage='WORD100_CONFUSION_LATE_TRAINING_DIAGNOSIS',activity_state='PUBLIC_READ_ONLY_ANALYSIS',next_single_action='공개 반복 오답 영상·저장 학습 후반 기록 분석, 공통 보완 계획 보고. 추가 학습·사용자 촬영 활용·운영 교체 없음.')
s[U.name]={'status':'ANALYSIS_IN_PROGRESS','unit':str(U),'contract':str(U/'CONTRACT.json'),'training_allowed':False,'user_data_used':False,'production_changed':False}
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('ANALYSIS_CONTRACT_FROZEN')
