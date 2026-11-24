from pathlib import Path
import json,datetime,shutil
U=Path(__file__).resolve().parent;R=U.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
s=read(U/'SUMMARY.json');a=read(U/'verification/FINAL_INDEPENDENT.json')
assert a['status']=='PASS' and read(U/'verification/SAVED9_INDEPENDENT.json')['status']=='PASS'
assert not (U/'RUNNING.json').exists() and read(U/'TRAINING_COMPLETE.json')['runs']==9
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
before=U/'before'/'final_sync';before.mkdir(exist_ok=True)
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U.parent/'STATE.json']
for p in paths:
    if not (before/p.name).exists():shutil.copy2(p,before/p.name)
chosen=next(r for r in s['models'] if r['name']==s['selection']['selected_model']);baseline=s['models'][0]
scorelines=' / '.join(f"{m['label']} 정답률{m['accuracy_mean']*100:.2f}%,균형F1{m['balanced_macro_f1_mean']*100:.2f}점" for m in s['models'])
summary=f"사용자가 요청한 CPU 모델 재선정을 실제9회학습·저장점수검산·CPU속도실측까지 완료했다. 기존V3와RepViT-M1.0/V4-Hybrid-Medium의 공식배포ImageNet가중치에서 새로시작하여 마지막단계와공식head경로를20epoch×3seed로비교했다. fit2489(01/02/04/05)·DEV488(03/06/07);45수형학습/개발30종포함·15종미평가. 결과 {scorelines}. 사전지정사람03/07균형macroF1평균과1점내nativeCPU p95우선규칙에따른개발후보는 {chosen['label']}. 전체정답률기준V3대비 {(chosen['accuracy_mean']-baseline['accuracy_mean'])*100:+.2f}%p. 이수치는TCN단어인식률/KSL교정정확도가아니다. 이전최종382·사용자·멘토자료미접근. 실제노트북Intel Core Ultra7 258V CPU4/FP32에서한장과120/300손사진의전체RGB경로를측정,부품속도와웹전체응답을구분한다. RepViT계산융합은별도보조결과이며원native선택규칙불변. 독립에이전트가입력/학습코드/저장점수/속도산술을검산했다. C/T3F·운영32핀보존,웹활성화없음. 학습·측정모두종료. 세계최적주장/새사람최종시험주장금지."
nxt=f"고정한 {chosen['label']} 개발후보를 대상으로 기존관절이정상일때보존하고손이보이지만검출만빠진때보완하는선택적RGB경로를새공개KSL근거로검증한다. 정상수형·회전·교차·짧은변화·관측부족의독립주석을우선확보하고개발선택에사용한488과이전최종382를새최종시험처럼재사용하지않는다. 성능/오판보존통과후실제촬영종료~피드백전체시간을측정한다. 약3초는부드러운선호·5초이상바람직하지않음,현재RGB부품시간으로전체응답보장금지. Holistic는별도검출기비교후결정하며이번에시험/도입하지않았다. 기존C/T3F유지,사용자테스터자료학습/튜닝/선택금지."
lat300=next(r for r in chosen['latency']['clips'] if r['hand_crops']==300)['rgb_decode_preprocess_full_model']['p95_seconds']
summary+=f' 선정후보의300손RGB처리p95는{lat300:.2f}초로전체피드백3초를보장하지못하며모든프레임기본모델로바로채택하지않는다. 정답률59.90%수준의단독손모양판정을교정정답으로사용하지않는다.'
context=paths[0].read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-28 CPU 모델 9회 비교·속도 검산 완료',summary),('## 9. 다음 단일 작업','2026-10-28 선정 후보의 선택적 RGB 보조 검증',nxt)]:
    context=context.replace(marker,marker+'\n\n### '+title+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.html, SUMMARY.json, verification/FINAL_INDEPENDENT.json.\n',1)
paths[0].write_text(context,encoding='utf-8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n현재목표: CPU에서신뢰할수있는손모양관측후보선정→선택적보조·피드백검증.\n\n'+summary+'\n\n다음한작업: '+nxt+'\n\n작업단위: research/front_keyphase_v1/ops/'+U.name+'\n\n실행종료. 학습9회/DEV점수/속도완료결과재실행불필요. 원native선택과RepViT융합보조를분리. 새추론없는별도산술검산과독립사람정답검토를혼동하지않는다. 최초오류및실행전수정보존: logs/PREFLIGHT_METRICS_REPAIR.json, PREFLIGHT_SOURCE_PIN.json, before/.\n',encoding='utf-8')
state=read(paths[2]);stage='CPU_MODEL_COMPARISON_COMPLETE_DEVELOPMENT_CANDIDATE_NOT_DEPLOYED'
state['current_stage']=stage;state['updated_at']=now;state['updated_date_kst']=now;state['next_single_action']=nxt
state[U.name]={'status':stage,'at':now,'unit':U.name,'summary':summary,'next':nxt,'models':[{k:m[k] for k in ['name','accuracy_mean','accuracy_range','balanced_macro_f1_mean','balanced_macro_f1_range']} for m in s['models']],'selected':chosen['name'],'completed_runs':9,'active_training_or_inference':False,'split':{'fit':2489,'dev':488,'final_test':0},'fit_classes':45,'dev_classes':30,'production_changed':False,'user_mentor_data_used':False,'old_final382_accessed':False,'end_to_end_latency_measured':False,'result':str(U/'RESULT_KO.html'),'verification':['verification/SAVED9_INDEPENDENT.json','verification/FINAL_INDEPENDENT.json']}
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
current=read(U/'CURRENT_EXECUTION.json');current.update(state='COMPLETE',active=False,completed_at=now,training_exit_code=0)
(U/'CURRENT_EXECUTION.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'STATUS.json').write_text(json.dumps({'state':stage,'active':False,'training_runs':9,'selection':chosen['name'],'at':now},ensure_ascii=False,indent=2),encoding='utf-8')
print('final synchronized')
