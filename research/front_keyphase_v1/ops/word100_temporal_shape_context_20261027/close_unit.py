from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent
R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert R.name=='WordsInSilence',R
assert not (U/'FINAL_STATUS.json').exists(),'Do not replace completed unit.'
report_review=read(U/'review/REPORT_REVIEW.json')
assert report_review['status'].startswith('PASS'),report_review
check=read(U/'review/COMPARISON_VERIFY.json'); assert all(check['checks'].values())
assert read(U/'UI_VERIFY.json')['status']=='PASS'
assert sha(U/'COMPARISON.json')==check['saved_comparison_sha256']
for name,current in [('WordsInSilence_PROJECT_CONTEXT.md',R/'WordsInSilence_PROJECT_CONTEXT.md'),
                     ('NEW_CHAT_HANDOFF.md',R/'NEW_CHAT_HANDOFF.md'),
                     ('STATE.json',U.parent/'STATE.json')]:
    assert sha(U/'before'/name)==sha(current),'Concurrent state change: '+name
t=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary='''앞뒤 흐름으로 불확실한 손모양을 보완하는 세조건 비교를 완료했다. 이전12표현묶음에서6표현×기록ID3개=18구간을 먼저 고정하고, 각원본13연속프레임(±6,약0.4초), 총234RGB/468특징칸을 준비했다. 현재이미지만A/이전200msB/앞뒤각200msC(기존TCN C와별개), 최소2앵커/방향, 현재관측보존, 재귀보간금지, 특징모순보류를 고정했다. 시간순을숨긴RGB초안334확정/134unknown. 알려진중간279칸을각각독립마스킹: B237일치/4충돌/38보류, C194일치/1충돌/84보류. 모양있음64/아님215의불균형, 인접프레임의종속성, AI초안대조임을명시했다. 충돌4실제이미지는별도검토자도원래판단에동의. T15별의짧은폄을주변접힘으로채우는C충돌이남았다. 실제unknown중심11칸에서B새추정1/C0, 전체unknown134칸에서B12/C0이므로실제가림복구향상주장불가. 직접334칸보존은설계검사이며검출정답보증아님. 중심36칸별도검토22확정일치/11양쪽unknown/3한쪽unknown; 추가14이미지28칸23확정일치/4양쪽unknown/1한쪽unknown. 시간순RGB검토18구간에서변화7/뚜렷변화없음6/제한5, 손연결이상징후관찰0(자동추적오류율아님). 별도구현으로234RGB해시·279독립mask·B미래변조·3645조합·시간한계·집계PASS. 18구간234슬라이더상태 및desktop/mobile UI PASS. 기존C/T3F/운영웹/DTW32핀유지. 신규학습0/사용자멘토접근0/운영교체0. 작은시간규칙시험완료이며TCN·DTW정확도향상이나전체시간모델한계로일반화하지않는다.'''
next_action='''다음 한 작업은 현재손RGB·관절에서 모양변화가보이는지와 단순이동·회전·가림인지를 구분하는 공통시간보완 입력/보호조건을 검증하는 것이다. 현재1/0초안을앞뒤로복사하는규칙만으로중간실제모양을확정하지않고, 현재관측/시간추정/관측부족을별도유지한다. 이번4충돌은원인진단용으로보존하고 동일결과를고치며성능주장하지않는다. 별도공개구간의안정된모양/짧은변화/겹침/휴지손과실제PTS·기록사람분리를결과전고정해, 정상유지와짧은변화훼손·추정가능범위를함께평가한다. 목표단어/Gloss/사전정답을관측입력으로사용하지않고 시간창임계값반복확대·새검출기추가로대체하지않는다. 이전36영상특징학습계획은유지하되 같은AI초안과시간추정을독립정답으로재학습하지않으며, 신뢰가능한보이는특징정답확보가학습의조건이다. 교육적정오·가려진정확손가락좌표는미확정. 사용자/멘토학습튜닝선택금지, 기존C·T3F·웹보존. 이단위실행종료/활성프로세스없음. 승인된일반검증의재승인은불필요하며운영교체는별도범위다.'''
title='2026-10-27 손모양 앞뒤 시간근거 세조건 비교 완료'
evidence='research/front_keyphase_v1/ops/word100_temporal_shape_context_20261027/RESULT_KO.html'
context=R/'WordsInSilence_PROJECT_CONTEXT.md'; text=context.read_text(encoding='utf-8-sig')
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    assert text.count(heading)==1
    text=text.replace(heading,heading+'\n\n### '+title+'\n'+t+'\n'+body+'\n근거: '+evidence+', FROZEN.json, COMPARISON.json, ADDITIONAL_REVIEW.json, review/COMPARISON_VERIFY.json, review/REPORT_REVIEW.json, UI_VERIFY.json.\n',1)
context.write_text(text,encoding='utf-8')
state_path=U.parent/'STATE.json';state=read(state_path)
state['updated_date_kst']=t;state['current_stage']='TEMPORAL_SHAPE_CONTEXT_PILOT_COMPLETE_REAL_OCCLUSION_GAIN_NOT_ESTABLISHED'
state['next_single_action']=next_action
state[U.name]={'status':'THREE_CONDITION_COMPARISON_VERIFIED_NOT_ADOPTED_FOR_CORRECTIVE_FEEDBACK',
 'at':t,'summary':summary,'next':next_action,'evidence':evidence,
 'counts':{'expressions':6,'recorded_people':3,'windows':18,'rgb_frames':234,'feature_cells':468,
 'direct_known':334,'direct_unknown':134,'known_mask_trials':279,'masked_positives':64,'masked_negatives':215,
 'training_runs':0,'operating_changes':0,'user_mentor_access':0,'protected_pins':32},
 'masked_comparison':read(U/'COMPARISON.json')['masked_summary'],
 'real_center_comparison':read(U/'COMPARISON.json')['center_summary'],
 'implementation':'Codex root; local_feature_inventory extraction/sequence review; linguistic_eval_review shuffled labels/independent arithmetic; ksl_source_audit center/additional image drafts.',
 'verification':'Separate code and actual RGB reviews; all same AI family, not human truth. Candidate limits explicitly retained.',
 'active_execution':None,'training_ready':False}
dump(state_path,state)
handoff=f'''# WordsInSilence 최신 인수인계

{t}

## 현재 목표와 마지막 완료
{summary}

## 다음 한 작업
{next_action}

## 실행·검증 구분
- 구현: root(동결·시간규칙·비교·보고/UI·보존검사). local_feature_inventory(원본234프레임·18시간순시각검토), linguistic_eval_review(234셔플이미지초안·별도코드검산), ksl_source_audit(18중심/추가14이미지초안).
- A/B/C는 현재만/과거/과거미래의 시간규칙이며 운영TCN C와 다른 명칭. 자동RGB손모양모델이나학습결과가아님.
- 같은AI초안에서의마스킹일치와실제미관측복구를분리. C실제unknown새추정0. 279마스킹64양성/215음성, 18표적구간이며독립279사람아님.
- T15별 f59: 실제네손가락폄을가리면앞뒤접힘으로채우는실패. 1~2프레임완전굽힘폄의신체한계를측정한것은아님. 해당영상의짧은변화는RGB로확인.
- FROZEN/IMAGE_LABELS/CENTER_LABELS/COMPARISON 최초결과보존. 후속확인SECOND_IMAGE_LABELS·ADDITIONAL_REVIEW별도. 조건·시간창·라벨을점수에맞춰변경하지않음.
- 334직접관측유지는코드보존계약. 손연결이상0/18은시각검토범위이며자동추적정확도증명아님. 저위휴지손·교차를수어오류로판정하거나삭제하지않음.
- 원본234RGB·PTS검증, 총540이미지생성(보기용crop포함). 학습관측ROI변경없음. 원본핀78·운영핀32유지.
- 학습/새검출기/사용자멘토접근/운영교체/Git/다운로드0. 실행종료·활성학습/서버/자동화새실행없음.

## 근거
- `{evidence}`: 성공·실패·실제모호구간을슬라이더비교.
- `{U.name}/COMPARISON.json`: 468항목/279독립mask 원시출력과집계.
- `{U.name}/review/COMPARISON_VERIFY.json`: 별도코드검산; 단순시간복사한계명시.
- `{U.name}/ADDITIONAL_REVIEW.json`: 중심/충돌이미지대조·보호해시.
- `{U.name}/sequence_review/SEQUENCE_REVIEW.json`: 18구간실제RGB시각검토.
- `{U.name}/UI_VERIFY.json`: 234프레임전환·4대표예·desktop/mobile PASS.

이번 단위는 완료했다. 실제가림복구·TCN인식률·DTW피드백향상으로승격하지않는다. 이전36영상공통특징학습준비와독립정답필요조건은유지한다.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(state_path)[U.name]['counts']['known_mask_trials']==279
assert title in context.read_text(encoding='utf-8')
pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'COMPARISON.json',U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'UI_VERIFY.json',U/'ADDITIONAL_REVIEW.json',U/'review/COMPARISON_VERIFY.json',U/'review/REPORT_REVIEW.json',context,state_path,R/'NEW_CHAT_HANDOFF.md']}
dump(U/'FINAL_STATUS.json',{'status':'COMPLETE','at':t,'active_execution':None,'training':False,'operating_replacement':False,'implementation_verified':True,'real_occlusion_gain_established':False,'pins':pins,'docs_synced':True})
print(json.dumps({'status':'COMPLETE','at':t,'docs_synced':3,'comparison_checks':len(check['checks'])},ensure_ascii=False))
