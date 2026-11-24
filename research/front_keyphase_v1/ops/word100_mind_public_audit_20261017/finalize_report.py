from pathlib import Path
import json,hashlib,html,re
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;REPO=U.parents[3];LAB=U.parents[1]
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
read=lambda p:json.loads(Path(p).read_text('utf-8'))
audit=read(U/'AUDIT.json');recount=read(U/'RECOUNT.json')
assert recount['status']=='PASS'
protect={
 LAB/'ops/word100_body_boundary_20261017/training_C_20261013_best.pt':'283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab',
 LAB/'runs/word100_tcn_v1/whole_context/best.pt':'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'}
for p,h in protect.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
text=(U/'RESULT_KO.txt').read_text('utf-8')
parts=re.split(r'(?m)^(\d+\. .+)$',text)
detail=[]
for i in range(1,len(parts),2):
    body=html.escape(parts[i+1].strip())
    body=re.sub(r'(https://\S+)',r'<a href="\1">\1</a>',body)
    detail.append('<section><h2>'+html.escape(parts[i])+'</h2><div class="text">'+body+'</div></section>')
table=''
for w,s in audit['summary'].items():
    d=s['core_seconds']
    table+=f'<tr><td>{html.escape(s["label"])}</td><td>9개</td><td>{d["median"]:.2f}초</td><td>{d["min"]:.2f}–{d["max"]:.2f}초</td></tr>'
gallery=''
for w,s in audit['summary'].items():
    row=next(x for x in audit['records'] if x['word_code']==w and x['signer_id']==1)
    gallery+=f'<article><h3>{html.escape(s["label"])}</h3><video controls preload="metadata" src="{Path(row["video"]).as_uri()}"></video><p>TRAIN01 원본 전체 영상. 다음 이미지는 TRAIN01~09 주석 구간 각10장면이다.</p><a href="{w}_CORE_TRAIN9.jpg"><img src="{w}_CORE_TRAIN9.jpg" alt="{html.escape(s["label"])} 공개 TRAIN9 장면"></a></article>'
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>마음에 들다 — 공개 자료 분석과 다음 계획</title><style>
*{box-sizing:border-box}body{margin:0;background:#f0f4f8;color:#192c40;font-family:"Malgun Gothic",sans-serif;line-height:1.75}
main{max-width:1120px;margin:auto;padding:48px 30px 80px}h1{font-size:34px;line-height:1.35;margin:12px 0 20px}h2{font-size:23px;margin:0 0 18px}h3{font-size:20px}
.tag{display:inline-block;background:#d8e9f7;padding:5px 12px;border-radius:20px;font-size:14px;color:#205b84}
.lead{font-size:21px}.card,section,article{background:#fff;border:1px solid #d9e2ec;border-radius:15px;padding:28px;margin:24px 0}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.grid .card{margin:0}strong{color:#174e73}
th,td{text-align:left;border-bottom:1px solid #dce4ea;padding:12px}table{border-collapse:collapse;width:100%}
.text{white-space:pre-wrap;font-size:16px;overflow-wrap:anywhere}a{color:#075e9a}img{width:100%;height:auto}video{width:100%;max-height:420px;background:#13263e}
.note{background:#fff4da;border-left:5px solid #d39d26;padding:18px}p{margin:10px 0}
@media(max-width:700px){main{padding:22px 14px}.grid{display:block}.grid .card{margin:16px 0}h1{font-size:26px}.card,section,article{padding:18px}}
</style><main><span class="tag">공개 자료 감사 완료 · 학습 전 검토</span>
<h1>‘마음에 들다’는 왜 다른 표현과 헷갈렸을까?</h1>
<p class="lead">부분적으로 닮은 장면은 있지만, 전체 동작은 다르다.<br><strong>관측 누락과 시간·순서 처리의 영향을 나누어 확인할 단계다.</strong></p>
<div class="grid"><div class="card"><h2>36개 공개 영상</h2><p>4개 표현 × TRAIN9<br>원주석·저장입력 대조</p></div><div class="card"><h2>핵심 시간 보존</h2><p>마음에 들다9개<br>주석 구간 프레임100% 포함</p></div><div class="card"><h2>새 학습 0회</h2><p>사용자 관측 재처리0<br>기존 C·M0·웹 보존</p></div></div>
<section><h2>표본 수가 달라서 생긴 문제인가?</h2><table><thead><tr><th>표현</th><th>학습 영상</th><th>핵심 구간 중앙값</th><th>범위</th></tr></thead><tbody>'''+table+'''</tbody></table><p>각 표현의 TRAIN ID는01~09로 동일하다. 이9개만으로 모든 사람·환경을 충분히 대표한다고 볼 수는 없다.</p></section>
<div class="note"><strong>확정할 수 없는 것</strong><br>아래 공개 장면과 코드 점검은 원인 후보를 좁힌다. 사용자의 오인식을 특정 손모양·속도 하나 때문이라고 확정하거나 개선 폭을 예측하는 결과는 아니다.</div>
<section><h2>남은 관측 문제: 겹친 왼손이 사라지는 구간</h2><p>마음에 들다 TRAIN06. 노란선은 입력 하단, 초록점은 왼손, 분홍점은 오른손이다. 뒤쪽은 손이 화면 안에 있어도 왼손 관측이 빠진다. 약0.47초 연속 누락이 확인됐다.</p>
<a href="NIA_SL_WORD0500_REAL06_F_DETAIL.jpg"><img src="NIA_SL_WORD0500_REAL06_F_DETAIL.jpg" alt="양손 겹침 구간의 관측 누락"></a><p>앞부분 왼손은 실제로 내려가 있다. 모든 미관측을 같은 오류로 취급하지 않는다.</p></section>
<section><h2>다음 작업 순서</h2><ol><li>공개100단어에서 같은 동작의 촬영 간격·구간별 취약성을 고정 모델로 진단.</li><li>간격 변화가 문제라면 학습·웹의 시간 처리 규칙을 공통화한 단일 후보 검토.</li><li>별도 후보로 처음·중간·끝의 특징을 함께 보존하는 방식 검토.</li><li>사용자 검토 후 단일 후보만3seed 비교. 전체100단어와 새 회귀까지 판정.</li></ol><p>사용자 영상으로 후보·계수·임계값을 고르지 않는다. 지금은 학습 전 계획이다.</p></section>
<details><summary>공개 영상과 네 표현의 장면 비교 펼치기</summary>'''+gallery+'''</details>
'''+''.join(detail)+f'<p>갱신 {now} · <a href="AUDIT.json">전체 측정 근거</a> · <a href="RECOUNT.json">별도 코드 검산</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p></main></html>'
(U/'RESULT_KO.html').write_text(page,'utf-8')
result={
'status':'PUBLIC_AUDIT_COMPLETE_PLAN_REVIEW_BEFORE_TRAINING','updated_kst':now,
'unit':'ops/word100_mind_public_audit_20261017','public_train_clips':36,'public_words':4,
'visual_sample_frames':360,'followup_detail_frames':32,'new_training':0,'new_inference':0,'new_detector_runs':0,
'user_observation_reads':0,'production_changes':0,'user_diagnostic_summary_reused':True,
'annotation_matches':36,'mind_whole_context_core_coverage':1.0,
'new_causal_root_cause_proven':False,'sensor_overlap_issue':{'clip_id':'NIA_SL_WORD0500_REAL06_F','left_palm5_observed_frames':10,'core_frames':34,'longest_missing_frames':14,'nominal_missing_seconds':14/30,'masked_left_frames':19,'native_left_frames':27},
'stored_dev':{w:s['stored_dev'] for w,s in audit['summary'].items()},
'verification':str(U/'RECOUNT.json'),'operator':'Codex root; separate-code recount, no other agent/expert review',
'protected_checkpoints':{str(p):h for p,h in protect.items()},
'report':str(U/'RESULT_KO.html'),'plan':str(U/'RESULT_KO.txt'),
'next_single_action':'사용자와 학습 전 계획 검토. 승인 후 전체100 공개 자료의 촬영간격/순서 취약성 진단과 입력 계약부터 진행. 현재 요청에서는 학습 시작하지 않음.',
'no_active_execution':'Audit exec66860 exit0; recount exit0; no training/extraction/server process launched',
'limitations':['Selected by previously reported confusion; no blind causal attribution','DEV3 people reused across views/seeds','RGB review sampled, not language expert certification','Improvement not demonstrated'] }
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
statep=LAB/'ops/STATE.json';state=read(statep)
state['updated_date_kst']=now;state['current_stage']='WORD100_MIND_PUBLIC_AUDIT_COMPLETE_PLAN_REVIEW'
state['next_single_action']=result['next_single_action']
state['word100_mind_public_audit_20261017']=result
statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
context=REPO/'WordsInSilence_PROJECT_CONTEXT.md';ctx=context.read_text('utf-8')
s8="""### 2026-10-17 마음에 들다 공개 자료 감사 완료 — 학습 전 계획
공개 TRAIN4표현×9=36영상 원주석·저장 native/masked/sensor 입력 감사. 원주석36/manifest일치, 표본9개씩동일. 마음에들다9개 whole/context 주석시간100%보존(관절정확도100%아님). 핵심길이중앙 마음1.22초/수어1.32/가평센터2.32/강진센터2.78. 공개360표본+32정밀장면에서 일부 비슷한 손·얼굴/양손접근 장면과 전체순서차이를 관찰; 어휘분해전문가인증·사용자오인식 인과확정 아님.
TRAIN06 마음 sensor왼손 palm5관측10/34(masked19/native27), 겹침 후반14프레임약0.47초누락잔존. 강진TRAIN07 sensor끝2프레임누락은 손내림마무리장면으로관찰. 저장DEV3명×2보기×3seed: 마음/수어/가평 각18/18,강진17/18(72독립시험아님). 재채점0.
코드상30fps학습/가변시각웹 간격비표준화, 마지막시간평균은 점검가설. 공개100간격·순서취약진단→조건부시간처리단일후보→별도순서보존후보→각3seed회귀검증 계획. 과거속도증강미채택반복금지. Codex 별도코드36주석/영상·255증거hash·216구간/216관측·72순위검산PASS,다른에이전트검토아님. 새학습/분류추론/검출/사용자관측읽기/웹변경0. C/M0해시보존.
근거 research/front_keyphase_v1/ops/word100_mind_public_audit_20261017/RESULT_KO.html,RESULT_KO.txt,AUDIT.json,RECOUNT.json.

"""
s9="""### 2026-10-17 공개 자료 진단·공통 보완 계획 검토 — 학습 미승인
최신요청은 마음에들다와 혼동된 공개 표현 비교 및 학습 전 계획보고. 완료근거 ops/word100_mind_public_audit_20261017. 다음한작업은 사용자와 계획확인이며 자동학습하지않는다. 이후승인시 공개100단어의 고정C 촬영간격(동일수행시간)/전체순서 취약성 진단·입력계약부터 진행. 계수/후보는 사용자자료가 아닌 공개TRAIN내부검증으로 결정하고 기존DEV는반복개발집합으로표기. 필요한경우 시간표준화와 구간순서보존을별도단일후보로3seed비교; 두 기본보기 평균Top1/5비회귀·지정취약조건2seed이상개선·중대집단회귀없음 제안은실행전고정. 속도보강재탐색/마음단어특례/센터명제외/미관측손임의복원금지. 기존C/M0/DTW보존, 사용자학습·튜닝·후보선택금지, 최종일반화는별도고정후평가.

"""
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
h9='## 9. 다음 단일 작업'
assert h8 in ctx and h9 in ctx
ctx=ctx.replace(h8,h8+'\n\n'+s8,1).replace(h9,h9+'\n\n'+s9,1)
context.write_text(ctx,'utf-8')
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {now}

## 현재 목표·마지막 완료
사용자 최신 요청: 마음에들다 혼동을 공개학습자료·혼동단어와 비교하고 공통 보완안을 학습 전에 보고. 이번 단위 완료, 학습은 시작하지 않음.
단위 research/front_keyphase_v1/ops/word100_mind_public_audit_20261017. RESULT_KO.html/RESULT_KO.txt/RESULT.json 참조.

## 확인 근거
공개 TRAIN4표현×9=36영상·원주석·저장3보기 감사. 주석36일치·표본균등. 마음에들다9개 주석시간 whole/context100%보존(손 관측완전성 뜻 아님).
360표본+32정밀장면 관찰에서 센터명 일부와 마음/수어의 부분 유사성, 전체순서차이. 전문가수어분해·사용자실패원인확정 아님.
마음TRAIN06 sensor에서 겹침후반 왼손14프레임약0.47초누락; 손바닥5점10/34(masked19/native27),오른손34/34.
강진TRAIN07끝2주석프레임누락은마무리내림장면. 저장DEV 마음/수어/가평 각18/18·강진17/18=같은3명×2보기×3seed,72독립영상아님.
별도recount.py:36주석/영상hash·255핀·216구간·216손관측·72저장순위PASS. 동일Codex담당 별도코드이며 다른에이전트/전문가검증아님.
새 학습/추론/검출/사용자관측읽기/웹변경0. 기존 사용자진단 요약만 읽음. 새성능개선 수치 없음.

## 다음 한 작업
사용자와 학습 전 계획 검토 후 승인 범위에서 진행. 현재 요청으로 자동학습금지.
계획: 공개100단어 고정C 간격·순서취약진단→필요할때 공통시간처리후보→별도 앞/중간/뒤 순서보존후보→동일3seed비교.
시작·끝 임의추가절단/빈손복원/센터클래스제외/마음점수상향/기존속도보강계수재탐색금지.
학습899/개발300/보류14~18 분리. 사용자·멘토 학습/튜닝/후보선택금지. 개발자료 반복사용을맹검최종시험처럼보고하지않음.
운영기본교체·카메라·다운로드·외부자료전송·Git없음. 감사exec66860종료0/recount종료0, 이단위실행중작업없음.

## 유지된 웹과 모델
직전완료: ops/word100_role_dtw_20261017의필요손기반DTW 적용. 멀다/공항/별requiredR,다른9연습단어양손유지.
사용자13회DTW계산가능10→13,양손10수치·TCN순위유지. 의미정오답검증아님.
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0유지.
마지막확인서버 PID38708 일반실행8765;이번턴서버·카메라점검/재시작없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab;M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 해시보존확인.
DTW ROLE_BANK SHA27da98d31469e70df8fc9172b3310d39d3867a6bd0e7e488cd1bd884d03717cb/config/word100_role_dtw.json.
이전첫12회 M0 8/12→C11/12는사용자1명탐색평가. 마음첫C8위,재시도2위. 사용자수행정오답/정확동작원인미확정.
st-gcn 자동화삭제상태유지.
"""
(REPO/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
# Final, narrow artifact validation.
assert read(statep)['word100_mind_public_audit_20261017']['new_training']==0
for p in ['RESULT_KO.html','RESULT_KO.txt','RESULT.json','AUDIT.json','RECOUNT.json','FOLLOWUP_DETAIL.json']:
    assert (U/p).is_file()
for word in audit['summary']:assert (U/f'{word}_CORE_TRAIN9.jpg').is_file()
assert 'word100_mind_public_audit_20261017' in (REPO/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
print(json.dumps({'status':'REPORT_AND_STATE_SYNCED','time':now,'report':str(U/'RESULT_KO.html'),'protected_models':len(protect)},ensure_ascii=False))

