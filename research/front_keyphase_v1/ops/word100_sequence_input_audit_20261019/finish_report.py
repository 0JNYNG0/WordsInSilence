from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,html
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=read(U/'SUMMARY.json');vm=read(U/'VISUAL_MANIFEST.json');vs=read(U/'VISUAL_SCOPE.json');ind=read(U/'INDEPENDENT.json')
assert ind['status']=='PASS' and len(vm['samples'])==108
notes='''두 손을 가슴 앞에서 번갈아 움직이는 흐름; 첫 표본 오른손 누락.
두 손 접근 뒤 얼굴 양옆으로 벌리는 변화.
얼굴 부근과 가슴 앞 동작이 바뀌는 다단계 흐름.
가슴 앞 두 손 접촉 후 좌우로 벌어짐.
양손→입 부근 한손→가슴 앞 동작; 내려놓은 반대손 미관측은 구분 필요.
머리 옆 동작 뒤 가슴 앞 두손 접촉; 머리 옆 실제 손이 있어도 한 표본 관절 없음.
초반 두손 분리 움직임→접근→한손 전방 이동이 표본에 남음.
입 부근 한손 동작 뒤 앞쪽으로 이동.
얼굴 양옆→가슴 앞→얼굴 양옆으로 손 위치·형태 변화.
양손 교차 뒤 좌우로 펴지는 흐름; 중간 한손 누락 표본.
팔 앞 접촉 뒤 양손을 올리는 흐름; 전환 표본 양손 관절 없음.
목 부근 한손 뒤 가슴 앞 두손 접촉.
머리 옆→입 부근→가슴 앞 두손 단계.
입 부근 동작 뒤 다른 팔 앞에서 동작.
머리 옆 동작 뒤 가슴 앞 아래로 이동.
눈 부근 동작 뒤 가슴 앞 한손 유지.
입 부근 동작 뒤 허리 앞 양손 움직임.
입 부근 동작 뒤 목 앞 한손 변화.
얼굴 부근의 손 형태·위치가 변한 뒤 가슴 앞으로 내려옴.
얼굴·가슴 사이 두손의 비교적 좁은 위치 변화.
얼굴·가슴 사이 두손의 비교적 좁은 위치 변화.
얼굴 부근 한손→가슴 앞 두손 접근.
가슴 앞 두손 접촉→분리·내림.
입 부근 한손→두손 접근→손 위치 교대.
입 부근 한손→가슴 앞 두손 접촉.
가슴 앞 양손→허리 부근→얼굴 양옆; 전환 표본 관절 누락.
가슴 앞 손 접촉·형태 변화 뒤 내림; 겹친 손의 세부 정확도는 별도.
가슴 앞 손 접촉 후 한손을 내리는 변화.
얼굴 옆 한손의 반복·작은 위치 변화.
입 부근→가슴 앞 한손→내림; 마지막 표본 잘림은 손 내린 뒤.
초반 양손→어깨 부근 한손→허리 옆 한손 단계.
팔 앞 두손 접촉·반복 뒤 내림.
가슴 앞 손 접촉·유지 뒤 내림.
가슴 앞 양손이 번갈아 올라오는 흐름.
가슴 앞 손 접촉·반복 뒤 내림.
가슴 앞→입 부근→양손 분리 단계.
턱·가슴 사이 양손 형태·위치 변화.
얼굴 옆 한손의 높이·형태 변화.
좌우 손이 번갈아 얼굴·가슴 부근으로 이동.
팔 앞 접촉 중 가린 손 관절 일부 없음; 뒤에서 두손 관절 확보.
입 부근→얼굴 앞 양손 교대→위쪽 한손 단계.
초반 양손→어깨 앞→허리 옆; 마지막 표본 손 내린 뒤 잘림.
초반 양손→어깨 앞→허리 옆 단계.
얼굴 옆·입 부근 한손→가슴 앞 두손 접촉.
가슴 앞 양손→입 부근 한손→가슴 앞 두손 접촉.
가슴 앞 양손→어깨 앞→허리 옆; 끝의 정지 자세 잘림.
가슴 앞 양손→입 부근 한손→가슴 앞 두손 접촉.
초반 가슴 앞 양손→입 부근 한손→두손 접촉; 첫 표본 한손 누락.
얼굴 옆 한손→가슴 앞 양손→입 부근→두손 접촉.
얼굴 옆 한손→가슴 앞→얼굴 옆 양손→내림.
가슴 앞 양손→어깨 앞→허리 옆→내림.
손을 올린 뒤 가슴 앞 양손 형태 변화와 내림.
양손→입 부근 한손→가슴 앞 두손; 처음 미검출·끝 잘림은 자세 전환 부분.
가슴 앞 두손 형태 변화 뒤 내림.
가슴 앞 양손→입 부근 한손→두손 접촉; 처음 양손 미검출.
가슴 앞 양손→어깨 앞 한손→허리 옆→내림.
얼굴 옆·가슴 앞 변화→입 부근 한손→가슴 앞 두손 접촉.
가슴 앞 양손→입 부근 한손→두손 접촉.
가슴 앞→얼굴 옆 한손→가슴 앞 양손 형태 변화.
가슴 앞→입 부근 한손→가슴 앞 두손 접촉.
얼굴 옆 한손→가슴 앞 양손 형태 변화.
가슴 앞 두손 접근 뒤 아래로 이동·유지.
팔 앞 두손 접촉→이마 부근 한손 단계.
눈·입 부근 한손→가슴 앞 두손 접촉.
바깥쪽 한손→입·얼굴 부근→가슴 앞 전방 이동.
양손 바깥 위치→가슴 앞 한손→바깥쪽 한손.
가슴 앞 한손의 형태·방향 변화.
입 부근 한손→가슴 앞 한손 변화.
가슴 앞 두손의 형태·작은 움직임; 정지 표본만으로 반복 횟수 판정 불가.
입 부근 한손의 방향·형태 변화.
가슴 앞 두손의 연속적인 형태 변화.
입 부근 한손→허리 앞→가슴 앞 두손 형태 변화.
가슴 앞 양손→얼굴 옆·입 부근 한손→두손 접촉.
가슴 앞 양손을 올려 형태 유지 후 내림; 마지막 표본은 정지 자세 잘림.
한손을 얼굴 위쪽까지 올린 뒤 내림; 상단 동작이 화면·관절에 남음.
가슴 앞 양손 교대 움직임; 시작·마무리 손 관측이 적음.
한손이 얼굴 위·앞을 지나 두손이 가까워짐; 처음 오른손 누락.
한손 형태 유지 후 내림; 처음·끝 정지 자세 잘림.
한손 손가락 형태가 단계적으로 변함; 끝 손 내린 뒤 잘림.
한손 형태를 얼굴·가슴 앞에서 유지한 뒤 내림.
한손 손가락 형태가 단계적으로 변한 뒤 내림.
한손 손가락 형태가 단계적으로 변한 뒤 내림.
한손 손가락 형태가 단계적으로 변함; 처음 정지 자세 잘림.
가슴 앞 한손 형태 유지 후 내림.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태가 순차적으로 변함.
가슴 앞 한손 형태·방향 변화.
가슴 앞 한손 형태·방향 변화.
가슴 앞 한손 형태·방향 변화.
가슴 앞 한손 형태·방향 변화.
가슴 앞 양손의 손가락 형태 변화.
가슴 앞 양손의 손가락 형태 변화.
한손을 먼저 올린 뒤 다른 손을 올려 양손 형태 변화.'''.splitlines()
assert len(notes)==100
review=[]
for i,(sample,note) in enumerate(zip(vm['samples'][:100],notes)):
    assert not sample['risk']
    review.append({'clip_id':sample['clip_id'],'label':sample['label'],'sheet':f'visuals/reference_{i//5+1:02}.jpg','row':i%5+1,'note':note})
risknotes=[
 '앞2프레임은 손을 들기 전, 뒤11프레임은 내린 뒤. 팔 앞 접촉 단계는 관절이 남음. 들어올리는 일부 표본에서 한손 관절 누락.',
 '입→가슴 앞 동작이 남으며 뒤11프레임은 손 내린 뒤.',
 '처음 준비·끝 정지 자세가 잘림. 초반 손이 위로 들어오는 동안 화면 안 손이 있어도 양손 관절이 간헐적으로 없음; 얼굴·어깨·허리 단계는 남음.',
 '얼굴 앞 두손 단계 후 내림. 뒤10프레임은 손 내린 뒤.',
 '가슴·어깨·허리 단계 후 내림. 뒤10프레임은 손 내린 뒤.',
 '얼굴 앞 두손 단계 후 내림. 끝의 긴 무손 관측은 손 내린 뒤.',
 '1.60~1.80초 머리 옆에 오른손이 실제 보이지만 손 관절 없음. 프레임은 입력 내부에 유지; 이후 재검출. 가슴 앞 양손 단계는 남음.',
 '2.17~2.37초 머리 옆에 오른손이 실제 보이지만 손 관절 없음. 프레임은 입력 내부에 유지. 끝 접촉·내림 때 오른손 관절 누락도 있음.'
]
rr=[{**r,'sheet':f'visuals/risk_{i+1:02}.jpg','note':risknotes[i]} for i,r in enumerate(vs['risk'])]
write(U/'VISUAL_REVIEW.json',{'status':'REVIEW_COMPLETE_WITH_RESIDUAL_SENSOR_UNCERTAINTY','method':'Codex visual inspection of all20reference sheets and8risk sheets; sampled frames, not continuous video or sign-language expert semantic annotation.',
 'reference_words':100,'reference_frames':800,'risk_samples':8,'risk_frames':sum(len(a['frames']) for a in vm['samples'][100:]),
 'reference_reviews':review,'risk_reviews':rr,'source_hashes_verified':len(vm['sources']),
 'conclusion':'Sampled macro stages and chronology remain. No new clear removal of an essential macro-stage was observed in these sampled risk scenes. Screen-in hand detection gaps remain; exact finger configuration and all899clip semantic completeness NOT certified.',
 'old_gate_reused':'word100_body_boundary_20261017/READINESS_VISUAL_REVIEW.json with its sampling and sensor-loss limitations; not silently upgraded to all-frame semantic ground truth.'})
plan=read(U.parent/'word100_believe_rearing_sequence_20261019/NEXT_PLAN.json')
plan.update({'status':'UPDATED_PROPOSAL_AFTER_100WORD_AUDIT_WAIT_REVIEW','next_single_step':'用户确认诊断后，可冻结具体训练合同并验证F-G16初始等价/冻结/掩码/时间，再按同条件3seed比较。当前不实现候选、不训练。',
 'next_single_step_ko':'이번 결과를 함께 확인한 후, 순서 보조 후보 F-G16의 초기 동등성·동결·시간/미검출 처리를 검증하고 같은 조건 3회 비교를 진행하는 방안. 이번에는 후보 구현·추가 학습 없음.',
 'audit_decision':'전처리로 중간 순서가 삭제되는 광범위한 문제는 확인되지 않음. 단계 흐름은 대체로 남아 있으므로 순서 보조 실험의 근거는 있음. 미검출 순간의 손 모양은 복원되지 않으며, 인식률 향상·완수 판정을 보장하지 않음.',
 'input_policy':'280채널·현재C입력·시간·미검출 마스크를 그대로 유지하고 손을 추정해서 채우거나 미검출 프레임을 삭제하지 않음. 검출기/ROI/정규화 변경과 순서 보조 학습을 한 실험에 섞지 않음.',
 'evaluation_addition':'정상100단어 정확도/상위5/사람별·미검출별 회귀가 우선. 순서변형·절반·반복 결과는 보조 민감도이며 낮은 유지율 자체를 성능 향상으로 선정하지 않음. 변형의 의미 정오에는 전문가 검증 필요.',
 'public_test_policy':'같은TRAIN899/DEV300·3seed 사용. 현재 진단은TRAIN 단일 고정seed이며 새로운 일반화 성적 아님. ID14~16은 결과 선택에 사용하지 않고 최종후보 고정 후 별도평가 승인 시만 사용.'})
plan['next_single_step']=plan['next_single_step_ko'];write(U/'NEXT_PLAN.json',plan)
conclusion={'status':'DIAGNOSIS_COMPLETE_NO_TRAINING','scope':'100word899TRAIN, ID01~09, 2views, fixed seed20261013 C/T3F, no user recordings',
 'findings':['Order/partial retention is not limited to 믿다/사육장. Sensor T3F reverse retention648/899, all samples retained in46/100words, majority in72/100words.',
 'C reverse retention888/899 versusT3F648/899; T3F responds more to altered order. Not an error-rate or generalization improvement.',
 'Sensor T3F prefix580/899,suffix544/899,double899/899 retain original top1. Partial recognition and repeat label retention can be appropriate for classification, insufficient as correctness/completeness proof.',
 'Frame chronology preserved for899/899 in both views. Annotation frame coverage sensor42687/42869=99.57545%,masked42803/42869=99.84604%. This is time coverage, not joint/semantic accuracy.',
 'Sensor145/899have at least one annotation frame with no hand input; four >=0.3s cases are reviewed terminal lowering/standing. Internal screen-in gaps remain e.g후원 REAL01/04.',
 'Masked31clips535coreframes keep1~3 partial left-hand points globally without local hand scale; corrected evidence shows no wrist-MCP pairs, not missing wrist. No complete hand was discarded by this local-mask count.'],
 'preserved':{'C':True,'T3F':True,'web':True,'DTW':True,'user_recordings_read':0,'training_runs':0,'new_detector':0},
 'limits':['No sign-language expert step annotations.8frame reference sheets cannot certify every distinguishing finger movement.','Single seed TRAIN diagnostic, not fresh test accuracy.','Synthetic block reversal creates discontinuities; no automatic invalid-sign labeling.','Actual user believe/rearing confusion cause remains unproven.'],
 'verification':ind}
write(U/'CONCLUSION.json',conclusion)

def metric(v,m,c):return next(x for x in s['metrics'] if (x['view'],x['model'],x['condition'])==(v,m,c))
def cell(v,m,c):
    x=metric(v,m,c);return f"{x['target_top1']}/899 · {x['retention_pct']:.2f}%"
names={'reverse_thirds':'앞·중간·뒤를 뒤·중간·앞으로','prefix_half':'앞 절반만','suffix_half':'뒤 절반만','double_sequence':'전체를 두 번 이어 붙임'}
table=''.join(f'<tr><td>{name}</td><td>{cell("sensor","C",c)}</td><td>{cell("sensor","T3F",c)}</td></tr>' for c,name in names.items())
bars=''
for c in names:
    bars+=f'<div class="barrow"><strong>{names[c]}</strong>'
    for model in ('C','T3F'):
        x=metric('sensor',model,c);bars+=f'<div class="barline"><span>{model}</span><div class="track"><i class="{model}" style="width:{x["retention_pct"]}%"></i></div><b>{x["retention_pct"]:.2f}%</b></div>'
    bars+='</div>'
perrows=''
refmap={x['clip_id']:x for x in review};coderef={x['clip_id'].split('_')[2]:x for x in review}
for w in s['per_word']:
    values=[w['conditions'][f'sensor/T3F/{c}']['target_top1'] for c in names];ref=coderef[w['word_code']]
    perrows+=f'<tr><td>{html.escape(w["label"])}</td><td>{w["clips"]}</td>'+''.join(f'<td>{n}/{w["clips"]}</td>' for n in values)+f'<td><a href="{ref["sheet"]}" target="_blank">장면 {ref["row"]}행</a></td></tr>'
exnames=['사육장','조상','장애인복지관','화상','독재자','특기','가평군수어통역센터']
extable=''
for label in exnames:
    w=next(x for x in s['per_word'] if x['label']==label);k=w['conditions'];extable+=f'<tr><td>{label}</td>'+''.join(f'<td>{k[f"sensor/T3F/{c}"]["target_top1"]}/{w["clips"]}</td>' for c in ('reverse_thirds','prefix_half','suffix_half'))+'</tr>'
gallery=''.join(f'<details><summary>대표 장면 {(i-1)*5+1}–{i*5} / 100</summary><a href="visuals/reference_{i:02}.jpg" target="_blank"><img loading="lazy" src="visuals/reference_{i:02}.jpg" alt="공개 대표 5단어의 시간순 원본과 저장관절" /></a></details>' for i in range(1,21))
riskhtml=''.join(f'<details><summary>{r["label"]} · {r["clip_id"]}</summary><p>{r["note"]}</p><a href="{r["sheet"]}" target="_blank"><img loading="lazy" src="{r["sheet"]}" alt="입력 위험 사례 원본과 관절" /></a></details>' for r in rr)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100단어 순서·관절 입력 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#18283b;font-family:'Malgun Gothic',sans-serif;line-height:1.7}main{max-width:1120px;margin:auto;padding:36px 28px 80px}header{background:#122b48;color:white;padding:42px;border-radius:22px}h1{font-size:32px;line-height:1.4;margin:10px 0}h2{font-size:24px;margin:0 0 14px}h3{font-size:19px}p{margin:12px 0}.eyebrow{letter-spacing:2px;color:#b5dbfd;font-size:13px}.lead{font-size:20px}section{background:white;border-radius:18px;margin-top:24px;padding:30px;border:1px solid #dce4ef}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{padding:20px;background:#edf4fa;border-radius:14px}.num{font-size:30px;font-weight:bold;color:#154f88}.note{background:#fff4d9;border-left:5px solid #ddaa33;padding:16px}.ok{background:#e8f4ee;border-left:5px solid #418a64;padding:16px}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;border-bottom:1px solid #e1e7ef;padding:12px 9px;vertical-align:top}th{background:#eff4fa}a{color:#145ca0}details{border:1px solid #dce4ef;border-radius:10px;padding:14px;margin:12px 0}summary{cursor:pointer;font-weight:bold}img{width:100%;height:auto;display:block}input{padding:12px;width:min(100%,360px);font-size:16px;border:1px solid #a6b9ca;border-radius:8px;margin:10px 0}.scroll{overflow:auto}.barrow{margin:20px 0}.barline{display:grid;grid-template-columns:50px 1fr 85px;gap:12px;align-items:center;font-size:14px}.track{height:15px;background:#e8eef5;border-radius:10px;overflow:hidden}.track i{height:100%;display:block}.C{background:#8193a8}.T3F{background:#2175bf}.small{font-size:14px;color:#45576c}.flow{display:flex;align-items:center;justify-content:center;gap:18px;flex-wrap:wrap}.step{padding:16px;background:#e9f1fa;border-radius:12px;text-align:center;flex:1}.foot{font-size:13px;margin-top:32px;color:#526579}li{margin:8px 0}@media(max-width:680px){main{padding:16px}header,section{padding:22px}.grid{grid-template-columns:1fr}h1{font-size:26px}.barline{grid-template-columns:40px 1fr 75px}}
</style><main><header><div class="eyebrow">WORDS IN SILENCE · PUBLIC INPUT AUDIT · 2026.10.19</div><h1>동작의 순서가 달라도<br>같은 단어로 판단할까?</h1><p class="lead">100개 단어로 넓혀 확인했습니다.<br>T3-F는 순서를 더 보지만, 동작의 완성도를 보증하지는 않습니다.</p><p>공개 학습 영상 899개 · 고정된 C / T3-F · 추가 학습 0회</p></header>
<section><h2>이번에 확인한 세 가지</h2><div class="grid"><div class="card"><div class="num">648 / 899</div><b>순서를 바꿔도 T3-F의 원래 단어가 1위</b><p>한두 단어에만 나타난 현상이 아닙니다. 46개 단어는 해당 8~9개 영상 모두에서 1위가 유지됐습니다.</p></div><div class="card"><div class="num">99.58%</div><b>주석 구간의 시간 프레임 보존</b><p>화면 범위 검출 입력의 수치입니다. 관절 정확도나 수어 의미 보존율은 아닙니다.</p></div><div class="card"><div class="num">미검출 잔존</div><b>화면 안 손도 잠깐 놓칩니다</b><p>후원처럼 손이 실제 보이지만 관절이 사라진 구간이 남습니다. 시간은 유지해도 손 모양 정보는 비어 있습니다.</p></div></div></section>
<section><h2>1. 어떻게 검사했나</h2><p>학습에 이미 사용한 공개 100단어·899영상(ID01~09)을 대상으로, 기존 고정 모델 한 쌍(seed 20261013)의 반응을 검사했습니다. 전체 화면 검출 뒤 범위를 적용한 입력(masked)과, 제한된 화면에서 다시 검출해 실제 웹캠의 관측 조건을 흉내 낸 입력(sensor)을 각각 확인했습니다.</p><div class="flow"><div class="step">원래 순서<br><b>A → B → C</b></div><span>진단용 변형</span><div class="step">구간 순서 변경<br><b>C → B → A</b></div><div class="step">절반만 / 전체 반복<br><b>앞 절반 · 뒤 절반 · 2회</b></div></div><p>구간은 실제 의미 단계가 아닌 시간 길이의 3등분입니다. 구간 안의 프레임 순서는 유지했고, 시간 간격 채널은 원래대로 두었습니다. 모델 가중치나 입력 규칙은 바꾸지 않았습니다.</p><div class="note"><b>아래 숫자는 정답률이 아닙니다.</b> 변형 뒤에도 원래 단어를 1위로 고른 비율입니다. 짧은 동작만으로 단어를 알아보는 것은 분류에는 타당할 수 있고, 정적·반복 동작은 순서를 바꿔도 의미가 유지될 수 있습니다. 모든 변형을 ‘틀린 수어’로 간주하지 않았습니다.</div></section>
<section><h2>2. 100개 단어에서도 나타났다</h2><p>공개 영상에 웹캠의 관측 조건을 흉내 낸 sensor 입력 기준. 정상 원본에서는 두 모델 모두 899/899를 맞혔지만, 학습 자료를 다시 본 결과이므로 새 사용자 성능으로 제시하지 않습니다.</p>__BARS__<div class="scroll"><table><thead><tr><th>변형</th><th>C: 원래 단어 1위</th><th>T3-F: 원래 단어 1위</th></tr></thead><tbody>__TABLE__</tbody></table></div><p><b>T3-F는 C보다 순서 변화에 더 반응했습니다.</b> 순서 변경 시 원래 단어 유지율은 26.70%p 낮아졌습니다. 원래 단어와 가장 강한 경쟁 단어의 내부 점수 차이도 중앙값 기준 C는 1.31, T3-F는 10.49 줄었습니다. 따라서 ‘순서를 전혀 보지 못한다’는 결론은 맞지 않습니다.</p><p>다만 T3-F도 100개 단어 중 72개에서 과반의 영상이 순서 변경 후 같은 단어 1위를 유지했습니다. 반복 영상을 모두 같은 단어로 분류한 것은 반복을 감지하지 못한다는 직접 증거가 아니며, 이 분류 출력으로 반복 횟수나 완수 여부를 판단할 수 없다는 한계를 보여줍니다.</p><details><summary>다른 관측 방식(masked) 결과도 보기</summary><table><tr><th>변형</th><th>C</th><th>T3-F</th></tr>__MASKED__</table><p>관측 방식에 따라 수치가 달라집니다. 두 관측 방식을 합쳐 1,798개의 독립 영상으로 세지 않았습니다.</p></details></section>
<section><h2>3. 단어마다 의존하는 부분이 다르다</h2><table><thead><tr><th>T3-F / sensor</th><th>세 구간 역순</th><th>앞 절반</th><th>뒤 절반</th></tr></thead><tbody>__EXAMPLES__</tbody></table><p>사육장·조상·장애인복지관처럼 원본에 여러 위치·행동 변화가 있는 경우에도 역순 뒤 원래 단어가 남았습니다. 반대로 독재자·특기 등은 일부 구간만 남겼을 때 결과가 크게 달라졌습니다. 이는 모든 단어에 동일한 ‘몇 초 이상’, ‘몇 단계 이상’ 규칙을 넣으면 안 되는 이유입니다.</p><p>수어통역센터 계열은 앞쪽 지명 표현이 달라도 뒤쪽 공통 동작이 비슷해 보였습니다. 특정 장면 하나의 유사도보다 전체 전환을 확인할 필요가 있습니다. 이 관찰은 전문 수어학적 단계 주석이나 혼동 원인의 확정은 아닙니다.</p></section>
<section><h2>4. 구별 동작은 관절 입력에 남아 있나</h2><p><b>큰 동작의 흐름은 대체로 남아 있지만, 모든 손가락 정보가 온전히 보존되었다고 할 수는 없습니다.</b></p><table><tr><th>확인 항목</th><th>전체 검출 후 범위 적용</th><th>화면 범위에서 검출</th></tr><tr><td>주석 구간 프레임 보존</td><td>42,803 / 42,869 · 99.85%</td><td>42,687 / 42,869 · 99.58%</td></tr><tr><td>유지 구간 내부 프레임 삭제</td><td>0 / 899영상</td><td>0 / 899영상</td></tr><tr><td>주석 중 양손 정보가 1프레임 이상 없는 영상</td><td>239 / 899</td><td>145 / 899</td></tr><tr><td>0.3초 이상 양손 정보가 없는 영상</td><td>0 / 899</td><td>4 / 899</td></tr></table><p>주석 구간에도 준비·내림 자세가 들어 있습니다. 긴 공백 4건을 원본으로 확인했을 때 모두 손을 내린 마무리 구간이었습니다. 이를 ‘필수 동작이 4건 잘렸다’고 해석하지 않습니다. 899개 입력 모두 내부 시간 순서와 원래 시간 간격은 유지되었습니다.</p><p>반면 <b>후원 REAL01의 1.60~1.80초, REAL04의 2.17~2.37초</b>에는 머리 옆 손이 화면 안에 실제로 있는데도 손 관절이 없습니다. 독재자·장애인복지관의 전환 표본에서도 화면 안 손의 누락이 보였습니다. 보조 모델을 붙여도 관측하지 못한 순간의 정확한 손 모양을 자동으로 복구할 수는 없습니다.</p><div class="note">100단어 대표 영상의 8장씩 총 800장과 입력 위험 8사례를 확인했습니다. 모든 899영상의 모든 프레임을 수어 전문가가 의미 단위로 검증한 것은 아닙니다. 좌우 배정·손가락 세부 각도의 정답률도 이번 수치에 포함되지 않습니다.</div><details><summary>손 위치와 손 모양 입력이 다른 경우</summary><p>masked 31영상·535주석 프레임에서는 왼손목 등 1~3점만 남아 손 위치 채널에는 반영되지만, 손바닥 관절과의 거리를 잴 수 없어 손 크기 기준을 만들 수 없었습니다. 손 모양 채널은 미관측으로 유지됩니다. 이는 정상 손 전체를 535번 버린다는 뜻이 아닙니다. 해당 사례를 억지 보간하거나 정상 손 모양으로 채우지 않았습니다.</p></details>__RISKS__</section>
<section><h2>5. 다음 보완 계획</h2><p>이번 결과는 <b>기존 T3-F를 그대로 고정하고, 프레임별 변화를 순서대로 요약하는 작은 보조 부분만 비교해볼 근거</b>가 됩니다. 효과가 입증된 것은 아닙니다.</p><ol><li><b>초기 상태부터 동일함을 검사:</b> 기존 T3-F 가중치·정규화 상태를 고정하고 보조 출력은 0으로 시작합니다. 처음 결과가 현재 모델과 같은지 확인합니다.</li><li><b>순서 보조만 학습:</b> 현재 TCN의 프레임별 특징을 작은 GRU16으로 시간순 처리하는 후보를 제안합니다. 새 학습 변수는 약 4천 개이며, 기존 C·T3-F 및 입력 규칙을 바꾸지 않습니다. 실제 미검출 시간은 유지합니다.</li><li><b>같은 공개 분할로 3회 비교:</b> 학습 899 / 개발 검증 300을 유지해 C·T3-F 저장 결과와 비교합니다. 정상 전체 동작 인식률, 사람별 성능, 미검출 사례의 회귀를 먼저 봅니다. 합성 순서변형 유지율이 낮아졌다는 이유만으로 채택하지 않습니다.</li><li><b>‘단어 인식’과 ‘올바른 수행’ 평가를 구분:</b> 생략·순서 오류·반복의 정오를 판정하려면 전문가가 확인한 정상/오류 자료와 별도 기준이 필요합니다. 정답 단어 라벨만으로 학습한 보조 모델이 이를 해결했다고 주장하지 않습니다.</li></ol><div class="ok"><b>현재는 검증과 계획 정리까지 완료했습니다.</b> 추가 학습 0회. C·T3-F·웹·DTW 유지. 사용자·멘토 촬영 자료는 읽거나 학습·튜닝·선택에 쓰지 않았습니다. 결과를 함께 확인한 다음 학습 단계로 넘어갑니다.</div></section>
<section><h2>100개 단어별 결과와 장면 근거</h2><p>표의 값은 T3-F / sensor에서 원래 단어가 1위인 영상 수입니다. 해당 영상은 대부분 단어당 9개, 1개 단어는 8개입니다. 점수가 나쁜 단어만 골라내지 않고 전체를 제공합니다.</p><input id="query" aria-label="단어 검색" placeholder="단어 검색"><div class="scroll"><table id="words"><thead><tr><th>단어</th><th>영상 수</th><th>역순</th><th>앞 절반</th><th>뒤 절반</th><th>2회 반복</th><th>원본+관절</th></tr></thead><tbody>__WORDS__</tbody></table></div><details><summary>100단어 장면 모음 열기</summary><p>원본 전체 화면을 유지했습니다. 청록선은 입력 하단 범위, 노란 선은 팔, 초록·분홍은 양손 저장 관절입니다. L/R은 촬영자 신체 좌우이며, 숫자는 해당 프레임 입력에 남은 손 관절 수입니다. 그림을 클릭하면 크게 볼 수 있습니다.</p>__GALLERY__</details></section>
<section><h2>검산과 범위</h2><p>저장된 17,980개 출력 벡터(1,798,000개 점수), 1,798개 관측별 입력, 시간·마스크를 별도 계산 코드로 검산했습니다. 같은 Codex가 다른 코드로 수행한 검산이며 별도의 사람·에이전트·수어 전문가 교차검토는 아닙니다. 보호된 입력·모델·웹 근거 4,513개 해시가 유지되었습니다.</p><p class="small">신규 DEV/최종보류 자료 재평가 없음. 이전 두 단어 진단의 180쌍은 저장 결과 재사용. RGB 생성 도중 미보관 영상 경로가 비어 있는 메타데이터 처리 오류를 보존하고 같은 원본 ZIP에서 8위험사례만 이어 생성했습니다. 손목/손크기 진단 초안의 원인 문구는 수치와 모순되어 최종 보고 전에 정정했고 최초 기록을 남겼습니다. 모델·점수 재실행은 없습니다.</p><p><a href="SUMMARY.json">전체 수치</a> · <a href="INDEPENDENT.json">독립 산술 검산</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="LOCAL_ANCHOR_CORRECTION.json">손 크기 근거 정정</a> · <a href="NEXT_PLAN.json">다음 계획</a></p></section><p class="foot">이번 검증은 학습 자료에서의 모델 반응 진단입니다. 수어 수행의 정오 판정, 새로운 사람의 정확도, 실제 사용자 오인식의 단일 원인을 입증한 결과가 아닙니다.</p></main><script>document.getElementById('query').addEventListener('input',e=>{const q=e.target.value.trim();document.querySelectorAll('#words tbody tr').forEach(r=>r.hidden=!r.cells[0].textContent.includes(q));});</script></html>'''
masked=''.join(f'<tr><td>{n}</td><td>{cell("masked","C",c)}</td><td>{cell("masked","T3F",c)}</td></tr>' for c,n in names.items())
for k,v in {'BARS':bars,'TABLE':table,'MASKED':masked,'EXAMPLES':extable,'RISKS':riskhtml,'WORDS':perrows,'GALLERY':gallery}.items():page=page.replace('__'+k+'__',v)
assert '__TABLE__' not in page
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
(U/'RESULT_KO.txt').write_text('공개100단어·899학습영상 순서 및 입력 검증 완료\n\n순서변경 후 원래단어1위: C888/899(98.78%),T3-F648/899(72.08%). 이것은 인식정확도나 오류율이 아닌 민감도 진단이다. T3-F는 순서에 반응하지만 완수 여부는 보증하지 않는다.\nT3-F 앞절반580/899(64.52%),뒤절반544/899(60.51%),전체2회연결899/899.\n입력 내부 시간순서·미검출 프레임 삭제는899개모두없음. 주석시간보존sensor99.58%,masked99.85%. 시간보존은손가락정확도보증아님. 후원 머리옆 실재손 미검출 등잔존. 대표100영상800표본과위험8사례를시각검토;전체899영상의의미·정밀손가락완전성미검증.\n다음제안: 기존T3-F전체고정+작은GRU16순서보조만3seed비교. 초기동등·동결검증과정상100단어성능/미검출회귀우선. 단어라벨만으로필수동작완수판정보장없음. 사용자결과확인전추가학습없음.\n기존C/T3-F/웹/DTW유지. 사용자자료읽기0/학습0/재추출0. 상세근거 RESULT_KO.html, SUMMARY.json, INDEPENDENT.json, VISUAL_REVIEW.json, NEXT_PLAN.json.\n',encoding='utf8')
print('REPORT COMPLETE; candidate training remains unauthorized')
