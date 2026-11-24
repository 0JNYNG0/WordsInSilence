"""Record completed manual visual audit; never execute an assigner or model."""
import json, html
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
U=Path(__file__).resolve().parent
def read(name):return json.loads((U/name).read_text('utf-8-sig'))
def save(name,obj):
 with (U/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

events=read('EVENTS.json');draft=read('VISUAL_DRAFT.json');dec=read('DETAIL_DECISIONS_DRAFT.json')
add={
292:('normal_identity',1,'위쪽 손바닥은 오른 전완과 이어지고 아래쪽 왼손과 구분된다. 접촉 중이어도 R 배정 유지가 정상.'),
293:('uncertain',None,'위쪽 오른 주먹과 아래쪽 왼 주먹이 겹친다. 해당 가설의 손가락점은 위쪽, 손목점은 아래쪽 손에 걸쳐 있어 한 손의 정상 21점인지 단정 불가. 단순 좌우 오류로 강제 분류하지 않음.'),
294:('error_identity',1,'위쪽 오른손의 손등·손가락에 R 후보와 L 후보가 중복된다. 아래쪽 실제 왼손의 팔 연결·손바닥 방향과 L 가설이 맞지 않음. f95~103 대조.'),
295:('uncertain',None,'위쪽 오른 주먹과 아래쪽 왼 손바닥의 겹침·펴짐 전환. L 가설의 점이 양쪽 손에 걸쳐 있어 왼손 관측인지 중복/혼합 가설인지 확정 보류.'),
309:('error_identity',1,'올라오는 실제 오른손 하나에 두 raw 후보를 R과 L로 중복 배정. 실제 왼손은 아래에 내려져 있음. f36~44 팔 연결로 확인.'),
312:('error_identity',0,'양손을 올리는 초입에서 실제 왼손을 R 배정. 최근 R 추적과는 멀리 떨어진 반대 손이며 팔 교차 없음. f22~30 대조.'),
409:('error_identity',1,'정지 자세의 실제 오른손 경계 검출을 L 배정. 손을 교차하지 않았고 전완 연결이 명확. f0~8 확인.')}
for k,(v,s,b) in add.items():
 assert str(k) not in dec
 dec[str(k)]={'verdict':v,'actual_side':s,'basis':b}
assert set(map(int,dec))==set(draft['needs_detail'])
assert draft['reviewed_ids']==list(range(1,457)) and draft['reviewed_pages']==list(range(1,39))
duplicate={89,91,294,309};jump={23,312}
records=[]
for e in events:
 d=dec.get(str(e['id']),{'verdict':'normal_identity','actual_side':e['side'],'basis':'전수 시트에서 해당 관측 손과 전완의 연결을 확인. 배정된 해부학적 좌우와 일치. 손가락 각 점의 정확도 판정은 아님.'})
 cat='normal'
 if d['verdict']=='error_identity':cat='duplicate_same_hand' if e['id'] in duplicate else 'opposite_track_takeover' if e['id'] in jump else 'wrong_first_entry'
 if d['verdict']=='uncertain':cat='overlap_mixed_landmarks'
 records.append({**{k:e[k] for k in ('id','clip_id','word','view','index','frame_id','raw_index','side','first_assignment_for_side','label_score')},**d,'category':cat,'sheet':f"sheets/{(e['id']-1)//12+1:02}.jpg",'panel':f"frames/{e['id']:03}.jpg",'detail':f"details/{e['id']:03}.jpg" if str(e['id']) in dec else None,'reviewer':'Codex visual inspection'})
save('VISUAL_REVIEW.json',{'status':'COMPLETE_WITH_EXPLICIT_UNCERTAINTY','reviewed_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'reviewed_events':456,'reviewed_sheets':38,'detailed_cases':23,'records':records,'correction':'ID36 preliminary L reading corrected to saved R; final normal Right. Original draft preserved.','limits':['Single operator visual anatomical identity audit, not independent human labels.','Normal identity does not certify finger landmarks.','Original pixels outside crop used only to establish offline anatomical truth.','No classifier accuracy measurement; known public TRAIN subset.']})
counts=Counter(x['verdict'] for x in records);cats=Counter(x['category'] for x in records)
summary={'status':'COMPLETE_AWAITING_REVIEW','events':456,'clips':len({x['clip_id'] for x in records}),'words':len({x['word'] for x in records}),'counts':dict(counts),'percent':{k:100*v/456 for k,v in counts.items()},'categories':dict(cats),'by_view':{v:dict(Counter(x['verdict'] for x in records if x['view']==v)) for v in ('sensor','masked')},'by_first_entry':{str(v):dict(Counter(x['verdict'] for x in records if x['first_assignment_for_side']==v)) for v in (True,False)},'error_ids':[x['id'] for x in records if x['verdict']=='error_identity'],'uncertain_ids':[x['id'] for x in records if x['verdict']=='uncertain'],'error_clips':len({x['clip_id'] for x in records if x['verdict']=='error_identity'}),'error_words':len({x['word'] for x in records if x['verdict']=='error_identity'}),'error_score_range':[min(x['label_score'] for x in records if x['verdict']=='error_identity'),max(x['label_score'] for x in records if x['verdict']=='error_identity')],'assignment_code_changes':False,'new_training':0,'new_classifier_inference':0,'production_changed':False,'user_mentor_dev_heldout_reads':0,'implementation':'Codex local audit tooling + visual review; independent arithmetic/source/hash checker separately.'}
save('SUMMARY.json',summary)
plan={'status':'PROPOSAL_ONLY_NOT_IMPLEMENTED','goal':'정상 교차/관측을 보존하면서 라벨 오배정의 확정·전파를 줄이는 공통 후보','items':[
 {'name':'관측과 좌우 확정의 분리','policy':'처음 들어온 손의 좌표는 보존하되 라벨만으로 붙인 좌우는 임시 상태로 기록. 화면 안 어깨·팔꿈치·전완/손 연결 또는 이미 독립 확인된 추적이 있을 때 확정. 라벨 단독 456건 일괄 삭제 금지.','measured_examples':[76,143,200,207,264,272,409]},
 {'name':'추적 근거의 출처 유지','policy':'라벨만으로 시작한 추적이 다음 프레임에서 가깝다는 이유만으로 독립 확인됨으로 승격되지 않게 한다. 기존 확정 손과 멀리 떨어진 반대 후보가 나타나면 정체를 즉시 덮어쓰지 않고 재획득 상태로 둔다.','measured_examples':[23,312]},
 {'name':'중복 가설과 실제 교차 분리','policy':'손목 거리 하나로 판단하지 않고 손 전체 가설의 겹침/형상과 팔 연결·독립 추적을 함께 검토. 같은 실제 손을 양쪽 슬롯에 채우지 않는다. 두 실제 손이 포개진 정상 장면은 별도 보호.','measured_examples':[89,91,294,309],'normal_controls':[68,69,184,289,291,292]},
 {'name':'불확실한 손 모양을 정답처럼 쓰지 않기','policy':'좌우가 맞아도 손가락 좌표가 정확한 것은 아니다. 가림으로 양손 점이 섞이면 미확정으로 남기고 관측한 위치와 손 모양 신뢰도를 분리. 숨은 손 모양을 보간 정답으로 생성하지 않는다.','measured_examples':[293,295]}],
 'validation_before_training':['현재 456건 판정을 보존한 진단 집합과 이전 정상 교차315프레임을 회귀 확인. 이 자료로 규칙을 개발하면 독립 테스트라고 부르지 않는다.','공개 TRAIN899 두 검출 조건 전체 재생, 변경된 모든 배정 원본 검토, 강제 반대 배정·정상 교차 회귀·미확정 증가를 함께 집계.','원본 전체 관측/실제 crop 관측 모두 확인. crop 밖 픽셀·미래 프레임을 온라인 좌우 결정에 사용하지 않음.','학습 데이터와 모델 입력을 바꾸는 후속 실험은 결과를 사용자와 확인하고 조건 동결 후 진행. 기존 C 유지.'],
 'not_claimed':['정확도 상승 수치','모든 손가락 관절 복원','완전 가림 해소','전수 영상 전체의 실제 좌우 오류율']}
save('NEXT_PLAN.json',plan)
labels={'normal_identity':'좌우 배정 정상','error_identity':'명확한 좌우 오류','uncertain':'판정 보류'}
lines=['남은 라벨 단독 배정 456건 공개 자료 감사','', '결론: 441건 정상 / 13건 명확한 오류 / 2건 불확실. 모두 버리거나 화면의 좌우 반으로 재배정하면 정상 관측과 교차를 손상시킨다.',
 '456건은 한 프레임의 raw 손 후보 1개 단위다. 영상/사람 수가 아니며 353개 공개 TRAIN 영상, 100개 단어에서 나왔다. 전체 배정에 대한 오류율 또는 수어 인식률이 아니다.',
 '38개 전수 시트의 456장면을 시각 검토하고 애매한 23건은 원본과 앞뒤 장면을 추가 대조했다. 독립 인물 정답표는 없으며 동일 Codex의 시각 판정이다.', '',
 '오류 유형: 첫 등장 반대 배정 7건, 기존 추적에서 반대 손으로 넘어감 2건, 같은 실제 손의 중복 가설 4건. 오류 후보의 라벨 점수가 높은 경우도 있어 점수 하나로 걸러내면 안 된다.',
 '겹침 2건은 손목·손가락 점이 양쪽 손에 걸쳐 있어 확정 보류. 좌우 정상 441건도 모든 손가락 점이 정확하다는 뜻은 아니다.', '',
 '공통 보완안 (제안이며 미구현):']
for p in plan['items']:lines.append(p['name']+': '+p['policy'])
lines+=['','검증 후 학습: 위 규칙 후보를 격리해 공개 TRAIN899 전체 재생 → 현재 오류 및 정상 교차 보존 확인 → 변경 장면 원본 검토 → 사용자 결과 확인. 성능 개선은 후속 고정 비교로 측정해야 한다.',
 '기존 C/TCN/DTW/웹 및 원본 불변. 학습/추론0. 사용자·멘토·DEV·보류 자료 사용0.',
 '수치·대상·해시는 INDEPENDENT.json 별도 검산. 시각 판정과 독립 집계 검산을 구분한다.','', '오류와 판정 보류 상세:']
for r in records:
 if r['verdict']!='normal_identity':lines.append(f"#{r['id']} {r['word']} {r['clip_id']} {r['view']} f{r['frame_id']} {labels[r['verdict']]}: {r['basis']}")
with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
e=html.escape
cards=[]
for r in records:
 kind=r['verdict'];path=r['detail'] or r['panel'];side=['왼손','오른손'][r['side']]
 cards.append(f'''<details class="case {kind}" data-kind="{kind}"><summary><b>#{r['id']:03} {e(r['word'])}</b> · {labels[kind]} · {side} 배정 <span>{e(r['clip_id'])} / {r['view']} / f{r['frame_id']}</span></summary><p>{e(r['basis'])}</p><a href="{path}"><img loading="lazy" src="{path}" alt="원본과 관측 근거"></a></details>''')
proposals=''.join(f"<li><b>{e(p['name'])}</b><p>{e(p['policy'])}</p></li>" for p in plan['items'])
doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>라벨 단독 456건 감사</title><style>
body{{font:17px/1.65 'Malgun Gothic',sans-serif;background:#f4f6fa;color:#172b42;margin:0}}main{{max-width:1100px;margin:auto;padding:36px 22px}}h1{{font-size:32px;line-height:1.3}}h2{{font-size:23px;margin-top:36px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.stat{{background:white;border-radius:14px;padding:18px 28px;flex:1}}.stat strong{{font-size:37px;display:block}}.good{{color:#17725e}}.bad{{color:#b64236}}.warn{{color:#967015}}section,.case{{background:white;border:1px solid #dde4ee;border-radius:12px;padding:20px;margin:14px 0}}.case{{padding:12px 18px}}summary{{cursor:pointer}}summary span{{display:block;color:#526579;font-size:13px}}img{{display:block;max-width:100%;max-height:1000px;margin:15px auto}}a{{color:#14589c}}button{{padding:10px 16px;border:1px solid #adbacb;background:white;border-radius:7px;cursor:pointer;margin:4px}}li{{margin:16px 0}}.note{{color:#53657b}}.flag{{border-left:5px solid #bd5845}}.bar{{height:22px;background:linear-gradient(to right,#218b73 0% 96.7105%,#ce6053 96.7105% 99.5614%,#c69e39 99.5614%);border-radius:7px;margin:20px 0}}[hidden]{{display:none!important}}</style><main>
<p>공개 학습 자료 · 좌우 배정 원인 분석 · 2026-10-18</p><h1>라벨만으로 배정한 456건,<br>대부분 정상이고 일부 오류는 구분 가능</h1>
<div class="cards"><div class="stat good">좌우 정상<strong>441건</strong>96.71%</div><div class="stat bad">명확한 오류<strong>13건</strong>2.85%</div><div class="stat warn">판정 보류<strong>2건</strong>0.44%</div></div><div class="bar"></div>
<p><b>456건은 영상 456개가 아니라, 한 프레임의 손 후보를 배정한 횟수다.</b> 공개 TRAIN 353개 영상·100단어에서 나온 대상이다. 이 비율은 수어 인식률이나 전체 검출의 오류율이 아니다.</p>
<section><h2>어디서 잘못됐나</h2><ul><li><b>처음 등장한 손의 좌우를 반대로: 7건.</b> 화면 아래 경계에서 손의 일부만 보이고 팔 손목 근거가 약할 때 라벨을 그대로 믿었다.</li><li><b>기존 손을 반대 손으로 바꿔 잡음: 2건.</b> 최근 추적과 떨어진 다른 손인데 라벨에 따라 같은 슬롯으로 들어갔다.</li><li><b>같은 실제 손을 두 손으로 중복 배정: 4건.</b> 겹침이나 왜곡된 검출 가설을 빈 좌우 칸에 채웠다.</li></ul><p>겹침 2건은 손가락과 손목 점이 두 손에 걸쳐 있어 확정하지 않았다. 정상 배정도 손가락 21점의 정확성을 보장하지 않는다.</p></section>
<section><h2>정상 교차는 그대로 살려야 한다</h2><p>예를 들어 #289는 왼팔이 몸 중앙을 넘어간 <b>정상 왼손</b>이다. 화면의 좌우 반으로 이름을 붙이면 오히려 틀린다. #68·69·184·291·292도 접촉·교차 중 정상 배정이다.</p><a href="details/289.jpg"><img src="details/289.jpg" alt="정상 왼손 교차를 보존해야 하는 사례"></a></section>
<section><h2>다음 공통 보완안</h2><ol>{proposals}</ol><p class="note">아직 적용하거나 학습하지 않은 제안이다. 현재 456건과 이전 정상 교차315프레임을 보존한 뒤 TRAIN899 전체 재생에서 회귀와 보류 증가까지 확인할 계획이다. 실제 인식률 상승은 이후 학습 비교가 필요하다.</p></section>
<section><h2>확인 범위와 한계</h2><p>456장면 전부를 검토했고, 애매한23건은 앞뒤 원본 장면을 추가 확인했다. 원본 영상의 잘린 영역 밖 부분은 오프라인 해부학적 판정에만 사용했다. 실제 웹캠에서 보이지 않는 픽셀을 사용할 수 있다고 가정하지 않는다.</p><p>시각 판정은 Codex 한 담당자의 검토이며 독립 사람 정답표가 아니다. 별도 검산 프로그램은 대상·집계·파일 보존을 확인한다. 현재 C·웹·DTW는 보존했고, 학습·추론 및 사용자 자료 사용은 없었다.</p><a href="RESULT_KO.txt">텍스트 보고서</a> · <a href="SUMMARY.json">집계 JSON</a> · <a href="INDEPENDENT.json">독립 계산 검산</a></section>
<h2>모든 판정과 원본 근거</h2><div><button onclick="filter('all')">전체456</button><button onclick="filter('error_identity')">오류13</button><button onclick="filter('uncertain')">보류2</button><button onclick="filter('normal_identity')">정상441</button></div>{''.join(cards)}
<script>function filter(k){{document.querySelectorAll('.case').forEach(x=>{{x.hidden=k!=='all'&&x.dataset.kind!==k;if(!x.hidden&&k!=='all'&&k!=='normal_identity')x.open=true;}});}}filter('error_identity');</script></main></html>'''
with (U/'RESULT_KO.html').open('x',encoding='utf-8') as f:f.write(doc)
print(json.dumps(summary,ensure_ascii=False,indent=2))
