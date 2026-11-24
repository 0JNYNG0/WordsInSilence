"""Publish already-verified local research results; no experiment execution."""
import json, hashlib, html, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
a=read(U/'ANALYSIS.json');r=read(U/'RESULT.json');v=read(U/'VERIFY.json')
assert v['status']=='PASS' and not (U/'RUNNING.json').exists()
checks={}; normals={}
for method in ('old','new'):
    sources=[s for s in r['sources'] if s['previous'][method]['inside_raw']==0]
    keys={(s['query'],s['kind']) for s in sources}
    pairs=[p for p in r['pairs'] if p['method']==method and (p['query'],p['kind']) in keys]
    c=a['cohorts'][method+'_missed'];assert len(sources)==c['n']
    for factor in ('alignment','range','neighbors'):
        hits=[p for p in pairs if p['factors'][factor]>0]
        computed={'variants':len({(p['query'],p['kind']) for p in hits}),'references':len(hits),'cells':sum(p['factors'][factor] for p in hits)}
        assert computed==c['factors'][factor]
    for k,val in c['source']['totals'].items():assert sum(s[k] for s in sources)==val
    checks[method]={'missed':len(sources),'factors':c['factors']}
    normals[method]={}
    for key in ('pool_1','pool_0','anchor_1','anchor_0'):
        rr=[p['config'][key] for p in r['normal'] if p['method']==method]
        x={k:sum(z[k] for z in rr) for k in ('bad','comparable','unknown','any_bad')}
        x.update(n=len(rr),difference_cell_percent=100*x['bad']/x['comparable'])
        assert x['bad']==sum(g[key]['bad'] for g in a['normal'][method].values())
        assert x['comparable']==sum(g[key]['comparable'] for g in a['normal'][method].values())
        normals[method][key]=x
write(U/'AGGREGATE_CHECK.json',{'at':now(),'status':'PASS','actor':'parent Codex, arithmetic independent of analyze.py','checks':checks,'normal_totals':normals,'pins':{n:sha(U/n) for n in ('RESULT.json','ANALYSIS.json','VERIFY.json')}})
prov=U.parent/'word100_motion_alignment_evidence_20261021'/'candidate.py'
write(U/'PROVENANCE_NOTE.json',{'at':now(),'source':str(prov),'sha256':sha(prov),'meaning':'e.observed=(usable & observed).any(point); at least one preserved usable handpoint in prepared input, not RGB visibility or raw all21 MediaPipe truth. Angle evidence separately requires state4 and finite interval. raw_observed is inherited field name only.','source_note':'Expected-hand counts exclude confirmed rest and unused hand. Counts are hand-frame / angle-frame cells, not independent persons or accuracy.'})
parts=[]
def add(title,text):parts.append((title,text.strip()))
add('이번 결론', '''DTW의 목표에 대한 이해는 맞다. 같은 수어를 빠르게 하거나 천천히 해도 대응하는 순간을 연결한 뒤 손 모양·위치·움직임의 차이를 살펴보는 기능이다. 다만 DTW만으로 모든 수어의 의미상 필수 단계를 이해하거나, 관측되지 않은 동작을 틀렸다고 확정할 수는 없다.
이번에는 새 모델을 학습하지 않고, 짧은 구간을 지웠을 때 차이가 추가되지 않은 이유를 저장된 공개 관절 입력과 시간 연결에서 분리 분석했다. 관절 정보 부족, 여러 사람의 넓은 범위, 앞뒤 장면까지 허용하는 비교가 함께 영향을 줄 수 있었다. 시간 정렬만의 원인이라고 단정할 근거는 얻지 못했다. 기존 C·T3-F·웹은 그대로다.''')
add('여러 사람의 평균 동작과 비교하는가?', '''현재 웹 코드에서는 여러 사람의 좌표를 평균 내서 하나의 가상 동작을 만들지 않는다. 선택한 연습 단어의 공개 참고 영상 각각에 대해 사용자의 동작을 따로 시간 정렬하고 비교한다. 그 결과들 중 비교 근거가 충분한 사람별 차이 수치를 모아 중앙값을 사용한다. 예를 들어 사람별 차이가 4, 5, 6, 7, 30이면 중앙값은 6이다. 극단적인 한 참고가 전체 결과를 끌고 가지 않게 하는 방식이다.
한 참고와 비교할 때에는 관측 가능한 각도·시간별 차이를 평균하는 부분이 있다. 이 시간 평균과 여러 참고 사이의 중앙값은 서로 다르다. 양손 단어는 두 손의 원시 차이를 평균하고, 한 손 단어는 필요한 손만 비교한다. 필요한 손을 못 봤다고 0점이나 정상으로 만들지 않는다.
현재 코드에 연결된 다중 참고 뱅크는 12개 표현×공개 TRAIN 촬영자 9명이다. TCN의 100개 학습 단어 전체와 같은 범위가 아니다. C/T3-F의 내부 가중치와 DTW를 직접 비교하는 것도 아니다. TCN은 단어 후보를 고르고, DTW는 선택한 연습 단어의 참고 관절 자료와 비교한다.
최근의 별도 연구는 평가 대상 한 사람을 제외한 8명의 자료로 시간대·항목별 관측 최솟값~최댓값 범위를 만들었다. 이것은 현재 웹의 중앙값 방식과 다르며 웹에 적용하지 않았다. 이번에 차이가 묻히는지 진단한 ‘여러 사람 범위’는 이 연구 방식이다. 위 웹 설명은 디스크의 코드·설정을 확인한 결과이며 실행 중 서버를 새로 호출한 실측은 아니다.''')
add('이번에 확인한 자료와 시험', '''수어·후원·멀다의 공개 TRAIN 영상 3단어×9명=27개를 사용했다. 각 영상의 핵심 구간 25%·50%·75% 부근에서 관절 입력 3프레임을 지운 81개 변형을 분석했다. 이는 짧은 누락을 시험하기 위한 인위적 변형이다. 지운 부분이 언어적으로 반드시 필요한 수어 동작이라고 확인된 것은 아니다.
각 변형을 다른 8명의 참고와 비교했고, 기존 정렬과 앞서 시험한 손·팔 위치/방향 보조 정렬의 저장 결과를 모두 확인했다. 81×8×2=1,296개 비교 기록에 대해 조건별 계산을 별도 검산했다. 원속도·빠르게·천천히·부분 속도 변화·일시 정지의 정상 대조는 2,160개 비교 기록이다. 2,160명의 사용자나 독립 영상이라는 뜻은 아니다.
새 정렬이나 모델은 실행하지 않았다. 같은 저장된 결과에서 (1) 원래 연결을 유지한 대조, (2) 한 참고만 쓰는 대조, (3) 앞뒤 장면 허용을 빼는 대조를 구성했다. 각 대조는 그 조건의 원본 영상 결과와 비교했고, 양쪽 모두 관측 가능한 곳에서만 추가 차이를 셌다.''')
add('짧은 누락 9건에서 무엇이 확인됐나?', '''앞서 위치·방향 보조 정렬에서 추가 차이를 찾지 못한 9개 변형을 기준으로 보면 다음과 같다. 아래 항목들은 겹칠 수 있으며 9건을 서로 다른 원인으로 나눈 표가 아니다.
• 입력 손 관측 일부가 없는 사례: 2/9. 그러나 9건 모두 팔꿈치·팔 손목 위치 근거는 남아 있었다. 모든 관절 정보가 없어서 발생한 것은 아니다.
• 손가락 각도의 비교 근거가 일부 부족한 사례: 7/9. 그중 후원 REAL02·REAL03의 75% 부근은 손 관측 자체는 6/6 손·프레임에 남았지만 신뢰 가능한 손가락 각도는 0/90이었다. ‘손은 대략 잡혔다’와 ‘손가락 모양을 믿고 비교할 수 있다’는 다르다. 이 두 사례에서는 이번 세 대조로 추가 차이가 드러나지 않아 원인이 완전히 설명되지 않았다.
• 여러 사람의 넓은 범위 대신 한 참고의 범위로 비교하면 추가 차이가 드러난 사례: 6/9. 넓은 범위가 작은 차이를 받아들이는 영향을 확인했다. 그렇다고 한 사람만 기준으로 쓰는 것이 더 정확하다는 뜻은 아니다.
• 대응 장면 앞뒤 한 프레임까지 허용하는 처리를 빼면 추가 차이가 드러난 사례: 7/9. 짧은 변화를 주변 장면이 덮을 수 있었다. 앞뒤 한 프레임의 시간 길이는 추출 간격에 따라 달라진다.
• 삭제 전 연결을 유지하고 지운 순간은 가장 가까운 남은 순간에 대응시키는 대조에서 새로 드러난 사례: 0/9. 이것만으로 시간 정렬이 올바르다고 결론 낼 수는 없다. 이 대조는 사람 간 정답 연결이 아니며, 전체 81개에는 같은 경로 대조에 민감한 변형이 11개 있었다.
또한 9건 모두 삭제 전부터 해당 분석 구간의 일부 참고에서 차이가 있었다. 삭제 후에도 8건은 일부 차이가 남았고, 한 건만 해당 구간에서 8개 참고 모두 차이 표시가 없었다. 그 한 건에도 관측 부족이 있었다. 따라서 ‘새 차이를 놓쳤다’는 말을 ‘수어 전체를 정상이라고 판정했다’로 해석하면 안 된다.''')
add('관측 정보가 남아 있는데도 놓칠 수 있는 이유', '''예를 들어 A라는 수행자는 손을 조금 높게, B는 조금 낮게 움직일 수 있다. 시간마다 두 사람의 최솟값~최댓값을 넓게 묶으면 그 사이의 값도 범위 안에 들어간다. 앞에서는 A, 중간에는 B, 뒤에서는 또 다른 사람의 특징을 따로 허용하면 실제 어느 한 사람이 수행한 연속 흐름과 달라도 부분별로 통과할 여지가 생긴다.
또 대응 순간의 앞뒤 장면까지 묶어 비교하면 정상 속도 차이를 잘 받아들이는 장점이 있다. 그러나 아주 짧은 모양 변화나 경로의 굴곡이 주변 장면에 섞일 수 있다. 이것은 DTW가 영상을 삭제했다는 뜻이 아니다. 시험에서 이미 지운 구간을 주변 장면과 연결·비교했을 때 변화가 작게 계산될 수 있다는 뜻이다.
이번 9개 제거 구간에서는 신뢰 가능한 손가락 각도가 375/720칸, 손바닥 위치가 56/96칸이었다. 손 관측 42/48 손·프레임과 다른 수치다. 준비 입력의 관측 근거이지 RGB에서 실제로 손이 보였는지 또는 모든 관절이 정확한지에 대한 정답률은 아니다. 원본에 수치상 구별되는 변화가 남아 있는 것도 확인했지만, 작은 수치 변화가 의미상 필수 단계라는 뜻은 아니다.''')
n=normals['new'];pct=lambda k:f"{n[k]['difference_cell_percent']:.2f}%"
add('정상 속도 차이와 손내림은 어떻게 지켰나?', f'''정상 대조에서도 같은 조건을 비교했다. 위치·방향 보조 정렬의 1,080개 비교에서 ‘비교 가능한 항목·시간 칸 중 차이가 표시된 비율’은 다중 참고+앞뒤 허용 {pct('pool_1')}, 앞뒤 허용만 제거 {pct('pool_0')}, 한 참고만 사용 {pct('anchor_1')}였다. 기준을 좁히면 빠진 구간이 더 드러나지만 정상 수행자 차이도 훨씬 많이 표시된다. 이 비율은 오답률이나 사용자 오류율이 아니다. 항목마다 분모·관측 가능성이 다를 수 있어 정답 판정 성능으로 비교하지 않는다.
모든 정상 비교에는 이미 어딘가에 작은 수치 차이가 있었으므로 ‘경고가 난 영상 수’만으로는 부작용을 구별할 수 없었다. 차이가 난 시간·항목의 양을 따로 확인했다. 따라서 단일 참고로 축소하거나 앞뒤 허용을 일괄 삭제하는 방법은 채택하지 않았다.
기존 정상 손내림·필요 손·관측 부족 표시는 그대로 보존했고, 손이 안 보인다는 이유로 참고에서 요구하는 비교 시간대를 줄이지 않았다. 손내림이 확인된 시간은 기존 조건대로 제외하되 불확실한 시간은 정상으로 바꾸지 않았다. 이전 135개 누락 대조의 범위 보존 검증은 재사용했다.
새 추적·새 시간 정렬을 실행한 작업이 아니므로 정상 교차 좌표를 수정하지 않았고, 손내림 시간 정렬이 새로 개선됐다고도 주장하지 않는다. 이전 보조 정렬의 손내림→활동 구간 연결 증가 문제는 그대로 남아 있다.''')
add('다음 보완 방향', '''1. 손 위치와 손가락 모양의 근거를 계속 분리한다. 손목을 알아도 손가락을 못 보면 위치는 비교하고 손 모양은 ‘비교 근거 부족’으로 표시한다. 현재 부족한 근거를 사용자 동작 오류로 바꾸지 않는다.
2. 여러 사람의 참고는 각각 연속된 동작으로 보존한다. 짧은 구간의 앞·중간·뒤를 한 참고 안에서 함께 비교한 뒤, 다른 참고들도 같은 흐름을 지지하는지 확인하는 후보를 준비한다. 순간마다 다른 사람의 넓은 범위에 들어가기만 하면 통과시키거나, 가장 차이가 적은 참고 한 명만 고르는 방식은 피한다.
3. 전체 평균 차이와 짧은 구간의 차이·관측 부족을 함께 보고한다. 정상 수행에서도 생기는 차이를 배우기 위해 한 사람씩 참고에서 제외하는 비교를 유지한다. 단순 압축 횟수나 짧은 이동량만으로 오류를 판정하지 않는다.
4. 정상 속도 변화·한 손·손내림·교차·반복 대조와 짧은 제거 대조를 함께 사전 고정한다. 중간 생략과 빠른 수행을 현재 촬영 정보로 구분할 수 없으면 ‘확인 불가’로 남긴다. 현재 3단어 결과로 모든 수어에 대한 효과를 보장하지 않는다. 새 후보도 조건부터 고정하고 공개 자료에서 검증한 뒤 확장한다.
이 내용은 다음 작업 제안이다. 이번 결과를 함께 검토한 다음 진행하며, 추가 TCN 학습이나 운영 DTW 교체는 하지 않았다. 사용자 촬영은 학습·튜닝·후보 선택에 사용하지 않고 접근하지 않았다.''')
add('확인 및 실행 기록', '''부모 Codex가 진단을 구현·실행했고, 별도 에이전트가 81개 원본 구간 요약·1,296개 비교·2,160개 정상 대조를 독립 검산했다. 다른 에이전트가 운영 코드와 연구 방식의 차이 및 결과 해석을 교차 검토했다. 수어 전문가의 언어적 정답 검토나 실사용 정확도 검증은 아니다.
최초 실행은 손 역할 설정에 기본값을 물려주는 부분이 빠져 9개 영상 처리 뒤 중단됐다. 원래 규칙의 기본값(양손)을 그대로 복구했고, 첫 오류와 이미 저장된 432개 결과는 보존·일치 확인했다. 단어별 예외나 실험 조건·점수는 바꾸지 않았다. 복구 실행과 독립 검산은 종료 코드 0으로 완료됐다.
기존 모델·운영 파일 32개 보호 핀과 동결 자료를 확인했다. 추가 학습·모델 추론·새 관절 추출·새 DTW 탐색·사용자/멘토 자료 접근은 0회다. 원본·과거 결과는 읽기 전용으로 유지했다.''')
text='짧은 구간 누락 원인 진단 및 DTW 비교 방식 설명\n\n'+'\n\n'.join(t+'\n'+x for t,x in parts)
with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(text+'\n')
esc=html.escape
body=[]
for i,(t,x) in enumerate(parts):
    body.append('<section><h2>'+esc(t)+'</h2>'+''.join('<p>'+esc(s)+'</p>' for s in x.split('\n')))
    imgs={3:['factor_cohorts'],4:['source_coverage','range_example'],5:['normal_tradeoff']}.get(i,[])
    for img in imgs:body.append(f'<figure><img src="visuals/{img}.png" alt="{esc(img)}"></figure>')
    body.append('</section>')
rows=[]
for x in a['misses']['new']:
    s=x['source'];f=x['factor_references']; person=x['query'].split('_')[-2]
    rows.append('<tr>'+''.join('<td>'+esc(str(y))+'</td>' for y in [x['label'] if 'label'in x else s['label'],person,x['kind'],f"{s['raw_observed_hand_frames']}/{s['expected_hand_frames']}",f"{s['supported_angle_cells']}/{s['expected_angle_cells']}",f['range'],f['neighbors'],f['alignment'],x['references_with_any_difference_after'],x['references_with_unknown']])+'</tr>')
body.append('<section><h2>9개 사례 근거표</h2><p>시간 위치 25/50/75는 원본 핵심 구간 비율이다. 요인 숫자는 추가 차이가 드러난 참고 수(8개 중)이며 중복된다. 마지막 두 열은 해당 짧은 분석 구간에 한정한다.</p><div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['단어','촬영자','제거','손 관측','신뢰 각도','범위 대조','이웃 대조','경로 대조','제거 후 차이 참고','관측부족 참고'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>')
body.append('<section><h2>근거 파일</h2><p>'+ ' · '.join(f'<a href="{n}">{n}</a>' for n in ['ANALYSIS.json','VERIFY.json','AGGREGATE_CHECK.json','RUNTIME_EXPLANATION.txt','INTERPRETATION_REVIEW.json','RECOVERY.json','RESULT_KO.txt'])+'</p></section>')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>짧은 누락 원인과 DTW 비교 방식</title><style>body{margin:0;background:#eef2f6;color:#172838;font-family:"Malgun Gothic",sans-serif;line-height:1.85}main{max-width:1080px;margin:auto;padding:40px 24px}header{padding:36px;border-radius:20px;background:#143d4c;color:white}h1{font-size:32px;line-height:1.4;margin:8px 0}section{margin:24px 0;padding:28px;background:white;border:1px solid #d9e2ea;border-radius:16px}h2{font-size:23px;margin:0 0 16px;color:#164859}p{white-space:pre-line;margin:14px 0}figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid #e3e8eb}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dde5ea;padding:10px;text-align:left;white-space:nowrap}th{background:#ecf4f6}.scroll{overflow:auto}a{color:#076a8a}small{color:#d1e5eb}@media print{body{background:white}section{break-inside:avoid;border:0}main{padding:0}}</style><main><header><small>WordsInSilence · 공개 자료 진단 · 2026.10.22</small><h1>짧은 동작의 차이가<br>어디에서 묻히는가?</h1><p>관측 근거 · 시간 연결 · 다중 참고 범위를 분리해서 확인했습니다.<br>C·T3-F 유지 / 사용자 촬영 접근 없음 / 학습·운영 적용 없음</p></header>'''+''.join(body)+'</main></html>'
with (U/'RESULT_KO.html').open('x',encoding='utf-8') as f:f.write(page)
print('REPORT WRITTEN',json.dumps(normals['new'],ensure_ascii=False))
