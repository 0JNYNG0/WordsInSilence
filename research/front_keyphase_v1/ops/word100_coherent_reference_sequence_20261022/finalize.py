"""Local report and handoff only; never changes frozen numerical results."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from html import escape
from collections import defaultdict
import hashlib, json, sys

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
OPS = U.parent
UNIT = U.name
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def now(): return datetime.now(timezone(timedelta(hours=9))).isoformat()

NEXT = ('사용자와 연속 참고 구간 후보의 소폭 추가 반응·비교 근거 손실·정상 속도 경고 증가를 공동 검토한다. '
        '현재 후보는 운영 미채택. 다음 제안은 기존 시점별 근거를 계속 유지하고, 앞·중간·뒤 값과 두 변화량을 하나의 최대값/허용범위로 합치는 영향을 분리해 진단하는 것이다. '
        '누락을 정상 처리하거나 문턱을 결과에 맞춰 조정하지 않고, 연속 비교 불가와 시점별 비교 가능을 구분한다. '
        '후보 재설계 시 같은 영상의 알려진 시간 대조와 정상 속도·손내림·교차를 먼저 고정 검증하고, 별도 공개 묶음에서 확인한다. '
        '같은 최소5명 근거 조건에는 대상·참고를 제외한 5명이 필요하므로 단어당 최소7명 확보 여부부터 확인한다. '
        'C·T3-F·운영 DTW/웹 유지, 사용자·멘토 자료 접근/학습/튜닝/선택 금지. 추가 학습은 결과 공동 확인 후.')

def aggregate_check(a, result, frozen):
    assert read(U/'VERIFY.json')['status']=='PASS'
    checks=0
    normal={'identity','fast075','slow15','pace','pause'}
    source=[r for r in result['cross'] if r['kind'] in normal]
    assert len(source)==1080
    for g, data in a['cohorts']['cross']['normal_all']['groups'].items():
        for k in ('expected','comparable','unknown'):
            target=data['expected'] if k=='expected' else data['modes']['sequence'][k]
            assert sum(r['groups'][g][k] for r in source)==target
            checks+=1
        for m in ('levels','sequence'):
            assert sum(r['groups'][g][m+'_bad'] for r in source)==data['modes'][m]['bad'];checks+=1
    rows=[r for r in a['rows']['cross'] if r['kind'].startswith('skip3_')]
    assert len(rows)==648
    d=defaultdict(lambda:defaultdict(int))
    for r in rows:
        for m in ('point','levels','sequence'):
            d[r['query'],r['kind']][m]+=int(r['inside'][m]['new_bad_common']>0)
    assert len(d)==81
    expected_counts={}
    for m in ('point','levels','sequence'):
        target=a['deletions']['cross']['short']['summary']['modes'][m]
        one=sum(v[m]>=1 for v in d.values());five=sum(v[m]>=5 for v in d.values())
        assert one==target['any_reference_new_local_common'];assert five==target['at_least5_references_new_local_common'];checks+=2
        expected_counts[m]={'one':one,'five':five}
    for group in ('known','cross','missing'):
        for r in a['rows'][group]:
            for g in r['groups'].values():
                for m,v in g['modes'].items():
                    assert v['comparable']+v['unknown']==g['expected']
                    assert v['bad']+v['within']==v['comparable'];checks+=2
    for p,h in frozen['protected'].items():assert sha(Path(p))==h
    out={'at':now(),'status':'PASS','checks':checks,'protected_files':len(frozen['protected']),
         'short_variant_counts':expected_counts,'basis':'Parent recomputed aggregate from saved case metadata; independent array reconstruction is VERIFY.json.',
         'source_sha256':{p.name:sha(p) for p in [U/'RESULT.json',U/'ANALYSIS.json',U/'VERIFY.json']}}
    write(U/'AGGREGATE_CHECK.json',out)
    return out

def report():
    a=read(U/'ANALYSIS.json');r=read(U/'RESULT.json');f=read(U/'FROZEN.json')
    aggregate_check(a,r,f)
    paragraphs=[
    ('결론', '여러 참고를 한 평균 동작으로 섞지 않고, 각각의 짧은 앞·중간·뒤를 비교하는 연구 후보를 구현하고 검산했다. 하지만 현재 후보를 운영에 적용할 만큼의 개선은 확인하지 못했다. 짧은 차이에 조금 더 반응하는 부분은 있지만 정상 속도 변화에도 차이가 늘고, 손가락 비교가 보류되는 구간이 크게 증가했다.'),
    ('어떻게 바꿨나', '예를 들어 A 수행자의 앞·중간·뒤는 끝까지 A의 동작과 비교한다. 앞은 A, 중간은 B, 뒤는 C의 값으로 섞지 않는다. B·C 등 다른 참고도 각각 비교한 뒤 결과를 모두 남긴다. 가장 유리한 한 사람만 고르거나 가상의 평균 손을 만들지 않았다.'),
    ('세 조건의 뜻', '① 기존 시점별 비교 ② 같은 사람의 앞·중간·뒤 위치·각도 비교 ③ 그 세 시점에 더해 앞→중간, 중간→뒤의 변화량 비교. ②도 이미 시간 순서가 있는 비교이며, ③은 변화량을 더한 것이다. ①→②에는 대표 시점 선택·관측 요구·개인차 범위 계산도 함께 달라지므로 순서 하나만의 효과라고 말할 수 없다. ②→③은 비교 가능한 구간은 같고 변화량과 그에 따른 기준 범위가 추가된다.'),
    ('실험 범위', '공개 TRAIN 자료의 수어·후원·멀다 3단어, 9명, 27영상을 사용했다. 원속도·속도 변형·짧거나 긴 구간 삭제·손/팔 관측 제거를 합해 2,808개 비교와 243개 기준 범위를 계산했다. 다른 사람 비교는 영상당 다른 8명을 각각 참고로 사용했다. 한 비교의 개인차 범위는 질의자와 해당 참고자를 제외한 7명 중 근거가 충분한 5명 이상으로 계산했다. 같은 영상 대조는 다른 8명으로 기준을 만든다. 이 묶음은 이미 개발 분석에 사용했으므로 독립 최종 시험이 아니다.'),
    ('짧은 구간의 크기', '앞·중간·뒤는 전체 수어의 시작·중간·끝이 아니다. 각 시점 주변 5개 입력 프레임 안에서 3시점을 보는 작은 창이며, 참고 영상에서는 약 0.133초 길이다. 이를 의미상 필수 동작 단계로 확정하지 않았다. 짧은 삭제는 27영상의 3위치에서 3입력프레임씩 제거한 총 81변형이다. 실제 사용자의 수어 오류 정답을 가진 시험은 아니다.'),
    ('짧은 삭제 반응', '81개 중 하나 이상의 참고에서 삭제 부위에 새 차이가 생긴 변형은 63→62→63개, 8개 참고 중 5개 이상이 함께 반응한 변형은 30→29→31개였다. ③은 ②보다 1개/2개 더 반응했지만, ①보다 전체적으로 좋아졌다고 보기는 어렵다. ①과 ③의 63개는 같은 구성이 아니다. 기존만 반응한 11개와 새 후보만 반응한 11개가 서로 바뀌었다. 5개 이상 반응 기준도 기존만13·새것만14·공통17개다. 이 수치를 인식률이나 의미상 오류 판정 정확도로 부르면 안 된다.'),
    ('비교 근거의 손실', '정상 원본과 속도 변형 1,080쌍에서 손가락 각도 비교 가능 비율은 61.9%→31.4%, 손 위치는 73.5%→45.7%, 팔꿈치와 손목은 각각 99.5%→89.5%였다. 분모는 고정한 참고 시간·손·항목 수다. 주변 구간이 연속해서 관측되고 여러 사람에게도 충분한 근거가 있어야 한다는 조건 때문에 보류가 늘었다. 원래 좌표와 시점별 결과를 지운 것은 아니지만, 새 연속 검사만으로는 판단할 수 있는 정보가 줄었다.'),
    ('정상 속도에 대한 손실', '정상적인 속도 변형 864쌍에서 세 조건의 원본·변형 모두 비교 가능한 567,457칸만 똑같이 보면 새 차이 표시가 8,213→11,866→12,118칸으로 늘었다. ②→③만 봐도 252칸 증가했다. 새 후보가 정상 속도 차이를 더 잘 허용했다고 말할 수 없다. 한 칸은 시점×손×각도 또는 좌표 항목이며 사람 수나 영상 정답 수가 아니다. 기존 조건보다 전체 경고 영상 수가 줄어 보이는 부분은 비교가 보류된 양을 함께 봐야 한다.'),
    ('같은 원본으로 한 대조', '시간 대응을 아는 같은 영상의 짧은 삭제에서는 반응이 80→10→9개였다. 기존 시점별 자기 영상 기준과 새 후보의 다른 사람 8명 허용 범위가 달라 순서 특징 하나의 퇴보로 단정할 수는 없다. 그러나 현재 기준이 작은 변화까지 개인차로 묻을 수 있음을 보여주는 중요한 한계다. ②→③에서도 한 변형의 반응이 사라졌다. 한 최대값으로 묶으면 변화량뿐 아니라 사람 간 기준값도 커질 수 있기 때문이다.'),
    ('손내림과 관측 부족', '참고에서 확인된 손내림은 그 손의 필수 비교 구간에 새로 넣지 않았다. 중간 관측이 없다고 원래 비교 범위를 줄이지 않고 비교 근거 부족으로 남겼다. 손·팔 전체나 짧은 중간 관측을 제거한 135개 대조에서 원래 필요 범위와 미관측 시간표를 보존했고, 근거 부족을 차이 0 또는 잘 맞음으로 올리지 않았다. 다만 정상 손내림을 가로지르는 주변 창도 새 연속 검사에서는 보류될 수 있다. 따라서 정상 손내림 전후를 충분히 비교하는 문제까지 해결한 것은 아니다.'),
    ('손 교차', '좌표와 좌우 손 배정은 변경하지 않았다. 기존에 검토한 후원 영상의 교차 12프레임은 각도와 손바닥의 신뢰 근거가 없으므로 계속 보류했고, 원래 관측된 팔 48개 점은 그대로 남겼다. 자기 영상 대조의 팔 좌표 비교도 유지됐다. 다른 참고와의 엄격한 주변 비교에서는 일부 팔 비교가 추가 보류됐다. 이것은 교차할 때의 손 모양을 정확히 맞혔다는 증거가 아니다.'),
    ('왜 한 번에 좋아지지 않았나', '연속성을 요구하면 잘못된 연결을 피할 수 있지만, 잠깐의 미관측 때문에 이웃 시점까지 비교하지 못할 수 있다. 반대로 여러 사람의 가장 큰 정상 차이를 허용하면 짧게 달라진 동작도 그 범위 안에 들어갈 수 있다. 또 변화량을 추가하면 관측 차이도 커지지만 정상 사람들의 기준 범위도 커진다. 이번 결과는 이 세 효과를 따로 확인한 것이며, 자료만 더 넣거나 즉시 재학습하면 해결된다는 근거는 아니다.'),
    ('다음 제안', NEXT),
    ('검증과 보존', '부모 Codex가 구현·실행했고 별도 에이전트가 후보 코드를 사용하지 않는 독립 수치 검산을 작성했다. 2,808개 배열 비교·243개 기준 계산, 합성 9조건, 결과 집계와 그림을 확인했다. 실제 미관측 구간·기존 비교·모델 및 운영 파일 해시 32개를 보존했다. 새 DTW 정렬 탐색, TCN 추론/학습, 사용자·멘토 촬영 접근, 운영 C·T3-F·웹 교체는 하지 않았다. 정상 반복 일반화와 새로운 단어/사람의 의미상 정확도는 이번 시험으로 입증하지 않았다.')
    ]
    text='여러 사람의 연속 참고 구간 비교 — 구현·검증 완료, 운영 미채택\n\n'+'\n\n'.join(h+'\n'+t for h,t in paragraphs)+'\n'
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    table='''<table><thead><tr><th>검사</th><th>기존 시점별</th><th>앞·중간·뒤</th><th>앞·중간·뒤 + 변화량</th></tr></thead><tbody>
    <tr><td>짧은 삭제: 1개 이상 참고 반응 / 81</td><td>63</td><td>62</td><td>63</td></tr>
    <tr><td>짧은 삭제: 5개 이상 참고 반응 / 81</td><td>30</td><td>29</td><td>31</td></tr>
    <tr><td>정상 자료의 손가락 비교 가능 비율</td><td>61.9%</td><td>31.4%</td><td>31.4%</td></tr>
    <tr><td>정상 속도 변화의 새 차이 / 공통 567,457칸</td><td>8,213</td><td>11,866</td><td>12,118</td></tr>
    </tbody></table>'''
    cards=''.join(f'<section id="s{i}"><h2>{escape(h)}</h2><p>{escape(t)}</p></section>' for i,(h,t) in enumerate(paragraphs))
    figures=''.join(f'<figure><img src="visuals/{name}.png" alt="{escape(cap)}"><figcaption>{escape(cap)}</figcaption></figure>' for name,cap in [
        ('short_response','반응 = 삭제된 부위에서 원본과 변형 모두 관측된 항목에 새 차이가 생김. 정답률이 아닙니다.'),
        ('coverage','비교 범위의 분모는 유지했습니다. 관측 부족과 주변 구간 근거 부족은 보류됩니다.'),
        ('normal_tradeoff','정상 속도 변화의 공통 근거와 전체 비교 가능량을 함께 봅니다.'),
        ('unknown_reasons','보류 이유는 서로 겹칠 수 있습니다. 합계가 고유 보류 수는 아닙니다.')])
    links=' · '.join(f'<a href="{p}">{n}</a>' for p,n in [('FROZEN.json','고정 조건'),('VERIFY.json','독립 검산'),('AGGREGATE_CHECK.json','집계 검산'),('INTERPRETATION_REVIEW.json','해석 교차검토'),('ANALYSIS.json','전체 분석'),('COMPONENT_DIAGNOSIS.json','차이·기준값 변화'),('RESULT_KO.txt','텍스트 보고서')])
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>연속 참고 구간 비교 · WordsInSilence</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#142337;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:42px 24px 70px}header{background:#172b49;color:white;padding:36px;border-radius:20px}h1{font-size:34px;line-height:1.4;margin:10px 0}h2{font-size:23px;line-height:1.45;margin:0 0 12px}p{margin:0}section,figure,.summary{background:white;border:1px solid #dbe3ee;padding:26px;border-radius:14px;margin:22px 0}figure{padding:12px}img{display:block;width:100%;height:auto}figcaption{padding:8px 14px;color:#43556b;font-size:15px}.tag{font-size:14px;letter-spacing:.06em;color:#aad6ed}.summary{border-left:6px solid #be8735}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:12px;border-bottom:1px solid #dce4ee;text-align:left}th{background:#edf3fa}a{color:#17668b}.table{overflow-x:auto}footer{font-size:14px;color:#526176}@media(max-width:650px){h1{font-size:27px}main{padding:18px 12px}section,header{padding:20px}table{min-width:650px}}@media print{section,figure{break-inside:avoid}body{background:white}}
    </style><main><header><div class="tag">WORDS IN SILENCE · 공개 자료 DTW 연구 · 2026.10.22</div><h1>짧은 동작을 앞·중간·뒤로 묶어 비교</h1><p>여러 참고를 각각 유지했습니다. 작은 추가 반응은 있지만,<br>관측 손실과 정상 속도 경고 때문에 현재 후보는 운영에 적용하지 않습니다.</p></header><div class="summary"><strong>구현·검산 완료 ≠ 성능 개선 입증</strong><p>공개 3단어 × 9명, 27영상. 기존 C·T3-F·웹 유지 · 사용자 촬영 사용 없음 · 추가 학습 없음.</p></div><div class="table">'''+table+'</div>'+cards+figures+'<footer>'+links+'</footer></main></html>'
    (U/'RESULT_KO.html').write_text(html,encoding='utf-8')
    print('REPORT_WRITTEN',U,flush=True)

def sync():
    for name in ['VERIFY.json','ANALYSIS.json','AGGREGATE_CHECK.json','INTERPRETATION_REVIEW.json','VISUAL_REVIEW.json']:
        assert (U/name).exists(),name
    f=read(U/'FROZEN.json')
    for p,h in f['protected'].items(): assert sha(Path(p))==h
    assert not (U/'RUNNING.json').exists()
    ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';state=OPS/'STATE.json';handoff=ROOT/'NEW_CHAT_HANDOFF.md'
    for p,b in [(ctx,'WordsInSilence_PROJECT_CONTEXT.md'),(state,'STATE.json'),(handoff,'NEW_CHAT_HANDOFF.md')]:
        assert sha(p)==sha(U/'before'/b),f'Concurrent edit: {p}'
    timestamp=now()
    title='### 2026-10-22 연속 참고 앞·중간·뒤 비교 — 소폭 추가 반응·큰 근거 손실, 운영 미채택'
    entry=title+'\n'+('\n'.join([
        '- 단위 `'+UNIT+'`. 공개 TRAIN 3단어×9명27영상, known297/cross2376/missing135=2808비교·243개개인차기준. 기존 저장 OLD DTW경로 고정, 재정렬/추론/학습0. 현재 C·T3-F·웹 불변.',
        '- 실제 참고별 5입력프레임(약133ms)에서 앞·중간·뒤3값+두변화량. 같은 사람 참고를 유지; 서로다른사람의 시점별 조각/가상평균/최소차이참고 선택 없음. 비교대상·참고 제외7명(자기대조8명) 중≥5명 관측의 최대거리범위. 단어별 예외없음. 의미상 단계 정답 아님.',
        '- 짧은3프레임삭제81변형: ≥1참고반응 point/levels/sequence63/62/63; ≥5참고30/29/31. 기존→sequence ≥1 기존만11·신규만11; ≥5 기존만13·신규만14. 모두 원본·변형 공통관측에서 고정삭제부위의 새차이. 기존raw69와정의다름. 추가반응을 인식정확도/수어오류정답률로 부르지않음.',
        '- 정상1080쌍 각도비교가능61.8721→31.4402%,손위치73.5269→45.7462%,팔꿈치/손목99.4744→89.5333%. 총분모1638000에서1148965→718558. 기존좌표·시점별결과/expected/nativeunknown 보존하나 엄격한연속창 때문에 추가보류. 불확실한한시점이 이웃창까지보류시킬수있음.',
        '- 정상속도864쌍 공통4관측567457칸 새차이8213→11866→12118. LEVEL→SEQ +252; 정상속도 허용개선 미입증. known short80→10→9는 자기시점기준과 다른사람허용범위 등 복합차이; 순서만의손실로단정금지. 변화량추가시 거리와정상범위 함께확대 가능.',
        '- 손내림역할/고정범위/135손팔미관측 시간표 보존,unknown을normal/0차이로승격없음. 정상손내림전후연속창 추가보류는남음. 교차12프레임 신뢰각도/손바닥0유지,팔48점보존; 교차정확손모양 검증아님. 반복일반화/새사람/전100단어/의미상필수누락 검증아님.',
        '- 부모구현·실행85377 exit0, 별도verify80087 exit0(2808/243),9합성PASS,분석21142 exit0. 해석에이전트/부모집계·4그림·보호32핀확인. 최초실험실패없음. 운영미채택. 다음은 결과공동검토와 위치값/변화량·허용범위 분리진단 계획; 추가학습은 공동확인후.'
    ]))+'\n\n'
    c=ctx.read_text(encoding='utf-8-sig')
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
    h9='## 9. 다음 단일 작업\n'
    assert c.count(h8)==1 and c.count(h9)==1
    c=c.replace(h8,h8+'\n'+entry,1).replace(h9,h9+'\n### 2026-10-22 연속 참고 구간 결과 공동 검토\n\n'+NEXT+'\n\n',1)
    ctx.write_text(c,encoding='utf-8')
    s=read(state);s['updated_date_kst']=timestamp;s['current_stage']='WORD100_COHERENT_REFERENCE_SEQUENCE_VERIFIED_NO_ADOPTION'
    s['next_single_action']=NEXT;s['resume_command']=NEXT
    s[UNIT]={'status':'VERIFIED_SMALL_RESPONSE_GAIN_COVERAGE_REGRESSION_NO_ADOPTION','at':timestamp,
        'unit':UNIT,'scope':{'words':3,'people':9,'clips':27,'known':297,'cross':2376,'missing':135,'calibrations':243},
        'short_omission_any_reference':[63,62,63],'short_omission_five_references':[30,29,31],
        'normal_angle_coverage':[723903/1170000,367850/1170000],
        'normal_speed_common_cells':567457,'normal_speed_new_difference':[8213,11866,12118],
        'independent_verify':'PASS','training':0,'new_dtw_search':0,'user_data_access':0,'production_changes':0,
        'remaining':['Severe loss of sequence-comparable angles','Normal speed variation produces more new differences','Reference-bound enlargement can hide short changes','No semantic stage ground truth or new independent cohort'],
        'report':str(U/'RESULT_KO.html'),'next':NEXT}
    write(state,s)
    hand='# WordsInSilence 最新 인수인계\n\n갱신 '+timestamp+'\n\n현재 목표: 여러 실제 참고를 각각 유지하며 짧은 동작 차이와 관측 부족을 구분하는 DTW 연구. TCN 학습 작업이 아니다.\n마지막 완료: '+UNIT+' 구현·실행·독립검산·보고. 새 후보의 추가 반응은 작고 정상 속도 경고·비교 근거 손실이 커 운영 미채택.\n다음 한 작업: '+NEXT+'\n\n'+entry
    hand+='구현·실행: 부모 Codex. 독립 배열 검산: phase_design_review 구현 verify.py/부모 실행. 집계·그림: omission_summary. 해석 교차검토: phase_evaluation_review. 부모 집계·시각확인 별도.\n현재 실행/잠금 없음. 완료 추출/실험/검산 재실행 금지. 원본·과거동결결과 읽기전용, 사용자·멘토 자료 접근0.\n근거 폴더: '+str(U)+'\nRESULT_KO.html/txt, FROZEN, RESULT, VERIFY, ANALYSIS, AGGREGATE_CHECK, COMPONENT_DIAGNOSIS, INTERPRETATION_REVIEW, VISUAL_REVIEW, FINAL_AUDIT, SYNC_CHECK.\n'
    handoff.write_text(hand.replace('最新','최신'),encoding='utf-8')
    assert read(state)['current_stage']=='WORD100_COHERENT_REFERENCE_SEQUENCE_VERIFIED_NO_ADOPTION'
    for p in (ctx,handoff):assert UNIT in p.read_text(encoding='utf-8') and NEXT in p.read_text(encoding='utf-8')
    write(U/'SYNC_CHECK.json',{'at':now(),'status':'PASS','files':{str(p):sha(p) for p in (ctx,state,handoff)},'next':NEXT})
    write(U/'FINAL_AUDIT.json',{'at':now(),'status':'PASS','adopted':False,'protected_files':len(f['protected']),
        'training':0,'classifier_inference':0,'new_dtw_search':0,'new_mediapipe':0,'user_data_access':0,'runtime_change':0,
        'runs':{'main':85377,'verify':80087,'analysis':21142},'scientific_run_exit_codes':{'main':0,'verify':0,'analysis':0},'ongoing':False,
        'report_qa':'Initial section-count assertion failed; report-only check corrected and passed. REPORT_QA_FIRST_ERROR.json retained; scientific outputs and conditions unchanged.',
        'report_sha256':{p.name:sha(p) for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt']}})
    print('SYNC_PASS',flush=True)

if __name__=='__main__':
    {'report':report,'sync':sync}[sys.argv[1]]()
