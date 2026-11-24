"""Write a local evidence report from completed, verified results; no new scoring."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, html, hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
esc=lambda s:html.escape(str(s))
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def main():
    s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json');f=read(U/'FROZEN.json')
    assert v['status']=='PASS'
    visual=read(U/'VISUAL_INDEX.json')
    notes={
      'NIA_SL_WORD0185_REAL11_F':('내려놓은 손과 영역 안 누락이 함께 나타남','1.83·2.10초에는 왼손이 노란선 아래에 내려와 있어 제외되는 것이 보인다. 3.30초에는 두 손이 가슴 앞에서 겹치며 C 두 경로의 사용 가능 손점이 모두 0개다. 기존 경로도 오른손은 0개여서 원래부터 관측 난점이 있던 장면이다. 이 장면만으로 누락의 원인을 검출기 또는 배정 필터 하나로 확정할 수 없다.'),
      'NIA_SL_WORD0298_REAL11_F':('영역 밖에 내려놓은 반대손 제외','1.63·2.53·3.43초의 왼손은 아래로 내려져 있고 오른손은 얼굴·가슴 앞에서 보인다. C 경로는 내려놓은 왼손을 제외한다. 이 표본의 왼손 측정 불가를 수행 오류로 세면 안 된다. 표본 장면만으로 단어 전체의 필수손을 새로 확정하지 않았다.'),
      'NIA_SL_WORD0298_REAL12_F':('다른 촬영자에서도 내려놓은 반대손 제외','1.37·2.13·2.90초에 같은 관측 양상이 보인다. 여러 촬영자에서 재현되지만, 이 결과로 전체 100단어의 왼손을 일괄 제외할 수는 없다.'),
      'NIA_SL_WORD0298_REAL13_F':('동작 중 손과 마무리 손 내림 구분 필요','1.60·2.60초에는 오른손이 보존되고 왼손은 아래에 있다. 3.57초에는 양손을 모두 내린 상태이며, sensor 경로는 두 손 모두 관측 근거가 없다. 마무리 정지 자세의 제외와 동작 중 누락은 다른 사건이다.'),
      'NIA_SL_WORD0500_REAL13_F':('손이 들어오는 경계에서 부분 관측','1.60·1.67·1.77초에 왼손이 하단선 부근에서 올라온다. 1.77초는 C 두 경로에서 손점 19/21개가 사용 가능하지만, 현행 굽힘 특징은 손 전체 관측과 각도 계산 조건을 요구하므로 이 프레임의 굽힘은 유효하지 않을 수 있다. 클립 전체의 왼손 굽힘 참고 수는 기존 9, masked 4, sensor 6이다.'),
      'NIA_SL_WORD1139_REAL12_F':('영역 안 동작손의 근거 부족','1.67초에는 왼손이 상반신 범위 안에 분명히 보이는데 sensor의 왼손 사용 가능 점이 0개다. 1.87초에는 손이 겹치는 상태에서도 두 손 관측이 돌아오며, 2.60초에는 sensor의 양손 근거가 다시 없다. 즉 모든 누락을 손이 화면 밖에 있어서라고 설명할 수 없다. 왼손 굽힘에 충분한 참고는 9→3개, 위치·궤적 참고는 9→7개다.'),
    }
    review=[]
    for item in visual:
        title,desc=notes[item['clip_id']]
        review.append({**item,'finding_title':title,'visual_observation':desc,'reviewer':'Codex visual inspection of 18 public RGB frames with saved observation overlays','limits':'posthoc diagnostic sample; not all missing intervals reviewed; no linguistic ground truth or new hand-role label'})
    write(U/'VISUAL_REVIEW.json',review)
    facts={
      'status':'COMPLETE_OFFLINE_COMPARISON_NOT_PROMOTED','classes':100,'references':899,'dev':300,
      'prior_multiref_classes':12,'prior_references':108,'views':s['views'],
      'recognition_accuracy_change':None,'training':0,'classifier_inference':0,'user_or_mentor_recordings_read':0,
      'production_changed':False,'known_roles':12,'unknown_roles':88,
      'recommendation':'Keep existing C/T3-F and production DTW. Review a common per-metric observability and trajectory-anchor candidate before recognition assistance or training.',
      'implementation':'Codex local','verification':'Codex separate arithmetic verifier, not independent human/agent review',
      'visual_scope':'6 public DEV clips, 18 selected frames; no full missing-interval census',
      'frozen_sha256':hashlib.sha256((U/'FROZEN.json').read_bytes()).hexdigest(),
    }
    write(U/'CONCLUSION.json',facts)
    names={'legacy':'기존 DTW 관측','masked':'전체 검출 후 범위 제한','sensor':'웹캠 경계 모사 관측'}
    cover='';axis='';wordrows=''
    for key in f['views']:
        n=s['views'][key]['totals']['measurable'];total=s['views'][key]['totals']['total']
        cover+=f'<div class="barrow"><b>{names[key]}</b><div class="track"><div style="width:{n/total*100:.3f}%"></div></div><strong>{n:,}/{total:,} · {n/total*100:.2f}%</strong></div>'
    for a,label in [('handshape_L','왼손 굽힘'),('handshape_R','오른손 굽힘'),('location_L','왼손 위치'),('location_R','오른손 위치'),('movement_L','왼손 궤적'),('movement_R','오른손 궤적')]:
        axis+='<tr><th>'+label+'</th>'+''.join(f"<td>{s['views'][k]['by_axis_side'][a]['measurable']}/{s['views'][k]['by_axis_side'][a]['total']}</td>" for k in f['views'])+'</tr>'
    bank=read(U/'BANK_legacy.json')['words']
    for code,w in bank.items():
        wordrows+=f'<tr><th>{esc(w["label"])}<small>{code}</small></th><td>{len(w["references"])}</td><td>{"기존 역할 유지" if w["role_known"] else "손별 기술 비교"}</td>'
        for k in f['views']:
            a=s['views'][k]['by_word'][code];wordrows+=f'<td>{a["measurable"]}/{a["total"]}</td>'
        wordrows+='</tr>'
    figures=''
    for item in review:
        figures+=f'<details><summary>{esc(item["label"])} · ID{item["clip_id"].split("REAL")[1][:2]} — {esc(item["finding_title"])}</summary><p>{esc(item["visual_observation"])}</p><a href="visual/{Path(item["sheet"]).name}"><img src="visual/{Path(item["sheet"]).name}" alt="{esc(item["label"])} 원본과 세 관측 경로" loading="lazy"></a></details>'
    text='''공개 100단어 다중 참고 · DTW 관측 경로 비교

결론: 연구용 다중 참고를 12단어 108영상에서 100단어 899영상으로 확장하고, 세 관측 경로 비교와 별도 검산을 완료했다. C 관측을 DTW에 그대로 바꾸어 넣는 것은 보류한다. 인식률이 좋아졌다는 결과는 이번 작업에서 측정하지 않았다.

1. 자료와 계산
TRAIN ID01~09의 899영상만 참고와 기준 분포에 사용했다. 99단어는 9명씩, WORD1614는 8명이다. 같은 사람끼리 비교하지 않은 TRAIN 7,184방향쌍, 다른 사람 ID11~13의 DEV300영상 대 TRAIN 참고 2,697쌍을 각 관측에서 계산했다. 합계 29,643건은 영상쌍×관측경로 수이며 독립 촬영 수가 아니다. 고유 영상은1,199개다.
기존12단어 역할은 그대로 유지, 새88단어는 필수손 미확정이므로 좌우별 차이만 기술한다. 손이 안 보인다는 이유로 불필요손이라고 정하지 않았다.

2. 비교 방법
legacy=기존 DTW 전체화면 관측. masked=C의 전체 검출 후 상반신 범위 안 관절 사용. sensor=C의 상반신 픽셀에서 검출한 저장 관측으로 실제 화면 밖 손을 모사. 각 조건 안에서는 참고와 질의에 같은 관측을 사용했다. 공식·정렬·최소 근거 조건은 동일하다. 검출/배정/범위/시간구간의 묶음 비교이므로 한 요소의 단독 효과가 아니다.
한 사람의 참고만 고르지 않고, 충분히 비교 가능한 여러 사람과의 차이를 중앙값으로 모았다. TRAIN 기준은 자기 촬영자를 제외해 만들었다. 최소 참고5개, 유효 대응50%, 서로 다른 프레임 각8개 조건을 유지했다.

3. 결과
DTW 수치를 낼 근거가 있는 항목: 기존1773/1773, masked1380/1773, sensor1369/1773. 이는 정확도가 아니라 공개 DEV300영상의 손×항목별 측정 가능 수다. 기존 오른손만 사용하는 것으로 지정된3단어의 반대손27항목을 분모에서 제외했다.
기존12연습단어 DEV36영상 중 세 항목 모두 측정/기준 비교가 가능한 영상은 기존36, masked30, sensor35다.
sensor에서 줄어든404항목 중403은 왼손,1은 오른손이다. 내려놓은 손을 제외한 결과와 화면 안 동작손의 누락이 함께 존재한다. 모든 측정 불가를 오류 또는 불필요손으로 해석할 수 없다.

4. 원본으로 확인한 문제
공개6영상18장면을 원본과 저장 관절로 확인했다. 흡연실의 표본은 내려놓은 왼손 제외가 보인다. 후원에서는 초기 내려놓은 왼손 제외뿐 아니라 가슴 앞 손 겹침 시 C 경로의 양손 근거 부족도 보인다. 믿다 ID12는 화면 안 왼손이 sensor에서 미관측으로 남는 장면이 있고, 굽힘에 쓸 참고가9→3개여서 최소5개를 못 채운다. 같은 영상의 위치·궤적은7개 참고가 유효하다. 마음에들다 ID13은 경계에서 손점19/21개만 남은 장면이 있어 위치 근거와 굽힘 근거가 다르다. 이것은 관측의 차이이며 해당 수어가 틀렸다는 판정이 아니다. 18장면 표본으로 전체 누락 원인을 분류하지 않았다.

5. 해석과 다음 계획
기존 관측은 쉬운 정지 자세까지 포함해서 계산이 잘 되는 경우가 있다. C 관측의 측정 비율이 낮아도 곧바로 더 부정확하다는 뜻은 아니다. 반대로 측정 가능한 부분만 남겨 차이가 작아져도 개선으로 볼 수 없다. 위치와 손가락 모양의 관측 신뢰를 분리해야 한다.
기존 굽힘 특징은 손 전체 관측 조건을 거친다. 다음에는 실제 관측된 각도와 부족한 각도를 분리하고, 자료가 부족하면 낮은 차이 점수 대신 보류하도록 하는 공통 후보를 먼저 검증할 것을 제안한다. 단어별 예외나 단순 기준 완화는 하지 않는다. 내려놓은 구간과 겹쳐 안 보이는 구간을 구분해 손별·구간별 근거를 함께 보고한다. 움직이지 않는 받침손은 무조건 제외하지 않는다.
궤적은 각 손의 첫 관측 위치를 원점으로 삼고 있어 화면 진입 시점이 달라지면 수치도 달라질 수 있다. 관측 시작 시점/정렬구간 영향을 분리하는 진단을 함께 제안한다. 이후에만 인식 보조나 추가학습 계획을 검토한다.
이번에는 C·T3-F·웹·기존 DTW를 바꾸지 않았고 사용자/멘토 촬영을 읽지 않았다. 추가 학습과 인식 순위 변경도 하지 않았다. DEV는 이미 활용된 개발자료이며 새 맹검이나 새 사용자 성능 결과가 아니다.

6. 검증
별도 코드로 원입력3597개/특징7312875값, 저장집계42725개, 정렬경로29643개와 고정선정300쌍의 거리계산을 검산했다. 기존12단어 결과 재현, 보호대상 해시 유지. 같은 Codex의 별도 산술 검산이며 다른 전문가나 별도 에이전트의 독립 검토는 아니다.
검산 초기 구문 오류와 float32 acos/float64 atan2의 미세 반올림 차이는 최초 증거를 보존했다. 후자는 코사인 및 수학적 각도 오차 한계로 검산을 보완했으며 비교 결과/조건/모델은 바꾸지 않았다. 최대 차이는 약0.036도였다.
'''
    (U/'RESULT_KO.txt').write_text(text,'utf-8')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 100단어 다중 참고 비교</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f2f5f9;color:#17273c;font:17px/1.7 "Malgun Gothic",sans-serif}}main{{max-width:1130px;margin:auto;padding:48px 26px 80px}}h1{{font-size:39px;line-height:1.3;margin:8px 0 18px}}h2{{font-size:25px;line-height:1.45;margin:0 0 20px}}h3{{font-size:20px}}p{{margin:12px 0}}section{{background:white;border:1px solid #dbe3ed;border-radius:18px;padding:30px;margin-top:22px}}.tag{{font-size:14px;font-weight:700;letter-spacing:.06em;color:#2563a3}}.lead{{font-size:21px;max-width:900px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.tile{{background:#edf4fa;border-radius:12px;padding:20px}}.tile strong{{display:block;font-size:30px;color:#155d91}}.tile span{{font-size:15px}}.notice{{background:#fff5df;border-left:5px solid #d58e17;padding:15px 20px;margin:20px 0;border-radius:7px}}.barrow{{display:grid;grid-template-columns:235px 1fr 210px;gap:14px;align-items:center;margin:16px 0;font-size:15px}}.track{{height:22px;background:#e5edf4;border-radius:6px;overflow:hidden}}.track div{{height:100%;background:#357eaf}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:13px 15px;border-bottom:1px solid #dfe7ef;text-align:left}}thead{{background:#edf4fa}}small{{display:block;font-size:12px;color:#4f6479}}.overflow{{overflow:auto}}li{{margin:10px 0}}details{{border:1px solid #dbe3ed;border-radius:10px;padding:17px;margin:14px 0}}summary{{cursor:pointer;font-weight:700}}img{{width:100%;height:auto;border-radius:8px;margin-top:15px}}a{{color:#176296}}input{{width:100%;padding:12px;font:inherit;border:1px solid #bccbd9;border-radius:8px;margin-bottom:12px}}.muted{{font-size:15px;color:#4f6479}}@media(max-width:800px){{main{{padding:24px 14px}}h1{{font-size:29px}}section{{padding:21px}}.grid{{grid-template-columns:1fr}}.barrow{{grid-template-columns:1fr}}}}
    </style><main><div class="tag">WORDS IN SILENCE · 공개 자료 비교 완료</div><h1>100개 단어, 여러 사람의 참고 동작으로<br>DTW 비교 범위를 넓혔습니다.</h1><p class="lead">확장은 완료했습니다. 다만 <b>C의 관측을 DTW에 그대로 연결하는 것은 보류</b>합니다. 내려놓은 손의 제외와 동작 중 관측 누락이 섞여 있어, 두 경우를 나누어 처리해야 합니다.</p>
    <div class="grid"><div class="tile"><strong>12 → 100단어</strong><span>연구용 다중 참고 범위 확장</span></div><div class="tile"><strong>899개</strong><span>9명의 공개 TRAIN 참고 영상</span></div><div class="tile"><strong>300개</strong><span>다른 3명의 공개 DEV 영상</span></div></div>
    <section><h2>01. 이번에 실제로 바뀐 것</h2><p>기존 다중 참고는 12개 연습단어·108영상이었습니다. 이번에는 <b>100개 단어·899영상</b>으로 연구용 참고 묶음을 준비했습니다. 99단어는 9명씩, 1단어는 8명의 자료입니다. 한 사람을 흉내 내는 정도만 보지 않도록, 여러 사람과 비교한 차이의 <b>중앙값</b>과 사람 간 차이 분포를 저장했습니다.</p><p>TRAIN은 사람 ID01~09, DEV는 다른 사람 ID11~13입니다. TRAIN 기준을 만들 때 비교 대상과 같은 사람은 제외했습니다. 고유 원본은 총1,199영상이며, 세 관측에서 계산한 29,643건은 <b>영상쌍×관측 경로</b> 수입니다.</p><div class="notice">이번 결과는 DTW의 비교 준비와 관측 근거 점검입니다. <b>TCN 인식률 개선, 수어 정답률, 새 사용자 성능 향상을 측정한 결과가 아닙니다.</b></div></section>
    <section><h2>02. 같은 영상으로 세 관측 경로 비교</h2><div class="overflow"><table><thead><tr><th>조건</th><th>무엇을 보는가</th><th>확인 목적</th></tr></thead><tbody><tr><th>기존 DTW 관측</th><td>전체 화면에서 얻은 기존 관절</td><td>현재 비교의 기준</td></tr><tr><th>전체 검출 후 범위 제한<br><small>masked</small></th><td>C의 전체 검출 뒤 상반신 범위 안 관절 사용</td><td>하단 영역 제외의 영향 포함</td></tr><tr><th>웹캠 경계 모사 관측<br><small>sensor</small></th><td>C의 상반신 픽셀에서 얻은 저장 관절</td><td>화면 밖 손을 볼 수 없는 조건</td></tr></tbody></table></div><p>참고와 질의는 각 조건 안에서 같은 관측을 사용했습니다. DTW 공식·시간 정렬·최소 근거 조건은 고정했습니다. 다만 검출·좌우 배정·범위·시작/끝 구간이 함께 달라지는 비교이므로, 숫자를 한 요소의 단독 효과로 해석할 수는 없습니다.</p></section>
    <section><h2>03. 얼마나 많은 항목에 수치를 낼 수 있었나?</h2>{cover}<p class="muted">분모1,773은 DEV300영상의 손×3항목 수에서 기존에 반대손이 불필요하다고 지정된27항목을 뺀 값입니다. 정확도가 아닙니다. 같은 영상의 세 항목은 독립된 시험이 아닙니다.</p><div class="overflow"><table><thead><tr><th>항목</th><th>기존 DTW</th><th>범위 제한</th><th>웹캠 경계 모사</th></tr></thead><tbody>{axis}</tbody></table></div><p>웹캠 경계 모사에서 측정 불가로 바뀐404항목 중 <b>403개는 왼손, 1개는 오른손</b>입니다. 여기에는 내려놓은 손이 제외된 정상적인 관측 변화도 포함됩니다. 반대로, 모든 왼손 누락을 무시해도 된다는 뜻도 아닙니다.</p><p>기존12개 연습단어의 DEV36영상만 보면 세 항목 모두 측정·기준 비교가 가능한 수는 <b>기존36 → 범위 제한30 → 웹캠 경계 모사35</b>였습니다. 이는 단어를 맞힌 횟수가 아닙니다.</p></section>
    <section><h2>04. 실제 장면에서는 서로 다른 문제가 섞여 있었습니다</h2><p>공개6영상에서 18장면을 확인했습니다. 아래를 펼치면 원본과 세 관측을 나란히 볼 수 있습니다. 노란선은 상반신 하단, 초록은 왼손, 분홍은 오른손입니다. 원본 전체 화면 위에 저장 관절을 표시했으며 새 검출을 실행하지 않았습니다.</p>{figures}<div class="notice"><b>핵심 사례: 믿다 ID12</b><br>왼손 굽힘에 쓸 수 있는 참고는9→3개로 줄어 최소5개 조건을 통과하지 못했습니다. 같은 영상의 위치·궤적은7개 참고로 비교 가능합니다. <b>위치를 안다고 손 모양까지 안다고 할 수는 없습니다.</b></div><p class="muted">이 장면들은 비교 후 원인 설명을 위해 선정한 진단 표본입니다. 전체 누락 구간을 검토하거나 언어적 정답을 확정한 것은 아닙니다. 특정 단어 규칙·필수손 라벨·계수는 바꾸지 않았습니다.</p></section>
    <section><h2>05. 좋아진 점과 아직 해결되지 않은 점</h2><div class="grid"><div class="tile"><h3>완료된 개선</h3><p>여러 사람의 참고를100단어로 확장. 좌우 손과 굽힘·위치·궤적을 따로 기록. 비교 근거가 부족한 이유를 추적할 자료 확보.</p></div><div class="tile"><h3>남은 문제</h3><p>범위 밖 정지손과 범위 안 동작손 누락이 혼재. 일부 손가락만 안 보여도 굽힘 전체가 보류됨. 관측 시작 시점에 따라 궤적 기준점이 달라짐.</p></div><div class="tile"><h3>아직 말할 수 없는 것</h3><p>수어가 더 정확해졌는지, 틀린 수행을 잘 거절하는지, C·T3-F 인식률이 높아졌는지. 이번에는 검증하지 않았습니다.</p></div></div><p>새88단어는 필수손을 확정하지 않아 <b>손별 차이만 기록</b>했습니다. 가만히 있는 손도 다른 손을 받치는 중요한 역할일 수 있습니다. 단순히 덜 움직이거나 안 보인다고 빼지 않았습니다.</p><p>기존 관측의 높은 측정 가능 수에는 쉬운 정지 자세를 비교하는 효과도 들어갈 수 있습니다. 공통으로 측정 가능한 항목에서도 C 경로의 차이가 전반적으로 더 커졌지만, 비교 구간·근거·첫 관측 위치가 달라져 그 자체를 성능 저하로 단정할 수 없습니다. 각 경로의 TRAIN 기준도 따로 만들었습니다.</p></section>
    <section><h2>06. 다음은 점수를 바꾸기보다, 근거를 나누는 보완</h2><ol><li><b>손 위치와 손 모양의 관측을 분리합니다.</b> 관측된 각도와 부족한 각도를 구분하는 공통 후보를 먼저 검증합니다. 잘 보이는 몇 개만으로 전체 손이 맞았다고 판정하지 않고, 근거가 부족하면 보류합니다.</li><li><b>손별·구간별 관측 상태를 함께 봅니다.</b> 내려놓은 구간과 화면 안 겹침/미관측을 나누고, 받침손을 무조건 제외하지 않습니다. 측정 가능 수만 늘리는 기준 완화는 하지 않습니다.</li><li><b>궤적의 시작 기준을 점검합니다.</b> 현재는 각 손이 처음 보인 위치가 원점입니다. 화면 진입 시점 차이가 궤적 수치에 미치는 영향을 분리한 뒤 같은 자료로 회귀를 확인합니다.</li></ol><p>위 후보의 구체 조건을 확인한 뒤 진행하는 것을 제안합니다. <b>인식 보조와 추가 학습은 이번 결과를 함께 검토한 다음</b>으로 남겨두었습니다. 전 단어의 수동 단계표 작성을 다시 선행조건으로 두지 않습니다.</p></section>
    <section><h2>07. 단어별 측정 현황</h2><p>숫자는 각 단어의 DEV3명×손×3항목 중 수치를 낼 수 있는 항목 수입니다. 적은 수가 곧 수행 오류를 뜻하지 않습니다.</p><input id="search" aria-label="단어 검색" placeholder="단어 또는 WORD 코드로 찾기"><div class="overflow"><table id="words"><thead><tr><th>단어</th><th>참고 수</th><th>역할 처리</th><th>기존</th><th>범위 제한</th><th>경계 모사</th></tr></thead><tbody>{wordrows}</tbody></table></div></section>
    <section><h2>08. 검증과 보존</h2><p>원입력3,597개와 특징7,312,875값, 저장집계42,725개를 별도 산술 코드로 검산했습니다. 전체29,643경로의 연결 조건과 고정선정300쌍의 거리계산을 확인했습니다. 기존12단어 결과도 재현됐습니다. 같은 Codex가 별도 코드로 검산했으며 별도 전문가·에이전트 검토는 아닙니다.</p><p><b>기존 C·T3-F·웹·DTW 유지. 사용자·멘토 촬영 미접근. 추가 학습·인식 순위 변경 없음.</b> DEV는 기존에 사용한 개발자료이므로 새 맹검 성능을 의미하지 않습니다.</p><details><summary>최초 검산 오류와 정정 기록</summary><p>검산 스크립트 구문 오류와 float32 acos / float64 atan2 계산의 작은 반올림 차이를 최초 파일로 보존했습니다. 후자는 코사인 및 수학적 각도 오차 한계로 검산을 보완했습니다. 최대 각도 차이는 약0.036도였으며 원래 비교 점수·조건·모델은 수정하지 않았습니다.</p><a href="VERIFICATION_RECOVERY.json">정정 기록</a></details><p><a href="RESULT_KO.txt">전체 설명 원문</a> · <a href="SUMMARY.json">저장 집계</a> · <a href="INDEPENDENT.json">별도 검산</a> · <a href="VISUAL_REVIEW.json">장면 검토 기록</a> · <a href="FROZEN.json">동결 조건</a></p></section></main><script>document.getElementById('search').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('#words tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (U/'RESULT_KO.html').write_text(page,'utf-8')
    print(json.dumps({'report':'RESULT_KO.html','visual_frames_reviewed':18,'word_rows':len(bank)},ensure_ascii=False))
if __name__=='__main__':main()
