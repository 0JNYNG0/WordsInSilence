"""Render already-verified diagnostics; no data extraction, inference, or training."""
from pathlib import Path
import json, hashlib, html
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,s):
    with p.open('x',encoding='utf-8') as f:f.write(s)
r=read(U/'RESULT.json');iv=read(U/'INPUTS_RECOUNT.json');c=read(U/'FROZEN_DIAGNOSTIC.json')
assert r['status']=='VERIFIED_SAMPLING_DIAGNOSIS_COMPLETE_NO_TRAINING'
assert iv['status']=='PASS' and r['new_training']==0 and r['user_recording_reads']==0
def m(v,n):return r['tables'][v][n]['development_only']
conds=['fps30','fps20_p0','fps15_p0']
labels=['30fps\n약 33ms마다','20fps 목표\n평균 약 50ms마다','15fps 목표\n약 67ms마다']
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,v,title,color in zip(axs,['masked','sensor'],['전체 영상 검출 → 상반신만 사용','상반신 영상에서 직접 검출'],['#5079a4','#168779']):
    vals=[m(v,n)['top1_percent'] for n in conds]
    bars=ax.bar(range(3),vals,color=color,width=.57)
    for i,b in enumerate(bars):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.2,
                f"{vals[i]:.2f}%\n{m(v,conds[i])['top1']}/300",ha='center',va='bottom',fontsize=11)
    ax.set_xticks(range(3),labels);ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100])
    ax.set_title(title,pad=34,fontweight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('정답 1위 비율 (%)')
fig.suptitle('같은 공개 개발 영상 300개 · 동작 시간은 그대로, 관측 프레임만 줄임',y=.99,fontweight='bold',fontsize=14)
fig.tight_layout(rect=[0,0,1,.93])
fig.savefig(U/'comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)

names={'WORD0500':'마음에 들다','WORD0003':'수어','WORD1642':'가평군수어통역센터','WORD1884':'강진군수어통역센터'}
for view in ['masked','sensor']:
    assert all(x['rank']==1 for rows in r['focus_words']['WORD0500'][view].values() for x in rows)
assert len(r['rgb_checks'])==12 and all(x['prediction']['target_rank']==1 for x in r['rgb_checks'])
rgbrows=[]
for wc,word in names.items():
    checks=[x for x in r['rgb_checks'] if wc in x['clip_id']]
    assert len(checks)==3
    rgbrows.append([word]+[str(next(x for x in checks if x['fps']==fps)['prediction']['target_rank'])+'위' for fps in [30,20,15]])
rows=[]
for v,label in [('masked','전체 검출 후 상반신만 사용'),('sensor','상반신 영상에서 직접 검출')]:
    for n,fps in zip(conds,[30,20,15]):
        d=m(v,n);delta=d['top1_percent']-m(v,'fps30')['top1_percent']
        rows.append([label,str(fps),f"{d['top1']}/300",f"{d['top1_percent']:.2f}%",'기준' if fps==30 else f'{delta:+.2f}%p'])
phase=[]
for v,label in [('masked','전체 검출 후 상반신만 사용'),('sensor','상반신 영상에서 직접 검출')]:
    for rate in [20,15]:
        phase.append([label,str(rate),f"{m(v,f'fps{rate}_p0')['top1']}/300",f"{m(v,f'fps{rate}_p5')['top1']}/300"])
def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(s)+'</th>' for s in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table>'

txt="""공개 자료 촬영 간격 영향 검증 — 2026-10-17

결론
이번 조건에서 30fps를 20·15fps 수준으로 성기게 관측한 영향은 작았다.
마음에 들다의 혼동은 재현되지 않았다. 촬영 간격이 주원인이라고 결론 내리거나
간격 보강 재학습을 바로 진행할 근거가 충분하지 않다. 새 학습과 웹 변경은 하지 않았다.

무엇을 비교했나
공개 AI Hub 100단어 1,199개 영상의 저장 관절 입력을 사용했다.
899개는 이미 학습에 사용한 영상이므로 참고 진단이며, 성능 표는 별도 촬영자 3명의
개발 영상 300개를 사용했다. 이 개발 자료도 과거 모델 선택에 사용했으므로 맹검 최종 시험이 아니다.
현재 웹에서 선택할 수 있는 C 한 개(seed20261013, epoch77)를 고정했다.
동작 시작·끝과 실제 시간은 그대로 두고 1초에 관측하는 장수만 줄였다.
빈 손을 채우거나 동작을 빠르게/느리게 만들지 않았다.
30fps 원래 개발 점수 600개(2입력 보기×300개)는 저장 결과를 재사용했다.

개발 결과 — 정답이 1위로 나온 비율
전체 검출 후 상반신만 사용: 30fps 294/300(98.00%), 20fps 293/300(97.67%), 15fps 291/300(97.00%).
상반신 영상에서 직접 검출한 입력: 30fps 295/300(98.33%), 20fps 295/300(98.33%), 15fps 294/300(98.00%).
두 번째 입력이 실제 화면 바깥을 관측할 수 없는 웹캠 조건에 더 가깝지만 실제 사용자 시험은 아니다.
모든 보기·간격 조건에서 상위5개 포함은 297/300(99.00%)이었다.
100단어 전체가 완전히 영향 없는 것은 아니다. 첫 번째 보기의 기본 15fps 조건에서
외상값·수영·김제시지부 각 1개가 정답→오답으로 바뀌었고, 두 번째 보기에서는 김제시지부 1개가 바뀌었다.

프레임 선택 시점도 바꿔 확인
주 비교는 첫 프레임을 기준으로 일정 간격 선택, 보조 비교는 선택 시점을 반 간격 이동했다.
전체 검출/20fps 정답293·295개, 15fps 291·293개.
상반신 직접 검출/20fps 정답295·294개, 15fps 294·294개.
가장 좋은 시점만 골라 채택하지 않았다. 위 숫자 범위는 신뢰구간이 아니다.

마음에 들다
개발 영상 3개가 두 입력 방식과 모든 간격·시점에서 정답 1위를 유지했다.
같은 3영상을 여러 조건으로 본 것이며 30명의 새 사용자 시험이 아니다.
따라서 이번 공개 자료에서는 규칙적인 간격 감소만으로 해당 혼동을 재현하지 못했다.
사용자 동작이 틀렸다는 뜻도 아니며 실제 사용자 실패 원인이 확정된 것도 아니다.

MediaPipe부터 다시 검출
마음에 들다·수어·가평군수어통역센터·강진군수어통역센터의 공개 TRAIN01 영상 각1개,
총4개를 30/20/15fps 조건으로 검출부터 다시 실행했다. 12조건 모두 해당 단어가 1위였다.
30fps 재검출은 기존 관측·사용 마스크 및 좌표와 정확히 같았다.
저속 조건은 손 검출 결과가 일부 달라졌으나 이4개에서는 분류 순위가 유지됐다.
관절 정답 좌표가 없으므로 좌표 변화량을 검출 정확도라고 부르지 않는다.
이미 학습한 4개 영상의 점검이므로 전체 단어·새 사람의 정확도를 보장하지 않는다.

검증 한계
30fps 원본에서 뽑은 20fps는 프레임 간격이 약33/67ms로 번갈아 나오는 근사 조건이다.
정규화 크기도 선택 관측에서 재계산했으므로 시간 채널 하나만의 순수 효과가 아니다.
주 비교의 동작 구간은 고정했고, 재검출4영상에서는 동작 구간 선택도 다시 계산했다.
실제 웹캠의 긴 멈춤·불규칙 지연·흔들림·조명·겹침 조합을 모두 검증한 것은 아니다.
한 C 모델 및 과거 개발 집합의 진단이며 새로운 개선 모델의 결과가 아니다.

보완 방향 — 사용자와 확인 후 진행
1. 촬영 간격 보강 재학습은 우선 보류. 현재 관측 범위에서 기대 이득이 입증되지 않았다.
2. 다음 진단은 공개 자료의 손 겹침·좌우 배정·관절 누락과, 비슷한 부분 자세 뒤의 동작 순서에 집중.
   지난 공개 감사에서 확인한 마음TRAIN06의 약0.47초 왼손 누락은 점검 근거이지 사용자 실패 원인 확정이 아니다.
3. 4표현의 공개 학습자료 36개에서 중요한 동작 구간의 관측 손실을 비교하고,
   앞/중간/뒤 구간이 각 단어를 구별하는 데 어떤 단서를 주는지 공통 기준으로 확인한다.
   단어명에 따른 예외나 마음에 들다 점수 상향은 하지 않는다.
4. 실제 개선 가능한 입력 문제가 확인되면 그 문제만 고치는 공통 후보를 제안한다.
   모델의 순서 처리 변경은 관측 문제와 분리해 검토한다. TCN도 원래 시간 순서를 학습하므로
   현재 모델이 순서를 전혀 못 본다고 설명하지 않는다.
5. 학습 전 조건·기준을 사용자와 확인한다. 이후 학습하더라도 공개TRAIN 내부에서 조건을 고정하고,
   기존 개발300으로 비교하되 반복 개발 평가임을 표시한다. 보류 촬영자14~18과 사용자 촬영은 사용하지 않는다.

수행·독립 검산
Codex 로컬 구현. 별도 검산 코드로 8,402개 증거 핀, 11,990개 조건 입력의 280채널 계산,
시간·마스크·원프레임 연결 및 저장 점수 정렬을 검산했다. 채널 최대 차이0, PASS.
이는 동일 담당자의 별도 코드 검산이며 다른 에이전트·수어 전문가의 검증이 아니다.
새 분류 계산11,402건은 반복 조건을 포함한 횟수다. 새 검출12조건, 새 학습0,
사용자 촬영 읽기0, 보류 촬영자 사용0, 모델·DTW·웹 변경0.
FROZEN_DIAGNOSTIC.json / INPUTS_RECOUNT.json / RESULT.json / FINAL_CHECK.json 참조.
"""
new(U/'RESULT_KO.txt',txt)
body=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>촬영 간격 영향 검증 | WordsInSilence</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f5;color:#203044;font-family:"Malgun Gothic",sans-serif;line-height:1.75}}main{{max-width:1080px;margin:auto;padding:42px 24px 70px}}h1{{font-size:32px;line-height:1.35;margin:8px 0 18px}}h2{{font-size:22px;line-height:1.4;margin:0 0 18px}}p{{margin:10px 0}}section{{background:white;border:1px solid #dce3e9;border-radius:14px;padding:28px;margin:20px 0}}.tag{{color:#186e65;font-weight:700}}.lead{{font-size:21px}}.key{{background:#e8f4f0;border-left:5px solid #168779;padding:18px 22px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e2e7ed}}th{{background:#f3f6f8}}img{{width:100%;height:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{border:1px solid #d9e4e9;border-radius:10px;padding:20px}}.number{{font-size:28px;font-weight:700;color:#147569}}a{{color:#136c9b}}li{{margin:8px 0}}@media(max-width:750px){{main{{padding:22px 14px}}.grid{{grid-template-columns:1fr}}section{{padding:20px}}table{{font-size:12px}}td,th{{padding:7px}}}}
</style><main><div class="tag">WordsInSilence · 공개 자료 진단 · 2026.10.17</div>
<h1>촬영 간격을 넓히면<br>‘마음에 들다’가 헷갈릴까?</h1>
<p class="lead">이번 실험에서는 혼동이 재현되지 않았다.<br><strong>촬영 간격만을 주원인으로 보기에는 근거가 약하다.</strong></p>
<div class="key">학습 전 점검 완료. 현재 C 모델을 그대로 검사했고, 새 학습이나 웹 수정은 하지 않았다.</div>
<section><h2>무엇을 바꿔 보았나</h2>
<p>동작을 하는 데 걸린 시간은 그대로 두고, <strong>영상에서 꺼내 보는 장수만 1초에 30장 → 20장 → 15장</strong> 수준으로 줄였다. 손이 보이지 않는 구간은 그대로 유지했다.</p>
<div class="grid"><div class="card"><div class="number">100단어 · 1,199영상</div><p>공개 AI Hub 자료. 학습에 이미 쓴 899영상과 별도 촬영자 3명의 개발 영상 300개를 구분했다.</p></div><div class="card"><div class="number">현재 C 모델 고정</div><p>추가 학습 없이 비교. 아래 성능은 개발300의 결과이며 실제 사용자 웹캠 정확도가 아니다.</p></div></div></section>
<section><h2>개발 영상 300개의 결과</h2><img src="comparison.png" alt="두 입력 방식의 30,20,15fps 정확도 비교">
{table(['입력 방식','목표 fps','정답 1위','정확도','30fps 대비'],rows)}
<p>상반신 영상에서 직접 검출하는 방식이 실제 웹캠의 화면 밖 미관측 조건에 더 가깝다. 주 비교에서는 <strong>20fps 정답 수 동일, 15fps에서 1개 감소(−0.33%p)</strong>였다. 모든 조건의 상위5개 포함률은 99.00%(297/300)로 같았다.</p>
<p>전체 검출 후 상반신만 남기는 방식에서는 15fps에서 3개 감소했다. 간격 변화의 영향이 완전히 없는 것은 아니다.</p></section>
<section><h2>‘마음에 들다’는 어떻게 됐나</h2><div class="key"><strong>개발 영상 3개 모두, 두 입력 방식과 모든 간격·선택 시점에서 정답 1위.</strong></div>
<p>따라서 이번 공개 영상에서는 규칙적으로 관측 장수를 줄이는 것만으로 앞서의 혼동을 만들지 못했다. 사용자 동작의 옳고 그름이나 실제 실패 원인은 이 결과로 판단할 수 없다.</p>
<p>한 손이 내내 미관측인 개발 영상도 상반신 직접 검출 입력에서 30·20·15fps 주 비교 모두 95/96개 정답을 유지했다. 이 집단은 관측 상태에 따른 구분이며 언어학적으로 한 손 수어라는 분류는 아니다.</p></section>
<section><h2>선택하는 순간을 바꾸어도 확인했다</h2>
<p>짧은 동작을 우연히 건너뛰는지 보기 위해 프레임 선택 시점을 반 간격 옮겼다. 더 잘 나온 결과만 고르지 않고 둘 다 보고한다.</p>
{table(['입력 방식','목표 fps','기본 시점 정답','반 간격 이동 정답'],phase)}
<p>전체 검출 입력의 기본15fps에서는 외상값·수영·김제시지부 각1개가 오답으로 바뀌었다. 직접 검출 입력에서는 김제시지부1개가 바뀌었고, 보조 시점에서는 강진군수어통역센터1개가 바뀌었다. 시점에 민감한 일부 사례는 남아 있다.</p></section>
<section><h2>관절 검출부터 다시 해도 같은가</h2>
<p>공개 학습 영상 4개는 저장 좌표만 줄이는 비교에 더해, 실제 영상의 장수를 줄인 뒤 <strong>MediaPipe부터 다시 검출</strong>했다.</p>
{table(['공개 영상 각1개','30fps','20fps 목표','15fps 목표'],rgbrows)}
<p>손 좌표와 관측 여부는 낮은 간격 조건에서 일부 달라졌지만, 이4영상에서는 정답1위가 유지됐다. 이미 학습한 작은 표본이므로 전체 단어나 새로운 사람에게도 같다고 일반화할 수 없다.</p></section>
<section><h2>어디까지 확인한 결과인가</h2><ul>
<li>30fps 원본에서 고른 20fps는 약33/67ms 간격이 번갈아 나오는 근사 조건이다. 실제 카메라의 모든 촬영 조건을 재현하지는 못한다.</li>
<li>실제 웹캠의 불규칙한 긴 멈춤, 손 겹침, 흐림, 조명 변화가 함께 일어나는 경우는 아직 확인하지 않았다.</li>
<li>같은 C 모델 한 개와 과거 모델 선택에 사용했던 개발 자료의 진단이다. 새 사람의 최종 정확도나 새 후보의 향상 수치가 아니다.</li>
<li>관측을 줄이면서 좌표 정규화도 같은 규칙으로 다시 계산했다. 시간 정보 하나만 바꾼 실험은 아니다.</li>
</ul></section>
<section><h2>보완 방향: 먼저 관측 손실과 동작 구별 단서 확인</h2>
<ol><li><strong>간격 보강 재학습은 우선 보류.</strong> 이번 결과로는 기대 이득을 뒷받침하기 어렵다.</li>
<li><strong>공개 영상의 손 겹침·좌우 배정·좌표 누락을 점검.</strong> 이전 공개 감사에서 마음에 들다의 일부 영상에 약0.47초 왼손 누락이 있었다. 중요한 동작과 겹치는지, 다른 표현에도 나타나는지 비교한다.</li>
<li><strong>비슷한 자세 이후의 전체 동작 순서 확인.</strong> 마음·수어·센터명 표현의 공개36영상을 공통 기준으로 검토한다. 현재 TCN도 순서를 학습하므로, 순서를 전혀 못 본다고 단정하지 않는다.</li>
<li><strong>확인된 문제에 맞춰 공통 보완안 제시.</strong> 입력 보완과 모델 변경은 분리한다. 특정 단어 점수 상향이나 단어별 예외는 두지 않는다.</li></ol>
<p><strong>결과와 방향을 사용자와 확인한 다음 학습 여부·조건을 정한다.</strong> 사용자·멘토 촬영을 학습·튜닝·후보 선택에 사용하지 않는다.</p></section>
<section><h2>검증 기록</h2><p>별도 계산 코드로 8,402개 증거 핀, 11,990개 조건 입력, 저장 점수 순위를 검산했다. 280채널 재계산 최대 차이는0이며 통과했다. 동일 Codex 담당자의 별도 코드 검산이고 다른 에이전트·전문가 검토는 아니다.</p>
<p>새 학습0 · 사용자 촬영 읽기0 · 보류 촬영자 사용0 · 모델/DTW/웹 변경0</p>
<p><a href="RESULT_KO.txt">상세 설명 원문</a> · <a href="RESULT.json">전체 결과</a> · <a href="INPUTS_RECOUNT.json">독립 계산 검산</a> · <a href="FROZEN_DIAGNOSTIC.json">사전 고정 조건</a></p></section></main></html>"""
new(U/'RESULT_KO.html',body)
# Final preservation check: hashes only, not re-extraction or re-inference.
pins=dict(c['pins'])
m0=ROOT/'research/front_keyphase_v1/runs/word100_tcn_v1/whole_context/best.pt'
pins[str(m0)]='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
for p,h in pins.items():assert sha(p)==h,p
assert all(x['old30_sameframe_observed_equal'] and x['old30_sameframe_usable_equal'] and x['old30_sameframe_max_joint_delta_pixels']==0 for x in r['rgb_checks'] if x['fps']==30)
final={'status':'PASS','at':datetime.now().astimezone().isoformat(timespec='seconds'),
       'protected_pins':pins,'report_sha256':sha(U/'RESULT_KO.html'),'text_sha256':sha(U/'RESULT_KO.txt'),
       'chart_sha256':sha(U/'comparison.png'),'result_sha256':sha(U/'RESULT.json'),
       'new_training':0,'user_recording_reads':0,'no_production_changes':True,
       'operator':'Codex root; separate-code verification, no external or subagent review'}
new(U/'FINAL_CHECK.json',json.dumps(final,ensure_ascii=False,indent=2))
print(json.dumps({'report':str(U/'RESULT_KO.html'),'preservation_check':'PASS'},ensure_ascii=False))

