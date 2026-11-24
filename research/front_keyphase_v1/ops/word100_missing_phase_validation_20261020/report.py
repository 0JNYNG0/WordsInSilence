from pathlib import Path
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U=Path(__file__).resolve().parent;P=U.parent/'word100_metric_evidence_20261020'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
S=read(U/'SUMMARY.json');V=read(U/'VERIFICATION.json');rows=read(U/'ROWS.json');real=read(U/'ACTUAL_GAPS.json')
assert V['status']=='PASS'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
(U/'visual').mkdir(exist_ok=True)
cid='NIA_SL_WORD0003_REAL11_F';row=next(r for r in rows if r['clip_id']==cid and r['side']=='L' and r['condition']=='peak10')
with np.load(P/'features/sensor'/(cid+'.npz'),allow_pickle=False) as z:q={k:z[k] for k in z.files}
ts=q['timestamp_us']/1e6;mask=np.zeros(len(ts),bool);mask[row['mask_indices']]=True
fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,layout='constrained')
line=q['position_L'][:,1].copy();line[~q['position_valid_L']]=np.nan
axes[0].plot(ts,line,color='#146c94',lw=2.5,label='원래 관측된 손목 위치');axes[0].set_ylabel('몸 기준 세로 위치\n(어깨 너비 단위)')
masked=line.copy();masked[mask]=np.nan
axes[1].plot(ts,masked,color='#146c94',lw=2.5);axes[1].axvspan(row['window']['start_s'],row['window']['end_s'],color='#f8b35c',alpha=.6)
axes[1].set_ylabel('짧은 구간을 가린 관측');axes[1].set_xlabel('영상 시간 (초)')
axes[1].text(.03,.12,'가린 부분은 비교하지 않음 → 남은 같은 위치끼리는 차이 0',transform=axes[1].transAxes,fontsize=12,color='#994200')
axes[0].set_title('공개 영상 자기 비교: 일부를 못 봤어도 남은 부분의 차이는 0',loc='left',fontweight='bold',pad=12)
axes[0].legend(loc='upper left',frameon=False)
fig.savefig(U/'visual/self_mask.png',dpi=150);plt.close(fig)

keys=['handshape','location','movement'];labels=['손가락 굽힘','손 위치','이동 궤적'];m=S['conditions']['peak10']['metrics']
fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained');yy=np.arange(3)
for i,(k,label) in enumerate(zip(keys,labels)):
    d=m[k];left=0
    for field,color,name in [('decreased','#bd4d33','더 낮아짐'),('unchanged','#9aa8b4','동일'),('increased','#2d708c','더 높아짐')]:
        value=d[field];ax.barh(i,value,left=left,color=color,label=name if i==0 else None)
        if value>20:ax.text(left+value/2,i,str(value),ha='center',va='center',color='white',fontweight='bold')
        left+=value
    ax.text(left+4,i,f"총 {left}건",va='center')
ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,505);ax.set_xlabel('동일 손·영상의 가림 전후 비교 건수');ax.legend(ncols=3,loc='upper center',bbox_to_anchor=(.5,1.23),frameon=False)
fig.savefig(U/'visual/masking_counts.png',dpi=150);plt.close(fig)

cid2='NIA_SL_WORD1139_REAL12_F'
with np.load(P/'features/sensor'/(cid2+'.npz'),allow_pickle=False) as z:q2={k:z[k] for k in z.files}
t=q2['full_timestamp_us']/1e6;side='L';inside=q2['full_L_inside_source_but_view_missing'];core=q2['full_L_annotated_action'];pts=q2['full_L_any_hand_points'];ang=q2['full_L_angle_count']
fig,ax=plt.subplots(figsize=(11,3.2),layout='constrained');ax.step(t,ang,where='mid',color='#146c94',lw=2,label='관측 가능한 손가락 각도 수')
for i in np.flatnonzero(inside&core):
    l=(t[i-1]+t[i])/2 if i else t[i];r=(t[i]+t[i+1])/2 if i+1<len(t) else t[i]
    ax.axvspan(l,r,color='#e3a74b',alpha=.45)
ax.set_ylim(-.6,16.5);ax.set_yticks([0,5,10,15]);ax.set_xlabel('영상 시간 (초)');ax.set_ylabel('왼손 각도 수 / 15');ax.set_title('실제 공개 관측: 원본 안에 손 근거가 있지만 잘린 영상에서 놓친 구간',loc='left',fontweight='bold');ax.legend(frameon=False,loc='upper right')
fig.savefig(U/'visual/real_gap.png',dpi=150);plt.close(fig)

def table(mode):
    data=S['conditions'][mode]['metrics'];out=''
    for k,label in zip(keys,labels):
        x=data[k];out+=f"<tr><th>{label}</th><td>{x['baseline_available']}</td><td>{x['masked_available']}</td><td>{x['paired_effective']}</td><td>{x['decreased']}</td><td>{x['unchanged']}</td><td>{x['increased']}</td></tr>"
    return '<div class="scroll"><table><thead><tr><th>항목</th><th>원래 수치 있음</th><th>가린 뒤 수치 있음</th><th>실제 관측 제거·전후 비교 가능</th><th>차이 감소</th><th>동일</th><th>차이 증가</th></tr></thead><tbody>'+out+'</tbody></table></div>'

text=f'''공개 자료 관측 누락 검증 — 2026-10-20

결론: 손 전체가 안 보이면 측정 보류가 정상 작동했다. 그러나 짧은 구간을 못 보아도 나머지의 평균 차이는 계속 출력되고, 오히려 작아지는 현상을 재현했다. 이는 인식률 변화가 아니라 DTW 차이 수치의 해석상 한계다.

범위: 공개 AI Hub 100단어1199영상, TRAIN899참고와 DEV300질의. 기존 역할 기준으로591손·영상 단위(600에서 기존 한손 역할9 제외). sensor관측 경로, 저장 DTW정렬 재사용. 원래 결과는 재채점하지 않았다. 가림4조건,21252참고쌍·손·조건 계산. 특정 단어별 규칙 없음.

1. 전체손 가림:591손단위 모두 손굽힘·위치·궤적 측정 불가(None). 미관측을0차이로 바꾸지 않았다. 전체손이 아예 없는 경우의 안전장치는 유지된다.
2. 짧은 가림: 공개 주석 동작시간의10%를 손목 이동량 최대 구간에서 가렸다. 그 길이는0.0578~0.3367초,중앙0.1581초다. 실제 구별에 필수인 동작이라는 정답 라벨은 없다. 이 조건에서 전후 비교가 가능한 손굽힘424건 중195건,위치445건 중215건,궤적445건 중220건은 차이가 오히려 낮아졌다. 평균에 들어갈 어려운 구간이 빠질 수 있고,유효 참고 구성도 달라질 수 있기 때문이다. 수치가 낮아졌다고 잘 수행한 것이 아니다.
3. 자기비교 대조: 같은 공개 관측을 자기자신과 비교하면서 한쪽의 짧은 관측만 가리면,관측 제거가 실제로 있었는데도435손단위는15각도 전부 차이0,455손단위는 위치차이0이 나왔다. 실제 사람 간 평가나 정확도가 아닌 측정의 빈틈을 보이는 대조시험이다.
4. 실제 저장 관측: 주석구간 안 원본에서 손 전체가 영역 안에 관측되지만 sensor에서 손점이 없는355손프레임이132손·영상 단위에 있었다. 그중104손단위는 전체손굽힘 수치가 여전히 있었고115는 위치 수치가 있었다. 관측증거는 MediaPipe 추정이며,실제 가림·필수단계 누락의 정답이 아니다. 정상 손내림도 별도로 구분하여 자동 오답 처리하지 않았다.

보완 계획:
- 평균차이 옆에 각손·각항목의 관측시간비율,가장 긴 연속미관측시간,구간별 관측지도 표시. DTW대응쌍 비율과 실제시간비율을 구분한다.
- 전체손 없음은 측정불가,일부 놓침은 관측된 부분만 비교한 값이라고 표시. 놓친 구간을0점/보간정답/동작 생략으로 단정하지 않는다.
- 공개 자료의 정상속도변화·정상반복·정상내림을 대조조건으로 삼아,새 관측경고가 정상수행을 오류로 몰지 않는지 먼저검증. 초단위 누락과구간상대누락을 함께 점검하고 임계값을DEV결과에맞추지않는다.
- '필수동작을 실제로 생략했는가'는 별도 평가과제. 먼저 관측불명표시를 안정화하고,내용이관측된경우의 단계차이를 연구한다. 전단어 수동판정은 요구하지 않되,필수단계정답없이 정오판정성능을 주장하지 않는다.

독립 검산: 같은Codex가별도코드로2364가림손단위/21252쌍조건/42552출력값/591자기대조를 재현했고 보호29파일과 이전결과불변을 확인. 구현자와별도코드검산,다른에이전트검증은아님. 자체8검사PASS.

유지: C·T3-F·웹·운영DTW 변경0,추가학습0,인식보조0,개인·멘토촬영읽기0,보류14~18입력읽기0. 결과를같이확인한뒤 다음보완 결정.
한계: 고정된기존정렬을재사용한측정검증이며가림후전체웹경로의정렬변화까지평가하지않았다. 가림은센서관측제거이지잘못된수어수행라벨이아니다.100단어밖일반성/정답거절성능/인식률개선은미검증.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>관측 누락 검증 | WordsInSilence</title><style>
:root{font-family:"Malgun Gothic",Arial,sans-serif;color:#183041;background:#f1f4f6;line-height:1.75}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:40px 24px 72px}header{padding:26px 0}h1{font-size:clamp(28px,4vw,42px);line-height:1.3;margin:12px 0 18px;letter-spacing:-1.4px}h2{font-size:25px;line-height:1.4;margin:0 0 18px}h3{font-size:18px}p{margin:12px 0}.tag{color:#2b6f8a;font-weight:700;letter-spacing:1px}.lead{font-size:20px;max-width:900px}.box{background:white;border:1px solid #dce4e9;border-radius:18px;padding:28px;margin:22px 0}.note{border-left:4px solid #d8943d;background:#fff6e7;padding:16px 20px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.num{font-size:36px;font-weight:800;color:#1e6785;line-height:1.25}.small{font-size:14px;color:#455b6c}.alert{color:#9a422c}img{display:block;width:100%;height:auto;margin:22px 0 10px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px 10px;border-bottom:1px solid #dce4e9;text-align:left}thead{background:#edf3f6}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}li{margin:12px 0}a{color:#146c94}details{overflow:auto}summary{cursor:pointer;font-weight:bold}footer{font-size:13px;color:#4f6170}@media(max-width:680px){main{padding:20px 14px}.box{padding:20px}.grid{grid-template-columns:1fr}.num{font-size:30px}h2{font-size:22px}}
</style><main><header><div class="tag">WORDS IN SILENCE · 공개 자료 검증 · 2026.10.20</div><h1>못 본 동작이 있어도<br>차이는 작게 나올 수 있습니다</h1><p class="lead">손 전체의 미관측은 걸러졌습니다. 하지만 <b>짧은 구간의 미관측은 평균 차이만으로 드러나지 않았습니다.</b> 앞으로는 관측한 부분의 차이와 놓친 구간을 함께 보여줘야 합니다.</p><p>이번 검증은 연구용 계산입니다. 기존 C·T3-F, 웹 테스트 환경, 운영 DTW는 그대로 유지했습니다.</p></header>
<section class="grid"><div class="box"><div class="num">100단어</div><p>공개 영상1199개<br>참고899 · 개발 검증300</p></div><div class="box"><div class="num">591손 단위</div><p>영상별 왼손·오른손을 분리<br>기존 한손 역할9단위 제외</p></div><div class="box"><div class="num">학습 0회</div><p>사용자·멘토 촬영 사용0<br>기존 정렬·저장 결과 재사용</p></div></section>
<section class="box"><h2>1. 손 전체가 안 보이면?</h2><p><b>591개 손 단위 모두 손가락 굽힘·위치·궤적의 측정을 보류했습니다.</b> 관측이 없는 손에 차이0을 주지는 않았습니다.</p><p>여기서 측정 보류는 “틀렸다”가 아니라 “볼 수 없어서 비교할 근거가 없다”는 뜻입니다. 다른 손이 보인다고 보이지 않는 손까지 잘했다고 판단하지 않습니다.</p></section>
<section class="box"><h2>2. 짧은 구간만 놓치면?</h2><p>공개 주석에 기록된 동작시간 중 <b>10% 길이의 구간</b>을 가렸습니다. 손목의 관측된 이동량이 가장 큰 구간을 공통 규칙으로 골랐습니다. 가림 길이는 약0.06~0.34초, 중앙값0.16초입니다.</p><div class="note">이 구간은 움직임이 큰 구간의 후보입니다. 수어 구별에 반드시 필요한 단계라는 뜻은 아닙니다. 실제 동작을 바꾼 것이 아니라 관측 정보만 제거했습니다.</div><img src="visual/masking_counts.png" alt="짧은 관측 제거 후 일부 항목의 평균 차이가 오히려 낮아진 건수"><p><b>차이가 더 낮아진 경우:</b> 손굽힘195/424건 · 손위치215/445건 · 궤적220/445건.</p><p>분모는 실제 정보가 제거되었고 전후 모두 수치가 나온 손 단위입니다. 어려운 구간이 평균에서 빠지거나 비교 가능한 참고 영상 구성이 달라지면, 적게 보고도 더 비슷한 수치가 나올 수 있습니다.</p>'''+table('peak10')+'''<details><summary>중앙10%와 이동량 최대20% 조건도 보기</summary><h3>동작 구간의 중앙10% 가림</h3>'''+table('center10')+'''<h3>이동량 최대20% 가림</h3>'''+table('peak20')+'''<p>조건과 길이는 결과를 보기 전에 고정했습니다. 결과에 맞춘 임계값 선택이나 추가 탐색은 하지 않았습니다.</p></details></section>
<section class="box"><h2>3. “차이0”인데도 전체를 본 것은 아닙니다</h2><img src="visual/self_mask.png" alt="같은 공개 동작의 중간 관측을 가려도 남은 동일 위치의 차이는0인 예시"><p>공개 ‘수어’ 영상 REAL11 왼손의 자기 비교 예시입니다. 가운데 약0.18초, 관측5프레임을 가려도 나머지 좌표끼리는 같습니다. 따라서 <b>남은 부분의 위치 차이는0</b>입니다.</p><p>같은 방식의 전체 대조에서는 관측을 실제로 제거했는데도 <b>435손 단위의15각도 모두 차이0</b>, <b>455손 단위의 위치 차이0</b>이 나왔습니다. 같은 영상을 비교한 원리 확인용 시험이며 사람 간 정확도가 아닙니다.</p></section>
<section class="box"><h2>4. 실제 공개 영상에서도 관측 공백은 남아 있습니다</h2><p>원본 관측에는 영역 안 손 근거가 있지만 잘린 화면의 검출에는 손점이 없는 구간이 <b>132손 단위, 합계355손 프레임</b>에 있었습니다. 그중104손 단위는 전체 손굽힘 수치가,115손 단위는 손위치 수치가 계속 나왔습니다.</p><img src="visual/real_gap.png" alt="공개 믿다 REAL12 왼손의 시간별 각도 관측 수와 관측 누락 구간"><p>공개 ‘믿다’ REAL12 왼손 예시: 노란 부분은 주석 동작 구간 안에서 원본 관측과 잘린 화면 관측이 달랐던 시간입니다. 손굽힘 전체 수치는 이미 보류되었지만, 위치 차이는약14.99로 측정되었습니다. 이것이 놓친 시간까지 평가했다는 뜻은 아닙니다.</p><p class="small">원본 관측도 MediaPipe 추정입니다. 실제 가림 또는 수어 필수 단계의 정답 라벨은 아닙니다. 손을 정상적으로 내린 구간은 별도로 구분했으며 미관측을 곧바로 오답 처리하지 않았습니다.</p></section>
<section class="box"><h2>다음 보완 방향</h2><ol><li><b>차이와 관측 범위를 나란히 표시합니다.</b><br>각손·각항목의 평균차이, 실제 시간 기준 관측비율, 가장 긴 미관측시간, 시간대별 관측지도를 함께 제시합니다. DTW 대응쌍 비율과 시간 비율을 구분합니다.</li><li><b>못 본 부분은 ‘판단할 수 없음’으로 남깁니다.</b><br>전체손 없음은 측정불가, 짧은 공백은 관측된 부분만의 비교로 설명합니다.0점이나 임의 보간으로 빈칸을 정답처럼 채우지 않습니다.</li><li><b>정상 동작이 불이익을 받지 않는지 먼저 확인합니다.</b><br>정상 속도차이·반복·손내림을 대조조건으로 검사합니다. 관측 경고를 수행오류 판정과 분리하고, 필요하면 잘 보이게 다시 촬영하도록 안내하는 근거로 사용합니다.</li><li><b>필수 동작의 실제 생략 판정은 별도 검증합니다.</b><br>센서가 못 본 것과 사람이 안 한 것은 다릅니다. 먼저 관측불명 표시를 안정화한 뒤, 실제로 보이는 동작의 단계차이를 평가합니다. 전단어 수동 검토를 요구하지 않지만 정답 근거 없이 생략·오류 판정 성능을 주장하지 않습니다.</li></ol><div class="note">현재 결과로 확인한 것은 차이 수치의 한계입니다. 인식률 개선, 잘못된 수어를 거절하는 성능, 모든 수어에 대한 일반성은 아직 검증하지 않았습니다.</div></section>
<section class="box"><h2>검증 범위와 보존</h2><p>서로 다른 사람의 참고자료와 공개 개발 자료를 사용했습니다. 가림 뒤 정렬을 새로 최적화하지 않고 <b>기존 DTW 경로를 고정</b>하여 측정값의 영향을 분리했습니다. 실제 웹 전체 처리의 변화는 후속 검증 대상입니다.</p><p>별도 코드로2364가림 손단위·21252쌍조건·42552출력값과591자기대조를 재현했습니다. 자체8검사 및 보호29파일·이전 결과 해시 검사 통과. 구현과 검산은 같은 Codex가 서로 다른 코드로 수행했습니다.</p><p>추가 학습·인식 보조·운영 반영·사용자 촬영 접근은 모두0입니다. 결과를 함께 확인한 뒤 다음 단계로 진행합니다.</p></section>
<footer>근거: <a href="FROZEN.json">동결 조건</a> · <a href="SUMMARY.json">집계</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="ACTUAL_GAPS.json">실제 관측 공백</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></footer></main></html>'''
(U/'RESULT_KO.html').write_text(doc,'utf-8')
print('REPORT CREATED')
