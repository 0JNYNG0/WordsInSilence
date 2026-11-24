"""Readability pass only; all scientific scores/checkpoints stay unchanged."""
from common import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=read(U/'COMPARISON_INDEPENDENT.json');analysis=read(U/'ALL_CLASS_ANALYSIS.json')
draft=U/'report_draft_before';draft.mkdir(exist_ok=True)
for name in ['RESULT_KO.html','RESULT_KO.txt','late_accuracy.png','late_accuracy.svg']:
 assert not (draft/name).exists();shutil.copyfile(U/name,draft/name)
p=U/'RESULT_KO.html';s=p.read_text(encoding='utf8')
for end in [43,59]:s=s.replace(f'<td>[61, {end}]</td>',f'<td>해당 구간 없음 · {end}회 종료</td>')
s=s.replace('internal_reentry','중간 미관측 후 재등장').replace('one_hand_never_usable','한 손이 계속 미관측').replace('>other<','>기타<')
needle='<section><h2>3. 반복 혼동과 새 오답</h2>'
note='<p><b>sensor 900개 예측에서 정답이 합계 1개 늘었고, masked에서는 4개 줄었습니다.</b> 같은 300영상에 3개 모델을 적용한 합계이므로, 새 사람 900명에게 시험한 결과는 아닙니다. 동대문·백육십이·흡연실은 좋아졌지만, 상처·성남시수어통역센터 등은 나빠졌습니다.</p>'
assert s.count(needle)==1;s=s.replace(needle,needle+note)
needle='<p>미통과 항목: '
start=s.index(needle);end=s.index('</p>',start)+4
s=s[:start]+'<p>통과하지 못한 부분은 전체 검출 후 가린 조건의 평균 정확도, 촬영자 ID13, 중간 미관측 후 재등장 그룹 등입니다. 즉, 전체 평균의 작은 이득과 함께 기존에 맞히던 경우의 손실도 생겼습니다.</p>'+s[end:]
needle='<section><h2>4. 학습 후반의 흔들림</h2>'
s=s.replace(needle,needle+'<p>세 실행은 각각 43·67·59회에서 종료됐습니다. 선택용 손실이 12회 동안 더 좋아지지 않으면 멈추는 기존 규칙에 따른 정상 종료이며, 실행 오류가 아닙니다. 따라서 61회 이후 자료가 있는 두 번째 실행만 제한적으로 비교할 수 있습니다.</p>')
p.write_text(s,encoding='utf8')
p=U/'RESULT_KO.txt';s=p.read_text(encoding='utf8')
s+='\n개별 변화: sensor900예측에서 순증1, masked900예측에서 순감4. 동대문7→9/9, 백육십이8→9/9, 흡연실8→9/9 개선; 상처8→6/9, 성남시수어통역센터9→7/9 회귀. 같은3영상×3초기값이며 독립9명 아님.\n'
s+='종료43/67/59는 CE12회무개선 규칙의 정상 조기종료. 후반61회이후 비교는 2회째61~67만 가능. 안정화 일반결론불가.\n';p.write_text(s,encoding='utf8')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,3,figsize=(14,3.8),sharey=True,layout='constrained');colors={'C':'#758699','T3':'#087e8b'}
for i,ax in enumerate(axes):
 for arm in ['C','T3']:
  counts=r['late_per_seed'][arm][i]['sensor']['counts'];ax.plot(range(61,61+len(counts)),np.array(counts)/3,color=colors[arm],lw=2,label='기존 C' if arm=='C' else 'T3')
 end=analysis['runtime']['T3'][i]['epochs']
 ax.text(.04,.04,f'T3 {end}회 종료'+(' · 이 구간 기록 없음' if end<61 else ''),transform=ax.transAxes,fontsize=10,color='#086d79',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
 ax.set(title=f'실행 {i+1}',xlabel='학습 회차',ylim=(92.5,100.5),xticks=[61,65,70,75,80]);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('공개 개발 검증 정확도 (%)');axes[-1].legend(loc='upper right')
fig.savefig(U/'late_accuracy.png',dpi=170);fig.savefig(U/'late_accuracy.svg');plt.close(fig)
write(U/'REPORT_POLISH.json',{'at':now(),'changes':['label absent61+ windows instead of reversed range','explicit normal early stop explanation','plain Korean regression labels','aggregate gains/regressions and limitations','annotate missing T3 late curves'],'scores_changed':False,'new_inference':0,'source_sha256':sha(__file__)},new=True)
print('Report readability pass complete')
