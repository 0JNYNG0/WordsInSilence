from pathlib import Path
import json,html,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def read(n):return json.loads((U/n).read_text(encoding='utf-8-sig'))
info=read('INPUTS.json');rows=info['rows'];scores=read('SCORES.json');runtime=read('RUNTIME.json');seqs=read('LABELS.json')['sequences']
assert read('FINAL_VERIFICATION.json')['status']=='PASS'
out=U/'figures';out.mkdir(exist_ok=True)
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
targets=info['targets'];korean=dict(R_index_only='오른손 검지만 펴짐',R_open_four='오른손 네 손가락 펴짐',L_open_four='왼손 네 손가락 펴짐',R_face_overlap='오른손·얼굴의 화면상 겹침',hands_overlap='양손의 화면상 겹침')
probs={c:np.stack([np.load(U/'runs'/f'{c}_{seed}'/'probabilities.npy',allow_pickle=False) for seed in [42,43,44]]) for c in ['pose_only','pose_plus_signrep']}
arr=[dict(np.load(r['input'],allow_pickle=False)) for r in rows];lookup={r['alias']:i for i,r in enumerate(rows)}
roles={'held_person':'배운 표현 · 학습에 없던 사람','held_expression_seen_person':'학습에 없던 표현 · 배운 사람','held_expression_new_person':'학습에 없던 표현과 사람','all_held':'전체 평가','fit_candidate':'학습 자료'}
def pct(x):return f'{100*x:.2f}%'
table=''
for g in ['held_person','held_expression_seen_person','held_expression_new_person','all_held']:
 a=next(s for s in scores['summary'] if s['group']==g and s['condition']=='pose_only');b=next(s for s in scores['summary'] if s['group']==g and s['condition']=='pose_plus_signrep')
 table+=f'<tr><td>{roles[g]}<small>{a["clips"]}영상 / {a["known_cells"]}확인 항목</small></td><td>{pct(a["mean"])}<small>{pct(a["range"][0])}–{pct(a["range"][1])}</small></td><td>{pct(b["mean"])}<small>{pct(b["range"][0])}–{pct(b["range"][1])}</small></td><td>{100*(b["mean"]-a["mean"]):+.2f}%p</td></tr>'
featuretable=''
for k in targets:
 vals=[]
 for c in ['pose_only','pose_plus_signrep']:
  vals.append(np.mean([s['per_target'][k]['balanced_accuracy'] for s in scores['rows'] if s['group']=='all_held' and s['condition']==c]))
 featuretable+=f'<tr><td>{korean[k]}</td><td>{pct(vals[0])}</td><td>{pct(vals[1])}</td><td>{100*(vals[1]-vals[0]):+.2f}%p</td></tr>'
cards=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for s in seqs:
 i=lookup[s['source_alias']];a=arr[i];frames=s['frames'];first,last=frames[0]['frame_id'],frames[-1]['frame_id'];selected=(a['frame_id']>=first-4)&(a['frame_id']<=last+4);t=a['timestamp_us']/1e6
 feats=[s['side']+'_'+f for f in ['index_only','open_four'] if s['side']+'_'+f in targets]
 fig,axes=plt.subplots(len(feats),1,figsize=(11,2.5*len(feats)),squeeze=False)
 for ax,k in zip(axes.flat,feats):
  j=targets.index(k)
  for c,color,label in [('pose_only','#d87517','관절만'),('pose_plus_signrep','#2563b9','관절 + SignRep')]:
   pp=probs[c][:,i,:len(t),j];ax.plot(t[selected],pp.mean(0)[selected],color=color,label=label);ax.fill_between(t[selected],pp.min(0)[selected],pp.max(0)[selected],color=color,alpha=.15)
  known=a['target_mask'][:,j]&selected;ax.scatter(t[known],a['y'][known,j],s=30,color='black',marker='s',label='RGB 재확인 주석',zorder=4)
  unknown=~a['target_mask'][:,j]&selected;ax.scatter(t[unknown],np.full(unknown.sum(),-.08),color='#999',s=12,marker='x',label='주석 없음/불확실')
  ax.axhline(.5,ls='--',color='#aaa',lw=.8);ax.set(ylim=(-.15,1.15),ylabel='해당 특징의 예측값',xlabel='원본 영상 시간(초)',title=korean[k]);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right',ncol=2)
 fig.suptitle(f'{s["case_id"]} · {s["label"]} · {s["source_alias"]} — {roles[s["role"]]}',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'{s["case_id"]}.png',dpi=130);plt.close(fig)
 strip=Image.new('RGB',(1820,230),'#eef2f6');d=ImageDraw.Draw(strip)
 for c,fr in enumerate(frames):
  im=Image.open(fr['focus_image']).convert('RGB');im.thumbnail((136,195));strip.paste(im,(c*140+(140-im.width)//2,30));d.text((c*140+3,5),f'f{fr["frame_id"]}',font=font,fill='black')
 strip.save(out/f'{s["case_id"]}_frames.jpg',quality=92)
 cards.append(f'<details id="{s["case_id"]}"><summary>{s["case_id"]} · {s["label"]} · {s["source_alias"]} · {roles[s["role"]]}</summary><p>원본에서 연속한 13프레임. 검은 네모는 이번에 쓴 주석, 선은 모델의 추정, 옅은 띠는 3회 학습 차이야. 회색 ×는 정답을 확인하지 않은 부분이므로 선이 자신 있어도 맞았다고 평가하지 않았어.</p><img src="figures/{s["case_id"]}_frames.jpg"><img src="figures/{s["case_id"]}.png"></details>')
text=f'''SignRep 시간별 판정부 비교 — 일부 개선, 전체 적용은 보류

공개 한국수어 12표현·기록ID 3명·36영상, 4,351프레임을 사용했다. 기존108정지장면과36개 짧은 연속구간(각13프레임)을 RGB로 재확인했다. 불확실17항목을 추가 보류했고, 중복을 합친 확인항목은1,112개(학습461/평가651)다. 인접 프레임은 서로 닮았으므로1,112개의 독립 사례라는 뜻은 아니다. Codex가 재검토한 임시 물리 주석이며 독립 사람·수어교육자 정답은 아니다.

학습16영상(8표현×2명), 다른 사람8영상, 새 표현·기존사람8영상, 새 표현·새사람4영상. 이전에 살펴본 개발자료이며 새로운 외부시험이 아니다. C·T3-F 단어 인식률을 다시 측정한 것도 아니다.

공식SignRep 본체55,644,329개 파라미터는 고정했다. 관절만 쓰는 비교용 시간판정부23,141개와 관절+SignRep 판정부47,749개를 각3회, 고정160epoch 학습했다. 기존 운영DTW를 그대로 채점한 것이 아니라, 같은 임시 정답·시간판정부에서 영상정보의 추가 효과를 비교했다. 두 판정부의 크기도 다르므로 개선을 영상의 의미이해 한 가지 원인으로 확정할 수 없다.

5항목: 오른손 검지만 펴짐, 오른손 네 손가락 펴짐, 왼손 네 손가락 펴짐, 오른손과 얼굴의 화면상 겹침, 양손의 화면상 겹침. 엄지 모양·모든 손모양·정확한 접촉·필수 반복·수어의 교육적 정오는 이 정답에 없다. '네 손가락 펴짐 아님'은 곧 '주먹'이 아니다.

3회 평균의 항목별 균형정확도: 배운표현/새사람91.74→97.06%(+5.32%p), 새표현/기존사람62.60→60.38%(-2.23%p), 새표현/새사람66.68→61.36%(-5.33%p). 평가전체73.24→71.42%(-1.82%p). 항목별 긍정·부정을 균등하게 반영해 평균한 값이며 단어인식률이 아니다. 학습자료에서는 두 조건 모두100%지만 평가에는 그 성능이 유지되지 않았다.

손모양 변화: 평가7개 변화구간에서 변화 전후 모양을 모두 맞힌 수가 관절만1–3개(평균2개), SignRep추가4–5개(평균4.33개)였다. 7개를3회 반복한 결과이며21개 별개사례가 아니다. 정확한 변화프레임·필수단계 누락 검출을 증명하지 않는다. 같은 상태가 연속 확인된377프레임쌍에서는 잘못 바뀌었다고 한 수가4/12/17→8/2/6으로, 합계33→16이었다. 한 seed에서는4→8로 나빠졌다.

평가전체 오른손 네 손가락 구분84.67→88.54%는 개선됐지만, 왼손 네 손가락71.43→64.13%, 오른손·얼굴 겹침76.74→74.28%, 양손 겹침64.49→60.58%는 나빠졌다. 정상 교차/관절 좌표/좌우 배정은 수정하지 않았다. 겹침 자체를 잘못된 수어라고 지적하는 기능도 넣지 않았다. 새 표현의 겹침 판정까지 정상 유지했다고 주장할 수는 없다.

시간: IntelArc140V GPU, 원본3영상×2회, 워밍업 후 중앙{runtime['median_total_seconds']:.2f}초({runtime['range_total_seconds'][0]:.2f}–{runtime['range_total_seconds'][1]:.2f}초). 원본RGB읽기+고정상반신범위SignRep+저장된관절값읽기/변환+새판정부 포함. 실제MediaPipe추출·DTW·웹전송/화면표시 제외. 초기모델준비/워밍업은 별도{runtime['setup_compile_warmup_seconds']:.2f}초. 새판정부만 중앙약{runtime['head_only']['pose_plus_signrep']['median_seconds']*1000:.2f}ms다. 웹최종피드백3.2초 보장은 아니다.

원인에 대한 판단: 이번 작은 자료에서는 학습한표현의 모양은 잘 익혔지만 처음보는표현에 같은특징을 적용하는능력이 부족하다. 예를들어 학습의 '왼손 네손가락펴짐' 긍정은2개표현에서만 나온다. 표현의 전체화면흐름을 외워서 손모양을 추정했을 가능성, 224해상도에서작은손/16프레임요약으로짧은변화가흐려지는 가능성은 남지만 인과관계가 확정된 것은 아니다. 단순히GPU나모델이름을 바꿔 해결됐다고 볼 수 없다.

추가확인: 저장점수·분리·정규화·해시427검사 통과. 학습전 입력36개를 별도코드로 검산. 모르는정답값을 바꿔도 학습손실/기울기가동일. 모든 점수·선택epoch는 동결했으며 평가점수로 재학습/가중치조정하지 않았다. 표현시간을0.75/1.25로바꾼 저장특징 시험은651항목중1–3개결정변화였지만 원본영상을다른속도로다시추출한시험이 아니므로 실사용속도검증은 아니다.

결론: '익숙한 표현을 다른 사람이 할 때'와 '일부 짧은 손모양변화'에는 개선이 있었다. 하지만 전체성능과 새표현에서는 후퇴했으므로 운영후보 채택 보류. C·T3-F·운영DTW/웹은 유지했고 사용자·멘토자료 접근/학습/튜닝0, 새다운로드0이다.

다음 한 작업: 새 모델을 찾거나 지금 평가자료 점수에 맞추기보다, 같은 손모양·관계가 서로 다른 표현에서도 충분히 나타나는 학습용 주석을 보강한다. 특히 왼손모양·얼굴/양손관계·짧은변화와 이동방향의 시작/끝 근거를 분리하고, 필수단계·반복/교육적오류는 확인되지 않은 채 정답으로 넣지 않는다. 그다음 고정한 별도 사람/표현에서 재검증한다. 아직 모델이 수어설명의 규칙을 읽고 필수동작 누락을 판단하는 전체피드백 기능은 완성되지 않았다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SignRep 시간별 판정부 비교</title><style>
body{{font:17px/1.7 "Malgun Gothic",sans-serif;background:#f1f5fa;color:#17243a;margin:0}}main{{max-width:1140px;margin:auto;padding:32px}}h1{{font-size:32px}}h2{{margin-top:42px;font-size:25px}}section,details{{background:white;padding:24px;margin:18px 0;border-radius:14px;border:1px solid #dce4ed}}.lead{{background:#fff7df;border-left:6px solid #bc8524}}table{{width:100%;border-collapse:collapse}}td,th{{padding:13px;text-align:left;border-bottom:1px solid #dce4ed}}small{{display:block;color:#5e697a;font-size:13px}}img{{width:100%;height:auto}}summary{{cursor:pointer;font-weight:bold}}a{{color:#174d9a}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.metric{{font-size:29px;font-weight:bold;color:#215cb2}}.note{{color:#5a6676}}@media(max-width:700px){{main{{padding:12px}}.grid{{display:block}}table{{font-size:13px}}}}
</style><main><p>WordsInSilence · 2026-10-29 · 연구용 비교 완료 / 운영 변경 없음</p><h1>익숙한 표현에서는 개선.<br>새 표현까지 좋아지지는 않았어.</h1><section class="lead"><b>이번 후보를 전체 피드백에 적용하는 것은 보류했어.</b><p>공개 한국수어의 손모양 주석을 재확인하고, SignRep 본체를 고정한 채 작은 시간별 판정부를 실제로 학습했어. 손모양 변화는 더 많이 찾았지만, 처음 보는 표현에서 잘못 추정한 부분이 늘었어.</p></section>
<div class="grid"><section><small>확인 범위</small><div class="metric">12표현 · 36영상</div><p>기록ID 3명 / 4,351프레임<br>학습16영상 · 평가20영상</p></section><section><small>실제 학습</small><div class="metric">2조건 × 3회</div><p>동일 관절판정부와 SignRep추가 비교<br>본체 고정 · 최종160epoch</p></section><section><small>GPU 분석시간 / 워밍업 후</small><div class="metric">{runtime['median_total_seconds']:.2f}초</div><p>MediaPipe추출·DTW·웹시간 제외<br>전체서비스 응답시간이 아님</p></section></div>
<h2>무엇을 배우게 했나</h2><section><p>① 오른손 검지만 폈는가 ② 오른손 네 손가락을 폈는가 ③ 왼손 네 손가락을 폈는가 ④ 오른손이 얼굴과 화면상 겹치는가 ⑤ 두 손이 화면상 겹치는가. 이 다섯 가지를 시간에 따라 추정하게 했어.</p><p>손모양·위치 관계를 파악하는 작은 부품이야. 모든 손모양을 이해하거나, 정답 수어의 필수 단계·반복 횟수를 알고 잘못을 교정하는 전체 AI가 완성된 것은 아니야. 겹침도 그 자체로 오류가 아니야.</p><p class="note">108정지장면 + 36개 연속13프레임 구간을 RGB로 재검토. 불확실17항목 추가 보류. 중복을 합친 확인항목1,112개(학습461/평가651). Codex의 임시 물리 주석이며 독립 사람·수어전문가 정답이 아님. 관절기준도 이번 비교용으로 새로 학습한 판정부이고 운영TCN/DTW를 직접 채점한 수치가 아님.</p></section>
<h2>같은 조건으로 비교한 결과</h2><section><table><tr><th>평가 조건</th><th>관절만</th><th>관절 + SignRep</th><th>변화</th></tr>{table}</table><p class="note">굵은 값은 3회 평균, 아래는 범위. 다섯 항목의 긍정·부정 균형정확도 평균이며 수어 단어 인식률이 아님. 이미 살펴본 작은 개발묶음이며 새 외부검증이 아님. 다른 사람은 기록ID 기준, 실제개인식별을 새로 한 것이 아님.</p></section>
<h2>시간에 따른 변화는 더 찾았나</h2><section><p><b>평가 7개 손모양 변화구간</b>에서 변화 전후 모양을 모두 맞힌 수는 <b>1–3개 → 4–5개</b>로 늘었어. 같은 7개를 세 번 학습해 비교한 결과야. 정확한 변화 순간이나 수어의 중요한 단계 누락까지 검증된 것은 아니야.</p><p>같은 모양이 연속 확인된377프레임쌍에서 잘못 바뀌었다고 판단한 수는 세 번 합계 <b>33 → 16</b>으로 줄었지만, 한 번의 학습에서는4→8로 늘었어. 모든 경우가 개선된 것은 아니야.</p><p><a href="#N05">N05 모양 변화 보기</a> · <a href="#T15">T15 짧은 변화 보기</a> · <a href="#N14">N14 교차 중 관측 보기</a></p></section>
<h2>무엇이 좋아지고 나빠졌나</h2><section><table><tr><th>평가전체 항목</th><th>관절만</th><th>SignRep추가</th><th>변화</th></tr>{featuretable}</table><p>오른손 모양은 조금 좋아졌지만 왼손 모양과 겹침은 나빠졌어. 학습에 쓰인 모양·표현의 조합이 적어 다른 표현에서도 같은 특징을 알아보는 능력이 부족하다는 결과야. 원인을 특정 구조 하나로 확정하지는 않았어.</p><p>기존 정상 관절좌표와 좌우배정은 수정하지 않았어. 관측하지 못한 손모양을 모델이 자신 있게 추정하더라도 확인된 사실로 승격하지 않았고, 사용자를 잘못했다고 지적하는 기능으로도 연결하지 않았어.</p></section>
<h2>속도와 한계</h2><section><p>노트북 IntelArc140V에서 원본3영상×2회: <b>{runtime['range_total_seconds'][0]:.2f}–{runtime['range_total_seconds'][1]:.2f}초</b>, 중앙{runtime['median_total_seconds']:.2f}초. 원본읽기·SignRep·저장관절변환·새판정부 포함. 실제관절추출/DTW/웹표시는 제외. 초기준비·워밍업은 별도{runtime['setup_compile_warmup_seconds']:.2f}초, 새판정부 자체는 약{runtime['head_only']['pose_plus_signrep']['median_seconds']*1000:.2f}ms였어.</p><p>SignRep는16프레임을 묶어 만든 특징을 사용해. 매 프레임 출력은 낼 수 있어도, 그것만으로 매 프레임의 세밀한 손모양을 확실히 봤다는 뜻은 아니야.</p></section>
<h2>다음에 집중할 작업</h2><section><p>모델을 또 바꾸기보다 <b>같은 손모양과 관계가 여러 표현에 골고루 나타나는 학습 주석</b>을 보강하는 것이 다음 작업이야. 특히 왼손·얼굴과의 관계·양손 관계·짧은 변화와 이동 방향을 함께 확인해야 해. 필수 단계나 반복 횟수는 설명에 있다는 이유만으로 영상의 모든 순간에 정답을 자동 부여하면 안 돼.</p><p>C·T3-F와 운영 웹/DTW는 유지. 개인 촬영 접근/학습/튜닝0, 외부다운로드0. 입력36개 검산, 저장결과427검사 통과. 별도 검산코드는 같은 Codex가 작성했으며 독립 사람검토와 다름.</p><a href="RESULT_KO.txt">상세 설명</a> · <a href="SCORES.json">원수치</a> · <a href="LABEL_REVIEW.json">주석 검토</a> · <a href="FINAL_VERIFICATION.json">검산</a></section>
<h2>원본 장면과 시간별 예측 직접 보기</h2><p>학습자료의 잘 맞는 그래프와 평가자료를 구분해서 봐줘. 검은 정답표시가 없는 부분은 점수가 높아도 정확성이 확인된 것이 아니야.</p>{''.join(cards)}</main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print('Reports +36 plots/strips written')
