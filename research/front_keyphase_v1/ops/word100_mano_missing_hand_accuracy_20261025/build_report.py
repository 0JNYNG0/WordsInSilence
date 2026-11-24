from pathlib import Path
import json,html,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
S=read(U/'REVIEWED_SCORES.json');CROPS=read(U/'DISPLAY_CROPS.json')
FONT=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16);SMALL=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13)
LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
(U/'views').mkdir(exist_ok=True)
for r in S['rows']:
 im=Image.open(r['source_frame']);crop=CROPS[r['id']];z=np.load(r['comparison']);t,s=r['frame'],r['side'];raw=np.load(OLD/f"raw_predictions/{r['clip_index']:02d}_{t:04d}_{s}.npz")['xy_px']
 tiles=[]
 for c,title in [('B','기존 관측'),('rawH','HaMeR 원후보'),('H','통합 후보 H'),('HT','시간 확인 후보 HT')]:
  view=im.copy();d=ImageDraw.Draw(view)
  for side in [0,1]:
   valid=z['B_valid'][t,side];xy=z['B_xy'][t,side]
   for a,b in LINKS:
    if valid[a] and valid[b]:d.line([tuple(xy[a]),tuple(xy[b])],fill='#3be2cf',width=2)
  if c!='B':
   xy=raw if c=='rawH' else z[c+'_xy'][t,s];valid=np.isfinite(xy).all(-1) if c=='rawH' else z[c+'_valid'][t,s]
   for a,b in LINKS:
    if valid[a] and valid[b]:d.line([tuple(xy[a]),tuple(xy[b])],fill='#ffab40',width=2)
   for j in np.flatnonzero(valid):
    x,y=xy[j];d.ellipse((x-2,y-2,x+2,y+2),fill='#ffd082')
  for p in r['conditions'][c]['points']:
   if not p['scorable']:continue
   b=p['region'];d.rectangle(b,outline='#ffffff',width=1)
   if p.get('available'):
    x,y=p['xy'];color='#f86b7e' if not p['inside_region'] else '#b7ff74';d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
  cropped=view.crop(crop);cropped=ImageOps.contain(cropped,(316,410))
  tile=Image.new('RGB',(328,470),'#132135');tile.paste(cropped,((328-cropped.width)//2,55+(410-cropped.height)//2));td=ImageDraw.Draw(tile)
  td.text((10,7),title,font=FONT,fill='white');td.text((10,30),'대상 손: '+('있음' if r['conditions'][c]['any_hand'] else '없음 / 보류'),font=SMALL,fill='#b7cce6');tiles.append(tile)
 out=Image.new('RGB',(1312,470))
 for k,v in enumerate(tiles):out.paste(v,(k*328,0))
 out.save(U/'views'/f"{r['id']}.jpg",quality=96)

def table(kind):
 rows=[]
 for c,label in [('B','기존 관측 B'),('rawH','HaMeR 원후보'),('H','통합 후보 H'),('HT','시간 확인 HT')]:
  q=S['summary'][c][kind];m=q['mean_distance_to_center_px'];e=q['mean_distance_to_region_px']
  rows.append(f'<tr><td>{label}</td><td>{q["predictions_available"]} / {q["visual_regions"]}</td><td>{q["inside_visual_region"]}</td><td>{q["outside_visual_region"]}</td><td>{q["not_output"]}</td><td>{"—" if m is None else f"{m:.1f}"}</td><td>{"—" if e is None else f"{e:.1f}"}</td></tr>')
 return '<table><thead><tr><th>조건</th><th>좌표 출력 / 검토점</th><th>시각 범위 안</th><th>범위 밖</th><th>보류</th><th>범위 중심 거리(px)</th><th>범위 밖 거리(px)</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
cards=[]
for r in S['rows']:
 w=r['conditions']['HT']['points'][0];tips=[p for p in r['conditions']['HT']['points'][1:] if p['scorable']];av=[p for p in tips if p['available']]
 wrist='보류' if not w['available'] else ('시각 범위 안' if w['inside_region'] else f'시각 범위 밖 {w["distance_to_region_px"]:.1f}px')
 if not tips:tiptext='손끝 번호·위치 확인 어려움 → 검증 보류'
 else:tiptext=f'확인 가능한 손끝 {len(tips)}개 중 출력 {len(av)}개, 시각 범위 안 {sum(p["inside_region"] for p in av)}개'
 gate=r['gate'];why={'existing_hand_duplicate_or_ambiguous':'다른 손과 중복인지 불확실해 통합 보류','pose_transition_uncertain_not_error':'앞뒤 자세 변화가 커 시간 확인에서 보류','temporally_consistent_estimate':'앞뒤 움직임 검사 통과 — 정확하다는 증명은 아님','current_arm_identity_not_confirmed':'팔과의 좌우 연결 근거 부족'}.get(gate.get('HT_reason',gate['reason']),gate.get('HT_reason',gate['reason']))
 special='<div class="notice"><b>양손 통합 실패 사례.</b> 청록과 주황 뼈대가 같은 펼친 손을 가리켜. 기존 왼손 칸의 손가락은 실제 오른손을 따라가고, 손목은 다른 위치로 치우쳐 있어. 팔 연결 판정만 서로 다르다고 나오자 중복 검사를 먼저 통과했어. 손끝 위치가 맞아도 두 손을 제대로 복원한 성공 사례로 세면 안 돼.</div>' if r['id']=='E05' else ''
 cards.append(f'''<article id="{r['id']}"><div class="tag">{r['id']} · {html.escape(r['label'])} · 촬영 ID {r['clip_id'].split('_')[-2]} · {r['frame']}프레임 · 촬영자 {'왼손' if r['side']==0 else '오른손'}</div><h3>손목: {wrist}</h3><p>{tiptext}</p>{special}<a href="source_views/{r['id']}_full.jpg"><img loading="lazy" src="views/{r['id']}.jpg" alt="{r['id']} 기존 관측, HaMeR, 통합, 시간 확인 비교"></a><p>{html.escape(r['reference']['note_ko'])}</p><p class="muted">{why}. 누락 구간 {r['gap_start']}–{r['gap_end']}프레임. 이미지를 누르면 원본 전체 장면을 볼 수 있어.</p></article>''')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 누락 장면의 HaMeR 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#e9f1fc;font:17px/1.65 'Malgun Gothic',sans-serif}main{max-width:1380px;margin:auto;padding:42px 28px}h1{font-size:34px;line-height:1.3}h2{margin:40px 0 18px;font-size:26px}h3{font-size:22px;margin:8px 0}p{margin:10px 0}.muted{color:#b7c6db;font-size:15px}.lead{font-size:20px}.notice{border-left:5px solid #f4be60;background:#202b3d;padding:20px;margin:24px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric,article{background:#152236;border:1px solid #2b405d;border-radius:16px;padding:22px}.metric strong{display:block;font-size:34px;color:#7be1d4}.tag{font-size:15px;color:#7be1d4}article{margin:22px 0}article img{width:100%;display:block;margin:18px 0;border-radius:8px}a{color:#8dc5ff}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px 10px;text-align:left;border-bottom:1px solid #30415c}th{color:#a7c3e9;background:#152236}.scroll{overflow:auto}li{margin:9px 0}.nav{display:flex;flex-wrap:wrap;gap:10px}.nav a{padding:5px 10px;background:#1c3049;border-radius:6px}@media(max-width:800px){main{padding:20px 14px}.metrics{grid-template-columns:1fr}h1{font-size:28px}}</style><main>
<div class="tag">공개 영상 관측 보완 · 2026-10-25 · 기존 모델 유지</div>
<h1>빠진 손을 보완할 가능성은 확인.<br>같은 손을 두 번 담는 통합 오류도 발견.</h1>
<p class="lead">HaMeR가 실제 누락 장면에 내놓은 좌표를, 원본에서 눈으로 확인할 수 있는 손목·손끝 위치와 비교했어. 새로 학습하거나 사용자 촬영을 사용하지 않았어.</p>
<div class="metrics"><div class="metric"><strong>16개 누락 구간</strong>공개 7영상 · 4명 · 5표현<br>57개 누락 프레임을 구간별로 묶음</div><div class="metric"><strong>손목 14곳 출력</strong>통합 H / 16개 대표 장면<br>시간 검사 HT는 13곳</div><div class="metric"><strong>손끝 17곳 검토</strong>80개 손끝 자리 중 확인 가능한 부분<br>가림·흐림·번호 불확실 63곳 보류</div></div>
<div class="notice"><b>이번 숫자는 정확한 정답률이 아니야.</b><br>사람이 손목·손끝으로 판단한 작은 사각형을 임시 비교 범위로 썼어. 동일 Codex의 시각 검토이며 전문가 정답이나 독립 평가가 아니야. 사각형 안에 있어도 숨은 손가락 각도·3D 자세가 맞다는 뜻은 아니고, 1px 밖이라고 틀렸다고 확정할 수도 없어.</div>
<h2>무엇이 나아졌고, 무엇이 남았나</h2><ul><li><b>추가 단서 확보:</b> 기존 방식이 비워 둔 자리에서 손목과 일부 보이는 손끝의 위치를 얻었어. 원본의 펼친 손과 엄지 끝을 잘 따라간 예가 있어.</li><li><b>중요한 통합 오류:</b> E05 ‘후원’에서는 기존 손가락 좌표가 잘못된 좌우 칸에 들어간 것으로 보이고, HaMeR를 더하자 같은 펼친 손이 양쪽 칸에 겹쳤어. 새 후보가 손끝에 맞아도 양손 복원 성공은 아니야. <a href="#E05">이 장면 보기</a></li><li><b>손목 오차가 남음:</b> 통합 H의 손목 14곳 중 4곳은 임시 시각 범위 안, 10곳은 밖이었어. 범위 밖 평균 거리는 6.2px, 최대 23.9px야. 이 값은 정확한 관절 정답과의 오차가 아닌 범위까지 거리야.</li><li><b>손끝은 일부만 검증:</b> 확인 가능한 17곳 중 H가 15곳을 출력했고 14곳은 범위 안, 1곳은 약 1px 밖이야. 이 합계에는 E05의 후보 위치도 포함돼. 따라서 이것을 양손 복원 정답률로 읽으면 안 돼. 불확실한 63개 손끝 자리와 중간 마디 각도는 검증되지 않았어.</li><li><b>시간 검사 한계:</b> HT는 H의 좌표를 더 정확하게 고치는 방식이 아니라, 앞뒤가 불안정한 후보를 보류해. 이번 대표 장면 1곳이 추가 보류됐지만, 계속 조금 어긋난 위치나 E05의 중복은 통과했어.</li><li><b>기존 손 보존:</b> 원래 관측 14,149점과 지정 교차 20프레임의 840점이 그대로야. 기존 좌표 보존과 새 통합의 정확성은 다른 문제야.</li></ul>
<h2>손목과 손끝은 나눠 읽어야 해</h2><p>좌표가 아예 없던 B와 출력된 H를 평균 오차 하나로 비교하면 공정하지 않아. 그래서 출력·보류 수를 함께 적었어. 평균 거리는 출력된 점에서만 계산했어.</p>
<h3>손목 · 16개 시각 범위</h3><div class="scroll">WRIST_TABLE</div>
<h3>손끝 · 17개 시각 범위</h3><div class="scroll">TIP_TABLE</div>
<p class="muted">기존 처리에서 빈 손 칸으로 표시된 57프레임 중 H 33개, HT 31개에 추정 손을 추가했어. E05처럼 실제로는 좌우 배정이 어긋난 경우도 포함되므로 33/57·31/57은 정확 복원률이 아니야. 위치 검토는 구간별 대표 16장에 한정돼.</p>
<h2>화면에서 볼 것</h2><ol><li><b>흰 사각형:</b> 원본 RGB를 보고 정한 대략적인 손목·손끝 범위. 사용자 허용 오차나 학습 기준이 아니야.</li><li><b>청록색:</b> 기존 관측. <b>주황색:</b> HaMeR의 추정. 주황 선이 자연스러워도 가려진 관절의 정답은 확인할 수 없어.</li><li><b>연두·분홍 원:</b> 검토 가능한 점이 시각 범위 안·밖에 있는지 표시. 분홍은 바로 잘못된 수어라는 뜻이 아니야.</li><li>H에서 사라지면 중복/좌우 등 통합 조건 때문에 보류한 것. HT에서만 사라지면 시간 검사 때문에 보류한 것이야.</li></ol>
<p class="muted">아래 확대는 결과를 보기 위한 화면이야. 실제 입력은 원본 전체 영상에서 검출한 뒤 고정된 상반신 범위로 제한했어. 원본 전체 화면은 각 비교 이미지를 눌러 확인할 수 있어.</p>
<div class="nav">NAV</div>CARDS
<h2>검토 과정에서 바로잡은 점</h2><p>E11 ‘대지’의 손가락 5개 번호를 처음 표시했지만, 손끝이 붙고 흐려 손가락별 대응을 확정하기 어려웠어. 앞뒤 원본 프레임까지 재검토하고 이 5곳은 보류했어. 예측에 맞춰 사각형을 옮기거나 번호를 다시 붙이지 않았고, 초기 자료도 보존했어. 따라서 손끝 22개로 계산한 초기 합계는 정확도 근거로 쓰지 않아.</p><p class="muted">이 검토가 모델 좌표를 본 뒤 이루어진 점도 한계야. 최종 17점 결과 역시 정밀 정확도 주장이나 채택 기준으로 쓰지 않고, 다음 평가를 준비하는 진단 근거로만 사용해.</p><a href="source_views/E11_context.png">E11 앞뒤 원본 장면 확인</a>
<h2>다음 단계</h2><ol><li><b>중복과 좌우 충돌 검사부터:</b> ‘팔 연결이 다름’만으로 바로 별개의 손이라고 통과시키지 않고, 손가락이 실제 어느 이미지 영역을 차지하는지도 함께 확인하는 별도 후보를 검증해. E01·E16의 정상 겹침과 E02·E06의 과도한 보류 가능성도 함께 비교해야 해.</li><li><b>손목 중심 정의 통일:</b> 피부 연결 부위와 MANO 관절 중심의 차이를 구분할 수 있도록, 선명한 공개 영상에서 일관된 기준점을 마련해.</li><li><b>손목·손가락을 따로 채택:</b> 손목만 근거가 충분하면 위치 보조로 제한하고, 손가락 각도는 자동으로 정답처럼 연결하지 않아.</li><li><b>정밀 기준을 통과한 뒤 DTW 연구 연결:</b> 지금은 운영 채택 보류야. TCN 인식률이나 실제 피드백 성능 향상은 아직 측정하지 않았어.</li></ol>
<p>C·T3-F·운영 DTW·웹 모델 변경 0 · 새 학습 0 · 사용자 촬영 접근 0 · 기존 542개 결과 파일/보호 32개 파일 동일.</p>
<p class="muted">근거: <a href="REVIEWED_SCORES.json">최종 진단 수치</a> · <a href="ANNOTATION_REVIEW.json">주석 불확실성 기록</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="FROZEN.json">검토 범위</a></p></main></html>'''
page=page.replace('WRIST_TABLE',table('wrist')).replace('TIP_TABLE',table('tip')).replace('NAV',''.join(f'<a href="#{r["id"]}">{r["id"]} {html.escape(r["label"])}</a>' for r in S['rows'])).replace('CARDS',''.join(cards))
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
(U/'RESULT_KO.txt').write_text('실제 누락 HaMeR 시각 검증. 7영상4명5표현812프레임,57누락손프레임→16구간중앙. 새추론/학습0,기존예측재사용.\n손목 H14/16출력,4범위안/10범위밖/2보류. 중심평균23.69px,범위밖평균6.24px/최대23.88px. HT13출력/4안/9밖/3보류.\n손끝80자리중17시각범위,63불확실. H15출력14안1밖(1.00px)2보류. HT동일. 정밀GT/정답률/전체손가락각도검증 아님.\nE11초기5손끝ID불확실→앞뒤원본검토후5개모두보류. 초기22점합계정확도근거금지. 최종17점도동일Codex후검토편향한계.\n전체57손프레임 H33/HT31추정추가(정확복원수아님). 기존14149점/지정교차20프레임840점보존. 542과거파일32보호파일동일.\n다음: 손목중심정의/정밀기준통일,위치·수형별증거분리,중복보류검증. 통과후연구DTW연결. C·T3-F/운영/사용자자료유지.\n',encoding='utf-8')
with (U/'RESULT_KO.txt').open('a',encoding='utf-8') as fp:fp.write('중요: E05후원에서 기존왼손칸의손가락과HaMeR오른손후보가같은펼친손을중복표현. 팔연결다름을먼저통과하는relation분기확인. 손목거리50.65px/손끝거리2.01~17.15px. 새후보위치일치와양손통합성공은별개. E02/E06은기존팔확정보류로중복의심보류. 다음우선작업은양손통합충돌검사이며정상교차를보호해야함. 운영채택보류.\n')
print('wrote16 comparison views and explanatory report')
