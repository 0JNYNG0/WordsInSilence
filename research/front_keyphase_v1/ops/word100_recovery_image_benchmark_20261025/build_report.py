from pathlib import Path
import json, html
from datetime import datetime, timezone, timedelta

U=Path(__file__).resolve().parent
def read(p): return json.loads((U/p).read_text(encoding='utf-8-sig'))
def esc(x): return html.escape(str(x))
NAMES={'B0':'기존 검출','B1':'이미지에서 재검출','Linear':'앞뒤 좌표를 직선으로 연결',**{f'{c}_seed{s}':f'{"관절+시간" if c=="B2" else "RGB+관절+시간"} · {s}' for c in ['B2','B3'] for s in [17,29,43]}}
def table(data):
 out='<table><thead><tr><th>조건</th><th>두 범위 안</th><th>한쪽만 / 애매</th><th>두 범위 밖</th><th>좌표 없음</th><th>판독 보류</th></tr></thead><tbody>'
 for c,v in data.items():
  st=v['statuses']; out+=f'<tr><th>{esc(NAMES[c])}</th>'+''.join(f'<td>{st[k]}</td>' for k in ['inside_both','inside_one_ambiguous','outside_both','unavailable','unscorable'])+'</tr>'
 return out+'</tbody></table>'

def main():
 r=read('evaluation/RESULT.json'); rows=read('evaluation/ROWS.json')
 if isinstance(rows,dict): rows=rows['rows']
 sel=read('SELECTION.json')['selection']; amap={a['sample_id']:a for a in read('BLIND_REVIEW_A.json')['rows']}; bmap={a['sample_id']:a for a in read('BLIND_REVIEW_B.json')['rows']}
 cond=r['by_condition']; bs={c:r['by_stratum'][c]['B'] for c in cond}; ass={c:r['by_stratum'][c]['A'] for c in ['B0','B1']}
 words=len({s['clip_id'].split('_')[2] for s in sel}); people=len({s['signer_id'] for s in sel}); clips=len({s['clip_id'] for s in sel})
 observed_a=ass['B1']['statuses']; learned=[bs[f'B3_seed{s}']['statuses']['inside_both'] for s in [17,29,43]]; pose=[bs[f'B2_seed{s}']['statuses']['inside_both'] for s in [17,29,43]]
 summary=(f'실제 보완 대상 {24}장면({words}표현·{people}명·{clips}영상)을 고정해 원영상과 비교했다. '
 f'재검출 대상 8장면에서 B1 손목은 두 판독 범위 안 {observed_a["inside_both"]}, 한쪽만 {observed_a["inside_one_ambiguous"]}, 범위 밖 {observed_a["outside_both"]}, 판독 보류 {observed_a["unscorable"]}였다. '
 f'시간 추정 대상 8장면의 두 범위 안 개수는 선형 {bs["Linear"]["statuses"]["inside_both"]}, 관절+시간 {pose}, RGB+관절+시간 {learned}였다. '
 '이 값은 수어 정답률이나 손가락 21점 정확도가 아니다.')
 sections=[]
 sections.append(f'<header><div class="eyebrow">WORDS IN SILENCE · 공개 자료 진단 · 2026.10.25</div><h1>다시 찾은 손은<br>실제 손 위치와 맞을까?</h1><p class="lead">모델의 답을 먼저 보지 않고 원영상의 손목 위치를 표시한 뒤, 저장된 보완 결과와 대조했습니다.</p><div class="metrics"><div><b>24</b><span>실제 보완·보존 장면</span></div><div><b>{people}명 / {words}표현</b><span>공개 학습 촬영자 · {clips}영상</span></div><div><b>23 + 1</b><span>손목 비교 가능 + 흐림 보류</span></div></div></header>')
 sections.append(f'<section><h2>이번에 확인한 결과</h2><p><b>현재의 RGB 학습 보완기가 단순한 앞뒤 연결보다 낫다는 근거는 나오지 않았습니다.</b> 이미지에서 손을 다시 찾는 효과와, 학습 모델이 빈 손목을 추정하는 효과를 구분해 확인했습니다.</p><p>{esc(summary)}</p><div class="note"><b>운영 모델은 그대로 유지합니다.</b> 기존 C·T3-F·웹·DTW를 교체하지 않았고, 이번에 추가 학습하거나 사용자 촬영을 열어보지 않았습니다.</div></section>')
 sections.append('<section><h2>어떻게 공정하게 확인했나</h2><ol><li>점수와 좌표값을 보지 않고, 실제로 손목을 다시 찾았거나 빈 좌표를 채운 구간에서 표본을 고정했습니다.</li><li>두 AI 검토자가 각각 원영상에서 해부학적 좌우와 대략적인 손목 범위를 표시했습니다. 예측 좌표와 서로의 판독은 보지 않았습니다.</li><li>판독을 동결한 다음 9조건의 저장 좌표를 같은 장면에서 비교했습니다. 결과에 맞춰 범위를 넓히지 않았습니다.</li><li>두 범위 모두 안이면 영상 근거와 일치, 한쪽만 안이면 애매, 모두 밖이면 불일치 의심으로 남깁니다. 흐리거나 판단이 엇갈린 장면은 보류합니다.</li></ol><p class="muted">두 AI의 일치는 전문가가 표시한 정확한 관절 정답과 다릅니다. 손목 범위는 대략적인 2차원 영상 근거이며, 숨은 관절의 위치를 확정하지 않습니다.</p></section>')
 sections.append('<section><h2>① 이미지에서 다시 찾은 손목 · 8장면</h2><p>기존 방식에서 손목 좌표가 없었고, 원이미지를 다시 살펴본 B1에서 좌표가 생긴 장면입니다. 추정으로 손을 만들어 넣은 결과와 구분합니다.</p>'+table(ass)+'</section>')
 sections.append('<section><h2>② 앞뒤 움직임으로 채운 손목 · 8장면</h2><p>B1에서도 손목이 없던 순간입니다. 같은 앞뒤 시간 범위에서 단순 선형 연결과 이미 학습한 두 보완기를 비교했습니다. 17·29·43은 학습 초기값만 달리한 세 번의 실험입니다.</p>'+table(bs)+'<p class="muted">같은 8장면을 세 번 본 결과이므로 24개의 독립 장면으로 합산하지 않습니다. B2/B3가 채운 값은 계속 추정으로 표시하며 실제 관측으로 승격하지 않습니다.</p></section>')
 sections.append('<section><h2>③ 전체 결과 · 24장면</h2><p>재검출 8 + 손목 추정 8 + 일부 손가락 추정 4 + 원관측 보존 대조 4입니다. 마지막 두 묶음의 손목은 원래 보이던 값이어서, 그 손목이 맞았다고 손가락 복원도 성공했다고 해석하지 않습니다.</p>'+table(cond)+'</section>')
 cards=[]
 for s in sel:
  sid=s['sample_id']; rr=[x for x in rows if x['sample_id']==sid]
  st={'A':'재검출','B':'손목 추정','C':'손가락 추정 · 손목 보존','D':'원관측 보존 대조'}[s['stratum']]
  cards.append(f'<details><summary>{sid} · {st} · 촬영자 {s["signer_id"]:02d} · {"왼손" if s["anatomical_side"]==0 else "오른손"}</summary><p>상자는 예측을 보기 전에 고정한 두 검토자의 손목 범위입니다. 검출점과 상자의 일치만 확인하며, 손가락 전체 정확도를 뜻하지 않습니다.</p><a href="evaluation/overlays/{sid}.jpg"><img loading="lazy" src="evaluation/overlays/{sid}.jpg" alt="{sid} 9조건 대조"></a><p><a href="images/{sid}.png">원본 장면</a> · <a href="triptychs/{sid}.jpg">앞·현재·뒤 장면</a></p></details>')
 sections.append('<section><h2>같은 원영상 위에서 확인하기</h2><p>각 장면을 펼치면 조건별 손목 좌표를 볼 수 있습니다. 이미지를 누르면 크게 열립니다.</p>'+''.join(cards)+'</section>')
 sections.append('<section><h2>무엇이 아직 해결되지 않았나</h2><ul><li><b>재검출해도 위치가 애매할 수 있음:</b> R02·R13은 B1 좌표가 두 판독 범위 밖입니다. 범위까지 거리는 약 4~23픽셀로, 정확한 관절 오차와는 다릅니다. 보완기는 기존 관측을 보존하는 설계이므로 이 값을 그대로 물려받습니다.</li><li><b>RGB를 더했는데 추정이 이동한 경우:</b> R16은 선형 연결이 두 범위 안에 있지만 B3는 두 실행에서 범위 밖, 한 실행에서 한쪽 범위에만 들었습니다. 모든 RGB 융합법이 나쁘다는 뜻이 아니라 이번 작은 후보의 한계입니다.</li><li><b>손가락 정확도:</b> 두 검토자가 보이는 손끝 개수를 다르게 적은 장면이 12개였습니다. 이 주석으로 21관절 학습이나 손가락 교정의 정답을 만들지 않습니다.</li><li><b>실제 가림:</b> 손목 비교가 가능한 장면과 손가락 일부 가림을 구분했습니다. 완전히 가려진 손가락이 정확히 복원됐다는 증거는 없습니다.</li><li><b>대표성:</b> 24장면 중 한 표현이 8개입니다. B2 seed17의 추정 가능 마스크 등을 이용해 고른 조건부 진단이며, 전체 수어의 성공률이 아닙니다.</li><li><b>새 사람 검증:</b> 모두 기존 학습 촬영자입니다. 기존 DEV07·TEST08/09를 다시 튜닝에 사용하지 않았습니다. 사용자 촬영에도 손대지 않았습니다.</li><li><b>교차·정상 손 보존:</b> 기존 관측 좌표를 바꾸지 않는 계약을 확인했습니다. 현재 표본 밖의 모든 손 교차가 정확하다는 뜻은 아닙니다.</li></ul></section>')
 sections.append('<section><h2>다음에는 무엇을 해야 하나</h2><p><b>전체 모델을 다시 학습하기 전에, 손가락 위치를 믿고 평가할 기준을 확보하는 것이 우선입니다.</b> 지금의 대략적인 손목 상자를 손가락 정답으로 확장하지 않습니다.</p><ol><li>이번 실제 영상 진단과 이전 평가 결과를 함께 유지하고, 불확실한 시간 추정은 확실한 관측과 분리합니다. B2/B3 전체 적용이나 C·T3-F 교체 근거로 사용하지 않습니다.</li><li>외부 손 관절 자료는 먼저 작은 검출 평가에 사용합니다. <a href="https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html">FreiHAND 공식 안내</a>의 주석 포함 평가 묶음은 724MB입니다. 실제 ZIP 내부의 이미지·XYZ·카메라 정보 존재는 아직 확인하지 않았습니다.</li><li>양손 겹침과 시간 변화에는 <a href="https://mks0601.github.io/InterHand2.6M/">InterHand2.6M</a>이 더 가까운 후보입니다. 다만 큰 자료이며, 사람·시퀀스·동시 시점이 평가 양쪽에 섞이지 않도록 분리해야 합니다.</li><li>손만 나온 외부 사진에 없는 어깨·팔 좌표를 만들어 현재 보완기에 넣지 않습니다. 손 검출 부품의 평가와 한국수어 전체 효과 검증을 따로 설계합니다.</li></ol><p>외부 자료 조사만 완료했으며 새 자료 다운로드·외부 제출·새 학습은 실행하지 않았습니다. <a href="EXTERNAL_LABEL_OPTIONS.txt">자료별 상세 근거와 제한</a></p></section>')
 sections.append('<section><h2>근거 파일</h2><div class="links">'+''.join(f'<a href="{f}">{label}</a>' for f,label in [('evaluation/RESULT.json','조건별 집계'),('evaluation/ROWS.json','216개 비교 행'),('BLIND_REVIEW_A.json','독립 원영상 판독 A'),('BLIND_REVIEW_B.json','독립 원영상 판독 B'),('SELECTION_CHECK.json','선정·이미지 검산'),('INDEPENDENT_CHECK.json','비교 독립 검산'),('SCIENTIFIC_REVIEW.json','과학적 해석 검토'),('PROTECTION_CHECK.json','C·T3-F 등 32개 보호 확인'),('EXTERNAL_LABEL_OPTIONS.json','공식 자료 조사')])+'</div><p class="muted">구현·판독: Codex 및 협업 에이전트. 별도 에이전트의 산술 검산과 연구 해석 검토를 수행했습니다. 전문가의 관절 주석 검수는 아닙니다.</p></section>')
 css='''*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#172739;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:36px 24px 72px}header,section{background:white;border-radius:20px;padding:34px 42px;margin:0 0 22px;border:1px solid #dbe4ea}header{background:#142d48;color:#fff;padding:48px}h1{font-size:44px;line-height:1.3;margin:14px 0 20px}h2{font-size:26px;line-height:1.4;margin:0 0 18px}.eyebrow{font-size:13px;letter-spacing:2px;color:#98d6e8}.lead{color:#d7e8f2;font-size:19px;max-width:800px}.metrics{display:flex;gap:36px;flex-wrap:wrap;margin-top:36px}.metrics b{display:block;font-size:30px}.metrics span{display:block;color:#c2d2df;font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin-top:20px}th,td{padding:13px 10px;border-bottom:1px solid #dfe6ed;text-align:center}th:first-child{text-align:left}thead{background:#edf3f7}tbody tr:hover{background:#f5f9fc}.note{background:#eaf5f3;border-left:5px solid #248371;padding:18px;margin-top:20px}.muted{color:#607287;font-size:14px}li{margin:10px 0}a{color:#126b99}details{border:1px solid #d9e4eb;border-radius:10px;padding:16px;margin:12px 0}summary{font-weight:700;cursor:pointer}img{width:100%;height:auto;border-radius:8px}.links{display:flex;gap:12px;flex-wrap:wrap}.links a{padding:8px 14px;background:#edf3f7;border-radius:8px;font-size:14px}@media(max-width:700px){main{padding:12px}header,section{padding:24px 18px}h1{font-size:32px}table{font-size:11px}th,td{padding:8px 4px}}'''
 (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 손 복원 구간 원영상 검증</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>',encoding='utf-8')
 lines=['실제 손 복원 구간 원영상 검증',summary,'','재검출 8장면',json.dumps(ass,ensure_ascii=False,indent=2),'손목추정 8장면',json.dumps(bs,ensure_ascii=False,indent=2),'','C·T3-F/운영DTW/웹 보호32 유지. 사용자 촬영 접근/학습/튜닝0. 새학습0.','두 AI 판독의 대략적 손목 범위 진단이며 전문가21관절 정답이나 새사용자 수어인식률이 아님.','보이는 손끝 개수 불일치12장면. 손가락 정밀평가/수형학습에는 더 신뢰할 수 있는 관절 정답 필요.','후속: 공개 손 관절 자료의 작은 검출평가 설계. 실제 파일 확보/수신 없음. 상세 HTML과 EXTERNAL_LABEL_OPTIONS 참조.']
 (U/'RESULT_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
 print(summary)

if __name__=='__main__': main()
