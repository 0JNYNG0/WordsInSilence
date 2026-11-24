from pathlib import Path
import json,html,shutil,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def clean(a):
    a=np.asarray(a)
    if a.dtype.kind=='b':return a.tolist()
    return np.where(np.isfinite(a),np.round(a,2),None).tolist()
def main():
    f=json.loads((U/'FROZEN_PROTOCOL.json').read_text('utf-8'));r=json.loads((U/'COMPARISON.json').read_text('utf-8'));v=json.loads((U/'VERIFICATION.json').read_text('utf-8'))
    review=json.loads((U/'VISUAL_REVIEW.json').read_text('utf-8'))
    assets=U/'assets';assets.mkdir(exist_ok=True);data=[]
    for k,m in enumerate(f['manifest']):
        dest=assets/(m['clip_id']+'.mp4')
        if not dest.exists():shutil.copy2(m['rgb_path'],dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==m['rgb_sha256']
        with np.load(U/'comparisons'/(m['clip_id']+'.npz')) as z:c={n:z[n] for n in z.files}
        raw=np.full((m['frames'],2,21,2),np.nan)
        for p in (U/'raw_predictions').glob(f'{k:02d}_*.npz'):
            with np.load(p) as z:raw[int(z['frame_index']),int(z['side'])]=z['xy_px']
        data.append({'name':m['word_label_for_display_only']+' · 촬영자 '+str(m['signer_id']),
                    'video':'assets/'+dest.name,'anchor':m['anchors'][0],'n':m['frames'],
                    'times':clean(c['timestamp_us']/1e6),'roi':c['roi_box'].tolist(),
                    'raw':clean(raw),'B':clean(c['B_xy']),'H':clean(c['H_xy']),'HT':clean(c['HT_xy']),
                    'Bv':clean(c['B_valid']),'Hv':clean(c['H_valid']),'HTv':clean(c['HT_valid']),
                    'Hi':clean(c['H_inferred']),'HTi':clean(c['HT_inferred']),
                    'note':review['anchors'][k]['finding']})
    H=sum(x['H_added_estimate_hand_frames'] for x in r['rows']);HT=sum(x['HT_added_estimate_hand_frames'] for x in r['rows'])
    table=''.join(f"<tr><td>{html.escape(data[i]['name'])}</td><td>{x['frames']}</td><td>{x['attempted_missing_hands']}</td><td>{x['H_added_estimate_hand_frames']}</td><td>{x['HT_added_estimate_hand_frames']}</td></tr>" for i,x in enumerate(r['rows']))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MANO · 공개 영상 첫 비교</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0e1520;color:#e7edf8;font-family:Malgun Gothic,Arial,sans-serif;line-height:1.65}main{max-width:1360px;margin:auto;padding:36px 28px}h1{font-size:34px;margin:8px 0 18px}h2{font-size:23px;margin:30px 0 12px}.tag{color:#6edbd0;font-size:13px}.lead{max-width:950px;font-size:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stats div,section{padding:20px;border:1px solid #2b3c52;border-radius:14px;background:#152132}.stats b{font-size:27px;display:block}section{margin:22px 0}.muted{color:#acbed5}.warn{border-left:4px solid #f5bb63;padding:12px 18px;background:#302820}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #314258;padding:10px}button,select{font:inherit;color:#e7edf8;background:#233750;border:1px solid #516a85;border-radius:7px;padding:7px 12px;cursor:pointer}label{margin-left:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.panel{background:#0b111a;border-radius:10px;overflow:hidden}.panel p{padding:10px 14px;margin:0;font-size:14px}canvas{width:100%;display:block}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}input[type=range]{width:100%}.legend{font-size:14px;color:#b4c5db}.note{font-size:15px}a{color:#79c9ff}video{display:none}@media(max-width:900px){.grid,.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){.grid{grid-template-columns:1fr}main{padding:20px 12px}} </style>
<main><div class="tag">WordsInSilence · 로컬 비상업 연구 · 2026.10.25</div><h1>MANO 손 보완, 실제 영상에서 첫 비교</h1>
<p class="lead">공식 MANO 파일을 확인하고 HaMeR를 연결했다. 놓친 손을 추정하는 모습은 확인했지만, 생성된 모든 손가락이 실제로 정확하다는 뜻은 아니다. 기존 관측과 새 추정을 분리한 채 비교했다.</p>
<div class="stats"><div><b>7개 · 812프레임</b>공개 영상 4명 / 5표현</div><div><b>243개</b>실제 HaMeR 손 영역 추론</div><div><b>HNUM → HTNUM</b>추가 추정 손-프레임 / 시간 확인 전→후</div><div><b>SECONDS초 / 손</b>CPU 추론 중앙값 · 실시간 아님</div></div>
<section><h2>세 조건은 이렇게 다르다</h2><p><b>B · 기존:</b> 저장된 MediaPipe 관측과 재검출 결과.<br><b>H · 3D 손 보조:</b> 현재 RGB에서 HaMeR가 추정한 손을 팔 연결·중복 확인 후 누락 부분에만 추가.<br><b>HT · 시간 확인 추가:</b> 같은 후보의 앞뒤 관절 회전·손 구조·손목 이동이 크게 충돌하면 추정을 보류. 정상 빠른 변화도 보류될 수 있어, 이를 수행 오류로 판정하지 않는다.</p>
<p>원본 전체 화면의 관측과 후보를 합친 뒤, 정해 둔 상반신 범위를 마지막에 적용했다. 아래 확대 버튼은 <b>보기만 확대</b>하며 검출 범위나 학습 조건을 바꾸지 않는다.</p>
<div class="warn">추가 수치는 정답률이 아닌 ‘추가 모델 추정’ 수다. 손가락 정밀 정답·가려진 3D 정답이 없어 정확도 향상은 아직 확정하지 않았다. 운영 모델과 웹 분석은 변경하지 않았다.</div></section>
<section><h2>같은 순간을 직접 비교</h2><select id="clip"></select><div class="controls"><button id="play">재생 / 멈춤</button><button id="anchor">대표 장면</button><button id="prev">이전 프레임</button><button id="next">다음 프레임</button><label><input id="zoom" type="checkbox"> 손 주변 확대 (보기만)</label><label>재생 속도 <select id="speed"><option value="1">1배</option><option value=".5">0.5배</option></select></label><span id="time"></span></div><input id="seek" type="range" min="0" step="1"><video id="video" preload="metadata" muted playsinline></video>
<div class="grid"><div class="panel"><p>원본 RGB</p><canvas data-mode="original"></canvas></div><div class="panel"><p>B · 기존 MediaPipe</p><canvas data-mode="B"></canvas></div><div class="panel"><p>HaMeR 원출력 · 통합 전 진단용</p><canvas data-mode="raw"></canvas></div><div class="panel"><p>H · 기존 관측 + 3D 추정</p><canvas data-mode="H"></canvas></div><div class="panel"><p>HT · 앞뒤 시간 확인 추가</p><canvas data-mode="HT"></canvas></div><div class="panel"><p id="note" class="note"></p><p class="legend">초록: 기존 관측<br>주황: 새 모델 추정 (정확성 미확정)<br>노랑 선: 고정 상반신 하단 경계<br>빈 곳: 이번 방식으로도 확정하지 못함<br>HaMeR 원출력은 선정된 프레임만 계산했으므로 다른 순간은 빈 화면으로 표시된다.</p></div></div></section>
<section><h2>전체 보존 검사와 추가 추정</h2><table><thead><tr><th>공개 자료</th><th>전체 프레임</th><th>누락 대상 손-프레임</th><th>H 추가 추정</th><th>HT 추가 추정</th></tr></thead><tbody>TABLE</tbody></table><p>기존 POINTS개 관절-프레임과 교차 구간 CROSS개는 값과 상태가 그대로 유지됐다. 기존 관측이 틀렸던 곳까지 모두 고쳐졌다는 의미는 아니다. 전체 시간축을 유지했으며 손이 안 보인다고 비교 구간을 삭제하지 않았다.</p></section>
<section><h2>판정과 다음 단계</h2><p>DECISION</p><p>기존 C·T3-F 보존 · 새 학습 0회 · 사용자 촬영 접근·학습·튜닝 0 · 운영 교체 0. 이번 자료는 이전에도 점검한 작은 진단 묶음이며 새로운 사람 전체에 대한 성능을 증명하는 시험은 아니다.</p><p class="muted">구현·실행 및 별도 저장 결과 검산: Codex. 독립 전문가 검증은 수행하지 않았다. 공식 자산은 연구 폴더의 private_assets에 보관하고 원본 다운로드 파일도 보존했다.</p></section></main>
<script>const DATA=DATA_JSON;
const L=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
const vid=document.getElementById('video'),sel=document.getElementById('clip'),seek=document.getElementById('seek'),zoom=document.getElementById('zoom');let d=DATA[0];
DATA.forEach((x,i)=>sel.add(new Option(x.name,i)));document.querySelectorAll('canvas').forEach(c=>{c.width=960;c.height=540});
function nearest(){let t=vid.currentTime,j=0;while(j+1<d.n&&Math.abs(d.times[j+1]-t)<Math.abs(d.times[j]-t))j++;return j;}
function showFrame(i){vid.pause();vid.currentTime=d.times[Math.max(0,Math.min(d.n-1,i))];}
function load(){d=DATA[+sel.value];seek.max=d.n-1;vid.src=d.video;document.getElementById('note').textContent=d.note;vid.onloadedmetadata=()=>showFrame(d.anchor);}
function draw(){if(vid.readyState>=2){let t=nearest();seek.value=t;document.getElementById('time').textContent=`${t+1} / ${d.n} 프레임`;let rect=zoom.checked?[480,220,960,620]:[0,0,1920,1080];
document.querySelectorAll('canvas').forEach(c=>{let g=c.getContext('2d'),mode=c.dataset.mode;g.clearRect(0,0,960,540);g.save();g.scale(960/rect[2],540/rect[3]);g.translate(-rect[0],-rect[1]);g.drawImage(vid,0,0,1920,1080);g.strokeStyle='#f4d45f';g.lineWidth=2;g.setLineDash([10,8]);g.beginPath();g.moveTo(0,d.roi[3]);g.lineTo(1920,d.roi[3]);g.stroke();g.setLineDash([]);
if(mode!=='original'){let points=d[mode][t],valid=mode==='raw'?null:d[mode+'v'][t];for(let s=0;s<2;s++){let q=points[s],ok=i=>q[i][0]!==null&&q[i][1]!==null&&(!valid||valid[s][i]);let inferred=mode==='raw'?null:(d[mode+'i']||null);g.strokeStyle=mode==='raw'||(inferred&&inferred[t][s].some(Boolean))?'#ffa347':'#46ef94';g.fillStyle=g.strokeStyle;g.lineWidth=3;for(let [a,b] of L){if(ok(a)&&ok(b)){g.beginPath();g.moveTo(...q[a]);g.lineTo(...q[b]);g.stroke();}}for(let i=0;i<21;i++)if(ok(i)){g.beginPath();g.arc(...q[i],3,0,Math.PI*2);g.fill();}}}g.restore();});}requestAnimationFrame(draw);}
sel.onchange=load;seek.oninput=()=>showFrame(+seek.value);document.getElementById('anchor').onclick=()=>showFrame(d.anchor);document.getElementById('prev').onclick=()=>showFrame(nearest()-1);document.getElementById('next').onclick=()=>showFrame(nearest()+1);document.getElementById('play').onclick=()=>vid.paused?vid.play():vid.pause();document.getElementById('speed').onchange=e=>vid.playbackRate=+e.target.value;load();draw();</script></html>'''
    replacements={'HNUM':str(H),'HTNUM':str(HT),'SECONDS':f"{r['median_inference_seconds_per_hand']:.2f}",'TABLE':table,'POINTS':f"{v['baseline_point_frames_exact']:,}",'CROSS':str(v['crossing_point_frames_exact']),'DECISION':html.escape(review['decision']),'DATA_JSON':json.dumps(data,ensure_ascii=False,allow_nan=False)}
    # HTNUM must precede HNUM to avoid accidental substring replacement.
    for k in ['HTNUM','HNUM','SECONDS','TABLE','POINTS','CROSS','DECISION','DATA_JSON']:page=page.replace(k,replacements[k])
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    print('REPORT_CREATED',len(page))
if __name__=='__main__':main()
