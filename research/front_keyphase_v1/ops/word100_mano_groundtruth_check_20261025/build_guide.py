from pathlib import Path
import json,html
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
links=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
protocol=read(OLD/'FROZEN_PROTOCOL.json')
cards=[]
for idx,frame,caption in [(2,35,'남아 · 35프레임: 기존에 비어 있던 손에 주황색 추정이 추가된 예'),(3,35,'멀다 · 35프레임: 손목을 찾은 것과 손가락 모양의 정확성은 별도'),(1,20,'후원 · 20프레임: 기존 손은 그대로 두므로 두 결과가 비슷한 것이 정상')]:
    row=protocol['manifest'][idx];cap=cv2.VideoCapture(row['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();assert ok
    z=np.load(OLD/'comparisons'/f"{row['clip_id']}.npz")
    canvas=Image.new('RGB',(1500,342),'#102033');draw=ImageDraw.Draw(canvas)
    for col,(title,mode) in enumerate([('① 원본: 실제 손을 먼저 보기',None),('② 기존: 초록색 관절','B'),('③ 보조 후: 주황색 추정 추가','HT')]):
        im=bgr.copy()
        if mode:
            for side in range(2):
                xy=z[mode+'_xy'][frame,side];valid=z[mode+'_valid'][frame,side]
                inf=mode=='HT' and z['HT_inferred'][frame,side].any();color=(45,155,255) if inf else (110,235,70)
                for a,b in links:
                    if valid[a] and valid[b]:cv2.line(im,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),color,3)
                for j in range(21):
                    if valid[j]:cv2.circle(im,tuple(xy[j].astype(int)),4,color,-1)
        im=cv2.resize(im,(500,281));canvas.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(col*500,50));draw.text((col*500+10,10),title,font=font,fill='white')
    name=f'guide_scene_{idx}.jpg';canvas.save(U/name,quality=94)
    cards.append(f'<figure><figcaption>{caption}</figcaption><a href="{name}"><img src="{name}" alt="{caption}"></a><p>사진을 누르면 크게 볼 수 있어. 초록은 기존 관측, 주황은 새 추정이다. 주황색 손목과 손끝이 실제 손 위에 놓였는지 확인한다.</p></figure>')
s=read(U/'CORRECTED_SCORES.json');r=s['results'];verified=read(U/'VERIFICATION.json');f=read(U/'FROZEN.json');vis=read(U/'PREINFERENCE_VISUAL_REVIEW.json')
gt=np.load(U/'truth.npz')['xy_px'];data=[]
for j,row in enumerate(f['records']):
    with np.load(U/'corrected_predictions'/f"{row['id']:05d}.npz") as z:
        data.append({'id':row['id'],'image':f"images/{row['id']:05d}.jpg",'gt':gt[j].tolist(),'mp':z['mp_xy'].tolist(),'hamer':z['hamer_xy'].tolist(),'visible':vis['clearly_visible_fingertips'][str(row['id'])],**s['per_image'][j]})
table=''.join(f'<tr><th>{label}</th><td>{r["mp"][key]:.2f}px</td><td>{r["hamer"][key]:.2f}px</td><td>{note}</td></tr>' for key,label,note in [
('wrist_mean_px','손목 평균 오차','HaMeR가 더 작음'),('finger20_mean_px','손가락 20점 평균 오차','MediaPipe가 조금 더 작음'),('visible_tip_mean_px','보이는 손끝 83점 오차','HaMeR가 더 작음 · 가시성은 별도 시각 판독'),('all21_mean_px','전체 21점 평균 오차','거의 비슷하며 HaMeR 우위가 아님')])
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MANO 결과 읽는 법과 정답 좌표 시험</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#edf3ff;font:17px/1.7 'Malgun Gothic',sans-serif}main{max-width:1300px;margin:auto;padding:38px 28px 80px}h1{font-size:36px;line-height:1.4}h2{font-size:25px;margin:0 0 15px}h3{font-size:20px}.tag{color:#79c8ff;font-size:14px}.lead{font-size:21px;max-width:1020px}section{padding:28px;background:#142236;border:1px solid #2e435d;border-radius:16px;margin:25px 0}.callout{border-left:5px solid #f2ba68;background:#312b23;padding:16px 20px}.good{border-left-color:#67dbaf;background:#153a33}table{width:100%;border-collapse:collapse}th,td{padding:13px 12px;text-align:left;border-bottom:1px solid #34475f}a{color:#88d0ff}strong{color:#fff}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.tile{background:#0d1929;padding:18px;border-radius:10px}.tile b{font-size:23px;display:block}.small{font-size:14px;color:#bbc9db}figure{margin:28px 0}figcaption{font-weight:bold;font-size:18px;margin-bottom:10px}img{max-width:100%;display:block;border-radius:8px}button,select{font:inherit;background:#263e59;color:white;border:1px solid #7394b7;border-radius:7px;padding:7px 11px}button{cursor:pointer}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}.four{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.four div{background:#0a1320;border-radius:9px;padding:10px}.four p{margin:3px 0 7px;font-weight:bold}canvas{width:100%;display:block}details{margin-top:20px}li{margin:8px 0}.bar{display:inline-block;height:9px;background:#74bfff;border-radius:6px} @media(max-width:900px){.four{grid-template-columns:repeat(2,1fr)}.three{grid-template-columns:1fr}main{padding:20px 12px}section{padding:18px}h1{font-size:28px}table{font-size:14px}}</style>
<main><div class="tag">WordsInSilence · 공개 자료 / 연구용 · 2026.10.25</div><h1>손을 더 찾았다고, 모두 정확해진 것은 아니다</h1>
<p class="lead">지난 화면은 <strong>“놓친 손을 보충할 수 있나?”</strong>를 본 시험이다. 이번에는 정답 좌표가 있는 별도 공개 사진으로 <strong>“그려진 점이 실제 관절에 얼마나 가깝나?”</strong>를 확인했다.</p>
<div class="callout good"><b>현재 결론:</b> 3D 손 보조의 가능성은 있다. 하지만 기존 MediaPipe를 전부 바꿀 근거는 없다. 손목·보이는 손끝에서는 이득이 있었고, 손가락 전체에서는 우위가 확인되지 않았다. 수어 인식률 향상은 아직 측정한 것이 아니다.</div>
<section><h2>1. 지난 결과 화면은 이 순서로 보면 된다</h2><p><a href="../word100_mano_threeway_20261025/RESULT_KO.html">기존 영상 비교 화면 열기</a> · 기존 결과 파일은 그대로 보존했다.</p>
<table><tr><th>화면</th><th>쉽게 말하면</th><th>확인할 점</th></tr>
<tr><td>원본 RGB</td><td>카메라에 찍힌 실제 손</td><td>손목·손끝이 어디에 있고, 어느 손가락이 가려졌는지 먼저 본다.</td></tr>
<tr><td>B · 기존 MediaPipe</td><td>기존 검출과 재검출 결과</td><td>실제 손이 보이는데 관절이 비어 있는지 확인한다.</td></tr>
<tr><td>HaMeR 원출력</td><td>3D 손 모델이 제안한 후보</td><td>아직 합치지 않은 진단용이다. 일부 프레임만 계산했으므로 중간의 빈 화면 모두가 검출 실패는 아니다.</td></tr>
<tr><td>H · 3D 손 보조</td><td>기존 손을 두고, 누락 손에 후보를 추가</td><td>주황색 손이 실제 손 위에 있는지, 다른 손으로 잘못 붙지 않는지 본다.</td></tr>
<tr><td>HT · 앞뒤 확인</td><td>추정이 앞뒤 움직임과 크게 충돌하면 보류</td><td>사라진 후보는 확인 근거가 부족한 것이다. 사용자가 틀렸다는 뜻은 아니다.</td></tr></table>
<div class="three" style="margin-top:20px"><div class="tile"><b>① 위치</b>손목 점이 손목에 붙어 있는가?<br>다른 손이나 소매를 따라가지 않는가?</div><div class="tile"><b>② 모양</b>주먹이면 관절도 접혔는가?<br>보이는 손끝과 점이 맞는가?</div><div class="tile"><b>③ 시간</b>앞·뒤 프레임에서 갑자기 튀거나 좌우가 바뀌지 않는가?<br>빠른 정상 움직임을 놓치지 않는가?</div></div>
<p>화면의 선은 <strong>3D 손 모델이 추정한 관절을 원본 화면 위에 2D로 옮겨 그린 것</strong>이다. 손 인형의 피부 표면을 보여주는 화면은 아니다.</p><p><strong>가려진 손가락은 “보기 자연스럽다”만으로 맞았다고 확인할 수 없다.</strong> 초록색도 모두 정답이라는 뜻은 아니다. 이번 보조가 기존 값을 바꾸지 않았다는 뜻이다. 확대 버튼은 보기만 확대한다.</p>
<p class="callout">H 33 → HT 31은 “33번 정답 → 31번 정답”이 아니다. 한 손이 한 프레임에 추가된 추정 수다. 시간 검사로 2개가 보류됐지만, 실제 오류 2개를 고쳤다고 확정한 것은 아니다.</p>
SCENES
<p><b>먼저 볼 장면:</b> 남아 선택 → 대표 장면 → 이전/다음 프레임으로 이동. 원본과 B에서 빈 손을 확인하고, H/HT의 주황색 손목·손끝을 본다. 후원 일부처럼 H와 HT가 같게 보이는 것은 기존 손을 유지했기 때문이다.</p></section>
<section><h2>2. 이번에 실제로 실행한 정답 좌표 비교</h2><p>연구 저자가 공개한 <a href="https://github.com/3d-hand-shape/hand-graph-cnn/tree/master/data/real_world_testset">CVPR 2019 실사 손 시험 자료</a> 583장 중, 결과를 보기 전에 번호를 일정하게 나눠 <b>32장</b>을 선정했다. 각 사진은 256×256이다. 같은 사진에 MediaPipe와 HaMeR를 실행하고 저자 제공 관절 좌표와 비교했다.</p>
<div class="three"><div class="tile"><b>32장 / 672점</b>손목 1점 + 손가락 20점<br>두 모델에 같은 표본</div><div class="tile"><b>83개 손끝</b>보이는 것이 명확한 손끝만 별도 분석<br>전문가 판정은 아님</div><div class="tile"><b>새 학습 0회</b>기존 가중치로 추론만 수행<br>C·T3-F와 사용자 자료 그대로</div></div>
<h3>오차는 낮을수록 정답 좌표에 가깝다</h3><p>예를 들어 8px은 점이 정답에서 평균 약 8픽셀 떨어졌다는 뜻이다. <b>8% 틀렸다는 뜻도, 동작이 92% 맞았다는 뜻도 아니다.</b></p>
<table><tr><th>항목</th><th>MediaPipe</th><th>HaMeR + MANO</th><th>이번 표본 해석</th></tr>TABLE</table>
<p>개별 사진의 21점 평균에서는 HaMeR가 더 가까운 사진이 <b>10/32장</b>이었다. 따라서 “모든 손가락이 더 정확해졌다”고 말할 수 없다. 관절 중심의 정의·저자 정답 자체의 오차도 이 값에 포함될 수 있다.</p>
<p><b>누락 때만 보조하는 조건:</b> 이번에는 MediaPipe가 32장 모두 손을 찾았다. 보조가 호출될 사례가 없어 합친 결과는 기존과 같았다. <strong>이번 32장으로 실제 누락 손 복원 성능을 입증한 것은 아니다.</strong></p>
<p>CPU 추론 중앙값: MediaPipe MPMSms / HaMeR HMSms. 모델 로딩·전체 영상 처리 시간은 제외한다. 기기 부하의 영향이 있으며 웹 실시간 성능을 보장하지 않는다.</p>
<details><summary>자료·설정 검증과 한계</summary><ul><li>FreiHAND 공식 평가 ZIP은 이번 접근에서 404, 공식 대체 서버는 연결 실패였다. 임의 재배포본 대신 원저자 GitHub의 다른 소규모 시험 자료를 사용했다.</li><li>첫 평가에서 내가 사진의 좌우를 오른손으로 잘못 지정했다. 원본 재검토와 현재 MediaPipe의 32장 왼손 판독을 확인하고, 왼손 입력/역변환으로 바로잡았다. 잘못된 첫 결과는 별도 보존하고 위 표에서 제외했다. 사진마다 점수가 좋은 좌우를 골라 넣지 않았다.</li><li>사진 번호·정답 좌표·모델·크기·비교 지표는 그대로다. 보이는 손끝 83개도 첫 추론 전에 정했으며, 나머지 손끝은 가려짐/불확실로 분리했다. 나머지가 모두 실제로 안 보인다는 뜻은 아니다.</li><li>저자 제공 3D 관절을 카메라 정보로 2D에 투영한 비교다. 검출기 간 일치를 정답으로 사용하지 않았다. 이미지 번호는 사람 수가 아니다.</li><li>한 손 정지 사진으로 시간 보완 HT·양손 교차·한국수어 실사용을 검증하지 않았다. 사전학습 자료와의 중복 가능성도 완전히 배제하지 못한다.</li><li>저장 점수 별도 검산 통과. 기존 542개 결과 파일과 보호 대상 32개 파일의 해시 동일. 구현·시각검토·검산은 같은 Codex이며 독립 전문가 검토는 아니다.</li></ul></details></section>
<section><h2>3. 정답 점과 직접 비교하기</h2><p>노랑은 저자가 제공한 기준, 초록은 MediaPipe, 보라는 HaMeR다. 선택한 사진에서 같은 손을 나란히 비교할 수 있다. 흰 원은 기준 손목·손끝이며 연결선은 해당 예측점과의 거리를 보여준다.</p>
<div class="controls"><button id="prev">이전 사진</button><select id="select"></select><button id="next">다음 사진</button><label><input type="checkbox" id="truth" checked> 기준 손목·손끝 표시</label></div>
<div class="four"><div><p>원본</p><canvas data-kind="original"></canvas></div><div><p>저자 제공 기준</p><canvas data-kind="gt"></canvas></div><div><p>MediaPipe</p><canvas data-kind="mp"></canvas></div><div><p>HaMeR + MANO</p><canvas data-kind="hamer"></canvas></div></div><p id="readout"></p><p class="small">첫 사진은 00000번이다. 00037번은 주먹, 00150번은 일부 손가락을 편 모습이다. 빈 곳을 그럴듯하게 그렸는지가 아니라 보이는 손가락과 맞는지 확인한다.</p></section>
<section><h2>4. 다음에는 무엇을 확인할까?</h2><p><b>기존에 잘 잡히는 손은 그대로 두고, 실제로 놓친 순간의 보조만 검증한다.</b> 앞의 수어 영상에서 보이는 손목과 손끝에 좁은 기준 영역을 마련해, HaMeR가 그 안으로 들어오는지 확인한다. 가려진 손가락은 정답으로 확정하지 않고 추정 상태를 유지한다.</p><ol><li>실제 누락 장면의 보이는 손목·손끝을 기준과 대조한다. 좌우 설정도 먼저 검산한다.</li><li>손목 위치가 맞는 것과 손가락 모양이 맞는 것을 따로 판정한다. 모양이 불확실하면 그 각도로 사용자 교정 피드백을 만들지 않는다.</li><li>정상 손·교차·빠른 변화가 유지되고 정확한 보완이 확인될 때만 별도 연구 DTW로 비교 이득을 확인한다. 운영 교체나 학습은 이번에 하지 않았다.</li></ol><div class="callout">현재는 <b>“보조 후보의 정확성 검증”</b> 단계다. 3D 손이 자연스러워 보이는 것과 사용자 동작을 정확히 교정할 수 있는 것은 다르다.</div></section>
<p class="small">자료 출처: Ge et al., 3D Hand Shape and Pose Estimation from a Single RGB Image, CVPR 2019. 저자 저장소의 연구·교육 이용 조건에 따라 로컬 연구에 사용. 별도 공개 배포 없음.</p></main>
<script>const DATA=DATA_JSON, LINKS=LINKS_JSON;const sel=document.getElementById('select');let idx=0,im=new Image(),token=0;DATA.forEach((d,i)=>sel.add(new Option(String(d.id).padStart(5,'0'),i)));document.querySelectorAll('canvas').forEach(c=>{c.width=512;c.height=512});
function render(){const d=DATA[idx];document.querySelectorAll('canvas').forEach(c=>{const g=c.getContext('2d'),k=c.dataset.kind;g.clearRect(0,0,512,512);g.drawImage(im,0,0,512,512);if(k==='original')return;let q=d[k],color=k==='gt'?'#ffe365':k==='mp'?'#4df49b':'#c696ff';g.strokeStyle=color;g.fillStyle=color;g.lineWidth=2;for(const [a,b] of LINKS){g.beginPath();g.moveTo(q[a][0]*2,q[a][1]*2);g.lineTo(q[b][0]*2,q[b][1]*2);g.stroke()}for(const p of q){g.beginPath();g.arc(p[0]*2,p[1]*2,3,0,Math.PI*2);g.fill()}if(k!=='gt'&&document.getElementById('truth').checked){g.strokeStyle='#fff';g.lineWidth=1;for(const j of [0,4,8,12,16,20]){const a=d.gt[j],b=q[j];g.beginPath();g.moveTo(a[0]*2,a[1]*2);g.lineTo(b[0]*2,b[1]*2);g.stroke();g.beginPath();g.arc(a[0]*2,a[1]*2,5,0,Math.PI*2);g.stroke()}}});document.getElementById('readout').textContent=`사진 ${d.id}: 21점 평균 오차 — MediaPipe ${d.mp_all21_px.toFixed(2)}px / HaMeR ${d.hamer_all21_px.toFixed(2)}px. 낮은 쪽이 이 사진의 저자 기준에 더 가깝다.`}
function load(i){idx=(i+DATA.length)%DATA.length;sel.value=idx;const t=++token;im=new Image();im.onload=()=>{if(t===token)render()};im.src=DATA[idx].image}sel.onchange=()=>load(+sel.value);document.getElementById('prev').onclick=()=>load(idx-1);document.getElementById('next').onclick=()=>load(idx+1);document.getElementById('truth').onchange=render;load(0);</script></html>'''
doc=doc.replace('SCENES',''.join(cards)).replace('TABLE',table).replace('MPMS',f"{r['mp']['inference_median_seconds']*1000:.1f}").replace('HMS',f"{r['hamer']['inference_median_seconds']*1000:.1f}").replace('DATA_JSON',json.dumps(data,ensure_ascii=False)).replace('LINKS_JSON',json.dumps(links))
(U/'RESULT_KO.html').write_text(doc,encoding='utf-8')
(U/'RESULT_KO.txt').write_text('MANO 결과 해설 및 공개 정답 좌표 32장 점검\n\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n\n첫 수동 좌우 지정 오류는 기록 보존 후 Left로 정정. 기존 C/T3-F, 운영DTW 변경0, 새학습0, 사용자자료 접근0.\n다음: 실제 누락 장면의 보이는 손목/손끝 정밀 범위 비교. 공개 32장에는 MP누락 없음. 정적사진만으로 시간/교차 검증 아님.\n',encoding='utf-8')
print('GUIDE_CREATED',len(doc))
