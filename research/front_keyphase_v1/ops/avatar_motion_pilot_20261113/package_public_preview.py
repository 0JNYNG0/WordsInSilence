"""Package actual127-frame derived animation and honest side-by-side review; no new inference."""
from pathlib import Path
import av,cv2,json,hashlib,sys,numpy as np
from fractions import Fraction
U=Path(__file__).resolve().parent;BODY_AXIS='--body-axis'in sys.argv;D=U/('public_approximate_preview_body_axis'if BODY_AXIS else'public_approximate_preview');V=U/'video_only'
out=D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4'
if not out.exists():
    with av.open(str(out),'w')as c:
        s=c.add_stream('libx264',rate=30);s.width=512;s.height=512;s.pix_fmt='yuv420p';s.options={'crf':'20','preset':'fast'}
        for f in range(127):
            im=cv2.imread(str(D/'frames'/('frame%03d.png'%f)));assert im is not None
            fr=av.VideoFrame.from_ndarray(im,format='bgr24');fr.pts=f;fr.time_base=Fraction(1,30)
            for packet in s.encode(fr):c.mux(packet)
        for packet in s.encode():c.mux(packet)
with av.open(str(out))as c:
    decoded=list(c.decode(video=0));assert len(decoded)==127
    times=[float(f.pts*f.time_base)for f in decoded];assert np.max(np.abs(np.array(times)-np.arange(127)/30))<1e-6
plate=[]
frames=[48,56,64,72,80,88,96,104]
source=V/'NIA_SL_WORD1567_REAL02_F.mp4';source_frames={}
with av.open(str(source))as c:
    for f,im in enumerate(c.decode(video=0)):
        if f in frames:source_frames[f]=im.to_ndarray(format='bgr24')
for group in [frames[:4],frames[4:]]:
    a=[];b=[]
    for f in group:
        im=source_frames[f];h,w=im.shape[:2];im=im[:,(w-h)//2:(w+h)//2];im=cv2.resize(im,(360,360))
        cv2.putText(im,'PUBLIC F | %d / %.2fs'%(f,f/30),(10,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        ar=cv2.resize(cv2.imread(str(D/'frames'/('frame%03d.png'%f))),(360,360))
        cv2.putText(ar,'APPROXIMATE | %d'%f,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,40,65),2);a.append(im);b.append(ar)
    plate.extend([np.concatenate(a,1),np.concatenate(b,1)])
cv2.imwrite(str(D/'SOURCE_AVATAR_SEQUENCE_REVIEW.jpg'),np.concatenate(plate,0))
report=dict(status='PUBLIC_APPROXIMATE_PREVIEW_PACKAGED',source_clip='NIA_SL_WORD1567_REAL02',frames=127,fps=30,duration_seconds=127/30,
            decoded_frame_timestamps_exact=True,mp4_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),comparison_source_frames=frames,
            caveats=['Inferred cameras, not official calibration','Face/torso/legs remain static','No manual source keyframe correction or invented sign poses','Joint response is not visibility confidence','Teaching/sign correctness not verified'],
            first_failures_preserved=['initial_synthetic_diagnostic','FIRST_SYNTHETIC_FBX_ERROR.json','video_only/calibration_v1_all_body'])
(D/'PACKAGE_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개5뷰 → 캐릭터 근사 동작</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf1f6;color:#172b47;font:17px/1.55 system-ui,"Malgun Gothic",sans-serif}main{max-width:1120px;margin:30px auto;padding:0 25px}h1{font-size:29px;margin:0 0 10px}p{margin:7px 0 17px}.notice{border-left:5px solid #526783;padding:12px 18px;background:white}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:white;border-radius:12px;overflow:hidden}.panel b{display:block;padding:10px 16px}video{width:100%;aspect-ratio:1;display:block;object-fit:cover;background:#c9cdd3}.controls{display:flex;align-items:center;gap:12px;margin:18px 0}button{border:0;border-radius:8px;background:#223d61;color:white;padding:10px 15px;font:inherit;cursor:pointer}input{flex:1;accent-color:#223d61}.limits{display:grid;grid-template-columns:1fr 1fr;gap:16px}.limits section{background:#fff;padding:15px 20px;border-radius:10px}h2{font-size:20px;margin:0 0 7px}ul{margin:8px 0;padding-left:22px}details{margin:20px 0}img{width:100%;border-radius:10px}small{color:#526783}a{color:#223d61}</style>
<main><h1>공개5방향 영상에서 캐릭터 동작으로</h1><p>‘조상’ 공개 영상 1개 · 원본 속도30fps · 같은 순간 나란히 보기</p>
<p class="notice"><b>원본 카메라 정보가 없는 근사 복원입니다.</b> 영상에서 추정한 팔·손 움직임을 캐릭터에 옮긴 기술 시연입니다. 올바른 수어를 보장하는 교육용 애니메이션은 아직 아닙니다.</p>
<div class="pair"><div class="panel"><b>공개 참고 영상 · 정면</b><video id="source" src="video_only/NIA_SL_WORD1567_REAL02_F.mp4" muted playsinline preload="auto"></video></div><div class="panel"><b>같은 순간의 캐릭터 · 근사 동작</b><video id="avatar" src="public_approximate_preview/APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4" muted playsinline preload="auto"></video></div></div>
<div class="controls"><button id="play">함께 재생</button><button id="prev">이전</button><input id="slider" type="range" min="0" max="126" value="64"><button id="next">다음</button><span id="stamp">64 / 126</span></div>
<div class="limits"><section><h2>이번에 실제로 만든 부분</h2><ul><li>같은 영상의 정면·좌·우·위·아래 관절 대응</li><li>영상 기반 카메라 추정과 다른 시점의 대응 검사</li><li>양팔·손목·양손 손가락의 캐릭터 적용</li><li>127프레임 동작 저장과 FBX 재가져오기 검사</li></ul></section><section><h2>아직 확인해야 하는 부분</h2><ul><li>손가락과 교차 순간의 세밀한 모양·접촉</li><li>사람과 캐릭터 비율 차이에 따른 얼굴·가슴 위치</li><li>앞뒤 깊이와 가림 순간의 추정 오차</li><li>몸통·얼굴·다리는 현재 고정</li></ul></section></div>
<details><summary>주요 순간을 한눈에 비교</summary><img src="public_approximate_preview/SOURCE_AVATAR_SEQUENCE_REVIEW.jpg" alt="동일 원본 프레임과 근사 캐릭터를 나란히 비교한 시퀀스"></details>
<details><summary>다섯 방향의 실제 관측 근거</summary><img src="video_only/FIVE_VIEW_SAME_TIME_OBSERVATIONS.jpg" alt="동일시점5뷰 관절관측"></details>
<small>프레임은 원본0~126에 대응합니다. 왼쪽은 보기 편하게 중앙을 확대했습니다. 새 모델 학습·개인 촬영 사용·기존 운영 모델 변경은 없습니다.</small></main>
<script>const a=document.querySelector('#source'),b=document.querySelector('#avatar'),s=document.querySelector('#slider'),t=document.querySelector('#stamp'),p=document.querySelector('#play');let playing=false;function seek(f){f=Math.max(0,Math.min(126,+f));a.pause();b.pause();playing=false;p.textContent='함께 재생';a.currentTime=f/30;b.currentTime=f/30;s.value=f;t.textContent=f+' / 126 · '+(f/30).toFixed(2)+'초'}s.oninput=()=>seek(s.value);document.querySelector('#prev').onclick=()=>seek(+s.value-1);document.querySelector('#next').onclick=()=>seek(+s.value+1);p.onclick=async()=>{if(playing){seek(Math.round(a.currentTime*30));return}if(a.currentTime>=126/30)seek(0);await Promise.all([a.play(),b.play()]);playing=true;p.textContent='일시 정지'};function tick(){if(playing){if(Math.abs(a.currentTime-b.currentTime)>.05)b.currentTime=a.currentTime;let f=Math.min(126,Math.floor(a.currentTime*30));s.value=f;t.textContent=f+' / 126 · '+a.currentTime.toFixed(2)+'초';if(a.ended||f>=126)seek(126)}requestAnimationFrame(tick)}Promise.all([new Promise(r=>a.onloadedmetadata=r),new Promise(r=>b.onloadedmetadata=r)]).then(()=>seek(64));tick();</script></html>'''
if BODY_AXIS:html=html.replace('public_approximate_preview/','public_approximate_preview_body_axis/').replace('<details><summary>주요 순간을 한눈에 비교</summary>','<details open><summary>원본·첫 시도·몸통축 수정</summary><img src="BODY_AXIS_BEFORE_AFTER_KO.png" alt="실제 공개영상과 이전 수정 후 캐릭터 비교"></details><details><summary>주요 순간을 한눈에 비교</summary>')
(U/('PREVIEW_KO_BODY_AXIS.html'if BODY_AXIS else'PREVIEW_KO.html')).write_text(html,'utf-8');print('PACKAGED',str(out))
