"""Presentation-only view of five frozen public cases. No new inference."""
from pathlib import Path
import json,hashlib,shutil,html,datetime
import numpy as np,cv2
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
SOURCE=O/'word100_yolo_selective_wrist_20261025';OLD=O/'word100_yolo_hand_fourway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def e(x):return html.escape(str(x))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
COL=(35,180,255)
def draw(im,p,v,side,box=None):
    im=im.copy()
    for s in (0,1):
        color=COL if s==side else (170,170,170)
        for a,b in LINKS:
            if v[s,a] and v[s,b]:cv2.line(im,tuple(p[s,a].astype(int)),tuple(p[s,b].astype(int)),color,3,cv2.LINE_AA)
        for j in np.flatnonzero(v[s]):cv2.circle(im,tuple(p[s,j].astype(int)),7 if j==0 else 3,color,2,cv2.LINE_AA)
    if box:
        x0,y0,x1,y1=map(int,box)
        cv2.rectangle(im,(x0,y0),(x1,y1),(230,200,50),2)
    return im
def pair(im,folder,name,focus):
    x0,y0,x1,y1=focus
    cv2.imwrite(str(folder/f'{name}_full.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,92])
    cv2.imwrite(str(folder/f'{name}_focus.jpg'),im[y0:y1,x0:x1],[cv2.IMWRITE_JPEG_QUALITY,95])
DESC={
 'R01':('손목은 가까워도, 손 모양 후보가 서로 달라요','펼친 손과 주먹이 겹쳐 있습니다. YOLO 선이 두 손의 모양을 섞어 따라가는지 봐 주세요.','같은 팔로 연결되는 확대 후보가 2개입니다. 손목은 가까워도 손가락 배치가 달라 하나를 확정하지 않았습니다.','보류 · 대상 손을 새로 채우지 않음'),
 'R02':('손이 없다고 나온 게 아니라, 위치가 어긋났어요','MediaPipe의 손목 동그라미와 YOLO의 동그라미가 어디에 있는지 비교해 보세요.','MediaPipe가 이미 손을 검출한 상태라 누락 전용 YOLO 보조는 호출하지 않습니다. 전체 화면 YOLO의 손목은 기존 검토 범위 안이지만 확대 재검출은 손을 찾지 못했습니다.','기존 검출 유지 · 위치 오류는 남음'),
 'R14':('한 손 주변에 서로 다른 확대 후보가 생겼어요','같은 주먹에 대해 확대 후보 1과 2가 어떤 선을 그리는지 비교해 보세요.','두 후보의 손목은 기존 허용폭 안이지만 손가락 배열이 다릅니다. 중복 후보 중 하나를 임의로 정하지 않았습니다.','보류 · 대상 손을 새로 채우지 않음'),
 'R21':('손가락뿐 아니라 손목부터 달라요','전체 화면 YOLO와 확대 YOLO가 표시한 손목 동그라미의 위치를 봐 주세요.','손목 위치 차이가 약32.6픽셀로 기존 허용폭 약24.1픽셀을 넘었습니다. 손가락 모양도 달라 손목만 믿고 넣을 수 없었습니다.','보류 · 대상 손을 새로 채우지 않음'),
 'R22':('옆으로 펼친 손에서 위치와 모양이 달라요','손목 동그라미와 손가락 선이 손의 실제 옆모습을 따라가는지 봐 주세요.','전체 화면과 확대 검출의 손목 차이가 약36.5픽셀로 기존 허용폭 약24.2픽셀을 넘었습니다. 위치와 모양을 모두 보류했습니다.','보류 · 대상 손을 새로 채우지 않음')}
def main():
    f=read(SOURCE/'FROZEN.json');cases=read(SOURCE/'FIVE_SCENE_AUDIT.json');mr={r['clip_id']:r for r in f['old_rows']};oi={r['clip_id']:r for r in read(OLD/'INDEX.json')};ni={r['clip_id']:r for r in read(SOURCE/'INDEX.json')}
    labels={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_A.json')['rows']}
    status={(r['id'],r['condition']):r for r in read(OLD/'SCORED_ROWS.json')}
    assets=U/'assets';assets.mkdir(exist_ok=True);sections=[];manifest=[];nav=[]
    for no,c in enumerate(cases,1):
        sid=c['id'];cid=c['clip_id'];i=c['frame_id'];s=c['side'];r=mr[cid];z=load(r['arrays']);q=load(ni[cid]['arrays']);rec=read(oi[cid]['records'])['frames'][i]
        assert sha(r['rgb_path'])==r['rgb_sha256']
        folder=assets/sid;folder.mkdir(exist_ok=True);video=folder/'original.mp4';shutil.copyfile(r['rgb_path'],video);assert sha(video)==r['rgb_sha256']
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();cap.release();assert ok
        p=np.asarray(rec['full'][c['full_index']]['xy'],float);cx,cy=p[0];x0=max(0,min(1920-760,int(cx)-380));y0=max(0,min(1080-520,int(cy)-260));focus=[x0,y0,x0+760,y0+520]
        pair(im,folder,'original',focus);box=labels[sid]['wrist_box'];stamp=float(z['timestamp_us'][i])/1e6
        base=z['baseline_roi_xy_px'][i,33:].reshape(2,21,2);bv=z['baseline_roi_usable'][i,33:].reshape(2,21)
        m1=z['redetected_roi_xy_px'][i,33:].reshape(2,21,2);v1=z['redetected_roi_usable'][i,33:].reshape(2,21)
        yp=np.full((2,21,2),np.nan);yv=np.zeros((2,21),bool);yp[s]=p;yv[s]=np.isfinite(p).all(-1)
        for name,pp,vv in [('baseline',base,bv),('mp',m1,v1),('yolo',yp,yv),('final',q['p_xy'][i],q['p_valid'][i])]:pair(draw(im,pp,vv,s,box),folder,name,focus)
        cropcards=[]
        for n,a in enumerate(c['crop_candidates'],1):
            j=a['index'];d=rec['crop'][j];pts=np.asarray(d['xy'],float);pp=np.full((2,21,2),np.nan);vv=np.zeros((2,21),bool);pp[s]=pts;vv[s]=np.isfinite(pts).all(-1)
            pair(draw(im,pp,vv,s,box),folder,f'crop{j}',focus)
            arm='좌우 연결 불확실' if a['arm']<0 else ('촬영자의 왼팔에 연결' if a['arm']==0 else '촬영자의 오른팔에 연결')
            delta=a['wrist_delta_pixels'];line=f'전체 화면 YOLO와 손목 차이 {delta:.1f}px' if delta is not None else '신뢰할 손목 좌표 없음'
            cropcards.append(f'<figure><figcaption><b>확대 후보 {n}</b><span>{arm} · {line}</span></figcaption><img class="compare" src="assets/{sid}/crop{j}_focus.jpg" data-focus="assets/{sid}/crop{j}_focus.jpg" data-full="assets/{sid}/crop{j}_full.jpg" alt="{sid} 확대 후보 {n}"></figure>')
        if not cropcards:cropcards=['<div class="empty"><b>확대 영상에서 손을 찾지 못했습니다.</b><p>잘 맞았다는 뜻이 아니라, 대조할 두 번째 검출이 없는 상태입니다.</p></div>']
        title,look,why,result=DESC[sid];side='왼손' if s==0 else '오른손';num=f'{no:02d}'
        nav.append(f'<a href="#{sid}"><b>{num}</b> {e(c["label"])} <small>{sid}</small></a>')
        conds=[('baseline','① 기존 관측','대상 손이 비어 있음'),('mp','② MediaPipe 재검출','검출됐지만 손목 범위 밖' if sid=='R02' else '대상 손이 비어 있음'),('yolo','③ 전체 화면 YOLO','손목은 기존 검토 범위 안 · 손가락 정확성은 별도'),('final','④ 최종 보조 결과',result)]
        grid=''.join(f'<figure><figcaption><b>{title2}</b><span>{e(sub)}</span></figcaption><img class="compare" src="assets/{sid}/{name}_focus.jpg" data-focus="assets/{sid}/{name}_focus.jpg" data-full="assets/{sid}/{name}_full.jpg" alt="{sid} {title2}"></figure>' for name,title2,sub in conds)
        # Derive baseline text instead of assuming the five M0 target masks.
        if bv[s].any():grid=grid.replace('① 기존 관측</b><span>대상 손이 비어 있음','① 기존 관측</b><span>기존 관측 있음')
        section=f'''<section id="{sid}" data-target="{stamp:.6f}" data-fps="30"><header><p class="eyebrow">장면 {num} / 05 · {sid} · 공개 촬영자 {r['signer_id']:02d} · 촬영자의 {side}</p><h2>{e(c['label'])} — {title}</h2><p class="result">{result}</p></header>
<div class="overview"><div><video controls preload="metadata" playsinline poster="assets/{sid}/original_full.jpg" src="assets/{sid}/original.mp4"></video><div class="controls"><button class="target">문제 순간으로</button><button class="around">앞뒤 움직임 0.5배속</button><button class="step" data-delta="-1">이전 프레임</button><button class="step" data-delta="1">다음 프레임</button></div><p class="caption">원본 전체 영상 · 문제 시점 약 {stamp:.2f}초. 영상은 자유롭게 움직일 수 있고, 아래 비교 이미지는 이 시점에 고정됩니다.</p></div>
<aside><h3>여기를 봐 주세요</h3><p>{look}</p><h3>왜 보류했나요?</h3><p>{why}</p><p class="note">보류는 사용자가 수어를 틀렸다는 뜻이 아닙니다. 이 검출 결과를 믿고 추가하기 어렵다는 뜻입니다.</p></aside></div>
<div class="toolbar"><h3>같은 순간, 네 결과</h3><button class="mode">원본 전체 화면으로 보기</button></div><p class="caption">주황 선·동그라미 = 확인할 손과 손목 / 회색 = 다른 손 / 파란 상자 = 이전 검토의 대략적인 손목 범위. 파란 상자는 정밀 정답 주석이 아닙니다.</p><div class="grid">{grid}</div>
<details open><summary>확대 재검출은 무엇을 봤을까? · 후보 {len(c['crop_candidates'])}개</summary><p class="caption">각 후보는 별도의 추정입니다. 후보 수가 실제 손 개수는 아닙니다. 좌표는 원본 영상 위치로 되돌려 표시했습니다.</p><div class="grid">{''.join(cropcards)}</div></details>
<p class="caption"><a href="assets/{sid}/original_full.jpg" target="_blank">문제 순간 원본 사진 크게 열기</a> · <a href="#top">장면 목록으로</a></p></section>'''
        sections.append(section);manifest.append({'id':sid,'clip_id':cid,'label':c['label'],'side':s,'frame':i,'timestamp_seconds':stamp,'focus_rect_for_display_only':focus,'video_sha256':sha(video),'source_video_sha256':r['rgb_sha256'],'source_video':r['rgb_path'],'result':result})
    page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보류된 5장면 · 직접 비교하기</title><style>
:root{color-scheme:light;--ink:#192f44;--line:#dbe5ed;--accent:#136d8b}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f6;color:var(--ink);font:16px/1.7 "Malgun Gothic",sans-serif}main{max-width:1440px;padding:40px 28px;margin:auto}.hero h1{font-size:38px;line-height:1.3;margin:8px 0 18px}.hero p{max-width:1000px}.eyebrow{font-size:13px;color:#5a7184;letter-spacing:.03em}nav{display:flex;flex-wrap:wrap;gap:10px;padding:14px 0 24px}nav a{background:white;border:1px solid var(--line);padding:12px 20px;border-radius:12px;text-decoration:none;color:var(--ink)}nav a:hover{border-color:var(--accent)}nav b{color:var(--accent);margin-right:8px}small{color:#697f90}section{scroll-margin-top:18px;background:white;padding:30px;border-radius:18px;margin:26px 0;box-shadow:0 4px 22px #19395208}h2{font-size:27px;line-height:1.4;margin:6px 0 14px}h3{font-size:18px;margin:10px 0}.result{display:inline-block;border-radius:8px;padding:6px 12px;background:#fff0d4;color:#825709;font-weight:bold}.overview{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(250px,1fr);gap:28px;margin:20px 0}video{display:block;width:100%;background:#152334;aspect-ratio:16/9;border-radius:10px}.controls{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}button{font:inherit;font-size:14px;background:#eef5fa;border:1px solid #cadbe7;padding:8px 13px;color:var(--ink);border-radius:8px;cursor:pointer}button:hover{background:#dbeef7}.around{background:#146f8e;color:white;border-color:#146f8e}.note{background:#f0f6fb;border-left:4px solid #5593b3;padding:14px;font-size:14px}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:15px;margin-top:28px}.caption{font-size:13px;color:#62778a;margin:10px 0 16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#f6f9fb}figcaption{padding:13px 16px;min-height:82px}figcaption b{display:block;font-size:17px}figcaption span{display:block;color:#64798b;font-size:13px;margin-top:3px}img{display:block;width:100%;height:auto}.empty{padding:40px 20px;background:#f5f8fb;border:1px dashed #a9bdcd;border-radius:12px}details{margin-top:24px}summary{cursor:pointer;font-size:18px;font-weight:bold;padding:16px 0}a{color:#126e92}.intro{padding:16px 20px;border-radius:12px;background:#e1edf4}footer{padding:18px;color:#607586;font-size:13px}@media(max-width:900px){.overview{grid-template-columns:1fr}main{padding:22px 14px}section{padding:20px}h2{font-size:22px}}@media(max-width:600px){.grid{grid-template-columns:1fr}.hero h1{font-size:29px}.toolbar{align-items:flex-start;flex-direction:column}}
</style></head><body><main id="top"><div class="hero"><p class="eyebrow">기존 결과를 보여주는 시각 자료 · 새 검출/학습 없음</p><h1>보류된 5개 장면을 직접 비교해 보세요</h1><p>먼저 원본 움직임을 보고, 아래 네 결과에서 손목 동그라미와 손가락 선을 비교하면 됩니다. 각 장면 아래에는 확대 재검출이 내놓은 후보도 모두 표시했습니다.</p><p class="intro"><b>검출은 원본 전체 화면에서 시작했습니다.</b> 아래 손 주변 확대는 자세히 보기 위한 표시입니다. 얼굴이나 머리 위를 잘라 검출한 화면이 아닙니다. 이번 5장면은 모두 보완이 확정되지 않은 사례입니다.</p></div><nav>__NAV__</nav>__SECTIONS__<footer>공개 AI Hub 영상만 사용했습니다. C·T3-F, 운영 웹·DTW, 학습 조건은 변경하지 않았습니다. 사용자 촬영 자료는 사용하지 않았습니다.<br><a href="../word100_yolo_selective_wrist_20261025/RESULT_KO.html">앞선 전체 검증 보고서로 돌아가기</a></footer></main><script>
document.querySelectorAll('section[data-target]').forEach(section=>{
 const v=section.querySelector('video'),t=Number(section.dataset.target),fps=Number(section.dataset.fps);let stop=null,whole=false;
 const seek=x=>{v.currentTime=Math.max(0,Math.min(Number.isFinite(v.duration)?v.duration:x,x));};
 v.addEventListener('loadedmetadata',()=>{v.pause();seek(t+.001)});
 section.querySelector('.target').onclick=()=>{stop=null;v.pause();seek(t+.001)};
 section.querySelector('.around').onclick=()=>{stop=Math.min(v.duration,t+.8);v.pause();seek(t-.8);v.playbackRate=.5;v.play().catch(()=>{})};
 section.querySelectorAll('.step').forEach(b=>b.onclick=()=>{stop=null;v.pause();seek(v.currentTime+Number(b.dataset.delta)/fps)});
 v.addEventListener('timeupdate',()=>{if(stop!==null&&v.currentTime>=stop){v.pause();stop=null}});
 section.querySelector('.mode').onclick=ev=>{whole=!whole;section.querySelectorAll('img.compare').forEach(im=>im.src=whole?im.dataset.full:im.dataset.focus);ev.currentTarget.textContent=whole?'손 주변 확대해서 보기':'원본 전체 화면으로 보기'};
});
</script></body></html>'''.replace('__NAV__',''.join(nav)).replace('__SECTIONS__',''.join(sections))
    (U/'RESULT_KO.html').write_text(page,encoding='utf8');save(U/'VISUAL_MANIFEST.json',manifest)
    for p,h in f['protected'].items():assert sha(p)==h
    # Exact source-pixel previews and local videos; no inference or drawing-generated replacements.
    import re
    for link in re.findall(r'(?:src|href)="([^"]+)"',page):
        if not link.startswith(('#','http')):assert (U/link).exists(),link
    save(U/'VALIDATION.json',{'status':'PASS','scenes':5,'videos_copied_hash_identical':5,'source_frames_read':5,'new_inference':0,'training':0,'user_data':0,'protected_files':len(f['protected']),'relative_links_exist':True,'at':datetime.datetime.now().astimezone().isoformat()})
    print('BUILT',len(manifest),'scenes',sum(p.stat().st_size for p in assets.rglob('*') if p.is_file()),'bytes')
if __name__=='__main__':main()
