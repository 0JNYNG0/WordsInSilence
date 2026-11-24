import inspect_alignment as S
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image,ImageDraw,ImageFont
import cv2,zipfile,json,html,sys

U=S.U
FONT='C:/Windows/Fonts/malgun.ttf'
plt.rcParams['font.family']=FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
COLORS=['#1778ad','#de721d']
def curve(z,prefix,g,k,scale=1):
    sl=S.A.OFFSETS[g];v=z[prefix+'_features'][:,sl][:,k].astype(float)*scale
    valid=z[prefix+'_present'][:,S.A.F.GROUP_ORDER.index(g)]&np.isfinite(v)
    return np.where(valid,v,np.nan)
def progress(t):return (t-t[0])/(t[-1]-t[0])*100
def savefig(fig,p):fig.savefig(p,dpi=145,bbox_inches='tight');plt.close(fig)

def plots(row,z):
    folder=U/'assets'/row['word_id'];folder.mkdir(parents=True,exist_ok=True)
    qtime=(z['q_timestamp_us']-z['q_timestamp_us'][0])/1e6;rtime=(z['r_timestamp_us']-z['r_timestamp_us'][0])/1e6
    pq=progress(z['q_timestamp_us']);pr=progress(z['r_timestamp_us']);qi,ri=z['path'].T
    specs=[('handpos_R',1,-100,'오른손 높이 · 어깨선 위가 +'),('handpos_L',1,-100,'왼손 높이 · 어깨선 위가 +'),
           ('handpos_R',0,100,'오른손 좌우 위치 · 영상 오른쪽이 +'),('bend_R',4,180,'오른 검지 중간 관절의 2D 내부각')]
    fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
    for k,(g,c,scale,label) in enumerate(specs):
        a=curve(z,'q',g,c,scale);b=curve(z,'r',g,c,scale)
        axes[k,0].plot(pq,a,color=COLORS[0],lw=2,label='사용자');axes[k,0].plot(pr,b,color=COLORS[1],lw=2,label='공개 참고')
        axes[k,1].plot(pq[qi],a[qi],color=COLORS[0],lw=2,label='사용자');axes[k,1].plot(pq[qi],b[ri],color=COLORS[1],lw=2,label='DTW가 연결한 참고')
        finite=np.concatenate([a[np.isfinite(a)],b[np.isfinite(b)]])
        if len(finite):
            lo,hi=float(finite.min()),float(finite.max());pad=max((hi-lo)*.1,3)
            for ax in axes[k]:ax.set_ylim(lo-pad,hi+pad)
        for j,ax in enumerate(axes[k]):
            ax.grid(alpha=.18);ax.set_xlim(0,100);ax.set_title(label);ax.set_xlabel('각 영상의 경과 비율 (%)' if j==0 else '사용자 경과 비율 (%) · 참고 시점만 재배치')
            ax.set_ylabel('도 (°)' if g.startswith('bend') else '어깨너비=100');ax.legend(loc='best',fontsize=9)
            if not np.isfinite(a).any():ax.text(.5,.4,'사용자 손 관측 없음\n0으로 채우지 않음',ha='center',transform=ax.transAxes,color=COLORS[0])
    fig.suptitle(row['label']+' | 왼쪽: 정렬 전 / 오른쪽: DTW 정렬 후\n선이 끊긴 곳은 비교 근거가 없음 · 곡선의 높이/각도는 바꾸지 않음',fontsize=15)
    savefig(fig,folder/'signals.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.2),layout='constrained')
    for p,t,name,c in [('q',qtime,'사용자',COLORS[0]),('r',rtime,'공개 참고',COLORS[1])]:
        axes[0].plot(t,curve(z,p,'handpos_R',1,-100),label=name,color=c,lw=2)
    axes[0].set(xlabel='손 관측 구간 시작 이후 실제 초',ylabel='오른손 높이 (어깨너비=100)',title='원래 시간 그대로 겹쳐 보기');axes[0].legend();axes[0].grid(alpha=.18)
    axes[1].imshow(z['cost'].T,origin='lower',extent=[0,qtime[-1],0,rtime[-1]],aspect='auto',cmap='Greys',vmin=0,vmax=np.percentile(z['cost'],95))
    axes[1].plot(qtime[qi],rtime[ri],color='#db393c',lw=2,label='DTW 연결 경로')
    axes[1].plot([0,qtime[-1]],[0,rtime[-1]],'--',color='#2aa4b5',label='길이만 비례해서 맞춤')
    axes[1].set(xlabel='사용자 초',ylabel='공개 참고 초',title='어느 시점끼리 연결했나');axes[1].legend(fontsize=9)
    savefig(fig,folder/'time_path.png')

def thumbs(row,z):
    folder=U/'assets'/row['word_id']; data={};frames={}
    with zipfile.ZipFile(row['base']+'.capture.zip') as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['observation_sha256']==S.sha(row['base']+'.npz') and manifest['mirrored']==False
        for ix,f in enumerate(manifest['frames']):
            blob=archive.read(f['image']);assert S.hashlib.sha256(blob).hexdigest()==f['sha256']
            import io
            im=Image.open(io.BytesIO(blob)).convert('RGB');im.thumbnail((640,420));p=folder/f'q_{ix:03}.jpg';im.save(p,quality=88)
            frames[ix]=dict(file=str(p.relative_to(U)).replace('\\','/'),time=f['timestamp_us']/1e6,frame=f['frame_id'])
    cap=cv2.VideoCapture(row['reference']['video_path']);assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);refimages={}
    for j,fid in enumerate(z['r_frame_id']):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,im=cap.read();assert ok
        # Container average FPS can include a different final-frame duration.
        # Check the actual decoded frame index and presentation timestamp.
        assert int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1==int(fid)
        pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        assert abs(pts-z['r_timestamp_us'][j]/1e6)<.002
        im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im.thumbnail((640,420));p=folder/f'r_{j:03}.jpg';im.save(p,quality=88)
        refimages[j]=dict(file=str(p.relative_to(U)).replace('\\','/'),time=float(z['r_timestamp_us'][j]/1e6),frame=int(fid),video_pts=pts,container_average_fps=fps)
    cap.release()
    for mode,key in [('dtw','path'),('linear','linear')]:
        pairs=[]
        for i,j in z[key]:
            qt=float(z['q_timestamp_us'][i]/1e6);best=min(frames,key=lambda k:abs(frames[k]['time']-qt))
            q=frames[best];r=refimages[int(j)]
            pairs.append(dict(query=q,reference=r,query_observation_time=qt,rgb_delta_ms=round((q['time']-qt)*1000,2),
                query_observation_frame=int(z['q_frame_id'][i]),cost=float(z['cost'][i,j]),evidence=float(z['evidence'][i,j])))
        data[mode]=pairs
    # Four deterministic quartiles including endpoints, for visual QA of actual stage matching.
    font=ImageFont.truetype(FONT,19);small=ImageFont.truetype(FONT,15)
    sheet=Image.new('RGB',(1280,4*310+55),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,10),row['label']+' · DTW 경로 0 / 33 / 67 / 100% 장면',font=font,fill='#173b56')
    for k,v in enumerate([0,1/3,2/3,1]):
        p=data['dtw'][round(v*(len(data['dtw'])-1))]
        for col,side in enumerate(['query','reference']):
            im=Image.open(U/p[side]['file']);im.thumbnail((610,270));x=col*640+15;y=55+k*310
            sheet.paste(im,(x,y));draw.text((x,y+273),('사용자' if col==0 else '공개 참고')+f" {p[side]['time']:.2f}초 / frame {p[side]['frame']}",font=small,fill='black')
    sheet.save(folder/'paired_contact.png')
    return data

def fmt(v):return '—' if v is None else f'{v:.2f}'
def main():
    S.check(S.read(U/'FROZEN.json')['sources']);rows=S.read(U/'RESULT.json')['rows'];views=[]
    page_only='--page-only' in sys.argv
    if page_only:views=S.read(U/'VIEW_DATA.json')
    else:
        for word in S.DETAIL_WORDS:
            row=next(x for x in rows if x['word_id']==word)
            with np.load(U/'arrays'/f'{word}.npz') as z:
                plots(row,z);pairs=thumbs(row,z)
            views.append(dict(word_id=word,label=row['label'],required=row['required'],pairs=pairs))
    table=''.join(f"<tr><td>{html.escape(r['label'])}</td><td>{r['query_span'][1]-r['query_span'][0]:.2f}s / {r['reference_span'][1]-r['reference_span'][0]:.2f}s</td><td>{r['linear_mean']:.3f}</td><td>{r['dtw_mean']:.3f}</td><td>{r['path_pairs']}</td><td>{r['evidence']*100:.1f}%</td></tr>" for r in rows)
    cards=[]
    for word in S.DETAIL_WORDS:
        r=next(x for x in rows if x['word_id']==word);metrics=''
        for before,after in zip(r['before'],r['after']):
            metrics+=f"<tr><td>{'왼손' if after['side']=='L' else '오른손'} {'위치 차이' if after['group']=='handpos' else '15개 굽힘각 차이'}</td><td>{fmt(before['difference'])}</td><td>{fmt(after['difference'])}</td><td>{after['paired']}/{after['pairs']}</td><td>{'비교 대상' if after['required'] else '참고 기준 비대상'}</td></tr>"
        cards.append(f'''<section class="case" id="{word}"><h2>{r['label']} · 원래 DTW가 사용한 증거</h2>
        <p>사용자 {r['query_span'][0]:.2f}–{r['query_span'][1]:.2f}초 / 참고 {r['reference_span'][0]:.2f}–{r['reference_span'][1]:.2f}초. 사용자 {r['query_sampled']}개·참고 {r['reference_sampled']}개 표본을 {r['path_pairs']}쌍으로 연결했다.</p>
        <img class="chart" src="assets/{word}/time_path.png" alt="원래 시간 그래프와 DTW 시점 연결 경로">
        <div class="viewer" data-word="{word}"><div class="toolbar"><label>비교 방식 <select class="mode"><option value="dtw">DTW 연결</option><option value="linear">길이만 비례해서 연결</option></select></label><button class="play">자동 넘기기</button><span class="step"></span></div>
        <input class="slider" aria-label="연결한 동작 시점" type="range" min="0" value="0"><div class="pair"><figure><img class="query" alt="저장 사용자 RGB"><figcaption class="qcap"></figcaption></figure><figure><img class="ref" alt="학습 참고 RGB"><figcaption class="rcap"></figcaption></figure></div><p class="pairnote"></p></div>
        <p class="note">슬라이더를 움직여 같은 단계끼리 맞았는지 눈으로 확인할 수 있다. 자동 넘기기는 대응쌍 보기이며 원래 속도의 동영상 재생이 아니다. 사용자 RGB는 최대 10fps 저장 표본이므로 실제 사용한 관절 시점과의 차이를 표시한다. 두 영상 모두 좌우 반전하지 않았다.</p>
        <details open><summary>정렬 전후 위치·각도 그래프</summary><img class="chart" src="assets/{word}/signals.png" alt="정렬 전후 손 위치와 검지 각도 그래프"></details>
        <div class="scroll"><table><thead><tr><th>항목</th><th>길이만 비례</th><th>DTW 후</th><th>DTW 비교 가능 쌍</th><th>손 역할</th></tr></thead><tbody>{metrics}</tbody></table></div>
        <p class="note">위치 단위는 어깨너비=100, 각도 단위는 2D 도(°). 값은 차이의 크기이며 틀린 비율이나 합격 점수가 아니다. 검지 그래프의 내부각 하나와 표의 15개 평균 차이는 서로 다른 값이다. 전후 관측 가능한 대응쌍이 달라질 수 있어 수치 감소만으로 향상을 판단하지 않는다.</p></section>''')
    header='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 촬영으로 확인하는 TCN과 DTW</title><style>
    *{box-sizing:border-box}body{margin:0;background:#eef3f7;color:#19344a;font:17px/1.65 'Malgun Gothic',sans-serif}main{max-width:1180px;margin:auto;padding:30px 20px}header{background:#173b56;color:white;padding:38px;border-radius:20px}h1{font-size:32px;line-height:1.4}h2{font-size:24px}section{background:white;border-radius:16px;padding:28px;margin:24px 0}.lead{font-size:20px}.note{background:#fff6e7;border-left:4px solid #df9d28;padding:14px;font-size:15px}.good{background:#e7f5f1;padding:16px}.chart{width:100%;height:auto}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce5eb}th{background:#eaf1f6}nav a{display:inline-block;margin:6px 16px 0 0;color:#1778ad}header a{color:#bfe6ff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;background:#f0f3f5;border-radius:8px;padding:8px}figure img{width:100%;height:260px;object-fit:contain;background:#18222b}figcaption{font-size:14px}.toolbar{display:flex;gap:14px;flex-wrap:wrap;align-items:center}.slider{width:100%;margin:20px 0;accent-color:#1778ad}button,select{padding:9px;border:1px solid #b3c8d7;border-radius:6px;background:white;color:#173b56;font:inherit}.pairnote{font-size:14px}.viewer{border:1px solid #dce5eb;padding:16px;border-radius:12px}a{color:#1778ad}summary{cursor:pointer;font-weight:bold}li{margin-bottom:8px}@media(max-width:650px){main{padding:12px}header,section{padding:20px}h1{font-size:25px}.pair{grid-template-columns:1fr}figure img{height:auto}.lead{font-size:18px}table{min-width:600px}}
    </style><main><header><small>기존 결과 재구성 · 개인 촬영은 설명과 평가에만 사용 · 2026.10.27</small><h1>TCN은 단어를 고르고,<br>DTW는 동작의 시점을 연결한다</h1><p class="lead">최근 시험한 YOLO·HaMeR·좌표 보정 없이, 저장된 MediaPipe 관측과 원래 웹 DTW를 그대로 확인했다.</p><nav><a href="#overview">핵심 결과</a><a href="#WORD1290">감사</a><a href="#WORD1139">믿다</a><a href="#WORD0949">멀다</a><a href="#tcn">TCN과 속도</a></nav></header>
    <section id="overview"><h2>이렇게 보면 이해하기 쉽다</h2><ol><li>‘감사’로 내려가 <b>정렬 전후 그래프</b>를 본다. 파란 선은 네 촬영, 주황 선은 공개 참고다. 오른쪽은 DTW가 연결한 시점으로 참고를 재배치했다. 위치나 각도 자체를 같게 고치지 않았다.</li><li><b>장면 슬라이더</b>를 움직여, 실제로 같은 행동을 하는 순간끼리 짝지었는지 본다. ‘길이만 비례’로 바꾸면 DTW 전과 비교할 수 있다.</li><li>선이 겹친다고 수어가 올바르다는 뜻은 아니다. 사진에서 동작 단계가 맞는지, 손이 실제로 보이는지도 함께 확인한다.</li></ol><h2>먼저 확인한 핵심</h2><p>12개 촬영 모두 기존 웹에서 사용한 공개 시연 참고 한 개와 비교했다. 저장된 DTW 수치와 이번 재구성 수치가 일치한다. 점수가 가장 좋은 참고를 새로 고르지 않았다. 이 표는 전체 다중 참고의 최종 집계가 아니라, 보여 준 참고 한 개와의 계산이다.</p>
    <div class="note"><b>중요한 입력 차이:</b> 현재 웹 DTW는 양쪽 영상에서 각각 ‘손이 처음 관측된 순간부터 마지막까지’를 사용한다. 참고 촬영은 내려놓은 손도 보이므로 준비·마무리가 남지만, 사용자는 손이 화면에 들어온 뒤부터 포함된다. ‘감사’는 참고 약4.47초, 사용자 약2.09초다. 의미상 핵심 동작만 정확히 잘라 비교한 상태가 아니다.</div>
    <p>정렬 후 전체 비용이 낮아져도 모든 손가락·위치 차이가 줄어들지는 않는다. DTW는 손·팔·손 모양 등의 가중 비용 합계를 줄이는 경로를 찾는다. 아래 경로는 정답 동작 시점을 사람이 인증한 것이 아니다.</p>
    <div class="note"><b>실제 장면에서 확인한 어긋남:</b> ‘멀다’의 33번째 대응쌍은 사용자 0.900초(손이 얼굴 옆에 올라온 상태)를 참고 1.267초(손을 내린 대기 상태)와 연결한다. 이 사용자 RGB는 관절 시점과도 정확히 일치한다. 따라서 준비·마무리가 섞이는 문제는 숫자만의 추측이 아니다. 다만 모든 구간의 원인이나 수정 효과까지 입증한 것은 아니다.</div>
    <div class="scroll"><table><thead><tr><th>표현</th><th>관측 구간 길이 사용자/참고</th><th>길이 비례 비용</th><th>DTW 비용</th><th>대응쌍</th><th>경로 관측 근거</th></tr></thead><tbody>'''
    middle='''</tbody></table></div><p class="note">비용은 여러 특징의 단위가 섞인 내부 비교값이다. 정확도나 수행 점수가 아니다. 경로 관측 근거는 실제 양쪽에서 비교한 특징 가중치 비율이며, 손가락 좌표가 정확하다는 확률이 아니다. DTW는 누적 비용을 최소화하며, 표에는 해석을 위해 대응쌍당 평균 비용을 표시했다.</p></section>'''
    tail='''<section id="tcn"><h2>TCN은 속도가 달라도 어떻게 인식하나?</h2><p><b>현재 C·T3-F 안에는 DTW가 없다.</b> 참고 영상과 직접 프레임을 짝짓는 대신, 시간 방향의 합성곱으로 관절 변화 패턴을 읽고 100개 단어 중 하나를 분류한다. 길이가 다른 입력을 처리하고 C는 시간별 특징을 평균 요약한다. T3-F는 실제 경과시간을 0~1로 바꾼 앞·중간·뒤 요약도 추가한다.</p>
    <p>현재 입력은 원래 시퀀스 길이와 프레임 간격 정보를 사용한다. 모든 촬영을 같은 프레임 수로 바꾸거나 동작 속도를 완전히 같게 만드는 구조는 아니다. 패턴을 학습했기 때문에 약간의 속도 차이에서도 맞힐 수 있지만, 변화 폭과 단어에 따라 달라지고 같은 답을 보장하지 않는다.</p>
    <p>지난 12회 동일 촬영 시험에서는 C 10/12, T3-F 12/12였다. 그 시험은 속도를 따로 통제한 실험이 아니므로 ‘비슷한 속도로 따라해서만 성공했다’거나 ‘속도에 완전히 무관하다’ 중 어느 쪽도 증명하지 않는다. 추가 공개 199영상의 화면 밖 미관측 재현 조건에서는 C 193/199(96.98%), T3-F 196/199(98.49%)였다. 이는 학습한 100개 단어 범위의 결과이며 모든 수어·모든 사용자 성능은 아니다.</p>
    <p><b>이전 개선을 어떻게 이어갈까?</b> 손 검출 보완의 작은 공개 문제 표본에서는 MediaPipe 재검출이 손목 위치 근거를 3/23→12/23으로 늘린 기록이 있다. 손가락 전체의 정확도나 TCN 인식률을 뜻하지 않는다. T3-F 등 단어 분류 개선은 그와 별개로 유지한다. 현재 웹 DTW는 원래 관측을, C·T3-F는 별도 C 입력을 사용한다. 재검출 개선을 이미 웹 DTW에 모두 적용한 것으로 혼동하면 안 된다.</p>
    <div class="note">기존 공개 시험에서 30fps→20/15fps로 관측 간격을 바꿔 본 것은 수행 속도를 빠르게/느리게 한 시험과 다르다. 이번에는 새 TCN 추론·학습·속도 보강을 하지 않았다.</div></section>
    <section><h2>다음에 바로잡을 부분</h2><ol><li>준비·마무리와 핵심 동작 범위를 양쪽에서 일관되게 다루는지 공개 자료로 점검한다. 중간 손내림·가림은 잘라내지 않는다. 화면 밖 손을 보였다고 채우지 않는다.</li><li>일부 참고 프레임이 같은 사용자 프레임에 몰려 연결되는 부분에서 실제 동작 단계가 맞는지 확인한다. 낮은 비용만으로 채택하지 않는다.</li><li>정상 속도·손내림·교차를 보존하는지 고정 공개 대조로 검증한 뒤에만 별도 웹 후보에 연결한다. 위치·궤적 근거부터 확보하고, 관측이 불확실한 손가락은 잘못된 동작으로 단정하지 않는다.</li><li>TCN 속도 시험은 별도 진행한다. 공개 영상의 정상적인 시간 변화에 대한 분류 변화를 확인하되, 단순 재생속도 변경 시험과 실제 빠른 손동작의 흐림·검출 누락 시험을 구분한다. 이번 촬영 결과로 속도 보강이나 모델 선택을 하지 않는다.</li></ol><p>이번 사용자 영상은 문제를 설명하는 평가 자료다. 범위·임계값·학습 조건을 이 촬영에 맞추지 않는다. 기존 C·T3-F·운영 DTW·웹과 원본 파일은 유지했다. YOLO·HaMeR·MANO 및 최근 좌표 보정은 이 비교와 후속 기본 경로에서 보류한다. 기존 연구 산출물을 삭제하거나 운영 모델을 되돌린 것은 아니다.</p><p><a href="FROZEN.json">고정 비교 조건</a> · <a href="RESULT.json">수치와 출처</a> · <a href="VERIFIED.json">별도 산술검산</a></p></section></main><script>const DATA=__DATA__;
    document.querySelectorAll('.viewer').forEach(v=>{const data=DATA.find(x=>x.word_id===v.dataset.word),slider=v.querySelector('.slider'),mode=v.querySelector('.mode');let timer=null;
    function update(){const arr=data.pairs[mode.value];slider.max=arr.length-1;slider.value=Math.min(+slider.value,arr.length-1);const p=arr[+slider.value];v.querySelector('.query').src=p.query.file;v.querySelector('.ref').src=p.reference.file;v.querySelector('.qcap').textContent=`사용자 RGB ${p.query.time.toFixed(2)}초 · frame ${p.query.frame}`;v.querySelector('.rcap').textContent=`공개 TRAIN 참고 ${p.reference.time.toFixed(2)}초 · frame ${p.reference.frame}`;v.querySelector('.step').textContent=`${+slider.value+1} / ${arr.length} 대응쌍`;v.querySelector('.pairnote').textContent=`사용한 사용자 관절: ${p.query_observation_time.toFixed(3)}초 (frame ${p.query_observation_frame}). 표시 RGB와 ${p.rgb_delta_ms}ms 차이. 이 쌍 비용 ${p.cost.toFixed(3)}, 양쪽 관측 근거 ${(100*p.evidence).toFixed(1)}%.`;}
    slider.addEventListener('input',update);mode.addEventListener('change',update);v.querySelector('.play').onclick=()=>{if(timer){clearInterval(timer);timer=null;v.querySelector('.play').textContent='자동 넘기기';return;}v.querySelector('.play').textContent='멈추기';timer=setInterval(()=>{slider.value=(+slider.value+1)%(+slider.max+1);update()},200)};update();});
    </script></html>'''
    datajson=json.dumps(views,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (U/'RESULT_KO.html').write_text(header+table+middle+''.join(cards)+tail.replace('__DATA__',datajson),encoding='utf-8')
    if not page_only:S.write('VIEW_DATA.json',views)
    S.write('RENDER_MANIFEST_V2.json' if page_only else 'RENDER_MANIFEST.json',dict(files={str(p):S.sha(p) for p in U.rglob('*') if p.is_file() and p.suffix in ('.jpg','.png','.html')},private_local_only=True))
    print('RENDERED3',flush=True)
if __name__=='__main__':main()
