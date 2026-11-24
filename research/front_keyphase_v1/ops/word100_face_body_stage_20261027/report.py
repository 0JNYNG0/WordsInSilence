import json,sys,html,math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.face_body_relations import region_geometry,REGIONS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
F=read(U/'FROZEN.json');R=read(U/'RESULT.json');V=read(U/'VERIFIED.json')
fontfile='C:/Windows/Fonts/malgun.ttf';font=ImageFont.truetype(fontfile,20);small=ImageFont.truetype(fontfile,15)
region_ko=['이마 부근','오른 광대 부근','왼 광대 부근','입 부근','턱 부근','코 부근','목 부근','가슴 부근']
colors=['#f5ad42','#bf8bff','#73b5ff','#f2769e','#70d3c1','#d5dd83','#d9dce4','#84bdaa']
(U/'images').mkdir(exist_ok=True)
visual=[]

def fixed_scope():
    rows=[]
    for section in ('known','cross'):
        for q in F['rows']:
            if section=='cross' and q['signer']==1:continue
            ref=q if section=='known' else next(a for a in F['rows'] if a['word']==q['word'] and a['signer']==1)
            qo=Observation.load(q['observation']);ro=Observation.load(ref['observation'])
            with np.load(U/'data'/q['clip_id']/('known_paths.npz' if section=='known' else 'cross_paths.npz')) as z:
                truth=z['source_mapping'] if section=='known' else np.arange(qo.n_frames)
                query_times=qo.timestamp_us[truth]
                required=np.zeros(len(truth),bool)
                for a,b in q['intervals']:required|=(query_times>=a*1e6)&(query_times<=b*1e6)
                ids=np.flatnonzero(required)
                for cond in 'ABC':
                    a,b=z[cond+'_reference'],z[cond+'_query']
                    inside=ids[(ids>=b.min())&(ids<=b.max())]
                    # Evaluate all annotated query frames on the same denominator; nearest
                    # saved path sample is diagnostic sampling, not new ground truth.
                    refs=[]
                    for i in inside:
                        near=np.abs(b-i).min();refs.append(int(np.median(a[np.abs(b-i)==near])))
                    refs=np.array(refs,dtype=int)
                    action=np.zeros(len(refs),bool)
                    for start,end in ref['intervals']:action|=(ro.timestamp_us[refs]>=start*1e6)&(ro.timestamp_us[refs]<=end*1e6)
                    d=dict(section=section,query=q['clip_id'],reference=ref['clip_id'],condition=cond,
                           required_frames=len(ids),outside_path_window=len(ids)-len(inside),
                           mapped_to_reference_action=int(action.sum()),mapped_to_reference_rest=int((~action).sum()))
                    if section=='known':d['mapping_mae_ms']=float(np.mean(np.abs(ro.timestamp_us[refs]-qo.timestamp_us[truth[inside]])/1000))
                    rows.append(d)
    summary={}
    for section in ('known','cross'):
        summary[section]={}
        for cond in 'ABC':
            rr=[r for r in rows if r['section']==section and r['condition']==cond]
            summary[section][cond]={k:sum(r[k] for r in rr) for k in ('required_frames','outside_path_window','mapped_to_reference_action','mapped_to_reference_rest')}
    write(U/'FIXED_SCOPE.json',dict(method='All annotated query frames; no exclusion by condition. Coarse action-vs-rest audit, not internal phase GT.',rows=rows,summary=summary))
    return summary

S=fixed_scope()
for r in F['rows']:
    cid=r['clip_id'];out=U/'data'/cid;done=read(out/'DONE.json')
    obs=Observation.load(str(out/'masked.npz'));face=join_face_observation(out/'face_roi.npz',obs)
    with np.load(out/'relations_roi.npz') as z:rel={k:v for k,v in z.items()}
    tmid=int(np.argmin(abs(obs.timestamp_us-np.mean(r['intervals'][0])*1e6)))
    # Add the strongest projected hand/face overlap if any, otherwise nearest tip/face.
    counts=rel['projected_hand_overlap'][:,:6].sum(axis=1)
    if counts.max()>0:stress=int(np.argmax(counts));why='손-얼굴 겹침 경고가 가장 많은 시점'
    else:
        ds=rel['distances'][:,:,:,:6];nearest=np.nanmin(np.where(np.isfinite(ds),ds,np.inf),axis=(1,2,3))
        stress=int(np.argmin(nearest));why='손끝이 얼굴 부근에 가장 가까운 시점'
    picks=list(dict.fromkeys((tmid,stress)))
    cap=cv2.VideoCapture(done['video'])
    for part,t in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(obs.frame_id[t]));ok,bgr=cap.read();assert ok
        pts=int(round(cap.get(cv2.CAP_PROP_POS_MSEC)*1000));assert abs(pts-int(obs.timestamp_us[t]))<=1000
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);overlay=Image.fromarray(rgb);d=ImageDraw.Draw(overlay)
        d.line((0,r['roi'][3],rgb.shape[1],r['roi'][3]),fill='#ffcf57',width=4)
        # Plot all face coordinates faintly, then body-region centres distinctly.
        for x,y in face['xy_px'][t]:
            if np.isfinite((x,y)).all():d.ellipse((x-1,y-1,x+1,y+1),fill='#71d6d0')
        for side,base in enumerate((33,54)):
            for j in range(21):
                if obs.usable[t,base+j]:
                    x,y=obs.xy_px[t,base+j];d.ellipse((x-3,y-3,x+3,y+3),fill=('#6da8ff','#ffbd67')[side])
        centers,axes,radii,valid=region_geometry(face['xy_px'][t],obs.xy_px[t],obs.usable[t])
        for j in range(8):
            if not valid[j]:continue
            x,y=centers[j];d.ellipse((x-5,y-5,x+5,y+5),fill=colors[j],outline='white',width=1)
            d.text((x+7,y-7),str(j+1),font=small,fill='white',stroke_width=2,stroke_fill='black')
        frame=Image.new('RGB',(1200,680),'#101923');draw=ImageDraw.Draw(frame)
        draw.text((22,16),f"{r['label']} · 공개 촬영 ID {r['signer']:02} · {obs.timestamp_us[t]/1e6:.3f}초",font=font,fill='white')
        frame.paste(overlay.resize((730,411)),(20,63))
        xy=face['xy_px'][t];xs=xy[:,0];ys=xy[:,1]
        x0=max(0,int(np.nanmin(xs)-55));x1=min(rgb.shape[1],int(np.nanmax(xs)+55))
        y0=max(0,int(np.nanmin(ys)-55));y1=min(rgb.shape[0],int(np.nanmax(ys)+55))
        crop=overlay.crop((x0,y0,x1,y1));crop.thumbnail((420,460));frame.paste(crop,(770,63))
        draw.text((20,495),'왼쪽: 원본 전체 화면 / 노란 선: 관절 사용 범위',font=font,fill='#e3e8f0')
        draw.text((20,528),'오른쪽: 얼굴 위치 확인용 확대 · 손/몸 검출 화면을 자른 것이 아님',font=small,fill='#abb9cd')
        for j,label in enumerate(region_ko):
            x=20+(j%4)*295;y=570+(j//4)*35
            draw.text((x,y),f'{j+1} {label}',font=small,fill=colors[j])
        path=f'images/{cid}_{part}.jpg';frame.save(U/path,quality=90)
        visual.append(dict(clip_id=cid,label=r['label'],signer=r['signer'],image=path,frame=int(obs.frame_id[t]),
                           pts_us=int(obs.timestamp_us[t]),selection='주석 구간 가운데' if part==0 else why,
                           overlap_regions=[region_ko[i] for i in np.flatnonzero(rel['projected_hand_overlap'][t])]))
    cap.release()

plt.rcParams['font.family']='Malgun Gothic'
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,word in zip(axes,('WORD1290','WORD0949')):
    r=next(x for x in F['rows'] if x['word']==word and x['signer']==1)
    with np.load(U/'data'/r['clip_id']/'known_paths.npz') as z:
        truth=z['source_mapping'];ax.plot(np.arange(len(truth))/30,truth/30,'k--',label='정확히 아는 원본 대응',linewidth=2)
        for cond,color in zip('ABC',('#db815c','#199a9b','#9e7bc9')):
            ax.plot(z[cond+'_query']/30,z[cond+'_reference']/30,label=cond,color=color,alpha=.85)
    ax.set(title=r['label'],xlabel='시간을 늘린 공개 영상 (초)',ylabel='연결된 원본 영상 (초)');ax.legend(fontsize=8)
fig.savefig(U/'images/timing.png',dpi=150);plt.close(fig)
write(U/'VISUAL_INDEX.json',visual)
rec=[read(U/'data'/r['clip_id']/'RECOVERED.json') for r in F['rows']]
speed=np.median([x['inference_ms_median'] for x in rec]);frames=sum(x['frames'] for x in rec);found=sum(x['found_frames'] for x in rec)
summary=f'''얼굴·신체 부위 연결 및 공개 시간 정렬 검증

얼굴 관측 기능 추가: 기존 Pose33+Hand42는 그대로 유지하고 FaceLandmarker 478점을 별도 저장한다.
얼굴 검출: 원본 전체 프레임의 기본 얼굴 검출은 {frames}프레임 모두 실패했다. 현재 Pose 머리/어깨로 정한 머리 영역 재확인 후 {found}/{frames}프레임에서 얼굴을 얻었다. 얼굴이 작게 찍힌 조건에서 확대 확인이 효과적임을 확인했다. 손/몸 검출 범위는 바뀌지 않았다. 얼굴 검출률은 정확도나 가림 해결률이 아니다.
이마·좌우 광대 부근·입·턱·코·목·가슴의 근처 관계를 5개 손끝 각각과 연결했다. 광대는 얼굴 표면 부근 추정이며 뼈 위치의 정밀 측정이 아니다. 2D 중첩은 실제 접촉을 확정하지 않는다. 손-얼굴 투영 겹침 {V['checks']['face_patch_overlap_warnings']}개 프레임-부위 경고는 얼굴 관측 불확실성으로 분리했다. 실제 가림 정답 건수는 아니다.
얼굴 추론 추가 비용: 영상별 중앙값의 중앙값 {speed:.2f}ms/프레임(CPU, 기본+재확인 합; 디코딩/TCN/DTW 제외). 실제 웹 실시간 성능은 미검증.

시간 대응을 아는 동일 공개 영상 변형 18개: A(참고 원본/비교 상반신) 평균오차 {R['known_mapping_summary']['A']['mean_clip_mapping_error_ms']:.2f}ms, B(양쪽 동일 상반신) {R['known_mapping_summary']['B']['mean_clip_mapping_error_ms']:.2f}ms, C(B+추가 시작끝 제한) {R['known_mapping_summary']['C']['mean_clip_mapping_error_ms']:.2f}ms. 이는 같은 관절을 시간만 늘린 통제시험이며 실제 다른 수행자의 단계 정렬 정확도가 아니다.
다른 공개 수행자 12쌍도 저장했다. 손실 없이 동일한 동작 구간 분모로 재검산한 수치는 FIXED_SCOPE.json 참조. 내부의 세부 동작 시점 정답은 없어 완전한 단계 일치나 개선률을 주장할 수 없다.
추가 시작끝 제한 C는 별 REAL01 주석49프레임 중6프레임을 제외했다. 따라서 채택하지 않는다. 실험 A/B/C와 기존 TCN 후보 C는 별개다.
관측 부족을 낮은 차이로 통과시키지 않는다. 손 전체 미관측시 비교 불가, 이웃 손끝 정보 보존, 좌표 이동/크기/회전 불변, 시간 연결 오류 거부 검산 완료. 관측한 범위와 제외/불확실 구간은 별도 기록한다.

현재는 연구용 얼굴 연결·다중 참고 진단 구현이다. 웹 practice/TCN C·T3-F/DTW 운영 코드 교체는 하지 않았다. 사용자·멘토 영상 접근/학습/튜닝/선택, 새 학습, YOLO/HaMeR/MANO 사용은 없다.
남은 일: 얼굴 부위 실제 위치/경계/가림의 더 넓은 검증, 기존 관측범위 B를 기반으로 다른 수행자 세부단계 정렬 검증, 언어적으로 허용되는 부위/모양 차이 기준. 손이 광대 근처라는 측정과 그 수어가 맞다는 판정은 다르다. 표정 점수/접촉 확정/교정문장은 아직 제공하지 않는다.
검증 담당: Codex 구현·실행 및 별도 저장산출물/불변조건 검산, 독립 전문가 평가 아님.
'''
(U/'RESULT_KO.txt').write_text(summary,encoding='utf-8')
cards=[]
for word in dict.fromkeys(r['word'] for r in F['rows']):
    refs=[r for r in F['rows'] if r['word']==word];c=[]
    for r in refs:
        vi=[x for x in visual if x['clip_id']==r['clip_id']]
        for v in vi:
            c.append(f'<figure><img loading="lazy" src="{v["image"]}"><figcaption>{html.escape(v["selection"])} · 프레임 {v["frame"]} · 겹침 경고: {html.escape(", ".join(v["overlap_regions"]) or "없음(완전 가시성 보증 아님)")}</figcaption></figure>')
    cards.append(f'<details><summary>{refs[0]["label"]} · 공개 ID 01 / 02 / 03</summary>{"".join(c)}</details>')
fixed=[]
for cond in 'ABC':
    d=S['cross'][cond]
    fixed.append(f'<tr><td>{cond}</td><td>{d["required_frames"]}</td><td>{d["mapped_to_reference_action"]}</td><td>{d["mapped_to_reference_rest"]}</td><td>{d["outside_path_window"]}</td></tr>')
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>얼굴 부위와 동작 단계 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#1e3042;font:17px/1.75 'Malgun Gothic',sans-serif}main{max-width:1180px;margin:auto;padding:42px 24px}header{padding:34px;border-radius:20px;background:#102a3a;color:white}h1{font-size:35px;line-height:1.35;margin:10px 0}h2{font-size:25px}section,details{background:white;border:1px solid #d9e3e9;border-radius:16px;padding:24px;margin:22px 0}.tag{color:#86ddd2;letter-spacing:.08em}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.box{background:#edf5f5;padding:22px;border-radius:12px}.big{font-size:30px;font-weight:bold;color:#167d80}img{width:100%;height:auto;border-radius:10px}figure{margin:22px 0}figcaption{font-size:14px;color:#526878}summary{font-size:20px;font-weight:bold;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #dde6ec;padding:10px;text-align:left}.warn{background:#fff4df;border-left:5px solid #db9b31;padding:18px}a{color:#127d85}.flow{font-weight:bold;font-size:19px;line-height:2}pre{white-space:pre-wrap;font:14px/1.7 'Malgun Gothic';overflow-wrap:anywhere}@media(max-width:700px){main{padding:18px 12px}h1{font-size:27px}header,section,details{padding:18px}.grid{grid-template-columns:1fr}table{font-size:12px}td,th{padding:5px}}</style><main>
<header><div class="tag">WordsInSilence · 공개 자료 검증 · 2026.10.27</div><h1>손이 얼굴의 어느 부근에 있는가?</h1><p>얼굴 위치 기준점을 추가하고, 같은 동작 시점에 비교되는지 함께 확인했습니다.</p></header>
<section><div class="grid"><div class="box"><div class="big">6표현 × 3 ID</div>공개 영상 18개</div><div class="box"><div class="big">'''+str(found)+'''프레임</div>얼굴 기준점 확보 · 정확도와 다름</div><div class="box"><div class="big">기존 모델 보존</div>연구 후보 구현 · 웹 운영 미교체</div></div><h2>달라진 부분</h2><p>기존에는 손목과 코의 상대 위치를 사용했습니다. 이제는 <b>각 손가락 끝이 이마·광대 부근·입·턱 등에 얼마나 가까운지</b> 별도로 기록합니다. 몸과 얼굴의 크기에 맞춰 거리 기준을 정리하고, 경계가 겹치면 한 부위로 억지로 확정하지 않습니다.</p><p class="flow">원본 전체 영상 → 기존 몸·손 검출 유지<br>→ 얼굴 검출 / 실패하면 머리 영역 재확인 → 원본 좌표로 복원<br>→ 상반신 관측 범위 적용 → 손끝–부위 관계 → DTW 대응 시점에서 비교</p><div class="warn">얼굴 기준점은 추정값입니다. 손이 얼굴을 가리면 비교 근거를 보류할 수 있습니다. 화면에서 겹쳐 보인다는 것만으로 실제로 만졌다고 판정하지 않습니다.</div></section>
<section><h2>얼굴 검출에서 발견하고 해결한 문제</h2><p>전체 화면에 얼굴이 작게 찍혀 기본 얼굴 검출은 모든 프레임을 놓쳤습니다. 몸 검출로 얻은 머리·어깨 위치를 이용해 <b>현재 장면의 얼굴 주변만 확대 확인</b>하자 얼굴 기준점을 얻었습니다. 손과 몸의 관절 검출은 원본 전체 화면 그대로입니다.</p><p>얼굴 확인 추가 비용: 영상별 중앙값의 중앙값 <b>'''+f'{speed:.2f}'+'''ms/프레임</b>. 영상 읽기·TCN·DTW는 제외한 CPU 추론 비용이며, 웹 전체 속도는 아직 시험하지 않았습니다.</p></section>
<section><h2>시간 정렬: 무엇을 비교했나?</h2><p><b>A</b> 참고는 원본, 비교 쪽만 상반신 범위 / <b>B</b> 양쪽 모두 같은 상반신 범위 / <b>C</b> B에서 시작·끝을 추가로 좁힌 후보입니다. 여기서 C는 기존 TCN 모델 C와 다른 실험 이름입니다.</p><img src="images/timing.png"><p>검은 점선은 시간을 늘리기 전의 정확한 대응입니다. 선이 가까울수록 이 통제시험의 시점이 잘 맞습니다. <b>동일한 공개 영상의 시간만 변형</b>했기 때문에 이 결과를 다른 사람의 실제 수어 정확도로 해석하면 안 됩니다.</p><div class="grid">'''+''.join(f'<div class="box"><b>{c}</b><div class="big">{R["known_mapping_summary"][c]["mean_clip_mapping_error_ms"]:.2f}ms</div>영상별 평균 대응오차의 평균</div>' for c in 'ABC')+'''</div><h2>다른 수행자 12쌍 · 같은 분모로 확인</h2><p>모든 조건에서 비교 쪽의 주석 동작 프레임을 분모로 유지했습니다. 준비·마무리에 잘못 연결되거나 창 밖으로 빠진 프레임도 숨기지 않았습니다. 이 주석은 단어 수행 구간이며, 안쪽의 세부 동작 단계 정답은 아닙니다.</p><table><tr><th>조건</th><th>동작 프레임</th><th>참고 동작 구간에 연결</th><th>참고 준비·마무리에 연결</th><th>비교 창 밖</th></tr>'''+''.join(fixed)+'''</table><div class="warn"><b>추가 시작·끝 제한 C는 미채택.</b> ‘별’ 공개 ID01의 주석 동작 49프레임 중 6프레임이 비교 범위에서 제외됐습니다. 낮은 정렬 비용만 보고 더 좋아졌다고 판단하지 않았습니다.</div></section>
<section><h2>실제 장면으로 확인하기</h2><p>아래 항목을 펼치면 각 공개 촬영자의 전체 화면과 얼굴 확대를 함께 볼 수 있습니다. 색 점은 얼굴 부근의 대표 위치이며, 광대뼈의 정확한 경계나 정답 영역으로 확정한 표시는 아닙니다. 손가락 점은 기존 관측을 그대로 사용했습니다.</p>'''+''.join(cards)+'''</section><section><h2>현재 할 수 있는 것 / 아직 남은 것</h2><p><b>구현:</b> 얼굴 관절 별도 저장, 원본 시점과 좌표 연결, 손끝별 부위 거리, 얼굴 가림 의심 표시, DTW 시점별 관계·관측 비율 계산.</p><p><b>남은 검증:</b> 얼굴 부위 경계와 가림 상황의 실제 정확성, 다른 사람끼리 세부 동작이 맞춰지는지, 작은 자연 차이를 어디까지 허용할지. 표정 평가와 ‘틀렸으니 이렇게 고치세요’라는 자동 교정은 아직 연결하지 않았습니다.</p><p>같은 가슴 부근에서 조금 다르게 움직이는 것과, 얼굴의 다른 부위에서 수행하는 것을 구분할 측정 기반을 마련한 단계입니다. C·T3-F 및 기존 웹은 보존했고 사용자 촬영은 접근하거나 학습·튜닝에 사용하지 않았습니다.</p><details><summary>검증 기록과 한계</summary><pre>'''+html.escape(summary)+'''</pre><p><a href="VERIFIED.json">별도 검산</a> · <a href="FIXED_SCOPE.json">같은 분모의 시간 정렬 감사</a> · <a href="RESULT_KO.txt">설명 원문</a></p></details></section></main></html>'''
(U/'RESULT_KO.html').write_text(doc,encoding='utf-8')
print(json.dumps(dict(images=len(visual),frames=frames,found=found,speed_ms=speed,fixed_scope=S),ensure_ascii=False))
