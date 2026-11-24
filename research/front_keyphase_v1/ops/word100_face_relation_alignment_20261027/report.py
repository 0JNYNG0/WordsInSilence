import sys,json,html
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent;OLD=LAB/'ops/word100_face_body_stage_20261027'
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.face_body_relations import region_geometry
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()),encoding='utf-8')
F=read(U/'FROZEN.json');R=read(U/'RESULT.json');A=read(U/'BOUNDARY_AUDIT.json');V=read(U/'VERIFIED.json')
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
reg=['이마','오른 광대 부근','왼 광대 부근','입','턱','코'];endpoint=['엄지끝','검지끝','중지끝','약지끝','소지끝','손목(손 모델)','손바닥 중심','손목(몸 모델)','팔꿈치','팔 구간 최근접점']
cases=[]
for status,preferred in ((2,'WORD0106'),(3,'WORD2551'),(4,'WORD2551')):
 selected=None
 for q in F['public_rows']:
  if q['word']!=preferred:continue
  with np.load(U/'data'/f"{q['clip_id']}_reference_comparison.npz") as c,np.load(q['observation']) as obs:
   active=np.zeros(len(obs['frame_id']),bool)
   for a,b in q['intervals']:active|=(obs['timestamp_us']>=a*1e6)&(obs['timestamp_us']<=b*1e6)
   positions=np.argwhere((c['projection_status']==status)&active[:,None,None,None])
   if not len(positions):continue
   t,side,part,region=map(int,positions[len(positions)//2]);rr=c['reference_rows'][:,t].tolist();refs=c['reference_ids'].tolist()
   selected=dict(status=status,query=q['clip_id'],label=q['label'],frame=t,side=side,part=part,region=region,references=refs,reference_frames=rr);break
 if selected:cases.append(selected)
descriptions={2:'같은 단계에서 두 참고도 투영 겹침을 보임',3:'두 참고의 투영 겹침 근거가 서로 다름',4:'두 참고와 다른 투영 겹침 · 오답 확정 아님'}
for ci,case in enumerate(cases):
 canvas=Image.new('RGB',(1440,780),'#112434');draw=ImageDraw.Draw(canvas);draw.text((20,12),case['label']+' · '+descriptions[case['status']],font=font,fill='white')
 table=[]
 for col,(cid,t) in enumerate([(case['query'],case['frame'])]+list(zip(case['references'],case['reference_frames']))):
  row=next(x for x in F['public_rows'] if x['clip_id']==cid);old=OLD/'data'/cid;d=read(old/'DONE.json');obs=Observation.load(str(old/'masked.npz'));face=join_face_observation(old/'face_roi.npz',obs)
  with np.load(U/'data'/f'{cid}_evidence.npz') as z:
   dist=z['distance'][t,case['side'],:,case['region']];availability=z['available'][t,case['side'],:,case['region']]
   table.append(dict(clip_id=cid,frame=int(t),distance=[float(x) if np.isfinite(x) else None for x in dist],available=availability.tolist()))
  cap=cv2.VideoCapture(d['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(t));ok,bgr=cap.read();assert ok;pts=int(round(cap.get(cv2.CAP_PROP_POS_MSEC)*1000));cap.release();assert abs(pts-int(obs.timestamp_us[t]))<=1000
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));di=ImageDraw.Draw(im)
  centers,axes,radii,valid=region_geometry(face['xy_px'][t],obs.xy_px[t],obs.usable[t])
  for side,base in enumerate((33,54)):
   color=('#70b4ff','#fdbd6a')[side]
   ids=[11+side,13+side,15+side]
   for a,b in zip(ids,ids[1:]):
    if obs.usable[t,a] and obs.usable[t,b]:di.line((tuple(obs.xy_px[t,a]),tuple(obs.xy_px[t,b])),fill=color,width=3)
   for j in range(21):
    if obs.usable[t,base+j]:
     x,y=obs.xy_px[t,base+j];di.ellipse((x-3,y-3,x+3,y+3),fill=color)
  for j in range(6):
   if valid[j]:
    x,y=centers[j];color='#fa737f' if j==case['region'] else '#78d9bf';di.ellipse((x-5,y-5,x+5,y+5),fill=color)
  x=col*480+12;draw.text((x,55),('비교 영상' if col==0 else '다른 사람 참고')+f" · ID{row['signer']:02}",font=font,fill='white');draw.text((x,90),f'{pts/1e6:.3f}초 / 프레임 {t}',font=small,fill='#b5d0e2')
  canvas.paste(im.resize((456,257)),(x,123))
  fxy=face['xy_px'][t];lo=np.floor(np.nanmin(fxy,axis=0)-65).astype(int);hi=np.ceil(np.nanmax(fxy,axis=0)+65).astype(int)
  zoom=im.crop((max(0,lo[0]),max(0,lo[1]),min(im.width,hi[0]),min(im.height,hi[1])));zoom.thumbnail((300,310));canvas.paste(zoom,(x+75,397))
 draw.text((20,728),f"빨간 기준점: {reg[case['region']]} · {'왼쪽' if case['side']==0 else '오른쪽'} {'손 영역' if case['part']==0 else '팔 구간'}의 화면상 겹침",font=small,fill='#d1e2ee')
 draw.text((20,754),'시점 정렬과 검출 오차가 남아 있어 실제 접촉·가림 원인·수어 정오를 확정하지 않음',font=small,fill='#a7bed2')
 path=f'images/reference_case{ci}.jpg';canvas.save(U/path,quality=95);case['image']=path;case['measurements']=table
write(U/'REFERENCE_VIEWS.json',cases)
review={g['id']:'양쪽 모두 상승·시작 모양 또는 마무리·하강을 보임. 주석 밖이라고 곧 정지 자세는 아님; 세부 의미정답은 미확인.' for g in A['groups']}
review['G08']='비교 영상의 손 접근이 진행되는 동안 참고 f52에 여러 시점이 붙음. 실제 단계 진행 불일치 의심이 남으며 정상 속도 차이와 구별해 검증 필요.'
write(U/'VISUAL_REVIEW.json',dict(operator='Codex contact-sheet visual review; not independent expert labels',
 reviewed_groups=review,reviewed_group_frames='each contiguous run first/middle/last; not every intermediate frame manually labeled',
 correction='45 frames are annotation-discordant, not 45 visually confirmed erroneous alignments. 33 within100ms of reference boundary.',
 hand_missing='only1 of45 has no usable hand on one side of paired clip; hand count alone does not certify shape/identity',
 future_expressions_excluded=True))
active_counts={str(k):0 for k in range(5)}
for q in F['public_rows']:
 with np.load(q['observation']) as obs,np.load(U/'data'/f"{q['clip_id']}_reference_comparison.npz") as c:
  ids=np.zeros(len(obs['frame_id']),bool)
  for a,b in q['intervals']:ids|=(obs['timestamp_us']>=a*1e6)&(obs['timestamp_us']<=b*1e6)
  for k in range(5):active_counts[str(k)]+=int((c['projection_status'][ids]==k).sum())
write(U/'ACTIVE_SUMMARY.json',dict(unit='frame x anatomical side x hand/forearm x face patch; correlated cells, not independent sign errors',counts=active_counts))
B,D=R['summary']['B'],R['summary']['D']
summary=f'''손·팔과 얼굴 부위 관계 비교 / 표정 제외

목표: 손가락끝, 손목, 손바닥, 팔꿈치, 팔 구간이 얼굴 어느 부근에 놓이는지 여러 수행자의 같은 단계와 비교한다. 표정 정보·표정 점수는 완전히 제외한다. 기존18공개영상/얼굴478점과75점몸손을재사용, 새검출·다운로드·학습없음.

1. 이전45프레임 표현 정정
45는 주석에서 단어 수행 구간 밖에 연결된 프레임 수다. 실제 영상에서 모두 잘못 연결된 것으로 확인된 수가 아니다. 33은 경계0.1초이내,44는0.2초이내. 13연속구간의 처음/가운데/끝 장면을 보니 시작/마무리 동작이 주석 경계에 걸친 경우가 많다. 단순 손 미관측이 있는 것은1/45이며, 믿다 일부에서 참고한시점에 여러진행시점이붙는 문제가남는다. 새정렬시점정답주석/언어정답검증은아님.

2. 구현한 얼굴 관계 분석
손가락끝5/손목(손모델)/손바닥/손목(몸모델)/팔꿈치/팔구간최근접점의10종근거를각손별로6얼굴부위와연결했다. 손위치와팔추정근거를합치지않고출처를구분한다. 가린위치도측정추정값은남기되직접비교가능값과분리한다. 직접투영겹침은2D근거일뿐실제접촉·가림원인확정이아니다.
각사람을제외한나머지2명을각각참고로사용했다(18개비교대상,36방향쌍). 두참고가함께보이는겹침/참고끼리다른겹침/둘과다른겹침/근거부족을분리했다. 두참고와다른경우도오답판정이아닌확인대상이다. 두사람의최솟값~최댓값은관측참고구간이지정답허용범위가아니다.
정상공개동작에서도가림투영이서로다르다. 그러므로정답참고에없는얼굴가림을곧바로사용자의잘못으로채점하면안된다. 시점불일치,카메라각도,수행자차이,검출오차를먼저분리해야한다. 양쪽참고와같아도가려진손가락모양은검증된것이아니다.

3. 시간정렬보완후보는미채택
몸모델팔꿈치/손목의관측된이동방향만기존비용에0.25추가했다. 전체프레임유지,새시작끝자르기없음. 36방향쌍의같은{B['required_frames']}주석동작프레임에서불일치{B['annotation_discordant']}→{D['annotation_discordant']},누락0. 시간대응을아는54통제시험평균오차{B['mean_known_mae_ms']:.2f}→{D['mean_known_mae_ms']:.2f}ms,25시험소폭악화/1개선/28동일. 개선입증부족으로기존B정렬유지. 후보가중치재탐색안함. 시간변형은관절재표본화시험이고실제빠른RGB촬영시험아님.

4. 검증과남은범위
75출처핀/32운영핀유지. 18영상관측보존/손누락때팔근거분리/얼굴없으면근거생성금지/미관측간격넘는이동방향금지,64개겹침판정조합,72교차쌍지표와108통제지표별도검산PASS. 같은Codex검산이며독립전문가정답아님.
기존C·T3F와운영웹DTW미변경,사용자·멘토자료접근/학습/튜닝/선택0. 이번단계는연구용측정기능과원인분리개선이며TCN인식률개선이아님. 자동교정문장/맞다틀리다채점은아직연결하지않음.
다음: 주석끝점맞추기경쟁을중단하고,관측가능한실제단계(손접근/방향전환/거리최소점등)의대응불확실성을표시하는작은공개기준을정한다. 새검출모델추가나무작정가중치탐색없이현재얼굴관계·관측상태를진단웹후보에연결할범위를정리한다. 정렬이불확실하면얼굴관계차이를수행오류로확정하지않는다. 표정과TCN재학습은이번범위밖.
'''
(U/'RESULT_KO.txt').write_text(summary,encoding='utf-8')
cases_html=''
for case in cases:
 cols=[]
 for k in (1,5,6,7,8,9):
  values=[m['distance'][k] for m in case['measurements']]
  cols.append('<tr><td>'+endpoint[k]+'</td>'+''.join('<td>'+('미관측' if x is None else f'{x:.2f}')+'</td>' for x in values)+'</tr>')
 cases_html+=f'<section><h2>{case["label"]}: {descriptions[case["status"]]}</h2><img src="{case["image"]}"><details><summary>이 장면의 위치 측정 근거</summary><p>{reg[case["region"]]} 중심과의 상대 거리입니다. 얼굴 크기로 정리한 거리이며, 퍼센트·정답 점수가 아닙니다. 다른 수행자의 위치가 조금 다르다는 이유로 오류라고 채점하지 않습니다.</p><table><tr><th>측정점</th><th>비교 영상</th><th>참고 1</th><th>참고 2</th></tr>'+''.join(cols)+'</table></details></section>'
audit_html=''.join(f'<details><summary>{g["id"]} {g["label"]} · {len(g["rows"])}프레임</summary><p>{review[g["id"]]}</p><img loading="lazy" src="{g["image"]}"></details>' for g in A['groups'])
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손·팔과 얼굴 관계 검증</title><style>*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#18313e;font:17px/1.75 'Malgun Gothic',sans-serif}main{max-width:1180px;margin:auto;padding:36px 22px}header{background:#103745;color:white;border-radius:18px;padding:32px}h1{font-size:33px;line-height:1.4}h2{font-size:24px}section{background:white;border:1px solid #dce6ea;border-radius:16px;margin:24px 0;padding:26px}.tag{color:#8bddd4}img{display:block;max-width:100%;height:auto;border-radius:12px}details{padding:16px;border:1px solid #d8e5e8;border-radius:10px;margin:14px 0}summary{cursor:pointer;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.box{background:#edf6f5;padding:20px;border-radius:12px}.big{font-size:28px;font-weight:bold;color:#127e85}.warn{padding:18px;background:#fff4df;border-left:5px solid #dca13d}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border-bottom:1px solid #dce7ea;text-align:left}a{color:#067c86}pre{white-space:pre-wrap;font:14px/1.8 'Malgun Gothic';overflow-wrap:anywhere}@media(max-width:700px){main{padding:15px 10px}section,header{padding:18px}h1{font-size:26px}.grid{grid-template-columns:1fr}td,th{padding:5px;font-size:12px}}</style><main><header><div class="tag">WordsInSilence · 공개 자료 검증 · 표정 제외</div><h1>얼굴을 가렸는가보다,<br>어떤 부위 관계로 움직였는가</h1><p>손가락·손목·팔의 위치와 참고 동작의 관계를 분리해서 확인했습니다.</p></header>
<section><h2>이번에 연결한 기능</h2><div class="grid"><div class="box"><div class="big">10종 위치 근거</div>손끝·손목·손바닥·팔꿈치·팔 구간을 따로 측정</div><div class="box"><div class="big">다른 2명과 비교</div>자기 자신은 참고에서 제외, 각 수행을 따로 유지</div><div class="box"><div class="big">가림 ≠ 오답</div>참고에서 지지 / 참고끼리 차이 / 다른 관계 / 근거 부족</div></div><p>얼굴 표정은 사용하지 않았습니다. 얼굴 기준점은 손과 팔이 이마·광대 부근·입·턱·코 주변에서 어떻게 움직이는지를 파악하는 용도로만 사용합니다.</p><div class="warn">참고와 다른 가림이 보여도 바로 잘못된 수어라고 말하지 않습니다. 시간 정렬·시점·수행 차이·검출 오차가 원인일 수 있기 때문입니다. 실제 위치 차이가 충분히 확인된 뒤에 교정 근거로 삼아야 합니다.</div></section>
<section><h2>먼저 바로잡은 해석</h2><p>이전에 남았다고 말한 <b>45프레임은 모두 확인된 정렬 오류가 아닙니다.</b> 단어 주석 구간 밖에 연결된 프레임 수였습니다. 33개는 경계에서 0.1초 이내였고, 실제 장면에서는 양쪽 모두 손을 올리거나 내리는 비슷한 동작이 많이 보였습니다.</p><p>참고의 한 시점에 비교 영상의 진행 중인 여러 동작을 붙잡아 두는 사례도 남았습니다. 이 문제와 주석 경계 차이는 구분해서 다룹니다.</p></section>'''+cases_html+f'''<section><h2>이동 방향을 더한 정렬 후보: 유지하지 않음</h2><table><tr><th>같은 평가 범위</th><th>기존 B</th><th>방향 보조 D</th></tr><tr><td>다른 사람 36방향쌍, {B['required_frames']}동작 프레임<br>주석 구간 불일치</td><td>{B['annotation_discordant']}</td><td>{D['annotation_discordant']}</td></tr><tr><td>정확한 시간 대응을 아는 54시험<br>영상별 평균 오차의 평균</td><td>{B['mean_known_mae_ms']:.2f}ms</td><td>{D['mean_known_mae_ms']:.2f}ms</td></tr></table><p>방향을 더하면 무조건 좋아지는 것은 아니었습니다. 주석 지표는 조금 줄었지만 통제시험 25개에서 오차가 소폭 늘어, 충분한 개선 근거 없이 교체하지 않았습니다. 프레임을 추가로 자르거나 잘 안 되는 영상을 제외하지 않았습니다.</p><p>이 수치는 TCN 인식률도, 수어 수행의 정답률도 아닙니다. 기존 C·T3-F 및 운영 웹은 유지했습니다.</p></section>
<section><h2>45프레임의 실제 장면 확인</h2><p>연속된 13구간의 처음·가운데·끝을 표시했습니다. 화면 확대는 보기 위한 것이며 원본 관절 검출 범위는 바꾸지 않았습니다.</p>'''+audit_html+'''</section><section><h2>현재 상태와 다음 방향</h2><p><b>완료:</b> 손·팔–얼굴 관계 측정, 다른 두 참고와 비교, 관측 부족과 투영 겹침을 분리하는 연구 기능. 원본 관측과 운영 모델 보존, 공개 자료 검산 완료.</p><p><b>남은 문제:</b> 가린 손가락의 실제 모양, 정확한 접촉, 사람 간 세부 동작 시점, 허용 가능한 수행 차이는 아직 확정할 수 없습니다. 두 참고의 최솟값~최댓값을 정답 범위로 사용하지 않습니다.</p><p><b>다음:</b> 주석 경계 점수를 계속 낮추기보다, 실제 손 접근·방향 전환 등의 대응이 확실한지와 불확실한지를 표시하는 데 집중합니다. 정렬이 불확실한 시간의 얼굴 관계 차이를 곧바로 교정 오류로 바꾸지 않습니다.</p><p>새 검출 모델·다운로드·학습·사용자 영상 접근 없이 공개 18영상을 재사용했습니다. 표정은 범위에서 제외했습니다. 연구 기능은 별도 검증 화면에 있으며 기존 실사용 웹을 교체하지 않았습니다.</p><details><summary>검증 기록</summary><pre>'''+html.escape(summary)+'''</pre><a href="VERIFIED.json">별도 검산</a> · <a href="RESULT.json">결과 수치</a> · <a href="RESULT_KO.txt">설명 원문</a></details></section></main></html>'''
(U/'RESULT_KO.html').write_text(doc,encoding='utf-8');print('REPORT',len(cases),'cases',len(A['groups']),'audit groups')
