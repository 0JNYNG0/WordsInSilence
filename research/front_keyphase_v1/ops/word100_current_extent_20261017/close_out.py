import json,hashlib,re,html,shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (U/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(n,x):
 with (U/n).open('x',encoding='utf-8') as f:f.write(x)
s=read(U/'SCOPE.json');v=read(U/'INDEPENDENT_CHECK.json');o=read(U/'OUTPUT.json');audit=read(U/'ASSERTION_REVIEW.json')
assert v['status']=='PASS' and len(audit['items'])==68 and all(x['cue_consistent'] for x in audit['items'])
for p,h in {**s['source_pins'],**s['protected_pins'],**s['dependencies']}.items():assert sha(p)==h,p
at=datetime.now().astimezone().isoformat(timespec='seconds')
next_action='사용자와 현재 Pose 손끝 영역 진단 결과 확인. 기존 경계2사례 단서 보완·겹침29중22교차 단서, 새 경계누락1/겹침누락7과 원인미확정은 남음. 진단 기록 전용이며 자동 좌표복구·프레임제외·운영연결 없음. 학습은 결과 함께 확인한 다음 별도 입력계약/비교조건을 확정한다.'
stats=Counter()
for x in o['comparisons']:
 h=x['candidate'];pos=h['current_overlap']['status']=='current_2d_intersection_suspected'
 if x['visual_category']=='inside_overlap':
  stats['visual_overlap']+=1;stats['current_intersection']+=pos;stats['previous_proximity']+=h['overlap']['proximity_hint']
  stats['previous_area']+=h['overlap']['area_evidence']=='area_supported_suspicion'
  stats['new_only_vs_proximity']+=pos and not h['overlap']['proximity_hint']
  stats['old_only_proximity_preserved']+=h['overlap']['proximity_hint'] and not pos
result={'status':'CURRENT_EXTENT_CUES_VERIFIED_DIAGNOSTIC_ONLY_REVIEW_BEFORE_TRAINING','at':at,'unit':U.name,
 'implementation':'Codex local; independent scalar saved-output audit by same operator, no other-agent/expert review',
 'changes':['Add current Pose wrist/pinky/index/thumb extent with original confidence threshold .5','Use current extent intersection with actually observed other-hand region as separate 2D suspicion',
 'Preserve all previous position/shape/overlap evidence; no coordinates/masks/assignment changed; no claim of whole-hand visibility or causal occlusion'],
 'public_scope':{'videos':68,'word_types':len({r['word_code'] for r in s['records']}),'train_people':len({r['signer_id'] for r in s['records']}),
 'hand_states':17774,'previous_states_preserved':15926,'known_visual_samples':141,'new8_preoutput_visual_samples':26,'observed_controls_preserved':40},
 'known_boundary_cases_with_new_crossing_evidence':2,'overlap_cue_counts_same29_known_samples':stats,
 'new8_visual_results':v['tables']['new8_before_output'],
 'audit':{'flagged_overlap_intervals':14,'flagged_interior_wrist_boundary_intervals':31,'representative_frames_reviewed':68,'obvious_cue_contradictions_seen':0,
 'note':'Post-output start/middle/end review; not all112 positive hand-frames or a blind independent test.'},
 'remaining':['Current4-point pose extent is not exact hand outline. P015전근L90: actual fingertips reach lower boundary but Pose extent stays inside; confidence~.94 does not certify geometry.',
 '7/29 known visual overlap samples have no current polygon intersection. Three retain earlier proximity evidence; absence of intersection must not negate earlier evidence or prove no overlap.',
 'New8 longest-gap labels contain no overlap positives; new unseen-overlap generalization not established.',
 '2D contact/intersection does not establish detector failure cause, depth ordering, hand shape or recovery. Model accuracy gain unmeasured.'],
 'verification':{'synthetic_cases':16,'scalar_polygon_pairs':v['scalar_polygon_pairs'],'max_area_diff_px2':v['max_intersection_area_difference_pixels2'],
 'source_pins':len(s['source_pins']),'protected_pins':len(s['protected_pins'])},
 'new_training':0,'classifier_inference':0,'new_detection':0,'user_recording_reads':0,'production_changes':0,
 'running_processes':[],'next_single_action':next_action,
 'evidence':{n:str(U/n) for n in ['SCOPE.json','spatial.py','CONTRACT_TESTS.json','OUTPUT.json','VISUAL_LABELS.json','INDEPENDENT_CHECK.json','ASSERTION_REVIEW.json']}}
save('RESULT.json',result)
notes=[]
for x in o['comparisons']:
 if x['visual_category']=='inside_overlap' and x['candidate']['current_overlap']['status']!='current_2d_intersection_suspected':
  notes.append({'review_id':x['review_id'],'clip_id':x['clip_id'],'frame':x['frame'],'side':x['side'],
   'finding':'RGB shows contact/overlap at hand/wrist region, but coarse Pose4 region and detected other-hand hull do not have positive area intersection. This is insufficient extent/position evidence, not proof of no overlap.',
   'preserved_proximity_hint':x['candidate']['overlap']['proximity_hint'],'cause_confirmed':False})
save('MISS_ANALYSIS.json',{'overlap_misses':notes,'boundary_miss':{'review_id':'P015','clip_id':'NIA_SL_WORD2193_REAL07_F','frame':90,'side':0,'pose_max_y':804.7927856445312,'input_bottom':855,'min_pose_confidence':0.9424260258674622,'finding':'Visual fingertips cross/meet lower crop boundary; coarse pose extent under-represents true hand. No threshold tuning or fictitious expansion performed.'},'visual_files':['miss_01.jpg','miss_02.jpg']})

# Compact evidence montage from already-reviewed public panels.
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
examples=[('audit_08.jpg',0,'경계 사례 1: 손목은 안쪽, 손끝 추정은 경계 밖'),
          ('audit_09.jpg',4,'경계 사례 2: 화상 · 하단 걸침 단서 추가'),
          ('audit_01.jpg',3,'마음에 들다: 현재 손 영역의 2차원 교차 단서'),
          ('miss_02.jpg',1,'남은 한계: 전근 · 실제 손끝보다 작게 추정')]
tiles=[]
for name,index,title in examples:
 im=Image.open(U/name);tileh=im.height//(3 if name!='miss_02.jpg' else 1);x=(index%2)*700;y=(index//2)*tileh
 tile=im.crop((x,y,x+700,y+tileh));panel=Image.new('RGB',(700,tileh+45),'white');panel.paste(tile,(0,45));ImageDraw.Draw(panel).text((10,10),title,font=font,fill='#123b57');tiles.append(panel)
montage=Image.new('RGB',(1400,tiles[0].height*2),'#eef2f5')
for i,t in enumerate(tiles):montage.paste(t,((i%2)*700,(i//2)*t.height))
montage.save(U/'EXAMPLES.jpg',quality=95)

text='''손 전체 경계와 겹침 미검출 — 추가 보완 결과

결론
기존 위치·모양·겹침 근거는 그대로 두고, 현재 프레임의 몸 관절 검출에 포함된 손목·새끼·검지·엄지 위치(Pose4)로 거친 손 영역을 추가했다. 기존에 남아 있던 경계2사례 모두에서 화면 밖으로 이어지는 손끝 단서를 얻었다. 겹침으로 판독했던29개 기존 표본 중22개에서 현재 영역 교차 단서를 얻었다. 관절 검출/TCN 인식률 향상, 완전한 원인 판정 또는 손 모양 복원 결과는 아니다.

무엇을 바꿨나
기존: 손목 한 점의 위치와 과거 정밀 손 영역 추정을 사용. 손목은 안쪽이지만 손끝은 화면 밖인 경우를 놓쳤고, 과거 모양의 불확실성이 빠르게 커졌다.
이번: 현재 Pose의 손목+손끝3점으로 별도 영역을 만들었다. 경계 바깥 점이 있으면 ‘손 영역 경계 걸침 단서’를 남긴다. 영역이 안쪽이고 반대손이 실제 관측되어 있으면 두 영역이 교차하는지 기록한다. 부정확할 수 있는 거친 추정이며 실제 Hand21 관측으로 바꾸지 않는다.
화면 밖 좌표는 현재 잘린 영상을 본 검출기의 외삽이다. 원본 화면 밖 RGB·미래 프레임을 알고리즘에 넣지 않았다. 원본 전체 RGB는 시각 판독용으로만 사용했다.
몸 관절의 신뢰값0.5는 기존 기준 그대로다. 교차 면적의 극소값은 수치 오차 기준이며 점수에 맞춰 조정하지 않았다. 단어·사람 이름에 따른 규칙은 없다.

검증 자료와 기존 보존
AI Hub 공개 TRAIN68영상,36표현,9명,17774손-프레임. 기존60영상의15926개 위치·모양·겹침 기록을 모두 그대로 보존했다. 기존에 위치 판독이 맞았던101개 표본도 유지된다.
이전141개 시각 표본과 새8영상26개 표본을 비교했다. 새26은 후보 출력 전 시각 판독했다. 정상 대조40개(이전32+새8)는 모두 유지했다. 자료는 반복 개발과 진단용이며 새 사용자 정확도가 아니다.

경계 결과
유성구주민지원센터 TRAIN03 왼손70프레임과 화상 TRAIN04 왼손136프레임: 손목은 안쪽이지만 손끝 추정 영역은 화면 밖. 두 사례 모두 추가 경계 근거가 생겼다.
기존 경계 표본63개에서62개 경계 단서/1개 보류. 새 경계 표본15개는13개 경계 단서/1개 보류/1개 놓침. 새 내부 표본3개에서는 잘못된 경계 단서가 없었다.
새로 놓친 전근 TRAIN07 왼손90프레임은 실제 손끝이 하단에 걸리지만 Pose 손끝 최대Y804.8px,화면하단855px로 추정된다. 최소 신뢰값도 약0.94라, 신뢰값이 높다고 정확한 손 외곽이라는 뜻은 아니다. 특정 단어에 맞춰 영역을 늘리지 않았다.

겹침 결과
기존에 시각적으로 겹침/접촉을 확인한29개 표본 중 새 교차 단서는22개,단서 없음7개. 14개는 마음에 들다 한 영상의 연속 프레임이므로29개 독립 동작으로 보면 안 된다.
같은29개에서 기존 근접 단서8개,과거 영역의 강한 교차 근거1개였다. 새 단서는 이전 근접 단서와 다른17개를 추가로 포착했다. 이전 근접 단서가 있던3개는 이번 교차 근거가 없지만,기존 단서는 삭제하지 않고 남겼다. 하나의 통합 원인 판정으로 바꾸지 않는다.
이전 공항시장N023의 잘못된 겹침 의심에는 이번 교차 단서가 붙지 않았다. 이는 이전 근접 단서를 삭제/정답으로 고쳤다는 의미는 아니다.
남은7개는 RGB의 손/손목 접촉 부위와 거친 Pose4/다른 Hand 영역의 위치·범위가 어긋나거나 맞닿음만 나타나 양의 교차 면적이 없었다. 교차 없음은 겹침 없음이라는 뜻이 아니다. 손-팔 가림이나 얼굴 가림도 이 규칙으로 모두 해결하지 못한다.
새8영상의 최장 누락 표본에는 겹침 양성이 없었으므로 새로운 겹침 영상에서의 일반화 성능은 아직 확인하지 못했다.

추가 시각 감사와 수치 검산
전체 출력에서 새 겹침 단서78손-프레임(14연속구간),손목안쪽+새 경계 단서34손-프레임(31구간)을 찾았다. 모든 구간의 시작/중간/끝68개 장면을 별도로 검토했고 명확한 단서 오탐은 발견하지 못했다. 모든112프레임을 판독한 것이 아니며 출력 후 검토다.
독립 수치 코드로17774손상태의 조건과 이전15926기록 보존을 검산했다. OpenCV를 쓰지 않은 별도 다각형 계산418쌍과 일치했고 최대 면적차이는0.00212px²미만이다. 합성·변형 검사16개 통과. 구현과 별도 검산의 담당자는 같은 Codex이며 전문가/다른 에이전트 검토가 아니다.

적용 범위와 다음 단계
현재는 진단 부품과 공개 검증 결과만 추가했다. 웹·TCN·DTW·관절 좌표·좌우 배정·관측 마스크는 변경하지 않았다. 네 촬영/멘토 촬영 읽기·학습·튜닝0,새 모델 학습0,TCN 추론0,새 MediaPipe 검출0.
손이 없어서 검출되지 않았는지,손이 있는데 겹친 것인지 구분할 단서는 늘었다. 하지만 손가락을 새로 찾아낸 것은 아니고 모델이 이 정보를 사용하도록 학습한 것도 아니다.
결과를 함께 확인한 후 이 진단 정보를 어떻게 입력에 연결할지 비교 조건부터 정한다. 놓친 손을 임의로 채우거나 정상 프레임을 자동 제외하지 않는다. 기존C/M0와 운영 웹은 유지하며 학습은 사용자와 결과 검토 후 진행한다.
'''
write('RESULT_KO.txt',text)
sections=text.strip().split('\n\n')
body=''.join('<section><h2>'+html.escape(x.split('\n')[0])+'</h2><p>'+html.escape('\n'.join(x.split('\n')[1:])).replace('\n','<br>')+'</p></section>' for x in sections[1:])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 경계·겹침 보완 검증</title><style>
body{margin:0;background:#edf2f5;color:#193344;font-family:Malgun Gothic,Arial,sans-serif;line-height:1.75}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:32px;line-height:1.4}h2{font-size:22px;margin:0 0 12px}section{background:white;border-radius:16px;padding:26px;margin:20px 0}p{margin:0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards div{background:#143e54;color:white;padding:20px;border-radius:14px}.cards b{display:block;font-size:30px}.cards span{font-size:14px}img{width:100%;height:auto;border-radius:12px}a{color:#006a95}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}@media(max-width:650px){.cards{grid-template-columns:1fr 1fr}main{padding:20px 12px}h1{font-size:26px}}
</style><main><h1>손 전체 경계와 겹침 미검출<br>현재 손끝 위치를 활용한 추가 보완</h1><p>공개 자료의 진단 근거 개선 · 사용자 자료 미사용 · 학습 전 결과 검토</p><div class="cards"><div><b>68개</b><span>공개 TRAIN 영상</span></div><div><b>2 / 2</b><span>기존 경계 사례에 단서 추가</span></div><div><b>22 / 29</b><span>기존 겹침 판독 표본의 새 교차 단서</span></div><div><b>0회</b><span>새 학습·사용자 자료 사용</span></div></div><section><h2>실제 장면에서 확인</h2><p>노랑: 입력 화면 하단 · 초록: 현재 Pose의 거친 손 영역 · 주황: 반대손의 실제 관측 영역<br>화면 아래 원본 영상은 검토용이다. 알고리즘이 화면 밖 영상을 본 것은 아니다.</p><img src="EXAMPLES.jpg" alt="두 경계 사례, 겹침 단서, 남은 경계 누락 사례"></section>'''+body+'''<details><summary>공개 자료 시각 검토와 독립 검산 근거</summary><p><a href="INDEPENDENT_CHECK.json">수치 검산</a> · <a href="RESULT.json">결과 요약</a> · <a href="MISS_ANALYSIS.json">남은 문제 분석</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p>'''+''.join(f'<p><a href="audit_{i:02}.jpg">추가 단서 검토 {i}</a></p>' for i in range(1,13))+'''<p><a href="miss_01.jpg">놓친 겹침 장면</a> · <a href="miss_02.jpg">남은 겹침·경계 사례</a></p></details></main></html>'''
write('RESULT_KO.html',page)

statepath=O/'STATE.json';state=read(statepath)
assert state['current_stage']=='WORD100_HAND_EVIDENCE_SPLIT_VERIFIED_DIAGNOSTIC_ONLY_REVIEW_BEFORE_TRAINING'
backup=U/'before';backup.mkdir(exist_ok=True)
for p in [statepath,R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']:
 assert not (backup/p.name).exists();shutil.copy2(p,backup/p.name)
state['current_stage']='WORD100_CURRENT_EXTENT_CUES_VERIFIED_DIAGNOSTIC_ONLY_REVIEW_BEFORE_TRAINING';state['next_single_action']=next_action
state[U.name]={**result,'report':str(U/'RESULT_KO.html'),'final_check':str(U/'FINAL_CHECK.json')}
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
summary=f'''### 2026-10-17 現フレーム 손끝 영역 추가 진단 완료 — 경계2단서 보완·겹침 일부 포착, 복원/학습 없음
최신 승인: 남은 손전체경계2사례·겹침 미검출 공통보완/공개검증, 사용자자료미사용·학습결과검토후. ops/{U.name} 완료. 현재 Pose손목/새끼/검지/엄지4점 범위와 실제반대손 Hand영역 교차를 별도기록. 기존위치/모양/겹침15926기록 모두보존,좌표/마스크/좌우배정변경없음. 기존confidence .5유지·단어예외/사후계수탐색없음. 화면밖Pose는현재입력검출기의외삽이며숨김RGB/미래미사용.
공개TRAIN68영상36표현9명17774손상태. 이전141시각판독+새8영상26출력전판독(미검출18/정상8). 기존정상32+신규8=40보존. 기존맞던위치101유지. R016센터L70/N001화상L136모두새경계단서. 기존경계63=62단서/1보류;새경계15=13단서/1보류/1누락(P015전근07L90). 새내부3오경계0. 손전체정확도보장아님.
기존겹침29중현재교차22(그중마음06연속14),놓침7. 기존근접8중현재교차5/기존단서만3도보존;새근거17추가,원인확정없음. 기존과거영역지지1과현재교차는서로다른근거. N023공항시장오의심에새교차없음. 새8최장누락엔겹침양성없어새겹침일반화미검증. 새경계/겹침45구간시작중간끝68시각감사명백오탐못찾음(전체112프레임전수판독아님).
별도scalar검산17774/이전15926보존PASS,다각형418쌍최대차.00212px²미만,합성16PASS. 자료272핀/보호10핀유지. 동일Codex별도코드검산,다른에이전트/전문가아님. 새학습/TCN추론/검출/사용자읽기/운영변경0·진행중없음. 근거 SCOPE/OUTPUT/INDEPENDENT_CHECK/ASSERTION_REVIEW/MISS_ANALYSIS/RESULT_KO.html/FINAL_CHECK.
거친Pose영역이진짜손윤곽을덜담는한계와손/손목접촉·가림원인미확정잔존. 코드/진단완료를TCN성능향상으로표기하지않음. 사용자와결과확인후학습입력계약/비교계획확정,자동학습없음.

'''.replace('現フレーム','현재 프레임')
project=R/'WordsInSilence_PROJECT_CONTEXT.md';t=project.read_text('utf-8')
for sec,insert in [(8,summary),(9,f'### 2026-10-17 현재 손끝 영역 진단 결과 검토 — 학습 미실행\n{next_action}\n\n')]:
 pattern=rf'(^## {sec}\. [^\n]*\n)';assert len(re.findall(pattern,t,re.M))==1
 t=re.sub(pattern,lambda m:m.group(0)+'\n'+insert,t,count=1,flags=re.M)
project.write_text(t,'utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

## 목표·마지막 완료
사용자 승인: 남은 손전체경계2사례와 겹침 미검출 공통보완·공개검증. 사용자자료학습/튜닝금지, 학습은 결과 함께 확인한 다음.
ops/{U.name} 구현/검산/시각감사완료. 진단기록전용·운영미연결. RESULT_KO.html/RESULT.json/FINAL_CHECK 재사용.

## 결과
현재 Pose손목/새끼/검지/엄지로 거친 손영역, 현재 실제관측 반대손 영역과 교차 단서 추가. 기존 분리위치/모양/겹침15926기록 전부보존. 단어예외·계수사후탐색·가상관절·마스크변경0.
공개TRAIN68영상36표현9명17774손상태. 이전141시각판독+새8영상26출력전판독. 정상대조40보존,기존맞던위치101보존.
R016센터L70/N001화상L136두경계사례 새걸침단서. 기존경계63=62단서/1보류,새경계15=13단서/1보류/1누락(P015전근L90). 신뢰값.94도정확한손외곽보장못함.
기존겹침29중22새교차/7놓침;14는마음06연속프레임. 기존근접8(새교차5/기존근접만3보존),현재교차17추가. 겹침원인확정아님. 새8최장누락엔겹침양성없어새겹침일반화미검증.
추가양성45구간시작중간끝68장면명백단서오탐못찾음;전체112프레임판독아님. 수치별도검산17774/이전15926보존,다각형418쌍오차.00212px²미만,합성16PASS. 272자료핀/10보호핀유지. 담당동일Codex별도코드,타에이전트/전문가아님.

## 남은 문제·금지
손끝4점은정밀Hand21이아님. 추가경계1놓침과겹침7놓침,손-팔/얼굴가림원인미확정. 현재출력이손전체안쪽/겹침원인을증명하지않음. 과거관절복원·실제센서미검출해결·TCN인식률향상주장금지.
사용자/멘토자료읽기0·학습0·TCN추론0·새검출0·운영변경0. 새8공개원본ZIP로컬발췌만,다운로드/카메라/외부전송/Git없음. 실행모두종료0·진행중없음.

## 다음 한 작업
{next_action}

## 기존 모델·웹 보존
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본M0. 이번 서버점검/재시작 없음.
C SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab; M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f 보존. 필요손DTW유지. st-gcn자동화삭제유지.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
for n in ['RESULT.json','SCOPE.json','CONTRACT_TESTS.json','INDEPENDENT_CHECK.json','ASSERTION_REVIEW.json','MISS_ANALYSIS.json']:read(U/n)
assert read(statepath)[U.name]['public_scope']['hand_states']==17774
for p,h in {**s['source_pins'],**s['protected_pins'],**s['dependencies']}.items():assert sha(p)==h,p
for name in re.findall(r'(?:src|href)="([^"]+)"',page):assert (U/name).exists(),name
save('FINAL_CHECK.json',{'status':'PASS','at':at,'source_pins_unchanged':len(s['source_pins']),'protected_pins_unchanged':len(s['protected_pins']),
 'root_docs_synchronized':True,'report_links_exist':True,'running_processes':[],'outputs':{p.name:sha(p) for p in U.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.html')}})
print(json.dumps({'status':'PASS','result':result['status'],'report':str(U/'RESULT_KO.html'),'scope':result['public_scope']},ensure_ascii=False,indent=2))
