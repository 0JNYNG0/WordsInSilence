from pathlib import Path
from datetime import datetime
import json,hashlib,collections,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111';T=U/'transition_shape';D=T/'parallel'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
checks=[]
def check(n,b):checks.append(dict(name=n,passed=bool(b)))
par=read(D/'EVALUATION.json');pl=read(T/'LABELS.json')['rows'];cfg=read(D/'FROZEN.json');pp=np.load(D/'PROBABILITIES.npz')
for p,h in cfg['pins'].items():check('parallel_frozen',sha(Path(p))==h)
for tr in read(D/'TRAINING_COMPLETE.json')['rows']:
 check('parallel_weights',tr['weights_changed']);check('parallel_checkpoint_sha',sha(Path(tr['path']))==tr['sha256'])
for tr in read(D/'INPUTS.json')['rows']:
 check('parallel_FIT_only',pl[tr['index']]['record_id']in cfg['fit_records'])
for row in par['rows']:
 r=pl[row['index']];p=pp[r['clip_id']+'_'+r['hand']][:,r['array_index']];votes=p.argmax(1)
 available=par['dense'][r['clip_id']+'_'+r['hand']]['available'][r['array_index']]
 good=available and len(set(votes.tolist()))==1 and float(p.max(1).min())>=.8
 check('parallel_saved_probability',row['middle_state']==(int(votes[0])if good else None))
probe=[r for r in par['rows']if r['split']=='exposed_probe'and r['label']is not None]
mid=[r for r in probe if r['label']=='transition'];end=[r for r in probe if r['label']!='transition']
check('parallel_arithmetic',sum(r['middle_state']==1 for r in mid)==par['summary']['middle_detected']and sum(r['middle_state']==1 for r in end)==par['summary']['false_middle'])
check('parallel_not_adopted',not par['adoption_passed'])
assert all(c['passed']for c in checks)
save(D/'VERIFIED.json',dict(passed=True,total=len(checks),checks=checks,independent_human=False,model_adopted=False))
assert read(U/'VERIFIED.json')['passed']and read(T/'VERIFIED.json')['passed']
now=datetime.now().astimezone().isoformat();base=read(U/'EVALUATION.json');dec=read(U/'DECOMPOSITION.json');ft=read(T/'VERIFIED.json')
next_ko='기존 끝 손모양 판정부와 공개100 모델을 고정한다. 공개 FIT의 다른 표현에서 펼침→굽히는 중→접힘 정답을 추가하되, 먼저 틀린 관절 관측인지 모델의 투영/중간상태 혼동인지 RGB-관절 대응으로 분리한다. 같은 단어의 임계값 조정이나 유지시간 늘리기는 하지 않는다. 별도 진행 상태 판정부에서 정상 끝 모양 회귀와 실제 원본 중간 잘림을 함께 비교하고, 설명-행동 연결 및 관측 부족을 포함한 G2/G3 전체 통과 전 웹/봉인8 승격하지 않는다. 이번 9학습과 완료 비교는 재실행하지 않는다.'
summary='손 닫힘·신체 관계·팔 자세의 직접 정답375항목을 준비하고 3seed 학습. 손모양 관측30/31, 영역23/27, 팔자세3/5. 손모양만 분리하면 공개2영상의 정상/짧은 끝/속도변형8조건 유지, 고정시작·끝자세만·역순·가림 구분. 추가 중간모양 학습은 전환4/4이나 기존정상30→26/31·짧은끝2→1/2 회귀로 미채택. 별도 시계열 보조판정부도 전환1/4·정상32중오경고1로 미채택. 기존 C·T3-F/공개100/웹 유지, 전체 완료 판정은 미완료.'
report=f'''짧게 수행한 동작과 중간에 멈춘 동작 구분 — 2026-11-14

이번 작업의 결론
시간이 짧다는 이유로 미완료라고 하지 않고, 실제로 요구한 모습에 도달했는지 따로 확인하는 부품을 만들고 학습·비교했다. 손모양에 한정된 구분은 확인했지만, 모든 동작의 완료 판정을 해결한 상태는 아니다. 새 후보가 정상 동작을 놓치는 회귀가 있어 웹 모델은 바꾸지 않았다.

무엇을 했나
1. 공개 자료 32영상(4표현 × 기록ID 02~09)에서 손의 펼침/접힘, 얼굴 앞/상체 관계, 팔 자세를 따로 주석으로 정리했다. 정답375항목은 영상375개가 아니다. 같은 프레임에 여러 특징을 붙인 항목 수다. 애매한 부분은 정답을 강제로 만들지 않았다.
2. 학습에는 기록02~07, 비교에는 이미 공개 연구에 노출된08~09를 사용했다. 최종 봉인8과 개인 촬영은 열지 않았다. 사람의 실제 수어 오답 자료가 아니라 정상 공개 영상과 이를 잘라/가려/역순으로 만든 제어 시험이다.
3. 새 작은 관측 판정부3개 seed를 학습했다. 위치·각도 대신 신체 기준의 상대적 관계와 네 손가락 특징을 입력했다. 기존 공개100 시간합성곱+GRU 및 C·T3-F는 그대로다.
4. 실제로 손을 굽히는 도중의 이미지24개도 추가 확인했다. 애매하거나 기존 정답과 중복인 것은 제외했다. 같은 신경망의 미세조정3회와 기존 손모양 판정부를 고정한 별도 시계열 보조판정부3회를 비교했다. 총9회이며, 동일 결과를 재학습해 좋은 점수를 고른 것이 아니다.

이전과 달라진 점
- 손이 움직인 거리 하나로 전체 완료를 대신하지 않는다. 손을 접는 변화가 보이는지, 손 위치가 확인되는지를 분리한다.
- 비교용2영상에서 손을 접은 직후까지만 남긴 짧은 구간도 두 경우 모두 손모양 변화를 확인했다. 정상2 + 짧은 끝2 + 속도변형4 = 8조건이다. 영상8개도 전체 정확도100%도 아니다.
- 처음 자세만 유지한 경우, 끝 자세만 보여준 경우, 역순, 손이 안 보이는 경우를 같은 실패로 합치지 않는다.
- 관측한 부분은 ‘손을 펴다가 접는 변화가 확인됐어요’, 가려진 부분은 ‘손모양이 충분히 보이지 않아 판단을 보류했어요’처럼 설명할 수 있게 연구용 문구를 연결했다. 단어 전체를 올바르게 했다는 문구는 내보내지 않는다.

학습 비교에서 확인한 한계
- 최초 새 판정부: 비교용 손모양 정답31항목 중30, 신체 영역27중23, 팔 자세5중3(나머지2보류). 위치와 팔 자세를 모두 요구하면 정상도 놓쳤다.
- 접는 중간 모습을 추가한 미세조정: 중간 상태4항목 모두 맞았지만 기존 손모양 정답은30/31에서26/31로 떨어졌다. 짧게 다 접은2경우 중1도 놓쳤다. 따라서 교체하지 않았다.
- 기존 손모양 판정부를 고정하고 앞뒤 관측 차이를 보는 별도 보조모델: 중간 상태4중1, 정상 끝 상태32개 중 중간이라고 잘못 본1개가 남았다. 원래 판정부는 보존했으나 이 보조모델도 통과하지 못했다.
- 실제 원본의 중간에서 잘라낸5개 구간에는 잘못된 ‘접힘 변화 완료’가 나오지 않았다. 이는 제한된 공개 편집 시험이며, 실제 사용자 미완료 판정 정확도가 아니다.
- 특히 비스듬한 손/팔과 얼굴·어깨 경계의 정상 변이가 어렵다. 입력 관절의 오류인지 학습된 구분의 부족인지 더 분리해야 한다. 데이터 수만 늘리면 해결된다고 단정할 근거는 없다.

검증과 보존
별도 검산: 첫 단계1563항목, 미세조정340항목, 별도 보조모델{len(checks)}항목 통과. 구현·수치·입력 분리 확인이며 성능 통과 수치가 아니다. 같은 Codex의 별도 실행이며 독립 수어 전문가 검토가 아니다.
이전246조건 결과의 원래 필드가 추가 진단으로 바뀌지 않음도 확인했다. 이는 인터페이스 회귀 검사로, 새 모델을246조건에 재추론해 성능을 검증한 것은 아니다.
JSON 저장 오류는 최초 파일을 보존하고 저장된 확률에서 복구했다. 첫 추가주석 준비 오류도 보존했다. 완료 학습과 추론을 좋은 점수를 위해 반복하지 않았다.
기존 C·T3-F, 공개100 모델, 웹, 원본, 사용자 촬영은 변경/학습하지 않았다. 서버/카메라/다운로드/Git 작업 없음. 처리시간 개선 실험은 하지 않았으므로 속도 개선 수치도 주장하지 않는다.

다음 한 작업
{next_ko}

전체 그래프 상태
관측/후보/기존 정상·오류 비교는 기록되어 있다. G2 구간 연결 REWORK, G3 전체 완료 판단 미통과, 이후 봉인 평가·웹 반영은 대기다. 이번 작업의 추가 실험은 종료했고 실행 중 학습은 없다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
evidence={}
for p in [U/'VERIFIED.json',T/'VERIFIED.json',D/'VERIFIED.json',U/'EVALUATION.json',U/'DECOMPOSITION.json',T/'EVALUATION.json',D/'EVALUATION.json']:
 evidence[str(p)]=sha(p)
write('RESULTS_REVIEW.json',dict(status='TERMINAL_TRAITS_AND_TRANSITION_TESTED_REWORK',at=now,summary=summary,
 evidence=evidence,new_training_runs=9,public_video_count=32,private_data_used=False,web_changed=False,final8_unopened=True,
 whole_graph_complete=False,implemented_by='Codex',verified_by='same Codex separate scripts',
 improvements='Disentangled shape observations from unqualified body/pose completion; verified short source shapes and uncertainty. No broad accuracy improvement claim.',
 regression='Third-class fine-tuning regresses old endpoints. Independent temporal head also fails admission.',next=next_ko))
nodes=[dict(id='A',text='공개 원본 확인\n손모양·위치·팔을 따로 표시',status='완료'),
dict(id='B',text='짧은 동작·중간 잘림\n관측 부족을 나눠 비교',status='완료'),
dict(id='C',text='학습 후 정상 동작 유지 확인\n중간 상태 구분은 회귀 발생',status='검증 완료 · 미통과'),
dict(id='D',text='다른 표현의 중간 모습 보강\n관절 오류와 학습 혼동 분리',status='다음 보완'),
dict(id='E',text='구간 연결 + 전체 완료 검증\n통과한 뒤 웹 적용',status='남음')]
write('GRAPH.json',dict(nodes=nodes,edges=[['A','B'],['B','C'],['C','D'],['D','B'],['C','E']],whole_graph_complete=False))
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(12,5.4),facecolor='white');ax.set_axis_off();ax.set_xlim(0,12);ax.set_ylim(0,5.4)
ax.text(.35,5.05,'짧게 수행한 동작과 중간에 멈춘 동작',fontsize=21,color='#17283f',weight='bold')
xs=[.35,4.35,8.35];ys=[3.0,3.0,3.0]
for i,n in enumerate(nodes):
 x=xs[i] if i<3 else (2.3 if i==3 else 7.0);y=ys[i] if i<3 else .55
 ax.add_patch(plt.Rectangle((x,y),3.35,1.5,facecolor='#eef1f5',edgecolor='#243b55',linewidth=1.4))
 ax.text(x+.18,y+1.12,n['status'],fontsize=12,color='#243b55',weight='bold')
 ax.text(x+.18,y+.52,n['text'],fontsize=12,color='#17283f',va='center',linespacing=1.6)
for a,b in [((3.75,3.75),(4.25,3.75)),((7.75,3.75),(8.25,3.75)),((9.9,2.9),(4.0,2.15)),((5.7,1.3),(6.9,1.3))]:
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#596a7f',lw=1.6))
ax.text(.35,.15,'정상 동작을 놓친 후보는 웹에 적용하지 않음 · 전체 피드백 완료는 아직 검증 중',fontsize=11,color='#596a7f')
fig.savefig(U/'PROGRESS_GRAPH.png',dpi=160,bbox_inches='tight');plt.close(fig)
# Synchronize current records only after actual results and verification exist.
backup=U/'before_final_records';backup.mkdir(exist_ok=True)
paths=[O/'STATE.json',G/'TASK_GRAPH.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']
for p in paths:
 if not(backup/p.name).exists():shutil.copy2(p,backup/p.name)
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']=next_ko
s[U.name]=dict(status='TERMINAL_TRAITS_AND_TRANSITION_TESTED_REWORK',summary=summary,next=next_ko,unit=str(U),updated_at=now,
 training_running=False,evaluation_running=False,web_changed=False,whole_graph_complete=False,report=str(U/'RESULT_KO.txt'),graph=str(U/'PROGRESS_GRAPH.png'))
save(O/'STATE.json',s)
g=read(G/'TASK_GRAPH.json');g2=next(n for n in g['nodes']if n['id']=='G2');g2['status']='REWORK';g2['latest_followup']=str(U/'RESULT_KO.txt')
g2.setdefault('attempts',[]).append(dict(at=now,status='REWORK',evidence=[dict(path=str(U/'RESULTS_REVIEW.json'),sha256=sha(U/'RESULTS_REVIEW.json'))],
 note='Direct goal/transition targets trained and tested. Normal endpoint regression in replacement model; separate temporal helper also insufficient. No promotion from partial observation success.'))
g3=next(n for n in g['nodes']if n['id']=='G3');g3['terminal_state_followup']=dict(status='TESTED_REWORK',path=str(U/'RESULTS_REVIEW.json'))
save(G/'TASK_GRAPH.json',g)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');title='### 2026-11-14 짧은 수행·중간 손모양 분리 학습 및 비교 완료 / 회귀로 웹 유지'
assert title not in text
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_ko)]:
 marker=header+'\n\n';assert marker in text;text=text.replace(marker,marker+title+'\n\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `RESULTS_REVIEW.json`.\n\n',1)
p.write_text(text,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-14 짧은 수행·중간 손모양 비교 완료 / G2·G3 미완료

{summary}

현재 단위: {U}
실행 중 학습/평가 없음. 기존 웹/서버/카메라 변경 없음.

완료 결과 재학습·재추론 금지
- LABELS:375특징 항목, 공개32영상; FIT02~07 / 노출probe08~09. Final8 봉인.
- 최초3개seed 작은 상태판정부. EVALUATION/DECOMPOSITION: 손모양만 분리한 짧은/정상/속도8조건 성공, 전체 완료는 아님.
- transition_shape:24고정분율 RGB프레임 검토(1중복/애매함 제외). 추가3class 미세조정3seed. 중간4/4이나 원래 정상30→26/31, 짧은끝2→1/2. 미채택.
- transition_shape/parallel: 원래 endpoint 고정 + 앞뒤 geometry차분 36→16→2 별도학습3seed. 중간1/4, 정상32중오경고1. 미채택. 총9학습 완료.
- 최초JSON 저장실패 FIRST_SERIALIZATION_FAILURE.partial.json 보존, DENSE_PROBABILITIES로 복구. 해당 신경망 재추론 안함. 추가주석 첫 준비실패도 보존.
- VERIFIED:첫1563, 전환340, 별도{len(checks)}체크 통과. 같은Codex 별도검산이며 교육자/독립사람 정답 아님. 정상/오류 성능 합격과 구분.
- 기존 전단위 goal_binding_integration의 visibility/ownership/association 보호와 후보 전체구간 보존 유지. 그 unit의RESULT_KO와TARGET_REVIEW도 계속 유효.

다음 한 작업
{next_ko}

목표: 충분히 짧은 동작은 인정하되, 중간 형태만 하고 멈춤/손이 안 보임/다른 행동 연결을 서로 구분. 이번 결과만으로 실제 사용자 미완료 정확도나 전체 웹 완성을 주장하지 않는다.
공개100/C·T3-F/운영웹/원본 보존. 사용자·멘토 자료 접근/학습/튜닝/선택 금지. 기록ID를 외형으로 사람 구분하지 않는다. 새 다운로드/외부전송/Git 없음.
''','utf-8')
save(U/'CURRENT_EXECUTION.json',dict(status='COMPLETED_BOUNDED_COMPARISONS_REWORK_REMAINS',at=now,training_running=False,evaluation_running=False,
 process_ids=[],web_changed=False,next=next_ko,no_background_job_claim=True))
for p,h in read(G/'PROTECTIONS.json').items():assert sha(Path(p))==h
assert read(O/'STATE.json')['current_followup']==U.name
assert str(U)in(R/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
assert title in(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8')
save(U/'RECORDS_SYNCED.json',dict(passed=True,at=now,files={str(p):sha(p)for p in paths+[U/'CURRENT_EXECUTION.json',U/'RESULT_KO.txt']},all_protected_unchanged=True))
print('Verified and synchronized. New candidates not deployed; no training/evaluation job remains.')
