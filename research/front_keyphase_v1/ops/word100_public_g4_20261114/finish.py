from prepare import U,O,read,sha,write,load,Path
import datetime,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
R=O.parents[2];G=O/'word100_web_completion_graph_20261111';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
v=read(U/'VERIFIED.json');audit=read(U/'LINEAGE_AUDIT_EXTENDED.json');assert v['passed']and not audit['hits']
summary=dict(status='G4_EVALUATION_COMPLETE_LIMITED_CAPABILITY',at=now,
 source_scope='Four words x public record17/18 =8 videos,977 native frames. Not used in active feedback/common100 lineage; previously used in TCN evaluation, so not globally unseen.',
 point_results=v['point_summary'],automatic_results=v['summary'],
 final_scope='Frozen raw models first evaluated; one seed had a false supported direction on one continuous reversal. Post-score presentation conflict guard regression-verified; its recheck is not an independent fresh accuracy estimate.',
 qualified_continuous_reverse=v['qualified_reverse_counts'],
 scientific_success='Face-region54/54, normal24 conditions no new correction; normal order18 per seed; shortened prefix2 not completed; missing8 not positive. No new detector/model training or checkpoint selection.',
 unresolved=['One reversed brush-arm source: seed43 interval includes other motion; raw branch false support remains. Common conflict guard displays uncertain instead of agreement; not repaired localization accuracy.',
 'Alternating repetition entirety, fingertip gathering, thumb and physical contact unsupported','No real learner error accuracy or complete-word pass','G5 web integration and G6 user-started camera/end-to-end acceptance remain'],
 protections=dict(files=v['protected_files'],C_T3F=True,common100=True,user_training=False,user_tuning=False,user_access=False,web_changed=False),
 next='G5: expose only measured feature evidence and uncertainty; preserve source time/hand/phrase links; conflict must stay gray, no full-word pass. G6 capture flow afterwards.',
 operator='Codex',verification='Same Codex separate source/hash/arithmetic audit; not independent human/KSL expert')
write(U/'RESULTS_REVIEW.json',summary)
write(U/'FAILURES_AND_REPAIRS.json',dict(
 execution=['Trait model path error and missing Path import stopped before inference; fixed local evaluator only.',
 'NumPy bool JSON serialization failed after dense8-source outputs saved. Kept partial TRAIT_RESULTS.json, reused dense outputs; only2 unsaved reverse forwards recomputed. TRAIT_RESULTS_V2.json is valid.',
 'New evaluation adapter placed nonmovement cards last, manufacturing reverse comparisons. Reassembled all saved outputs with established G3 guide-list orientation/adjacency. AUTO_COMPLETE remains historical; AUTO_RESULTS_V2 is corrected. No model forward for repair.',
 'Partial reversed clip segments introduced splice jumps. All original tests retained; six whole-video continuous reversals added as diagnostic repair, all native frames retained.',
 'Raw continuous reversal missed in seed43 on one source (same old/new rank reference). Initial verifier assumed every seed detects6 and failed. Actual raw result6/5/6 retained; no score or gate hidden. Final verifier distinguishes raw accuracy limitation from source/contract integrity.',
 'Added source-independent contradiction guard after final score. Development82 and normal public52 comparison rows unchanged; same source reversed under old/new comparison now uncertain. This is safety/uncertainty qualification, not improved learned recognition or new independent validation.'],
 model_training_runs=0,checkpoint_changes=0,web_changed=False,remaining_localization_failure=True,
 artifact_warning='TRAIT_RESULTS.json is deliberately preserved incomplete output; use TRAIT_RESULTS_V2.json. AUTO_COMPLETE.json includes superseded adapter ordering; use AUTO_RESULTS_V2.json plus CONFLICT_REGRESSION.json.'))
text='''G4 공개자료 검증·보완 결과

이번에 끝낸 일
공개 4단어의 8영상을 처음부터 끝까지 관절 추출하고, 기존 판정과 G3 판정을 같은 조건으로 비교했다.
이 영상들은 이번 피드백 모델 학습·개발에는 사용하지 않았다. 과거 TCN 평가에는 사용했으므로 “프로젝트 전체에서 미사용”이라고 부르지 않는다.
원본 977프레임을 처리했으며 사용자 촬영은 열거나 학습·튜닝에 사용하지 않았다. 새 모델 학습과 C·T3-F 교체도 하지 않았다.

이전 → 이번에 확인한 결과
• 손이 얼굴 앞인지 상체 쪽인지: 기존 54항목 중53 확인·1보류 → G3 보완은54 확인.
  54는 8영상에서 직접 확인한 특정 시점·손의 위치 항목이다. 영상54개나 수어 전체 정확도가 아니다.
• 손모양: 단일 시점 판정부는52항목 중1개를 오인식했다. 앞뒤 흐름 판정부와 비교하면 그1개는 의견 충돌로 걸러져 확정 교정을 보류한다. 나머지51항목은 맞았다.
• 정상 동작: 8원본과 두 속도 조건을 합친24조건에서 새 교정 경고가 없었다. 세 학습 실행 모두 기존 순서 판정을 유지했다.
• 중간에 끊은2조건은 완료로 처리하지 않았다. 손 관측값을 뺀8조건도 맞았다고 하거나 교정 명령으로 바꾸지 않았다.

찾아서 고친 문제
1. 평가 연결 코드가 설명 항목을 잘못된 배열 순서로 비교했다. 기존 G3 계약대로 고쳐, 없는 순서 오류를 만들지 않게 했다. 신경망 결과는 그대로 재사용했다.
2. 영상 중간만 뒤집은 시험은 편집 경계에 갑작스러운 손 이동이 생겼다. 이 시험도 남겨두고, 전체 영상을 연속 역재생한6조건을 추가했다.
3. 연속 역방향6조건에서 학습 실행별 차이 발견은6·5·6이었다. “잘하다” 한 사례에서 한 실행이 마무리 움직임까지 섞어 방향을 잘못 판단했다.
   의견이 충돌하는데도 맞았다고 표시하지 않는 공통 장치를 추가했다. 표시 결과는5조건에서 차이 안내,1조건에서 확인 필요다.
   기존 개발82조건과 이번 정상 조건은 바뀌지 않았다. 이 조치는 잘못된 확신을 막는 보완이며, 그 움직임 자체를 더 잘 알아보도록 학습한 것은 아니다.
   보완 후 같은8자료 재검산은 독립적인 새 성능 평가라고 하지 않는다.

G4의 의미
미사용 공개자료 평가와 문제 분석·출력 안전장치 검증을 마쳤다. 정상 보존과 관측 부족 분리는 확인했다.
모든 수어 동작이나 모든 오류를 정확하게 판단하는 기능이 완성된 것은 아니다. 새 자료에서 발견된 구간 선택 문제1개는 여전히 남으며 충돌로 표시한다.
‘조상’의 반복 전체, ‘이집트’의 마지막 손끝 모양, 엄지·접촉은 충분히 검증되지 않아 합격 판정을 내리지 않는다.
이8영상은 이제 평가에 노출됐다. 향후 모델을 수정하면 다시 미사용 평가자료라고 부르지 않는다.

다음 단계
G5: 검증된 항목만 웹의 손·팔 강조와 짧은 문장에 연결. 차이/비슷함/확인필요를 구분하고 전체 단어 합격과 혼동하지 않게 표시.
G6: 사용자가 시작한 촬영에서 자동 종료·결과 표시·전체 대기시간을 점검.
이번 작업에서 기존 웹은 바꾸지 않았다. 실행시간은 오프라인 검증 시간이며 실제 웹 대기시간으로 쓰지 않는다.

검증: 보호477파일 및 원본/모델 해시, 저장 계산과 문장 근거를 별도로 확인. 같은 Codex의 검산이며 독립 한국수어 전문가 검토는 아니다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
# Graph: show completed evaluation separately from remaining capability coverage.
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(16,9));fig.patch.set_facecolor('white');ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off')
navy='#172B46';muted='#5C6E86';light='#EEF2F7'
ax.text(.55,8.42,'피드백 기능 작업 현황',fontsize=25,weight='bold',color=navy)
ax.text(.6,7.85,'G4 공개자료 검증 종료  ·  확인한 항목만 다음 단계로',fontsize=16,color=muted)
def box(x,y,w,h,title,body,done=True):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.12',facecolor=navy if done else light,edgecolor=navy,linewidth=1.4))
 ax.text(x+.18,y+h-.42,title,fontsize=16,weight='bold',color='white'if done else navy)
 ax.text(x+.18,y+.26,body,fontsize=12,color='white'if done else navy,linespacing=1.5)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color=muted,linewidth=1.6))
for x,title,body in [(0.6,'G0·G1  완료','범위 고정\n설명·정답 주석'),(4.35,'G2  개발 검증 완료','동작 구간 찾기\n설명과 연결'),(8.1,'G3  개발 검증 완료','손모양·방향·위치\n같은 행동에서 비교'),(11.85,'G4  검증 완료','새 공개자료 확인\n충돌은 확인 필요로')]:box(x,5.48,3.45,1.75,title,body)
for x in [4.15,7.9,11.65]:arrow((x-.15,6.35),(x+.16,6.35))
box(.6,2.8,7.0,1.92,'이번에 확인한 것','정상·속도 차이 유지   /   관측 부족은 합격하지 않음\n역방향 시험: 5조건 차이 안내 + 1조건 판단 보류',False)
box(8.1,2.8,3.45,1.92,'G5  다음 작업','웹의 강조 표시와\n짧은 피드백 연결',False)
box(11.85,2.8,3.45,1.92,'G6  남음','실제 촬영 흐름\n자동 종료·대기시간',False)
arrow((13.55,5.4),(9.85,4.82));arrow((11.6,3.76),(11.8,3.76))
ax.text(.65,1.83,'남은 한계',fontsize=17,weight='bold',color=navy)
ax.text(.65,1.19,'구간 선택이 엇갈리는 1사례  ·  반복 전체  ·  손끝·엄지·접촉의 세부 판정',fontsize=15,color=navy)
ax.text(.65,.53,'G4 완료는 검증 단계 종료를 뜻합니다. 전체 단어의 정오 판정·웹 테스트 완료는 아직 남아 있습니다.',fontsize=13,color=muted)
fig.savefig(U/'PROGRESS_GRAPH.png',dpi=145,bbox_inches='tight',facecolor='white');fig.savefig(U/'PROGRESS_GRAPH.svg',bbox_inches='tight');plt.close(fig)
graph=load('g4finalgraph',G/'graph.py')
graph.record('G4','PASS',v['g4_checks'],[str(U/'RESULTS_REVIEW.json'),str(U/'VERIFIED.json'),str(U/'CONFLICT_REGRESSION.json'),str(U/'LINEAGE_AUDIT_EXTENDED.json')],
 'Frozen feedback-unexposed public8 evaluated; prior TCN use disclosed. Evaluation stage complete with capability limits. Raw reversed detection6/5/6, conflict guard ->5difference/1uncertain; no model accuracy gain or fresh post-guard evaluation claimed. Remaining localization/fine-shape gaps explicit. G5 only qualified evidence display, not full-sign success.')
next_action='G5: 검증된 관측·차이·불확실과 원본 시점/손/설명을 웹에 연결. 판단 충돌은 회색 확인필요, 반복/손끝/엄지/접촉은 미지원으로 명시. 보호 C·T3-F/공통100 유지, 사용자자료 금지. 공개 저장결과 화면 검산 후 G6 사용자 시작 촬영·자동종료·전체지연 확인.'
state=read(O/'STATE.json');state.update(current_stage=G.name,current_followup=U.name,next_single_action=next_action,last_updated=now)
state[U.name]=dict(status=summary['status'],path=str(U),result=str(U/'RESULTS_REVIEW.json'),verified=str(U/'VERIFIED.json'),graph=str(U/'PROGRESS_GRAPH.png'),training_running=False,web_changed=False,private_access=False,remaining=summary['unresolved'])
state[G.name].update(status='G4_EVALUATION_COMPLETE_G5_PENDING_LIMITED_CAPABILITY',active_unit=str(U),at=now,latest_evidence=str(U/'RESULTS_REVIEW.json'),progress='G0-G4 evaluation stages complete; one contradictory source withheld, incomplete fine-feature coverage retained.',remaining=next_action,training_running=False,web_changed=False)
(O/'STATE.json').write_text(__import__('json').dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig');heading='### 2026-11-14 G4 공개8 검증 종료 / 충돌1 확인필요 / G5 다음'
latest='피드백 학습·개발 미사용 공개17/18의4단어8영상977프레임 검증(과거TCN평가 이력 있음). 손위치 기존53확인+1보류→54확인, 손모양52항목51일치+1판정부 충돌. 정상/속도24조건 새 경고0, 순서18쌍/seed 유지, 부분2·관측없음8 완료금지. 연속역방향6조건의 원시검출은seed42/43/44에서6/5/6: 잘하다1구간 선택 오류 남음. 공통 충돌표시 보완으로 최종5차이+1확인필요; 개발82·정상결과 보존. 보완 후 동일자료 재검산은 독립새평가 아님. 모델학습/웹교체 없음. 반복전체·손끝·엄지·접촉·전체단어정오 미검증. G4는 평가 단계 완료이며 기능 전면 합격 아님. 근거: `research/front_keyphase_v1/ops/word100_public_g4_20261114/RESULTS_REVIEW.json`, `VERIFIED.json`, `PROGRESS_GRAPH.png`. 기존C·T3-F/공통100/보호477파일 불변, 사용자자료 미접근. 모든 실행 종료.'
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',latest),('## 9. 다음 단일 작업',next_action+' 이번8평가는 완료·노출됨. 과거 Final8 미개봉 문구를 현재 상태로 사용하지 않는다. 완료 추론/검증 재실행 금지. 구간학습을 다시 수정하면 새 독립자료가 필요하다.')]:
 assert marker in s;s=s.replace(marker,marker+'\n\n'+heading+'\n\n'+body+'\n',1)
p.write_text(s,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'# G4 검증 종료, 제한 범위 G5 다음 — {now}\n현재 단위: {U}\n'+latest+'\n\n다음 한 작업: '+next_action+'\n현재 모델은 변경하지 않았다. conflict_guard.py는 사후 출력 자격 보완이며 새 정확도 향상/독립 검증으로 주장하지 않는다.\n유효 결과: TRAIT_RESULTS_V2.json / AUTO_RESULTS_V2.json / CONTROL_COMPLETE.json / CONTINUOUS_COMPLETE.json / CONFLICT_REGRESSION.json / RESULTS_REVIEW.json / VERIFIED.json.\n초기 잘못된 배열순서 출력과 불완전 JSON은 실패 이력으로만 보존; FAILURES_AND_REPAIRS.json 확인.\nG4그래프 PASS는 미사용 공개평가/검산 종료를 뜻하며 모든 표현 정오판정 완료가 아니다. G5/G6/DONE 미완료.\n모든 실행 종료. 서버/카메라/학습/다운로드/외부전송/Git 없음.\n','utf-8')
for p in [U/'CURRENT_EXECUTION.json',G/'CURRENT_EXECUTION.json']:
 p.write_text(__import__('json').dumps(dict(status='G4_EVALUATION_COMPLETE_LIMITED_CAPABILITY',at=now,active_unit=str(U),running=False,next=next_action),ensure_ascii=False,indent=2)+'\n','utf-8')
print('G4 evaluation complete; limited capability only. Graph and handoff updated.')
