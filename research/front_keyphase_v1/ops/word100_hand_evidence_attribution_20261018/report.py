import sys
sys.dont_write_bytecode=True
from analyze import U,O,SRC,ROOT,read,write,sha,now,pincheck,SEEDS,GROUPS,VIEWS,CONDITIONS
import numpy as np

def main():
 assert read(U/'INDEPENDENT.json')['status']=='PASS'
 s=read(U/'SENSITIVITY.json');f=read(U/'FEATURE_STATS.json');d=read(U/'SAVED_DEV_ANALYSIS.json');ctx=read(U/'CONTEXT_v2.json')
 agg={v:{cond:{k:float(np.mean([r['views'][v][cond][k] for r in s['runs']])) for k in ['delta_ce','mean_abs_true_margin_change']} for cond in CONDITIONS} for v in VIEWS}
 for v in VIEWS:
  for cond in CONDITIONS:
   for k in ['prediction_changed','gained','lost','net_correct']:agg[v][cond][k]=sum(r['views'][v][cond][k] for r in s['runs'])
 regression={}
 for v in VIEWS:
  rows=[r for q in d['errors'].values() for r in q[v] if r['change']=='regression']
  regression[v]={'clip_seed_events':len(rows),'unique_clips':len({r['clip_id'] for r in rows}),'any_positive_intersection':sum(r['groups']['intersection']['fraction'][2]>0 for r in rows),'any_positive_proximity':sum(r['groups']['proximity']['fraction'][2]>0 for r in rows),'note':'Existing C0-to-C1 stored DEV regressions; repeated clip across seeds not independent.'}
 facts=[
 {'finding':'Base input corruption unsupported','evidence':'Prior exact280-channel audit, labels/split/masks/time preservation reused; no new extraction or input modification.'},
 {'finding':'View-dependent position/extent evidence differs','evidence':f['same_clip_spatial_view_disagreement'],'interpretation':'Same original timestamps aligned; extraction/window differences existed in C too. Added categories may expose these differences more directly; not proof of classification failure cause.'},
 {'finding':'Positive overlap cues cannot alone explain all regressions','evidence':regression,'interpretation':'Unavailable/no-intersection categories also have learned weights, so absence of positive cue does not exonerate the whole group.'},
 {'finding':'Frozen model is especially sensitive to forcing position=unknown','evidence':agg['sensor']['unknown_position'],'interpretation':'Unknown substitution alters information/distribution and possibly joint consistency. Sensitivity does not establish that position caused training degradation.'},
 {'finding':'Zeroing individual groups does not restore training predictions','evidence':{g:agg['sensor']['zero_'+g] for g in GROUPS},'interpretation':'All single-group zero perturbations preserve Top1 on899 TRAIN clips across3seeds; scores still change. Not unseen-user accuracy and not a retrained ablation.'},
 {'finding':'Seed/optimization outcome matters','evidence':d['histories'],'interpretation':'C1seed13 selected45/stopped57; other C1runs80. Saved history shows different optimization/early-stopping paths, not demonstrated sole cause. No patience/epoch retuning.'}
 ]
 plan={'status':'PROPOSED_AWAITING_USER_REVIEW_NO_TRAINING','at':now(),'goal':'Separate each added group effect during learning instead of infer from noisy post-hoc importance. No promised gain.','training_arms':[{'name':'minus_'+g,'change':'This group always0 in training and validation; other three original groups unchanged; 310shape retained.'} for g in GROUPS],'seeds':SEEDS,'new_runs_if_approved':12,'references':'Stored C/C0/C1 only; no retraining or rescoring original references','fixed':'Same899/DEV300, source inputs, model310/common fresh seed initialization/extra projection0, mixed Cschedule, AdamW .001/.001, batch32,max80/patience12, two-view CE selection. No ROI/detector/temporal/encoding change. No cherry-picked seed.','metrics':'Mean/range Top1/Top5/CE, paired gains/regressions, bysigner/fixedmissinggroups; clear DEV reuse/exploratory multiple-arm comparison. Strong candidate requires gain in >=2seeds vs fullC1 and mean above bothC/C0, no prior group/signer/Top5 regression. Otherwise no adoption.','importance_limit':'Removing one group while others learn isolates its effect in this particular combination and protocol; cannot fully resolve interactions or prove population-general causality.','separate_later_hypotheses':['Make channel meaning/availability consistent across views without leaking hidden raw points','Reduce redundant unavailable one-hot baseline/constant encoding only in a separate experiment','No simultaneous threshold/smoothing/epoch changes'],'time':'Previous6runs puretraining7.64min;12runs roughly15~25min as provisional puretraining budget, prep/verification additional.','user_mentor_heldout_used':False,'web_model_replacement':False,'execution_this_turn':{'training':0,'DEV_inference':0,'TRAIN_frozen_diagnostic_rows':59334}}
 write(U/'NEXT_PLAN.json',plan)
 write(U/'ANALYSIS.json',{'at':now(),'status':'COMPLETE_DIAGNOSTIC_NOT_CAUSAL_PROOF','findings':facts,'aggregate_sensitivity':agg,'DEV_regression_audit':regression,'conclusion':'Position/extent cross-view semantics and joint training response are priority hypotheses; no single group proven causal. Keep C. Next train-time leave-one-group-out requires user confirmation.','limits':['TRAIN sensitivity is not validation performance','zero is off-simplex; unknown replacement can be out-of-distribution/jointly inconsistent','DEV scores repeatedly used for selection, no new independent test','No RGB/manual occlusion labels added this turn','Separate numerical verifier by same Codex operator, not independent human/agent'],'new_training':0,'user_mentor_data_read':0,'new_DEV_inference':0,'protected_models_changed':False})
 lines=['추가 손 관측 정보의 성능 저하 원인 분석',now(),'',
 '결론: 원인 하나를 확정하지 못했다. 현재 우선 의심할 부분은 위치·경계 정보가 두 검출 방식에서 서로 다르게 표현되는 점과, 이 추정 정보들을 한꺼번에 학습했을 때의 최적화 변화다. 겹침 정보 하나 때문이라는 근거는 없다. 기존 C 유지, 새 학습 없음.',
 '', '1. 이번에 확인한 범위',
 '공개 TRAIN899/DEV300의 저장 추가 입력과 기존 점수·학습 이력을 분석했다. C1의3개 모델 가중치를 고정하고 TRAIN899에서11개 조건(원본/각묶음0제거/각묶음모름대체/전체제거·모름)을 양쪽보기로 진단했다. 총59,334예측행은 같은899영상의 반복 조건이며 새 테스트 표본 수가 아니다.',
 'DEV300은 이미 저장한 점수만 재사용했다. 새 DEV추론, 새 학습, 검출, 사용자/멘토/보류자료 읽기, 웹/DTW/모델 변경 모두0. 기존 결과는 덮어쓰지 않았다.',
 '', '2. 확인된 사실: 같은 순간도 위치·경계 설명은 자주 달랐다',
 '두보기는 원래 선택된 시작/끝 프레임이달라(1199개모두) 처음분석의동일배열길이가정은실패했다. 최초오류를보존하고같은원본프레임/시각끼리만비교했다. 기존입력이나점수를바꾼복구가아니다. 서로다른시간구간길이를새정보의효과로오해하지않는다.'
 ]
 names={'position':'관측·위치','extent':'손 영역의 경계','intersection':'현재 양손 영역 교차','proximity':'최근 관측·근접'}
 for g,q in f['same_clip_spatial_view_disagreement'].items():lines.append(f"  {names[g]} 표시 불일치: {q['different']}/{q['total']} 손-프레임 = {100*q['fraction']:.2f}%")
 a=f['distribution']['train_candidate']['masked'];b=f['distribution']['train_candidate']['sensor']
 lines.extend([
 f"영상별 비율을 동일가중 평균한 TRAIN에서 위치모름은 전체검출후가림 {100*a['position']['equal_clip_fraction'][5]:.2f}% / 실제잘린영상검출 {100*b['position']['equal_clip_fraction'][5]:.2f}%. 영역경계걸침은 {100*a['extent']['equal_clip_fraction'][2]:.2f}% / {100*b['extent']['equal_clip_fraction'][2]:.2f}%였다.",
 '왜 차이가 나는가: 전체검출후가림(masked)은 화면밖 원본 좌표를 차단하므로 영역을 모를 수 있다. 실제잘린영상(sensor)의 Pose는 화면밖 손목을 외삽할 수 있어 경계 단서가 생긴다. 추가정보의 가용성 자체가 두 보기에서 다르다. 이는 숨긴 원본정보를 사용하지 않기 위한 정상적인 차이이기도 하며, 잘못된 코드/성능저하의 확정 원인으로 단정하지 않는다.',
 '학습자료와 DEV의 전체 상태비율은 대체로 비슷했다. 이번 집계는 새로운 촬영자 전체가 전혀 다른 상태분포였다는 주장을 지지하지 않는다. 개인/단어별 차이와 관측추정의 실제정확도는 별도다.',
 '', '3. 겹침 의심을 잘못 넣은 것만이 원인은 아니다',
 f"sensor에서 C0가맞고 C1이틀린 경우는3seed합계{regression['sensor']['clip_seed_events']}건(고유{regression['sensor']['unique_clips']}영상). 겹침의심 표시가 한번이라도있던경우는{regression['sensor']['any_positive_intersection']}건, 최근근접양성은{regression['sensor']['any_positive_proximity']}건이었다.",
 '다만 미평가/교차없음도 각각학습된 채널이므로 양성표시가없다고 해당그룹의영향이0인것은아니다. 손가락 가림 복원 실패와 새 채널의 학습 영향도 서로 다르다.',
 '', '4. 고정 모델에서 정보를 가려본 결과',
 '아래는 TRAIN899×3seed에서 예측1위가 바뀐 횟수다. 성능향상치가 아니며 같은영상 반복 포함. 두 가리기 방식은 분포변화가 달라 직접 중요도 순위로 해석하면 안 된다.',
 '항목 | 추가항목을0으로지움(sensor) | 모름/미평가로대체(sensor)'
 ])
 for g in GROUPS:lines.append(f"{names[g]} | {agg['sensor']['zero_'+g]['prediction_changed']} | {agg['sensor']['unknown_'+g]['prediction_changed']}")
 lines.extend([
 '위치모름 대체에서 가장 큰 변화가 나타났다. 모델이 위치/관측구분을 쓰고있다는근거이지, 위치정보가 성능하락의 범인이라는증명은아니다. 다른그룹과함께학습한모델을사후에가리는것과그정보없이처음부터학습하는것은다르다.',
 '현재각단일그룹0제거로는sensor TRAIN예측이회복되거나변하지않았다. 익숙한학습자료는이미거의맞히므로변화가작을수있고, logit점수는변했다. 모름대체도관측좌표가존재하는데모름표시가되는등불일치가있을수있다. 이수치를새사용자인식률로사용하지않는다.',
 '', '5. 학습 경로 차이도 남아 있다',
 '첫seed는C1이45epoch를선택하고57epoch에서정해진patience12로끝났지만C0는80epoch까지진행해77epoch를선택했다. 다른두C1은80epoch까지갔다. 첫seed에회귀가집중됐다. 따라서여러채널추가가최적화/중단시점에영향을준가능성도있다. 학습을더시키면회복된다는증명은없으며결과를보고중단규칙을변경하지않았다.',
 '기존 C의좌표·마스크손실이나자료누락은이전전수검산상발견되지않았다. 이번추정채널들은실제미검출손가락을새로복원하지않으므로세밀한모양정보부족은그대로남는다.',
 '', '6. 다음 보완 계획 — 사용자 확인 후 실행',
 '1) 기존C를유지하고, 310입력구조·자료·seed·최적화·선택조건은고정한다.',
 '2) 위치만제외 / 영역경계만제외 / 현재교차만제외 / 최근근접만제외, 네조건을각3seed로처음부터학습한다. 총12회. 제외그룹은학습과평가양쪽에서항상0, 다른세그룹은동일. 기존저장C/C0/C1을비교기준으로재사용한다.',
 '3) 평균뿐아니라3회방향,새오답,한손미관측/중간누락,사람별회귀를함께보고한다. C1만이겨도기존C보다낮다면채택하지않는다. 여러조건을살펴보는개발비교임을명시하고보류촬영자는보존한다.',
 '4) 어떤그룹을빼야안정적으로회복되는지확인한후,그그룹의두보기간의미/가용성통일또는미평가표시의중복을줄이는별도후보를설계한다. 숨긴화면밖좌표를다시넣어맞추거나,좌표복원·평활화·학습횟수를한번에바꾸지않는다.',
 '이방법은이조합에서각그룹을추가학습하는효과를분리하는가장직접적인다음비교다. 단일원인이나보편적인성능상승을보장하지않고그룹간상호작용은여전히가능하다.',
 '직전6회순수학습7.64분기준12회순수학습약15~25분을임시예산으로보며입력준비·검증시간은추가. 이번에는계획만저장했고학습하지않았다.',
 '', '검증: 별도코드로추가특징1,316,688상태값/진단59,334예측행재집계PASS. 같은Codex의분리검산이지다른사람/에이전트검토는아님. 기존모델·자료해시보존. 근거CONTEXT_v2/FEATURE_STATS/SAVED_DEV_ANALYSIS/SENSITIVITY/INDEPENDENT/ANALYSIS/NEXT_PLAN.json.'
 ])
 (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 pincheck(ctx['pins']);pincheck(read(SRC/'PREP_CONTEXT.json')['pins'])
 stamp=now();latest='### 2026-10-18 추가 손 정보 성능 저하 분석 완료 — 위치·경계 불일치/학습경로 의심, 단일원인 미확정\n'
 latest+='ops/word100_hand_evidence_attribution_20261018. 새학습0, C1동결TRAIN899두보기11조건3seed민감도진단59,334행. DEV는저장점수만사용/새추론0. 원본시각정렬공통156,214손프레임에서위치22.08%/영역23.93%불일치. TRAIN영상평균영역경계masked0%/sensor20.31%,위치모름15.70%/1.48%. 정보가용성차이를새채널이직접표시하지만성능하락인과증명아님.\n'
 latest+='sensor C0→C1회귀12clip-seed/고유11영상중교차양성2/최근근접양성0. frozen TRAIN의각그룹0제거는sensorTop1변경0;위치모름대체35예측변경은분포교란/의존성진단이며학습원인확정불가. 첫seed C1선택45/중단57 vs C0선택77/80로최적화변화가능. 숫자재검산PASS/모델해시보존. 첫정렬가정오류보존후같은frameID/시각교집합으로정정(기존두보기선택구간다름).\n'
 latest+='기존C/M0/DTW/웹유지, 사용자/멘토/보류자료읽기0, 새검출/다운로드/Git/외부전송없음. 결과RESULT_KO.txt/ANALYSIS.json,다음12회제외학습계획NEXT_PLAN.json은사용자확인전실행금지.\n\n'
 nexttxt='### 2026-10-18 네 정보 묶음별 학습 제외 비교 계획 사용자 확인\n위치/영역경계/현재교차/최근근접각한묶음만학습·검증에서0제외하는4조건×3seed=12회제안. 동일899/DEV300/310구조/초기조건/80epoch·patience12/CE선택고정, 저장C/C0/C1기준재사용. 이번은진단/계획만승인, 새학습은사용자결과확인후진행. 단어특례·사용자튜닝·동시여러변경·DEV결과맞춤임계완화금지. 완료진단재실행불필요.\n\n'
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
 for mark,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n',latest),('## 9. 다음 단일 작업\n',nexttxt)]:assert mark in t;t=t.replace(mark,mark+'\n'+body,1)
 p.write_text(t,encoding='utf-8')
 state=read(O/'STATE.json');state['current_stage']='WORD100_HAND_EVIDENCE_ATTRIBUTION_COMPLETE_AWAITING_ABLATION_PLAN_REVIEW';state['updated_date_kst']=stamp;state['next_single_action']='사용자와 원인분석/12회묶음별제외학습계획확인. 새학습승인전실행없음. 기존C유지.';state['word100_hand_evidence_attribution_20261018']={'status':'COMPLETE_DIAGNOSTIC_NOT_CAUSAL_PROOF','at':stamp,'report':str(U/'RESULT_KO.txt'),'analysis':str(U/'ANALYSIS.json'),'next_plan':str(U/'NEXT_PLAN.json'),'verification':read(U/'INDEPENDENT.json'),'training':0,'DEV_inference':0,'user_mentor_heldout_reads':0,'production_changed':False,'running':False,'completed_exec':71994,'exit_code':0}
 write(O/'STATE.json',state,new=False)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+stamp+'\n\n'+latest+nexttxt+'기존 C sensor98.67%, C099.00%, C198.00% 평균은반복개발검증이며실사용정확도아님. C1미채택유지. 자동화st-gcn삭제유지. 원본/sources/W/과거동결모델읽기전용. 실행중프로세스없음. 이번가려보기의TRAIN정답수변화를새검증성능/학습개선으로표기금지.\n',encoding='utf-8')
 write(U/'STATUS.json',{'stage':'COMPLETE_AWAITING_USER_PLAN_REVIEW','at':stamp,'training':0,'source_and_models_preserved':True,'report':str(U/'RESULT_KO.txt')},new=False)
 write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now(),'source_model_pins_preserved':True,'latest_documents_synchronized':True,'report_sha256':sha(U/'RESULT_KO.txt'),'analysis_sha256':sha(U/'ANALYSIS.json'),'next_plan_sha256':sha(U/'NEXT_PLAN.json')})
 print('Complete: analysis, next plan, root documents and protected hashes recorded.')
if __name__=='__main__':main()
