from learning import *
import datetime,shutil
ROOT=OPS.parents[2]
def main():
 at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();proto=read(U/'PROTOCOL.json');checks={}
 for p,h in proto['protected'].items():assert sha(p)==h,p
 for p,h in proto['pins'].items():assert sha(p)==h,p
 for f in ['review/RESULTS_VERIFICATION_FINAL.json','review/CPU_RUNTIME_VERIFICATION.json','UI_VERIFICATION.json','UI_FILE_VERIFICATION.json']:
  d=read(U/f);assert d['status']=='PASS',(f,d.get('status'));checks[f]=sha(U/f)
 info=read(U/'LEARNING_INPUTS.json')
 for r in info['rows']:
  for key,hk in [('input','input_sha256'),('targets_file','targets_sha256'),('source','source_sha256')]:assert sha(r[key])==r[hk],(r['alias'],key)
 scores=read(U/'SCORES.json');cpu=read(U/'CPU_RUNTIME_RESULT.json');gpu=read(U/'RUNTIME_RESULT.json')
 completion=dict(status='FOUR_STAGE_RESEARCH_CYCLE_COMPLETE_NOT_ADOPTED',at=at,steps={'1':'Public80 labels, geometry masks, split and guide map complete within stated evidence limits','2':'Two matched heads x3 seeds complete, locked scores verified; aggregate quality gate not met','3':'Three individual references per24 query x3 paths; research difference/insufficiency viewer complete; educational correctness not established','4':'Raw CPU/GPU component timing, CPU-only timing and local HTTP/file UI tests complete; latency target not met'},product_goal_complete=False,production_changed=False,new_downloads=0,personal_or_mentor_material_used=False,training_runs=6,backbone_updates=0,fit_videos=36,development_videos=12,held_videos=32,selected_features=info['targets'],independent_checks=checks,protected_files=len(proto['protected']),same_Codex_not_independent_human=True,score_summary=scores['summary'],runtime_gpu=gpu['summary'],runtime_cpu=cpu['summary'],ui_http_seconds=.1092,ui_http_sample_count=1,adoption='Do not adopt SignRep feedback head: main10-feature BA regression, per-feature regressions, educational labels missing, raw-path wait target missed.',limitations=['All public videos previously in TCN TRAIN; newly held only for this temporal observer','Provisional RGB labels by Codex, not KSL educator truth','12expression full guide form match0, partial4; mandatory steps/repeats/contact not established','449 body-wrist-only target labels withheld; no invisible-hand recovery guarantee','Sparse6transition brackets do not establish exact transition timing','Reference-only low-rest heuristic and partial scope do not prove semantic irrelevance','CPU/GPU three-route timing shares measured detector and reports component sums; separate CPU-only benchmark provided','Local UI result load measured once; deployment/live webcam timing unmeasured'],next_focus='Keep C/T3F and baseline. Diagnose common under-supported features and interval-vs-point direction supervision using development data; obtain reliable guide/form/allowed-variation evidence for actual educational verdicts; measure capture-time MediaPipe scheduling before new model searches. Current locked32 now consumed; no tuning/reselection on them.')
 save('COMPLETION.json',completion)
 current=dict(stage=completion['status'],active=None,training_started=True,training_complete=True,remaining=[],research_viewer_server_session=None,stopped_server_session=41581,completed_sessions=[95887,18876,13143,69961,67919,84499],evidence='COMPLETION.json',note='All analysis finished; isolated browser tests closed. Static RESULT_KO.html works directly via file://. No recurring followup created.')
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf8')
 backup=U/'before/final_sync';backup.mkdir(exist_ok=False)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:shutil.copy2(p,backup/p.name)
 latest8='''### 2026-10-31 시간별 특징→다중 참고 피드백 1~4단계 실행 완료 — 미채택

- 현재 단위 `word100_stage_feedback_cycle_20261031`. 공개16표현80영상/9817프레임, fit36(ID1~3)/dev12(ID4)/held32(ID5~6,24배운표현+8새표현). 기존36은이번fit로개발사용, 전체80은과거TCN TRAIN이므로 C/T3F미사용시험아님. 사용자·멘토자료접근/학습/튜닝0, 새다운로드0.
- body-wrist fallback449타깃을제외해전체17후보3384known. 영상·프레임유지. 지원양/음4표현·3ID이상10특징학습,7보류. Codex RGB임시물리라벨,독립사람/교육자정답아님. 설명전체형태확인0/12,부분연결4/12,교육적규칙강제주입없음.
- SignRep본체고정,같은47914파라미터판정부 두조건각3seed/160epoch. 배운표현새사람10특징균형정확도80.47→79.37%(-1.09%p),새표현62.70→61.32%(-1.38%p). 영상별평균84.21→86.35%는집계방식차이로주지표대체금지. Rindex+7.59/양손겹침+12.30%p, Lopen-16.67/Rup-15.82%p. 단어인식률아님.
- 고정6영상의6변화브래킷: 관절4/3/3→영상6/5/4;같은물리상태144인접샘플예측변경5/6/5→1/2/4. 정확한변화시점·언어적단계정답아님. 24질의×3개별참고×3경로216쌍,합성360시험/미관측분모144검사통과. 시간변형오차3.34~3.35ms로정렬추가개선없음. 관측차이후보를수어오류로단정하지않는연구HTML완료.
- 원본MediaPipe CPU+SignRep GPU병행후head/DTW/기록합: 중앙5.49초,p956.74초(6회),준비8.30초별도. 별도CPU-only관절경로중앙4.62,p955.27초(6회),준비5.10초별도. 로컬HTTP전달/2화면갱신0.109초1회별도;실제제품웹캠종료대기시간아님. HTML HTTP/file 각각6조작PASS.
- 별도Codex에이전트 입력/라벨/코드/결과검산. 결과302658검사 및CPU118검사PASS,32보호파일동일. 최초오류/시간보조거리1건수정기록보존,모델재채점/튜닝없음. C/T3F·운영웹/DTW미교체. `COMPLETION.json`, `RESULT_KO.html/txt`가최신근거. 제품목표달성아님,현재4단계연구주기완료/미채택. 실행/서버종료,현재active=null.

'''
 latest9='''### 2026-10-31 완료 결과 검토 이후의 단일 우선순위

요청한 계획1~4 연구주기는 `word100_stage_feedback_cycle_20261031`에서 모두 실행·검산·보고 완료. 같은6회학습/잠금평가/완료추론을재실행하지 않는다. 현재진행중실행없음. C/T3F·운영DTW유지,사용자/멘토자료학습·튜닝·선택금지.
다음 후보를 시작한다면 새 모델 탐색보다 공통 특징의 학습 지원과 시간 범위를 먼저 점검한다. 현재fit의 R_motion_up양성9셀/L_open_four양성30셀처럼편중과, 구간방향을중간시점에학습한설계의영향을개발자료에서분리한다. 전체손미관측449셀은관절손목만으로ROI복귀시키지말고RGB범위근거가생긴후사용한다. 이번held32는이미평가를소모했으므로동일점수로규칙/후보선택튜닝금지.
교육피드백은설명과실제표현형의일치·필수단계/허용변이의확인근거가선행되어야한다. 현재출력은물리적차이와근거부족이며교정정답이아님. 속도는촬영중기존관절추출을지속할때종료후대기시간을실측하는것이우선이다. 임의로운영교체/새모델다운로드/개인자료학습으로범위를늘리지않는다. 상세이후계획은현재RESULT_KO.txt/COMPLETION.json,새사용자요청과함께이어간다.

'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert s.count(h8)==1 and s.count(h9)==1;s=s.replace(h8,h8+'\n'+latest8,1).replace(h9,h9+'\n'+latest9,1);p.write_text(s,encoding='utf8')
 state=read(OPS/'STATE.json');state['updated_date_kst']=at;state['current_stage']=completion['status'];state['next_single_action']='현재1~4주기완료·미채택. 결과보고검토후 공통특징지원/구간감독 원인검증과 촬영중처리실측 우선. 완료학습/잠금평가반복금지.';state['word100_stage_feedback_cycle_20261031']={**completion,'unit':str(U),'result_html':str(U/'RESULT_KO.html'),'result_txt':str(U/'RESULT_KO.txt'),'next_single_action':state['next_single_action']};(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
 handoff=f'''# WordsInSilence 최신 인수인계

{at}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: research/front_keyphase_v1/ops/word100_stage_feedback_cycle_20261031
현재 상태: FOUR_STAGE_RESEARCH_CYCLE_COMPLETE_NOT_ADOPTED
사용자 승인한 상세계획1~4 모두 실행. 준비만끝난것아님. 제품목표전체완성아님. CURRENT_EXECUTION.active=null, 학습/검산/브라우저시험/연구서버 종료. 후속자동화생성없음. 완료입력/6회학습/저장점수 재실행금지.

{latest8}
{latest9}
최신 근거: RESULT_KO.html/txt, COMPLETION.json, PROTOCOL.json, LABEL_AUDIT.json, LEARNING_INPUTS.json, TRAINING.json, SCORES.json, TEMPORAL_SCORES.json, PAIRED_FEEDBACK.json, CONTROLS.json, RUNTIME_RESULT.json, CPU_RUNTIME_RESULT.json, UI_VERIFICATION.json, UI_FILE_VERIFICATION.json. review/RESULTS_VERIFICATION_FINAL.json=302658PASS, review/CPU_RUNTIME_VERIFICATION.json=118PASS. 별도Codex에이전트검산이며독립사람/수어전문가정답아님.

구현: Codex root, 데이터RGB/가이드/입력검산·코드/점수검산은 guide_stage_map/rgb_labels_a/rgb_labels_b. 기존공개원본/과거동결읽기전용. C/T3-F/운영웹/DTW변경없음. 사용자·멘토자료접근/학습/튜닝/선택없음. 카메라·외부전송·Git·새다운로드없음.
환경: Core Ultra7 258V/32GB/Intel Arc140V. 실행Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. MediaPipe CPU, SignRep OpenVINO GPU,학습판정부torchCPU. SignRep본체는전혀학습안함. 얼굴표정·접촉판정모델없음.
브라우저도구초기화실패+샌드박스로컬소켓차단은로컬전용승인실행의별도Chrome headless/CDP·임시HTTP로검증완료. RESULT_KO.html은VIEW_DATA.js와함께파일로직접열어사용가능(6조작PASS), 공개비디오원본경로참조. UI_SCREENSHOT.png/좁은화면확인완료. 승인운영웹이아닌연구보고서임.
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
 save('FINAL_ARTIFACTS.json',dict(at=at,pins={str(p):sha(p) for p in [U/'COMPLETION.json',U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'VIEW_DATA.json',U/'VIEW_DATA.js',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']},protected_sha_checks=len(proto['protected']),independent_human_truth=False))
 print(json.dumps(dict(status=completion['status'],root=str(ROOT),protected=len(proto['protected']),files_synchronized=True)))
if __name__=='__main__':main()
