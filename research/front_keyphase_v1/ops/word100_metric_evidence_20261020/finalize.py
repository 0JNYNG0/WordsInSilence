from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;B=U.parents[1];ROOT=B.parents[1];OLD=U.parent/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
    f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');v=read(U/'VERIFICATION.json');at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    assert v['status']=='PASS' and read(U/'BROWSER_QA.json')['status']=='PASS'
    assert all(sha(p)==h for p,h in f['protected'].items())
    assert sha(U/'candidate.py')==f['candidate_sha256'] and sha(U/'run.py')==f['runner_sha256']
    oldaudit=read(OLD/'FINAL_AUDIT.json');assert all(sha(OLD/n)==h for n,h in oldaudit['artifact_sha256'].items())
    assert read(U/'COMPARISON.json')['counts']['pairs']==29643
    names=['FROZEN.json','PREPARE.json','SELFTEST.json','COMPARISON.json','VERIFICATION.json','SUMMARY.json','DEV_ROWS.json','DEV_QUERY_DETAILS.json','VISUAL.json','VISUAL_REVIEW.json','RESULT_KO.html','RESULT_KO.txt','BROWSER_QA.json']
    audit={'status':'PASS','at':at,'protected_files':len(f['protected']),'previous_frozen_results_unchanged':True,'protected_unchanged':True,
      'implementation_and_verification':'Codex local / same operator separate source-cosine and scalar verifier','public_clips':1199,'classes':100,'views':3,
      'new_training':0,'classifier_inference':0,'private_or_mentor_recordings_read':0,'production_changes':0,
      'visual_timelines_inspected':6,'previous_RGB_frames_reused':18,'semantic_labels_added':0,
      'minor_qa_repair':'Mobile expanded table overflow fixed in report CSS only; FIRST_BROWSER_QA preserved; final QA PASS',
      'artifact_sha256':{n:sha(U/n) for n in names}}
    write(U/'FINAL_AUDIT.json',audit)
    nextaction='사용자와 항목별 관측근거·구간진단·궤적점검 결과를 확인한다. 다음 후보는 연구용 DTW의 부분각도/관측불명 표시와 구간별 핵심근거 누락 검증 조건 검토. 기존 C/T3-F·웹·DTW 유지. 인식보조·추가학습·개인촬영튜닝은 시작하지 않는다.'
    status='COMPLETE_RESEARCH_EVIDENCE_SPLIT_NOT_DEPLOYED'
    statepath=B/'ops/STATE.json';state=read(statepath);state['current_stage']='WORD100_METRIC_EVIDENCE_VALIDATED_REVIEW';state['updated_date_kst']='2026-10-20';state['next_single_action']=nextaction
    state[U.name]={'status':status,'at':at,'unit':str(U),'public_scope':{'classes':100,'TRAIN':899,'DEV':300,'views':3,'saved_path_pairs':29643},
      'old_angle_values_preserved':6393705,'added_frame_angle_cells':73308,
      'DEV_hand_units':591,'DEV_measurability':{k:x['DEV'] for k,x in s['views'].items()},
      'sensor_source_witness_DEV':{'lowered_full_source_but_missing_view_handframes':28643,'annotated_in_ROI_full_source_but_missing_view_handframes':355,'not_ground_truth':True,'not_available_outside_real_webcam':True},
      'sensor_trajectory_common_units':461,'sensor_trajectory_absolute_change_median':0.0,'sensor_trajectory_absolute_change_p90':s['views']['sensor']['trajectory_common_measurable']['absolute_change']['p90'],
      'verification':v,'selftests':13,'visual_timelines':6,'model_and_production_unchanged':True,'training':0,'private_data_read':0,
      'execution':'prepare62400/compare14527/verify63094 exited0, no active process',
      'limits':['No new whole-hand measurable units; partial-angle evidence recovered only','No recognition improvement measured','Offline full-view diagnostics cannot infer out-of-camera reality','2D angle and DTW phase uncertainty remain'],
      'report':str(U/'RESULT_KO.html'),'next_single_action':nextaction}
    write(statepath,state)
    title='### 2026-10-20 손 위치·각도별 관측근거 분리 검증 완료 — 부분근거 확보, 운영 미적용'
    entry=f'''{title}
갱신 {at}. ops/{U.name}. 사용자 승인한 공개100단어 항목별관측/내림·미관측구간/궤적기준 검증 완료. TRAIN899/DEV300×3관측,기존29643DTW경로재사용. 원본/이전결과/관측경로/정렬/최소근거조건변경없음. candidate별도연구부품,운영미적용.
- 15굽힘각 각각3관절관측·비퇴화조건,손위치는기존값보존. 기존6393705프레임각도값수치오차내유지/73308관측셀추가(독립영상수아님). 누락각도0/보간없음,각도마다5참고·50%·8프레임,15각도모두아니면전체값보류. 집계순서변화있으므로기존대새원점수향상비교금지.
- DEV손591단위 기존전체가능→보완15각도가능 legacy591→591,masked426→426,sensor447→447. 기존가능손실0,전체가능추가0. 기존전체보류에서부분각도 masked19/sensor14건(각도근거+87/+75). 믿다REAL12왼손7/15각도가능·전체여전히보류. 인식성능향상아님.
- 현재관측의위치/각도/팔힌트/내부공백과전체native위치증거분리. 전체21점하단밖+낮은안정관측만내림표시;누락을내림으로확정안함. sensorDEV 전체원본낮은정지&view미관측28643손프레임,주석동작중native범위내&view미관측355. 오프라인추정근거조건건수/정답아님,실제웹캠밖추론불가. 자동손제외/새필수손라벨0.
- 궤적개별첫관측대첫공통대응시작비교:동일경로공통461손궤적,변화절댓값중앙0/p90 4.0609어깨너비100단위. 초반누락/실제시작위치제거위험으로새원점운영채택안함. 공백채움없음/동일의미단계보장없음.
- 자체13검사/별도source코사인6467013각도·106290집계·300고정쌍/3597입력/위치보존PASS. 같은Codex별도검산,타agent/전문가아님. 공개6타임라인육안/이전18RGB장면근거재사용,전수의미검토아님. 보호29파일/이전동결결과불변,보고모바일표넘침만수정·QA PASS.
- C/T3-F/웹/DTW유지,개인·멘토촬영미접근,인식보조·학습0. prepare62400/compare14527/verify63094 종료0. RESULT_KO.html/txt,SUMMARY,VERIFICATION,VISUAL_REVIEW,FINAL_AUDIT. 현재결과검토후후속결정.

'''
    cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ctx=cp.read_text('utf-8-sig');assert title not in ctx
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert ctx.count(h8)==1 and ctx.count(h9)==1
    ctx=ctx.replace(h8,h8+'\n'+entry,1).replace(h9,h9+'\n'+title+'\n'+nextaction+'\n\n',1);cp.write_text(ctx,'utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 공개다중참고 DTW 관측근거 개선. 마지막 완료: 손 위치·15각도별 근거 분리/내림·미관측 구간진단/궤적첫기준 비교 검증. 운영 미적용.

- 단위 {U}
- 공개100단어1199원본×3관측/기존29643DTW경로재사용. prepare62400,compare14527,verify63094 모두종료0;활성작업없음. 완료계산재실행불필요.
- 기존6393705각도값수치오차내유지,73308프레임각도셀추가. DEV591손단위 전체측정가능 legacy591/masked426/sensor447 유지,손실0/추가0. 부분근거 masked19/sensor14확보;전체손점수로승격안함. 믿다12왼손7/15각도·여전히전체보류.
- 내림은전체원본21점아래+관측안정일때만오프라인표식. sensorDEV 내림원본근거&view없음28643손프레임/주석동작범위내원본근거&view없음355. 전체원본도추정이며정답라벨아님. 실제화면밖은불명,자동제외없음.
- 궤적개별첫점대공통대응첫점:공통461손궤적,변화절댓값중앙0/p90 4.0609(어깨너비100단위). 낮아졌다고좋음아님;실제시작위치지워질수있어채택안함.
- 자체13검사/별도3597입력·6467013각도·106290집계·300쌍검산PASS;같은Codex별도코드,타agent아님. 6타임라인확인+이전18RGB장면재사용. 원본·C/TF·웹/DTW보호29파일/이전결과불변. 최초모바일표넘침CSS복구후QA PASS.
- 학습0/인식보조0/사용자·멘토자료미접근/14~18미접근. 특정단어규칙없음. 원본과이전동결결과읽기전용. RESULT_KO.html,SUMMARY,VERIFICATION,VISUAL_REVIEW,FINAL_AUDIT 확인.

다음 한 작업: {nextaction}
''','utf-8')
    write(U/'STATUS.json',{'status':status,'at':at,'verified':True,'report':str(U/'RESULT_KO.html')})
    write(U/'CURRENT_EXECUTION.json',{'status':'EXITED','prepare':{'session':62400,'exit':0},'compare':{'session':14527,'exit':0},'verify':{'session':63094,'exit':0},'active':False})
    assert nextaction in cp.read_text('utf-8') and nextaction in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
    assert read(statepath)[U.name]['status']==read(U/'STATUS.json')['status']
    print(json.dumps({'status':status,'protected':len(f['protected']),'docs_synced':True},ensure_ascii=False))
if __name__=='__main__':main()
