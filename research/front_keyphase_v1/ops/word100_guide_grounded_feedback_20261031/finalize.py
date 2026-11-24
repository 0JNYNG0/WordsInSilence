"""Validate saved artifacts and synchronize current state; no experiments."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from urllib.parse import urlparse,unquote
import json,hashlib,shutil
U=Path(__file__).resolve().parent;ROOT=U.parents[3];STATE=U.parent/'STATE.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    f=read(U/'FROZEN.json');res=read(U/'DELIVERY_RESULT.json');checks=[]
    for group in ['pins','protected']:
        for p,h in f[group].items():assert sha(p)==h,p;checks.append(group+':'+p)
    assert res['summary']['clause_statuses']=={'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE':1,'OBSERVER_RELATION_CONFLICT':0,'DESCRIBED_FEATURE_OBSERVED':5,'INSUFFICIENT_LINKED_EVIDENCE':14,'SOURCE_LINK_WITHHELD':21}
    assert len(res['rows'])==12 and sum(c.get('expected_anchor_count',0) for r in res['rows'] for c in r['clauses'])==224
    assert len([r for r in f['rows'] if r['role']=='fit'])==36
    assert len([r for r in f['rows'] if r['role']=='development'])==12
    assert set(r['person'] for r in f['rows'] if r['role']=='fit').isdisjoint(set(r['person'] for r in f['rows'] if r['role']=='development'))
    assert read(U/'REPORT_UI_VERIFICATION.json')['status']=='PASS'
    assert read(U/'review/REPORT_REVIEW.json')['status'].startswith('PASS')
    assert (U/'review/DELIVERY_REVIEW.json').exists()
    for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt']:
        assert '손 관측 근거가 부족해 비교할 수 없습니다' in p.read_text(encoding='utf8')
    data=read(U/'REPORT_DATA.json');media=set()
    for v in data.values():
        for url in [v['query'],*v['refs'].values()]+([v['sequence']] if v.get('sequence') else []):
            parsed=urlparse(url);assert parsed.scheme=='file';p=Path(unquote(parsed.path).lstrip('/'))
            assert p.exists(),p;media.add(str(p))
    verify=dict(status='PASS',frozen_pins=len(f['pins']),protected_models_and_code=len(f['protected']),references36_queries12_person_ids_disjoint=True,known_positive_anchor_cells_retained=224,report_local_media_files=len(media),report_public_pairs=12,
        observations_and_DTW_unchanged=True,all_original_results_preserved=True,new_model_inference=0,new_training=0,educational_accuracy=None,
        note='Report UI passed before a Korean wording/axis-unit correction; no interaction changes. Final local media/JSON/hash validation passes.')
    write(U/'FINAL_VERIFICATION.json',verify)
    now=datetime.now(timezone(timedelta(hours=9))).isoformat();name=U.name
    nextaction='설명에 연결된 공통 동작 단계·관계의 시간 근거를 fit/dev 공개자료에서 보강한다. 펼침/접근/팔을 따른 이동/분리의 관측 가능 시점과 표현형을 구분하고, 기존 이동 판정부·손목/팔 축·DTW 시점의 충돌 원인을 분리한다. 단계/접촉/반복 정답을 설명이나 미검출값으로 만들지 않으며 검증된 시간주석이 확보된 특징만 작은 단계·관계 판정부 학습 비교에 사용한다. 소모 held/fresh 재튜닝·새모델 순회·개인자료·운영교체 금지. 완료 source 비교/추론/속도시험 재실행 불필요.'
    status='COMPLETE_SOURCE_LINKED_PROTOTYPE_NOT_EDUCATIONAL_VALIDATION'
    complete=dict(status=status,at=now,unit=name,active=None,scope='Saved learned physical observer + unchanged DTW + source-linked partial features/order/body relations. Research prototype; no new semantic model training.',summary=res['summary'],delivery_controls=18,
        execution=dict(model_inference=0,training=0,report_ui_session=53652,report_ui_exit_code=0,chrome_closed=True,relational_session=81554,relational_result='COMPLETE_AND_INDEPENDENTLY_VERIFIED; execution handle no longer present at final check'),
        verification=dict(root='FINAL_VERIFICATION.json',independent='review/INDEPENDENT_REVIEW_FINAL.json',report='review/REPORT_REVIEW.json',delivery='review/DELIVERY_REVIEW.json',visual='review/VISUAL_RELATION_REVIEW.json',same_Codex_family_not_human_truth=True),
        not_completed=['Pedagogical normal/error/tolerance truth','Full semantic stages and mandatory repetition','Valid contact and 3D orientation','Current P048 motion discrepancy automatic resolution','Production feedback integration'],
        preservation=dict(C_T3F=True,production_web_DTW=True,user_mentor_data_unused=True,source_originals=True),next_single_action=nextaction)
    write(U/'COMPLETION.json',complete);write(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,at=now,completion='COMPLETION.json',resume='Use saved results; no unfinished run to restart'))
    before=U/'before';before.mkdir(exist_ok=False)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',STATE]:shutil.copy2(p,before/p.name)
    state=read(STATE);state['updated_date_kst']=now;state['updated_at']=now;state['current_stage']=status;state['next_single_action']=nextaction;state['resume_command']=nextaction
    state['activity_state']='CURRENT_UNIT_COMPLETE_RESEARCH_ONLY';state['latest_visual_review']=str(U/'RESULT_KO.html')
    state['goal_status']='INCOMPLETE: source-linked feature/order prototype verified on public development data; educational error vs acceptable variation and production semantic feedback remain unvalidated.'
    entry=dict(status=status,at=now,active=None,unit=str(U),report=str(U/'RESULT_KO.html'),summary=res['summary'],new_training=0,new_model_inference=0,person_roles='fit ID1/2/3 36; development ID4 12; held/fresh not tuned',
        independent_checks=dict(saved=1125,relational=1265,engine_counterexamples=48,root_controls=39,delivery_controls=18),
        visual_review='Separate Codex 5 clips/10images/65 samples, not expert truth',
        original_anchors=224,prior_candidate_cells=420,outside_contract_cells_preserved=351,
        failure='P048 binary down-motion candidate conflicts with visible/trajectory flow; added arm-relative diagnostic did not meet fixed direction agreement; automatic resolution0, candidate1 retained',
        residual='Partial source-feature mappings, broad draft anchors not full phases, order witness not complete sequence, no validated linguistic tolerance/contact/count',
        C_T3F_preserved=True,production_changed=False,user_mentor_used=False,next_single_action=nextaction)
    state[name]=entry
    # Keep current unit near the top without rewriting historical entries.
    ordered={}
    for k,v in state.items():
        if k==name:continue
        ordered[k]=v
        if k=='current_stage':ordered[name]=entry
    write(STATE,ordered)
    update='''### 2026-10-31 설명과 관측을 연결한 첫 비교 완료 — 교육적 정오 미확정

- 현재 단위 `word100_guide_grounded_feedback_20261031`. 사용자 최신 지시로 관측→수어 설명 연결을 즉시 구현. 기존학습10물리특징 seed42 저장출력/기존DTW고정, fit ID1/2/3 36영상·dev ID4 12영상. 새학습/추론0. 12표현 사전 원문/뜻·표현형/번호표기 감사, source clauses41·orders5. partial/proxy/미확인/기능미지원 분리. 설명으로 가려진 손을 채우거나 TCN정답으로 세부동작정답을 만들지 않음.
- 일부특징관측5(수어·마음에들다 R검지,감사 양손펼침,별 R펼침),차이후보1,근거부족14,설명연결/관측미지원21. 순종의 R상승2.300~2.900→하강2.967~3.100 일부순서1,미지원4. 전체의미단계/반복수/교육정오 아님. 멀다 두손설명 불일치/감사 동음이의/대체표현 섞기 방지. 알고있는fit양성앵커224셀은 미관측도보존하며 전체시간주석은아님.
- 잘하다P048 기존Rdown판정이 영상/궤적과상충. 공통양손→반대팔축투영을48영상96궤적에추가했으나짧은창방향합의불통과. 자동해결0·후보1보존,성과향상/사용자오류주장금지. 420기존frame×feature후보와새1설명항목은단위가달라감소율계산금지;설명밖351셀보존. 좌표불변성은언어허용폭아님.
- 자체39/별도반례48PASS,별도저장1125/상대좌표1265검사불일치0,보호32hash유지. 별도Codex가모델출력미열람상태로5영상10장65표본확인;교육자정답아님. NaN/inf·미정출처allowlist·.2경계문제수정/최초보존,engine_v2결과동일. 이후relational_overlay_v2가부분충돌시남은후보시각보존(18통제);DELIVERY결과41상태동일,초기결과보존.
- RESULT_KO.html/txt 공개12영상쌍·DTW시점버튼·출처별설명·8궤적그래프·좁은화면/JSError0 확인. C/T3F/운영웹/DTW보존,개인·멘토자료미접근. active=null,보고서Chrome종료. 완료연구결과재추론/재학습불필요. 전체수어의중요차이/허용폭판단AI는미완료.

'''
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf-8-sig')
    marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';assert marker in txt;txt=txt.replace(marker,marker+'\n'+update,1)
    marker9='## 9. 다음 단일 작업\n';assert marker9 in txt;txt=txt.replace(marker9,marker9+'\n### 2026-10-31 설명 연결 이후 — 공통 단계·관계의 시간 근거\n\n'+nextaction+'\n\n현재 보고서의 5관측/1순서를 전체 단어정답으로 확대하지 않는다. 현후보1건은사용자오류로배포하지않는다. SOURCE_CONTRACTS/DELIVERY_RESULT/COMPLETION/별도검산을재사용하며기존완료품질평가를반복하지않는다.\n\n',1)
    p.write_text(txt,encoding='utf8')
    handoff=f'''# WordsInSilence 최신 인수인계

{now}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고 중요한 차이와 허용 가능한 차이를 구분한다.

현재단위 `{name}`, 상태 `{status}`. 이번 source 연결·검산·보고서 완료, active=null. 서버/추론/학습 진행중 없음. 최종 연구목표인 교육적 정오·허용폭 판단은 아직 미완료이며 운영 교체 아님.

{update}
## 현재 실제 구조

공개영상 전체MediaPipe→고정ROI→69개2D기하특징(69관절아님)→기존pose_capacity_matched_42의10특징→기존개별3참고DTW→source-linked비교. source는관측후비교대상만정하고관측값은변경하지않는다. 순서는원래영상시간으로확인. 이번SignRep/TCN추론/새학습없음.

fit36(ID1/2/3),dev12(ID4)만사용. 이전held32/fresh32 소모평가점수튜닝금지. source의partial/proxy는부분물리비교만허용하며실제표현형/필수단계/허용폭확정아님. 시점앵커224는knownpositive초안표본이지전체영상정답아님. 서로다른사전후보/뜻을합치지않는다.1지검지/4지새끼/5지엄지 공식근거확인.

기능별지원gate는12비엄지각도필수+기존손목지원으로보완. 손이없으면비교분모를줄이거나없는특징=0으로판정하지않음. 접촉·3D손방향·정확반복·입표정은교육판정범위밖. 위→아래 witness는전체순서/반복검증아님.

## 결과 파일과 재개 주의

FROZEN/RESULT 원본불변. engine_v2는확률.2경계수정(개발출력동일). RELATIONAL_PROTOCOL/RELATIONAL_RESULT/FINAL_RESULT는팔축진단추가(충돌0). relational_overlay_v2/DELIVERY_RESULT는후보시각별충돌/잔여보존수정18통제,실제41상태동일. 초기add_relational_evidence의전체항목상태덮어쓰기분기를후속실행에그대로사용하지말것.
REPORT_UI_VERIFICATION공개12쌍·8그래프PASS,FINAL_VERIFICATION 보호32/동결153핀검사. review/INDEPENDENT_REVIEW_FINAL,REPORT_REVIEW,DELIVERY_REVIEW,VISUAL_RELATION_REVIEW. 같은Codex별도검토이지사람/교육자정답아님. 현재완료결과 재실행하지않는다.

## 다음 한 작업

{nextaction}

기존속도증거 word100_streaming_web_replay_20261031 종료→텍스트표시ack중앙0.411초는공개파일CPU관절+고정head+DTW경로만이며SignRep/TCN·실웹캠전체제품시간아님. 이번후단.000423초도제품시간으로보고금지.

Core Ultra7 258V/32GB/Arc140V. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. PYTHONIOENCODING=utf-8,PYTHONDONTWRITEBYTECODE=1. MediaPipe현재Windowswheel CPU만;기존SignRep IntelGPU경로존재하나이번미사용. C/T3F/운영웹/DTW 유지. 원본W:/sources/과거동결읽기전용. 사용자·멘토자료 접근·학습·튜닝·선택,카메라,외부전송,다운로드,Git,운영교체 금지.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
    assert read(STATE)['current_stage']==status
    assert name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
    # Pin deliverables and document updates without recursive inclusion.
    targets=[p for p in U.rglob('*') if p.is_file() and not any(part in ['before','report_profile','__pycache__'] for part in p.relative_to(U).parts)]
    write(U/'FINAL_ARTIFACTS.json',dict(at=now,files={str(p.relative_to(U)):sha(p) for p in targets},documents={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',STATE]}))
    print(json.dumps(dict(status=status,protected=32,pins=153,local_media=len(media),state_synced=True),ensure_ascii=False))
if __name__=='__main__':main()
