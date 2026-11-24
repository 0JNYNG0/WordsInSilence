"""Check planning facts and record the reviewed roadmap, not an experiment."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil
import numpy as np
U=Path(__file__).resolve().parent;ROOT=U.parents[3];OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    old=OPS/'word100_guide_grounded_feedback_20261031';saved=read(old/'COMPLETION.json')
    assert saved['active'] is None
    inputs=read(OPS/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json');counts={}
    for role in ['fit','development']:
        rows=[r for r in inputs['rows'] if r['role']==role];frames=knownframes=0
        for r in rows:
            z=np.load(r['targets_file'],allow_pickle=False);m=z['target_mask'][:,inputs['target_indices']]
            frames+=len(m);knownframes+=int(m.any(1).sum())
        counts[role]=dict(clips=len(rows),frames=frames,frames_with_any_target_annotation=knownframes)
    assert counts=={'fit':{'clips':36,'frames':4351,'frames_with_any_target_annotation':615},'development':{'clips':12,'frames':1874,'frames_with_any_target_annotation':67}}
    reviewfiles=[U/'review'/n for n in ['DATA_PLAN_REVIEW.json','ARCHITECTURE_PLAN_REVIEW.json','EVALUATION_PLAN_REVIEW.json']]
    for p in reviewfiles:assert read(p)['status'].startswith('PLAN')
    plan=(U/'PLAN_KO.txt').read_text(encoding='utf8')
    for phrase in ['최대12 fit영상','개발4영상','독립 정답','구간과 경계 구분','이미 소모한 held/fresh','정상 손모양','계획 재검토','단계3·4 각각']:
        assert phrase in plan,phrase
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    stages=[
      dict(id='S1',name='자료·표현 연결과 분할 고정',depends=[],owner='자료·출처 담당',reviewer='검증 담당',output='기존 지원표 재사용 + 사건 목표 지원/표현형/사용이력/역할표',gate='출처/역할/unknown 분리; 점수 전 미사용평가 후보 감사·잠금'),
      dict(id='S2',name='사건 시간 주석·충돌 원인 분리',depends=['S1 scope'],owner='영상·물리 담당',reviewer='자료 담당+검증 담당',output='최대12fit×2사건창+4dev×2창 및 명시 대조 묶음; 관측/판정부/정렬/표현형 원인표',gate='직접 원본 근거와 경계 불확실 범위; 양성/음성/unknown 및 다표현·ID 지원'),
      dict(id='S3',name='구간 움직임 후보 하나 비교',depends=['S1','S2'],owner='root 구현·실행',reviewer='검증 담당',output='고정기존/학습없는기하/새작은구간판정부 3seed 비교',gate='학습전 계약·후보예산·단위동결; 목표개선+보류/회귀함께; 주어진구간시험으로표시'),
      dict(id='S4',name='자동 사건 경계·단계 대응·순서',depends=['S3 qualified component'],owner='root 구현·실행',reviewer='영상 담당+검증 담당',output='정답시간 없는 전체영상 사건목록/참고별단계대응/원시간순서',gate='사건누락/허위사건/경계오차 검증; 원래범위·개별참고보존; 합성오류와실제수어오류구분'),
      dict(id='S5',name='중요차이·허용차이 타당성',depends=['S4','independent linguistic labels and fresh evaluation'],owner='자료·출처 담당+root',reviewer='검증 담당 및 해당 수어 근거 확인',output='지원범위의 정상오경고·중요오류·보류coverage·잘못된확신 평가',gate='독립 실제사례 정답/명시통과기준/새평가잠금 필수; 모르면PASS불가'),
      dict(id='S6',name='격리 웹 전체 경로',depends=['S5 for educational correction; S4 for research physical comparison'],owner='root 통합',reviewer='검증 담당',output='같은영상 전체종료→근거미디어/피드백표시 품질·대기·메모리',gate='선언범위의 품질/unknown전달/전체지연통과; 운영교체자동승인아님')]
    resolutions=[
      '코드검사 통과와 교육적 타당성 검증을 분리',
      '전체48 경계주석을 선결조건으로 만들지 않고 첫배치 상한·종료분기 설정',
      '주어진 구간 분류와 전체영상 자동경계찾기를 별도 구현/평가',
      '한표현에만 있는 팔관계를 범용학습 성공으로 확대하지 않음',
      '단계3/4 개발 계약도 결과 전 동결; 최종평가만잠그는실수 방지',
      '정상오경고/실제중요오류/보류coverage/관측불가확신을 동시에 검증',
      '소모평가 재튜닝금지; ID9는이력감사전fresh아님',
      'Codex합의/사전문장/합성변형을 독립실제수어정답으로 쓰지 않음']
    nextaction='기존48 fit/dev 지원표를 재사용해 공통 사건 목표·표현 연결·사용이력 계약을 고정하고, 최대12fit영상×2사건창+4dev영상×2창과 명시 대조의 첫시간주석 묶음을 준비한다. 단어결과 대신 원본RGB·관측·판정부·DTW·표현형 원인을 분리. 주어진구간분류와자동경계검증을구분하며 자료지원통과 후에만 고정후보1개 비교로 간다. 소모평가튜닝/개인자료/운영교체/새모델순회 금지.'
    contract=dict(status='PLAN_REVIEWED_NOT_EXPERIMENT_EXECUTED',at=at,unit=U.name,goal='한국수어 설명과 여러 참고 영상을 함께 이용해 손·모양·위치·관계·움직임·순서를 파악하고 중요한 차이와 허용 가능한 차이를 구분한 근거 피드백',
        first_practical_milestone='지원된 공통 움직임의 시간근거→후보비교→자동사건대응→설명과근거장면 연결. 교육적교정은별도정답gate통과후.',
        current_measured_unit=old.name,current_saved_summary=saved['summary'],annotation_recount=counts,
        stage_dependencies=stages,reviewers={p.stem:sha(p) for p in reviewfiles},review_resolutions=resolutions,
        limits=['No verified educational error/acceptable-variation truth bundle yet','No unused final evaluation declared before use-history audit','Numbers of events bound work, not statistical sufficiency','A rule/array test cannot prove KSL educational accuracy','No error-free guarantee'],
        not_run=['new training','model inference','new quality evaluation','new video extraction','production changes','downloads','user or mentor access','camera','external transfer','Git'],next_single_action=nextaction)
    write(U/'PLAN.json',contract)
    evidence=[old/'COMPLETION.json',old/'DELIVERY_RESULT.json',old/'source_audit/SOURCE_CONTRACTS.json',OPS/'word100_feature_support_streaming_20261031/COMPLETION.json',OPS/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json']
    write(U/'VERIFICATION.json',dict(status='PASS_PLANNING_FACTS_AND_REVIEW_INCORPORATION',at=at,annotation_recount=counts,review_tracks=3,
        learning_or_inference=False,evidence={str(p):sha(p) for p in evidence},review_hashes={str(p):sha(p) for p in reviewfiles},plan_sha256=sha(U/'PLAN_KO.txt'),note='Read-only saved-label arithmetic and plan audit, not new performance validation'))
    before=U/'before';before.mkdir(exist_ok=False)
    statepath=OPS/'STATE.json';docpaths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',statepath]
    for p in docpaths:shutil.copy2(p,before/p.name)
    state=read(statepath);stage='SEMANTIC_FEEDBACK_ROADMAP_REVIEWED_NO_NEW_TRAINING'
    state.update(updated_date_kst=at,updated_at=at,current_stage=stage,activity_state='PLAN_REVIEW_COMPLETE',next_single_action=nextaction,resume_command=nextaction)
    entry=dict(status=contract['status'],at=at,active=None,plan=str(U/'PLAN_KO.txt'),structured_plan=str(U/'PLAN.json'),last_executed_unit=old.name,
        annotation_recount=counts,review_tracks=3,resolutions=resolutions,new_training=0,new_inference=0,production_changed=False,user_mentor_used=False,next_single_action=nextaction)
    state[U.name]=entry;ordered={}
    for k,v in state.items():
        if k==U.name:continue
        ordered[k]=v
        if k=='current_stage':ordered[U.name]=entry
    write(statepath,ordered)
    update='''### 2026-10-31 핵심 피드백 목표·6단계 계획 재검토 완료 — 새 실험 없음

- 현재 계획단위 `word100_semantic_feedback_roadmap_20261031`. 사용자 현황쉬운설명/세부계획/에이전트검토 요청에3개검토(자료·출처/아키텍처/평가)완료. 실제 마지막실행은`word100_guide_grounded_feedback_20261031`이며5부분특징/1부분순서/후보1·자동해결0그대로. TCN·운영웹새향상주장없음.
- 저장주석재집계: fit36의4351프레임중615,dev12의1874중67에현재10특징주석있음. 연속단계/정확경계/중요오류/허용변이정답아님. 사전전체표현형/필수성/허용폭검증0. ID9의12표현은미사용평가후보일뿐이력·SHA감사전fresh선언금지.
- 6단계: 지원/표현/분할계약→상한있는시간주석/4갈래원인분리→기존고정·기하·새구간후보1개비교→정답구간없는자동사건/참고별순서대응→독립정오/허용근거및새평가→격리웹전체지연. 구간분류가자동경계탐지는아님을별도gate로수정. 기존GRU미채택재사용강요없고손모양/C·T3-F고정.
- 첫준비묶음은최대12fit×2사건창+4dev×2창과명시한휴식/전이/교차대조. 충분→후보1개,부족→해당관계보류/필요근거명시로마감. 전48단계전수주석을선결조건으로삼지않음. 한표현팔관계는범용학습근거로승격금지. Codex합의·사전·합성변형은실제수어오류정답아님.
- 단계3/4개발기준도결과전동결. 정상오경고·중요누락·보류coverage·미관측확신·전체UI시간을함께판정하며항상보류로개선주장금지. 교육정오/허용경계는독립근거없으면미검증. 이번추론/학습/새영상/다운로드/개인자료/운영변경0,active=null. PLAN_KO.txt/PLAN.json/VERIFICATION.json및review3파일에근거·역할·gate기록.

'''
    ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8-sig')
    for marker,addition in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n',update),('## 9. 다음 단일 작업\n','### 2026-10-31 검토된 첫준비묶음부터 진행\n\n'+nextaction+'\n\nPLAN_KO의단계별산출/gate/실패분기를따르며현재는계획검토완료상태이다. 이전완료실험/소모평가재실행없음.새모델학습이이미시작됐다고해석하지않는다. 전체교육피드백완성은아직아님.\n\n')]:
        assert marker in text;text=text.replace(marker,marker+'\n'+addition,1)
    ctx.write_text(text,encoding='utf8')
    hand=f'''# WordsInSilence 최신 인수인계

{at}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 이용해, 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고 중요한 차이와 허용 가능한 차이를 구분한다.

현재계획단위 `{U.name}`. 상태 PLAN_REVIEWED_NOT_EXPERIMENT_EXECUTED, active=null. 이번요청은현황·계획재검토이며새학습/추론/운영변경없음. 계획보고를학습완료로혼동하지말것.

{update}
## 마지막 실제 구현 결과

`{old.name}` 완료. 공개전체MediaPipe/고정ROI/69기하특징→기존pose_capacity_matched_42의10특징→개별3참고DTW→source-linked후단. 설명은관측값을바꾸지않는다. fit36 IDs1/2/3,dev12 ID4.5부분특징/순종1부분up→down,후보1P048,부족14/미지원21. 팔상대진단96트랙은후보자동해결0. 교육정오/허용폭미확정. 앵커224는희소양성표본;420기존셀과새1항목단위다름. 원본·DTW·C/T3F·운영웹보존.
원FROZEN/RESULT/FINAL보존,engine_v2(.2경계수정),relational_overlay_v2/DELIVERY(부분충돌뒤남은후보유지)사용. 원add_relational_evidence 전체항목덮어쓰기분기후속재사용금지. 보고서12공개영상쌍/8그래프UI검증완료. 별도검산1125/1265/410과코드39/48/18PASS는교육자정답이아님. 완료실험재실행불필요.

## 다음 단일 작업과 역할

{nextaction}

자료·출처담당=분할/표현/사건지원,영상담당=출력과분리한물리시간주석,root=계약고정·한후보구현/실행/통합,검증담당=원본/저장값독립산술·회귀/누출/분모검사. 독립된수어학적정오자료확보는별도이며모든프레임전수사람검토를요구하지않는다. 첫묶음은pilot상한이지충분한통계보장아님. 자동사건경계실패면웹교정으로올리지않는다.

## 계획 근거/상태

PLAN_KO.txt,PLAN.json,VERIFICATION.json; review/DATA_PLAN_REVIEW.json,ARCHITECTURE_PLAN_REVIEW.json,EVALUATION_PLAN_REVIEW.json. 같은Codex계열별도검토. 기존fresh/held는소모평가로재튜닝금지. ID9후보는이력감사전새자료선언금지. 현재교육오류양성/허용범위독립정답부족은모델/코드로대신하지않는다. 개발기준과최종평가기준을각결과전잠금.

## 환경·보존

Core Ultra7 258V/32GB/Arc140V. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. PYTHONIOENCODING=utf-8,PYTHONDONTWRITEBYTECODE=1. MediaPipe Windowswheel CPU;기존SignRep IntelGPU경로있으나이번미사용. 촬영중부분경로0.411초를전체제품속도로인용금지. UI/필요미디어/모든선택모델포함해추후실측.
원본W:/sources/과거동결읽기전용. 사용자·멘토자료학습·튜닝·선택금지,이번개인자료접근없음. C/T3-F·운영웹/DTW유지. 새다운로드/외부전송/카메라/Git/운영교체없음.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(hand,encoding='utf8')
    assert read(statepath)['current_stage']==stage
    for p in docpaths:assert U.name in p.read_text(encoding='utf-8-sig')
    write(U/'COMPLETION.json',dict(status=contract['status'],at=at,active=None,review_tracks=3,review_resolutions=resolutions,plan_sha256=sha(U/'PLAN_KO.txt'),documents={str(p):sha(p) for p in docpaths},execution='Saved labels read-only recount; no training, inference, evaluation rerun or production change',next_single_action=nextaction))
    print(json.dumps(dict(status=contract['status'],review_tracks=3,annotation_recount=counts,documents_synced=True),ensure_ascii=False))
if __name__=='__main__':main()
