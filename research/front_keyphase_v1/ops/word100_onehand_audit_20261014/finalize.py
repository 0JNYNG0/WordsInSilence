"""Record verified diagnosis and synchronize the current handoff only."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat();check=read(U/'INDEPENDENT_CHECK.json');assert check['status']=='PASS'
assert (U/'PRESENTATION.png').exists()
assert not (U/'COMPLETION.json').exists()
unit='ops/word100_onehand_audit_20261014'
completion=dict(status='USER_ROUND_AND_ONEHAND_CAUSE_DIAGNOSIS_VERIFIED',at_kst=now,unit=unit,
 report=unit+'/RESULT_KO.txt',verification=unit+'/INDEPENDENT_CHECK.json',audit=unit+'/AUDIT.json',public_sensitivity=unit+'/PUBLIC_SENSITIVITY.json',
 session_id='ses_083c3a62d345',participant_scope='One development participant, untrained; not independent new-user generalisation proof',
 trial_count=12,practice_words=12,class_count=100,top1=dict(correct=6,total=12,percent=50.0),top3=dict(correct=7,total=12,percent=100*7/12),top5=dict(correct=8,total=12,percent=100*8/12),
 public_profiles=dict(train=108,development=36),public_forwards=108,public_onehand_original_correct=9,public_onehand_removed_left_correct=0,public_onehand_total=9,
 user_reinference=0,training=0,model_change=False,operating_code_changed=False,dtw_threshold_change=False,camera_access=False,
 implementer='Codex local',validation='Separate raw saved-logit and observation recount + hashes + public control parity + actual browser; not a second agent review',
 original_hashes_preserved=check['checked_original_hash_count'],
 presentation=dict(path='Docs/interim_report_20261014/web/index.html',url='http://127.0.0.1:8766/#s9',pages=10,notes=10,browser_verified=True,overflow=False,screenshot=unit+'/PRESENTATION.png'),
 remaining=['Cause of two-handed errors is not isolated','No repaired-model performance measured','No expert handshape correctness or independent new-user generalisation claim'],
 next_single_action='Prepare one public-training-only hand-visibility robustness experiment: establish resting vs signing hand evidence, freeze masking and comparison rules, preserve original clean development results and bimanual signs. User trials excluded from training/tuning/candidate selection; no automatic model replacement.',
 running_processes_started_by_unit=[])
completion['artifacts_sha256']={name:sha(U/name) for name in ['SCOPE.json','AUDIT.json','PUBLIC_SENSITIVITY.json','INDEPENDENT_CHECK.json','RESULT_KO.txt','PRESENTATION.png']}
write(U/'COMPLETION.json',completion)
before=U/'docs_before';before.mkdir(exist_ok=True)
for path in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json']:
    (before/path.name).write_bytes(path.read_bytes())
state=read(LAB/'ops/STATE.json');state['updated_date_kst']=now;state['current_stage']='WORD100_ONEHAND_CAUSE_DIAGNOSIS_VERIFIED';state['word100_onehand_audit_20261014']=completion
state['next_single_action']=completion['next_single_action']
state['word100_pretest_clarity_20261014']['user_round_completed_in']=unit+'/COMPLETION.json'
write(LAB/'ops/STATE.json',state)
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8')
item8='''
### 2026-10-14 실사용12표현 전수 집계·한 손 누락 원인 재현 완료
사용자 ses_083c3a62d345의12표현 각1회 전수 확인.100후보에서Top1 6/12=50.0%,Top3 7/12=58.33%,Top5 8/12=66.67%.12결과/12이벤트ID/12확인기록 일치,취소0. 학습미포함 개발참여자1인으로 독립새사람일반화증명 아님. 과거3trial제외. session.json trial_ids=[]이므로 파일/이벤트를 함께 집계.
멀다·공항·별에서 사용자 왼손 관측3.0/7.1/8.3%,대응9학습자 모두100%.공개개발36개 원본/왼손제거/오른손제거 총108회 고정모델진단,사용자재추론0.몸/다른손/시간/원래구간 고정. 해당3표현9개는 원본9/9정답→왼손제거0/9,멀다56~62위/공항13~16위/별5~10위. 현재모델의 쉬는손 미관측 민감성 확인;양손오답 잘하다4/믿다7/감사2위 원인은미분리.새성능개선아님.
DTW 세한손표현 왼손유효참고0/종합보류,오른손비교유지.나머지수행도범위초과가있어초과=수어오답으로해석금지.62원본해시·촬영프레임해시검산,공개대조저장출력오차최대3.82e-6.모델/기준/운영코드변경0,학습0,카메라접근0.근거 ops/word100_onehand_audit_20261014/COMPLETION.json 및 RESULT_KO.txt. Codex구현+별도재집계,별도에이전트/전문가판정아님.
기존발표웹10장으로갱신:9장실제12회·누락진단/대본,10장후속우선순위.8766/#s9실제브라우저확인,가로넘침없음.기존AIHub96.67%와사용자50%집합구분.시험사이트/모델은변경없음.

'''
item9='''
### 2026-10-14 다음 — 공개 자료만으로 손 관측 조건 보완 실험 준비
실사용12회집계·원인진단·발표반영완료.다음은공개학습분할에서쉬는손과필수손의근거를확인해화면밖관측조건을재현하는단일보완후보를설계·사전고정하는것.본래개발성능과관측누락조건,양손수어회귀를같이평가하도록계획.이번진단의36공개개입과사용자12회는완료되어재실행하지않는다.사용자자료학습/튜닝/계수/후보선택금지.모델교체·새학습자동착수하지않음.잘하다/믿다/감사양손오답과DTW실제교정타당성은미분리/미검증으로유지.과거36회촬영권고를강제하거나실시된결과로보고하지않음.

'''
s=re.sub(r'(## 8\. 현재 상태[^\n]*\n)',lambda m:m[1]+item8,s,count=1)
s=re.sub(r'(## 9\. 다음 단일 작업[^\n]*\n)',lambda m:m[1]+item9,s,count=1);context.write_text(s,'utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}. AGENTS/CONTEXT 최신 §3·§8·§9/WORKFLOW 우선.

## 현재 요구와 완료
사용자는12표현을각1회촬영·저장후 한손수어에서 반대손이안보이면인식이떨어지는지전수원인진단요청. ops/word100_onehand_audit_20261014/COMPLETION.json 완료. RESULT_KO.txt 상세보고.
session ses_083c3a62d345(participant self)12결과/이벤트ID/확인기록 모두일치,취소0.직전 ses_abfadf0f647a는0trial.이전10/13 3trial은제외.목표순위:수어1,마음에들다1,멀다33,잘하다4,믿다7,즐겁다1,감사2,수영1,공항11,별7,사계절1,냉커피1. Top1 6/12 50.0%,Top3 7/12 58.33%,Top5 8/12 66.67%.100후보중순위.학습미포함개발참여자1인,독립새사람일반화증명아님.자기보고는전문가수행정답라벨아님.

## 확인된 원인과 한계
멀다/공항/별은사용자왼손관측3.0/7.1/8.3%,각9학습자100%.저장영상과참고영상에서쉬는왼손의화면밖/안차이확인.CONTACT_1/2.jpg는로컬검토,새촬영아님.
공개개발12표현×11/12/13=36개에원본/왼손제거/오른손제거 총108추론1회.다른손/몸/시간/원래입력구간고정.한손3표현9개원본9/9정답→왼손제거0/9,멀다56~62/공항13~16/별5~10위.공개대조는저장출력오차최대3.82e-6.현재모델누락민감성직접확인,전체웹캠오답원인확정/개선결과아님.양손잘하다/믿다/감사는다른원인미분리.
입력은누락좌표정규화후0+마스크0,반대손까지버리는버그없음.마스크채널제공만으로누락강건성보장안됨.현재AIHubfresh100단어학습은전체/핵심구간혼합만있고쉬는손전체누락보강없음.시간패딩마스크와관절관측마스크를혼동하지않을것.
DTW한손3표현PARTIAL:왼손유효참고0/종합판단보류,오른손비교유지.다른수행도위치/궤적범위초과많음;사용자오류로단정금지,촬영/정렬/수행차이분리미완료.

## 검증과 변경
Codex로컬도구 audit.py/public_sensitivity.py/verify.py.별도원본NPZ·저장logits재집계·62해시·모든replay프레임해시PASS.별도에이전트/언어전문가검증아님.사용자재추론0/학습0/운영코드·모델·DTW기준변경0/카메라접근0.개입108결과는 PUBLIC_SENSITIVITY.json 재실행금지.
Docs/interim_report_20261014/web/index.html/report.css만발표갱신.10장/대본10개,9장실제12회/공개누락진단,10장후속.기존96.67%와실사용50%분리. http://127.0.0.1:8766/#s9 실제화면확인,넘침없음,PRESENTATION.png.5~7장은기존10/13화면예시유지,새시험과명시분리.

## 다음 한 작업
공개학습자료에서쉬는손과필수손의근거를정리하여관측누락에강한단일보완후보설계/조건고정.원본개발성능·누락조건성능·양손수어회귀를함께검증하도록계획.사용자촬영은학습/튜닝/임계값/후보선택금지.이번요청은진단이며새학습/모델교체를자동시작하지않음.기존36회권고를완료결과로쓰지않고이번실제12회만보고.

## 고정 모델·기준·서버
모델 runs/word100_tcn_v1/whole_context/best.pt SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.100단어899학습01–09/300개발11–13,개발96.67%;보류10/14–18미사용.
DTW BASELINE SHA 0af5e66be355ab0a781fc0b7bfcbcee551fd92f9f1cf3bf2097f4c87d34440cf.12표현108학습영상/864사람간비교.손별최소5참고,사용자9참고.차이점수100r/(1+r),training_observed_max9_v1학습관측상한은합격선아님.좌우표시 explicit_target_hands_v1 유지.
연습8765/practice.html(이전확인PID65612),발표8766(이전PID58776).이번발표HTTP/화면만확인.새진단프로세스모두종료.사용자세션파일OPEN/camera:true/trial_ids=[]는실제장치상태/시도수증거아님.사용자촬영종료보고와독립적으로카메라/세션종료API접근하지않음.다음시도도파일·이벤트·메모union으로집계.
말뭉치다운로드/과거학습자동재개없음.Git변경없음.원본과 sources는읽기전용.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
for p in [U/'COMPLETION.json',LAB/'ops/STATE.json']:read(p)
assert unit in context.read_text('utf-8').split('## 8.')[1][:2000]
assert all((LAB/p).exists() for p in [completion['report'],completion['verification'],completion['audit'],completion['public_sensitivity']])
assert all(sha(p)==h for p,h in read(U/'ORIGINAL_GUARD.json').items())
print(json.dumps(dict(status=completion['status'],at_kst=now,documents_synced=True),ensure_ascii=False))
