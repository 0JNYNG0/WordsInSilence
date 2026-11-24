"""Check saved evidence and synchronize current project handoff; no model execution."""
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
assert ROOT.name == 'WordsInSilence'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
bench = read(U/'BENCH_RESULT.json')
v = read(U/'BENCH_INDEPENDENT_VERIFY.json')
obs = read(U/'OBSERVATION_TEST_RESULT_POST_REVIEW.json')
assert v['status']=='PASS' and v['checks']==186 and not v['failed']
assert sha(U/'BENCH_RESULT.json')==v['result_sha256']
assert obs['corrected_execution']['tests_run']==25 and obs['corrected_execution']['exit_code']==0
for filename,h in obs['tested_code_sha256'].items():
    assert sha(U/filename)==h,filename
assert sha(U/obs['prior_self_check_preserved']['path'])==obs['prior_self_check_preserved']['sha256']
review = (U/'OBSERVATION_INDEPENDENT_REVIEW.txt').read_text(encoding='utf-8-sig')
assert 'PASS_AFTER_FIX' in review
for p,h in read(U/'FROZEN.json')['source_pins'].items():
    assert sha(Path(p))==h,p

class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,val in attrs:
            if k in ('href','src') and val and not val.startswith(('http:','https:','#','data:')):
                self.links.append(val)
parser=Links(); parser.feed((U/'RESULT_KO.html').read_text(encoding='utf-8'))
assert parser.links
for link in parser.links:
    assert (U/link).is_file(),link
now=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
next_action = ('작은 공개 KSL 정답 묶음의 출처·확실한 항목·수행자 분리를 갖춘 뒤, 같은 자료에서 관절/관측상태만 쓰는 작은 시간별 판정부와 저장 SHuBERT 특징 추가 조건을 비교할 계약을 고정한다. '
               '속도와 실제 오류/잘못된 교정을 함께 판단한다. 현재 FP32/INT8 SHuBERT 경로는 제품 미채택. INT8 특징 변화가 커 품질유지 주장 금지. '
               'Intel 가속/15fps/얼굴 제외는 미실행 별도조건; 새 다운로드 자동실행 금지. C/T3F/개인자료 보존.')
summary = ('공개 고정3영상×2회×2조건 총12회 CPU비교완료. 같은30fps/224RGB3스트림·FP32SHuBERT를 유지하고 DINO Linear만 동적INT8로 변환하자 cachedRGB→특징 중앙15.3278→7.1595초(53.29%감소), '
           '개별INT8 6.7161~9.4263초. 최종특징 상대L2 중앙31.62~44.60%, cosine최소0.4137로 변화커서 품질유지/제품채택불가. '
           '총촬영후피드백대기·한국수어정답률미측정. 독립벤치186검사PASS. '
           '관측상태입력·partial-label loss 구현:12영상1438프레임, 손현재2764/이전복사112/blank0. '
           '초기22PASS후독립반례에서잘못된배치broadcast와몸/얼굴상태입력누락을발견·수정;최종25검사PASS+별도반례/수식/공개3검산PASS. '
           '손별108관측+768저장특징=876입력이나 학습판정부는아직없고기존특징의과거RGB문맥영향은미해결. '
           'KSL학습0/새다운로드0/사용자·멘토접근0/운영변경0. exec63880종료0,진행실행없음.')
stage='LATENCY_INPUT_GATE_COMPLETE_SPEED_GAIN_WITH_FEATURE_DRIFT_NO_DEPLOYMENT'
evidence=['FROZEN.json','BENCH_RESULT.json','BENCH_INDEPENDENT_VERIFY.json','OBSERVATION_TEST_RESULT.json','OBSERVATION_TEST_RESULT_POST_REVIEW.json','OBSERVATION_POST_REVIEW_TEST_LOG.txt','OBSERVATION_INDEPENDENT_REVIEW.txt','RESULT_KO.txt','RESULT_KO.html','observation_input.py','test_observation_input.py']
audit={'at':now,'status':'PASS_SAVED_EVIDENCE_AND_STATIC_REPORT_NOT_SEMANTIC_QUALITY','bench_checks':186,'observation_final_tests':25,'initial_observation_tests_preserved':22,'independent_observation_review':'PASS_AFTER_FIX','local_report_links':len(parser.links),'actual_browser_ui_test':False,'training_runs':0,'new_downloads':0,'user_mentor_access':False,'operating_changes':False,'frozen_sources_unchanged':True,'evidence_sha256':{f:sha(U/f) for f in evidence}}
(U/'FINAL_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(U/'CURRENT_EXECUTION.json').write_text(json.dumps({'at':now,'stage':'COMPLETE_REPORTED_EVIDENCE_READY','active_inference':False,'active_workers':[],'completed_exec_session':63880,'completed_exit_code':0,'training_runs':0,'unit_result':stage,'next_single_action':next_action},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=ROOT/'research/front_keyphase_v1/ops/STATE.json'
state=read(state_path)
state['current_stage']=stage;state['updated_at']=now;state['updated_date_kst']=now;state['next_single_action']=next_action
state[U.name]={'at':now,'status':stage,'unit':str(U),'summary':summary,'median_seconds':bench['summary'],'training_runs':0,'private_access':False,'operating_changes':False,'new_downloads':0,'independent_bench_checks':186,'observation_checks':25,'input_dimensions':{'observation':108,'saved_shubert':768,'joined':876},'report':str(U/'RESULT_KO.html'),'audit':str(U/'FINAL_AUDIT.json'),'next_single_action':next_action}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context_path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
context=context_path.read_text(encoding='utf-8-sig')
title='### 2026-10-28 CPU 속도·관측 입력 검증 완료 — INT8 속도 개선, 품질 미확인으로 미채택'
assert title not in context
ref=f'근거: research/front_keyphase_v1/ops/{U.name}/RESULT_KO.txt, BENCH_INDEPENDENT_VERIFY.json, OBSERVATION_TEST_RESULT_POST_REVIEW.json, OBSERVATION_INDEPENDENT_REVIEW.txt, FINAL_AUDIT.json.'
for heading,txt in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    assert context.count(heading)==1
    context=context.replace(heading,heading+'\n\n'+title+'\n'+now+'\n'+txt+'\n'+ref+'\n',1)
context_path.write_text(context,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

{now}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: research/front_keyphase_v1/ops/{U.name}
현재 상태: {stage}

{summary}

사용자 최신 요청: 다음 작업 진행, 14~16초 피드백 대기를 큰 문제로 보고 더 가벼우면서 성능 좋은 경로를 검토. 현재 14~18초 및 INT8 6.7~9.4초는 준비된RGB→특징 구간이며 총피드백대기 아님. 3초선호,빠를수록좋음. SHuBERT제품최적모델확정근거없음. 수어사전학습된연구비교후보로만유지. 새모델검색/다운로드순회로돌아가지않는다.

다음 한 작업: {next_action}

관측부품: observation_input.py/test_observation_input.py. v2는관측108+저장특징768=876. source current/previous/none+실제elapsed time,각손점/손가락geometryvalid,얼굴/몸출처전달. 센서결측과확정정답결측별개;항목별loss/padding검산. 모든공개자료의실제정답확보/학습완료아님. 좌우slot독립확정없음,visiblepoints도정답관절보장아님,storedSHuBERT이전RGB문맥을상태입력만으로복원했다는주장금지.

기존 KSL 비교: word100_shubert_ksl_probe_20261028. 공개4표현·3기록수행자12영상,모두개발자료. AI초안1080중711합의/5불일치/364unknown은독립사람GT아님. 활동공통132 기존120/SHuBERT118,움직임38중34/32;참고범위173 기존126/11/36 vsSHuBERT126/13/34. shape적격21,공통판독2뿐. nativefullframe관측이며웹상반신조건아님. KSLFT0;동결특징거리우위미확인이지FT불가능증명아님.

환경: Samsung960XHA,Ultra7 258V,32GB,IntelArc140V내장GPU/NPU. 이번Torch2.12CPU4threads/batch8/onednn. GPU있지만XPUfalse,OpenVINO/ORT/DirectML미설치. 가속성능미검증. 눈/입표정목표제외,손-얼굴부위관계유지;얼굴RGB생략이무해하다고확정하지않는다.

완료실행: exec63880 exit0. 현재추론·학습·다운로드·자동후속없음. BENCH_RUNNING은BENCH_FINISHED로전환. 벤치최초실패없음. 입력독립반례및수정기록보존. 코드자체검사/별도에이전트검산은수어전문가정답검증아님.

담당: root CPU벤치/보고/문서; observation_input_builder 구현·25시험; latency_quality_reviewer 186저장검산; latency_path_audit 환경감사/반례·공개3검산. C/T3F·운영웹·DTW·원본W:/sources/이전동결읽기전용. 개인/멘토접근·학습·튜닝·선택금지,외부전송/Git/임의모델다운로드금지.

{ref}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(state_path)['current_stage']==stage
assert title in context_path.read_text(encoding='utf-8')
assert stage in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
print(json.dumps({'status':'PASS','synced':['PROJECT_CONTEXT §8/§9','STATE','NEW_CHAT_HANDOFF'],'report_local_links':len(parser.links)},ensure_ascii=False))
