"""Record the completed local input diagnostic and synchronize handoff documents."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

unit=Path(__file__).resolve().parent
base=unit.parents[1]
root=base.parents[1]
run=base/'runs/rgb_temporal_roi_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
result=read(unit/'RESULT.json')
check=read(unit/'INDEPENDENT_CHECK.json')
raw=read(run/'GEOMETRY_RESULT.json')
assert check['status']=='PASS' and raw['geometry_pass']
assert result['hidden_accepted']==raw['hidden_accepted']['n']==4456
assert result['all21_containment']['proposal']==raw['hidden_accepted']['proposal']['all21']==4452
assert result['natural']['accepted_rgb']==raw['natural_accepted']==check['actual_rgb_crops_redecoded']==78
assert not read(run/'SOURCE_PRESERVATION.json')['changed']
assert read(unit/'SYNTHETIC_CHECK.json')['tests']==7
stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
context=root/'WordsInSilence_PROJECT_CONTEXT.md'
handoff=root/'NEW_CHAT_HANDOFF.md'
state=base/'ops/STATE.json'
before=unit/'before'
before.mkdir(exist_ok=False)
for p in (context,handoff,state): shutil.copy2(p,before/p.name)
text=context.read_text(encoding='utf-8')
header8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
header9='## 9. 다음 단일 작업'
assert text.count(header8)==1 and text.count(header9)==1
text=text.replace(header8,header8+'\n\n'+(unit/'HANDOFF_UPDATE.txt').read_text(encoding='utf-8'),1)
text=text.replace(header9,header9+'\n\n'+(unit/'NEXT_CONTEXT_UPDATE.txt').read_text(encoding='utf-8'),1)
role='- 2026-10-10 RGB 준비 후속: 사용자의 “너의 계획대로 다음 작업 바로 진행해줘”에 따라 같은 로컬 입력 준비 범위에서 `ops/rgb_temporal_roi_20261010`의 시간 기반 손 영역·실제 RGB 확보 도구와 검산을 완료했다. 직전 RGB 도구 Codex 허용의 후속이며 모델 구현·학습·추론으로 확대하지 않았다. 동결계획의 authorization과 RESULT_KO.txt 참조.\n\n'
text=text.replace('## 3. 운영 역할','## 3. 운영 역할\n\n'+role,1)
context.write_text(text,encoding='utf-8')
s=read(state)
s['updated_at']=stamp
s['current_stage']='RGB_TEMPORAL_ROI_INPUT_DIAGNOSTIC_COMPLETE_NOT_DEPLOYED'
s['next_single_action']=result['next']+' Encoder weights and independent paired evaluation videos are currently unavailable in inspected scope; main data path is user Modu KSL annotation application/access. No automatic downloads/model training/product changes.'
s['rgb_temporal_roi_20261010']={**result,'completed_at':stamp,
    'reports':['RESULT_KO.txt','RESULT.json','FROZEN_PLAN.json','INDEPENDENT_CHECK.json','VISUAL_REVIEW.json','NEXT_COMPARISON_CONTRACT.txt'],
    'roles':'Codex local input-tool continuation authorized by user; no external transmission or model implementation.',
    'failure_record':'FIRST_VERIFIER_FAILURE.json: boolean comparator defect corrected in verifier only; candidate executed once.'}
state.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
handoff.write_text(f'''# WordsInSilence 최신 인수인계
갱신: {stamp}. 현재 웹/인식 STATE는 `research/front_keyphase_v1/ops/STATE.json`이다.
시작 때 AGENTS.md, PROJECT_CONTEXT §3·§8·§9, WORKFLOW_HANDOFF.md를 따른다.

## 현재 목표와 마지막 완료
- 목표는 여러 사람의 수어 단어 인식과 이후 구체적 수행 피드백이다. 사용자 자료를 학습/튜닝/후보 선택에 쓰지 않는다.
- 최신 완료: `ops/rgb_temporal_roi_20261010/RESULT_KO.txt`·`RESULT.json`. 한 프레임 관절 누락에서 앞뒤 손 영역만 연결하고 같은 시점의 실제 RGB를 읽는 입력 도구다. 손가락·사진을 합성하거나 원관측을 덮어쓰지 않았다.
- 정상관측 숨김4,694손-프레임 중 정체성 검사4,456통과/238보류. 동일 통과집합에서21점 전체 포함률97.35%(4,338)→99.91%(4,452),+2.56퍼센트포인트. 개선116/후퇴2,후보실패총4. 평균 영역 면적비는 비슷하다. 기존105원영상/113구간/69단어/학습 수행자01·02의 영역 진단이며 단어 인식률/수형정확도/새사람 성능이 아니다.
- 실제 완전손누락 한프레임88개 중78개 실제RGB 확보/10개거절. 과거잘못된손7개모두거절,같은손74/77통과,불확실4개는통과해도불확실로유지. 원래263RGB무효칸 중185칸미보완. 사진의 흐림·손 겹침이 남는다. 추정영역=true/관절관측=false를 분리했다.
- 합성7검사,별도구현4,782영역/거절결정·9,388지표전수,78사진재디코드검산PASS. PTS오차0us,고정442파일불변. 자연13비교표와후퇴2비교표확인. 검산불리언비교오류만기록후수정,후보재실행/조건조정0. 상태INPUT_DIAGNOSTIC_PASS_NOT_DEPLOYED. 모델/운영코드/카메라/다운로드/Git변경없음. 실행종료/진행프로세스없음.
- 출력: `runs/rgb_temporal_roi_v1/`의 HIDDEN_EVENTS/NATURAL_EVENTS/GEOMETRY_RESULT/PIXEL_RECORDS/SOURCE_PINS/SOURCE_PRESERVATION와crops/visual/. ops의계획/검산/육안/완료기록을먼저읽는다.

## 다음 한 작업
`ops/rgb_temporal_roi_20261010/NEXT_COMPARISON_CONTRACT.txt`로 실제RGB가중치·전처리·이용조건과사람분리대응자료를확정한다. 같은자료·학습기회의관절만(A),정상관측RGB결합(B),추정영역RGB추가(C)를비교해RGB자체효과와공백보완효과를구분한다. 기존832는별도기준이다. 모델/학습예산/성공기준은아직미동결이다.

현재로컬사전학습RGB가중치와별도수행자의대응평가영상이조사범위에서미확보다. 자료전에는완료영역실험반복/공백길이확대/계수탐색/근거없는학습으로확대하지않는다. 새모델구현은기존준비도구역할예외에포함되지않는다. 조건을충족할수있는실제자료나명시적승인범위를먼저확인한다.

## 보존한 입력과 자료 조사
- `ops/rgb_pose_readiness_20261010/PAIRED_INPUT_MANIFEST.json`:105영상/113구간/69단어,2,789시간행/5,578손칸. 유효RGB5,315/관측누락191/화면밖72. 이전파일읽기전용. RGB는사용한손만선별한것이아니라관측된각손이다. RGB128,기존좌표/시간/몸대비크기보존.
- `ops/rgb_fusion_feasibility_20261010/`:공백128개/263칸,내부109중95개가1~2프레임. 이전손동일성반례/실패재사용. 표준가중치캐시없음,프로젝트조사에서범용RGB가중치미확인. MobileNetV3Small은후보만이며224전처리/이용조건/가중치확정필요. 다운받지않았다.
- 현재2346학습행/1882원본/93단어. AIHub81단어는확인수행자01~09,나머지12는NIKL시범1개씩/신원미확인. MP113은832·bed학습에포함되지않은별도준비자료다.44단어2명각1구간/25단어1명1구간으로새사람평가부족.
- 기존699/74평가는11/12/13의좌표이고대응RGB경로미확인. 연구폴더비압축NIA127영상은01/02만확인(전체디스크부재주장아님). KSL36은사람미확인,공부/먹다는준비69단어에없음. 기존855조건을855독립영상이나전체정확도로합산하지않는다.
- `ops/data_pretraining_inventory_20261010/`:INVENTORY.html,DATA_INVENTORY,FREQUENCY_TOP100,EXTERNAL_SOURCES,DATA_ACQUISITION_PLAN.txt. 기존AIHub빈도top100은모두의말뭉치빈도나독립사람수아님. ASLCitizen/How2Sign은조건부외부사전학습후보일뿐다운로드/입력변환미완료.

## 자료 확보와 전체 계획
- 사용자확인:모두의말뭉치한국수어아직신청·확보안함. 연구실3TB하드2개는커넥터없어미연결. 장치규격/포트미확인,구매/장치접근/다운로드자동승인아님.
- 주자료경로는한국수어주석+대응영상. 사용자신청후배포설명서/파일목록/주석부터확인하여단어별고유구간/원본/사람수와영상ID/최소압축묶음/공간을집계한다. 양손주석중복/문장번역빈도/연도별동일인혼동금지. 하드전작은주석가능여부는실제배포형식에따른다.
- 하드연결후원본/허용백업→소량추출·시각·의미검증→사람/원본분할→같은모델에서자료추가효과비교. 외부사전학습은별도후속이다.
- 전체계획/멘토후속: `docs/mentoring_audio_20261010/DATA_SELECTION_AND_SPLIT_FOLLOWUP.txt`. 단어당10건/top100/7:2:1·8:2는제안이며일반화보장/확정비율아님. 이름있는동작지도학습과이름없는자기지도사전학습을구분한다.

## 모델과 현장 시험
- 현재TCN832: `runs/partial_encoder_pair_v1/checkpoint_control_final.pt`, SHA83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91.
- 기하bed: `runs/tcn_geometry_pair_v1/checkpoint_treatment_final.pt`, SHAbed406d931cb5a747ae55f347a42c91a9ea55a38798dd8fbb2f3cdfed5a6ae36. `ops/tcn_geometry_pair_20261008/RESULT_KO.md`:832→bed691/699유지(98.86%),KSL22/34→23/34(64.71→67.65%);먹다개선은대조군도맞혀기하만의효과아님. bed웹미연결/운영기본불변. STGCNlocalshape662/699·8/34미채택. 완료실험반복금지.
- 멘토2명5회체험은저장됐지만정답확인0/사람ID혼재. `docs/mentoring_audio_20261010/SESSION_RECORD_REVIEW_KO.txt`·JSON. ses_33578ccdf974,5회연구모델union93/e7c9c31a로현재832·bed아님. 새사람정확도/수형정오답집계불가. 사용자자료재학습/후보선택금지.
- 웹DTW/task_match는구현됐으나수행합격/거절기준UNCONFIGURED. top1차이는목표단어로인식되지않음이고top2/3은경미한오류의증거아님. 다음정식시험의모델832/사람별ID/실제표현확인을명확히한다. UI선택과수행판정은별도후속.
- 멘토녹음37분38초로컬전사/중요4구간재전사완료. `docs/mentoring_audio_20261010/MENTORING_REVIEW.html`, `output/pdf/WordsInSilence_멘토링_피드백과_실행계획_20261010.pdf`. 자동전사불확실성유지/음성외부전송0. 반복전사불필요.

## 역할 및 실행 상태
- 기본Claude구현/Codex실행·검증·문서. 사용자10/10RGB준비도구Codex허용및이번후속진행으로입력도구를로컬구현했다. 모델구현·학습으로확대하지않는다. 외부Claude명세전송자동검토거절은이전기록에보존/전송재시도없음.
- 카메라/장치/자동다운로드/사용자자료학습/운영기본교체/Git변경금지. 루트Ops/STATE는옛Unity계보이며현재웹STATE와다르다.
- 이번모든실행종료. 서버/카메라시작없음. 예전웹8765생존여부미확인. st-gcn heartbeat는과거완료후PAUSED.
- 최신STATE키rgb_temporal_roi_20261010. 이전rgb_fusion_feasibility와rgb_pose_readiness는완료근거로재사용한다.
''',encoding='utf-8')
for p in (state,unit/'RESULT.json',unit/'INDEPENDENT_CHECK.json'): read(p)
assert 'rgb_temporal_roi_20261010' in context.read_text(encoding='utf-8')
assert 'rgb_temporal_roi_20261010' in handoff.read_text(encoding='utf-8')
evidence=['FROZEN_PLAN.json','RESULT.json','RESULT_KO.txt','INDEPENDENT_CHECK.json','VISUAL_REVIEW.json','SYNTHETIC_CHECK.json','NEXT_COMPARISON_CONTRACT.txt']
assert all((unit/p).is_file() for p in evidence)
completion={'status':'COMPLETE_INPUT_DIAGNOSTIC_NOT_DEPLOYED','completed_at':stamp,
 'evidence':{p:hashlib.sha256((unit/p).read_bytes()).hexdigest() for p in evidence},
 'documents_updated':[str(context),str(state),str(handoff)],'active_processes':[],
 'criteria_pass':True,'word_accuracy_claim':False,'source_preservation_files':442,
 'no_model_camera_download_product_git_changes':True}
with (unit/'COMPLETION_CHECK.json').open('x',encoding='utf-8') as f: json.dump(completion,f,indent=2,ensure_ascii=False)
print(json.dumps({'status':completion['status'],'completed_at':stamp,'documents':3}))
