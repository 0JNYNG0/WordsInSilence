"""Record verified input-preparation completion, leaving implementation decisions explicit."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import hashlib

UNIT=Path(__file__).resolve().parent
BASE=UNIT.parents[1]
ROOT=BASE.parents[1]
OUT=BASE/'runs/rgb_tensor_adapter_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    build=read(OUT/'BUILD_CHECK.json'); check=read(UNIT/'INDEPENDENT_CHECK.json')
    synth=read(UNIT/'SYNTHETIC_CHECK.json'); visual=read(UNIT/'VISUAL_REVIEW.json')
    assert all(r['status']=='PASS' for r in [build,check,synth,visual])
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    result={'status':'RGB_TENSOR_INPUT_VERIFIED_AWAITING_MODEL_IMPLEMENTATION_ROLE',
            'completed_at':now,'counts':build['counts'],'segments':113,'videos':105,
            'synthetic_checks':14,'independent_check':check,'visual_review':visual,
            'cache_bytes':sum(p.stat().st_size for p in OUT.glob('*.npz')),
            'training':0,'model_inference':0,'downloads':0,'web_changes':0,
            'next_role_question':'Pending user answer for Codex local implementation of isolated RGB fusion component and explicit web research model selection. Previous role exception covered input tools only.',
            'weight_question':'Pending user permission for official approximately10MB MobileNetV3-Small weights; no download.',
            'source_or_private_media_external_transmission':False,
            'active_processes':[]}
    write(UNIT/'RESULT.json',result)
    report=f'''RGB 입력 처리 구현·전수 검산 완료
완료: {now}

무엇을 완료했나
- 기존105영상/113구간의 동일 손 영역에서224 RGB를 직접 만들었다. 이전128사진을 확대하지 않았다.
- 2,789시간행/5,578손칸. 정상5,315 + 기존 추정영역78 = 5,393장. 남은185칸은 사진없음으로 보존.
- 사진이 없는 칸은 인코더에 넣을 목록에서 제외한다. 실제 검은 사진과 구분한다.
- 읽은 사진의 특징을 원래 시간·좌우로 돌려놓는 pack/scatter 인터페이스를 구현했다.
- 관절280개 입력·시간·좌우·누락 표시·원래 손크기 정보를 보존했다.
- 추정영역 사진을 관절검출 성공으로 바꾸지 않았으며 이전 불확실4건을 재판정하지 않았다.

검증
- 합성14검사 PASS: 좌우/색상/누락/검은사진/시간/패딩/출처/해시/특징배치/기울기.
- 별도코드로5,393장 모두 원영상 픽셀과 대조, 2,789시간행 대조. 정규화 최대오차 {check['normalization_max_abs_error']}.
- 정상관측5,315장의 검출21점이 저장영역 안에 있음. 실제해부학/가린손/추정78장의 수형정답을 보증하지 않음.
- 관련 입력·명세523파일 해시 유지. 직접 원영상 crop과 nearest 패딩 마스크를 독립 재구성했다.
- 같은Codex가 작성한 별도코드 검산이며 외부사람/기관검증은 아니다. 시각 검토는 별도 VISUAL_REVIEW.json.
- 파생 캐시 {result['cache_bytes']/2**20:.1f}MiB. 원본/이전실험 미변경.

아직 하지 않은 것
RGB인코더가중치다운로드0, 실제모델추론0, 학습0, 웹변경0. 정확도향상/새사람성능/수형정답평가가 아니다.

다음
사용자가 승인한 전체 진행방향은 입력준비→격리관절+RGB연결→웹시험명시선택이다.
입력준비는 완료. 모델/웹 구현 담당의 좁은 Codex 예외와 공식가중치 다운로드는 비동기 질문 답변 대기.
NEXT_IMPLEMENTATION_SCOPE.txt와 FUSION_INTERFACE_REVIEW.txt에 구체 변경/검사 범위를 준비했다.
모두의말뭉치 자료 수신 전에도 후속 연결검사는 가능하다. 새사람 성능비교는 별도 자료조건이 필요하다.
'''
    (UNIT/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'; handoff=ROOT/'NEW_CHAT_HANDOFF.md'; statefile=BASE/'ops/STATE.json'
    before=UNIT/'before'; before.mkdir(exist_ok=False)
    for path in [context,handoff,statefile]: (before/path.name).write_bytes(path.read_bytes())
    text=context.read_text(encoding='utf-8-sig')
    assert '### 2026-10-10 RGB224 모델 입력 처리 구현' not in text
    role='- 2026-10-10 사용자 “계획대로 진행해줘”: 새 말뭉치 대기 중 RGB 입력 준비를 기존 Codex 로컬 역할로 이어 `ops/rgb_tensor_adapter_20261010`까지 완료했다. 모델 결합부와 웹 코드의 Codex 구현 역할 확대 및 공식 RGB 가중치 다운로드는 비동기 질문 답변 대기이며 실행하지 않았다.\n\n'
    text=text.replace('## 3. 운영 역할\n\n','## 3. 운영 역할\n\n'+role,1)
    status='''### 2026-10-10 RGB224 모델 입력 처리 구현·전수 검산 완료 — 모델/웹 후속 역할 확인 대기

`ops/rgb_tensor_adapter_20261010/RESULT_KO.txt`·`INDEPENDENT_CHECK.json`이 최신이다. 원영상105개/113구간에서 기존동결영역 그대로224입력을생성했다. 2,789행/5,578손칸 중 정상5,315+추정78=5,393사진;185누락유지. 기존128확대/새검출/새손배정아님. 기존280관절·시각·좌우·몸대비손크기/출처보존,추정영역과관절관측분리,이전불확실4유지.

인코더입력목록은사용가능사진만포함하며학습특징을원래T/좌우로배치하는pack/scatter를구현했다. 합성14검사·별도코드5,393사진/2,789시각전수·정규화검산PASS. 정상사진5,315의검출21점이저장영역안에있음. 관련523파일해시보존. 입력코드동작의검증이며실제수형정확도/새사람성능/단어정답률이아니다. RGB추론·학습·다운로드·웹변경0.

다음모델/웹코드역할예외및약10MB가중치확보는사용자질문답변대기. 구체범위 NEXT_IMPLEMENTATION_SCOPE/FUSION_INTERFACE_REVIEW 참조. 입력준비역할을자동확대하지않는다. 현재실행종료.

'''
    text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+status,1)
    nexttext='''### 2026-10-10 다음 — 자료 대기 중 격리 RGB 결합·웹 시험 선택 구현

사용자는 새말뭉치수신전에도 진행하도록 승인했다. 입력준비는완료했으므로재실행하지않는다. `ops/rgb_tensor_adapter_20261010/NEXT_IMPLEMENTATION_SCOPE.txt`와 `FUSION_INTERFACE_REVIEW.txt`로 이어간다. 비동기질문 두개의 사용자답변부터 확인: (1)새 관절+RGB 부품 및 웹 명시선택 코드에한한 Codex로컬구현 역할예외, (2)공식 MobileNetV3-Small약10MB다운로드. 답변없음을승인으로간주하지않는다. Claude auth상태는로그인중이나이전프로젝트명세외부전송거절을자동재시도하지않았다.

역할확정후실제TCN280→64투영/시간처리를보존하는격리RGB연결및합성연결검사,연구체크시모델명시선택·기존촬영잠금/결과ID보존을수행한다. 현재운영기본·동결모델·학습결과는불변. 새말뭉치신청은병행하며새사람정확도평가와혼동하지않는다. 학습/카메라/장치/사용자자료튜닝/Git변경없음.

'''
    text=text.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+nexttext,1)
    context.write_text(text,encoding='utf-8')
    state=read(statefile)
    state['current_stage']=result['status']; state['updated_date_kst']='2026-10-10'
    state['rgb_tensor_adapter_20261010']={**result,'artifact_root':str(UNIT.relative_to(ROOT)).replace('\\','/'),'output_root':str(OUT.relative_to(ROOT)).replace('\\','/')}
    state['next_single_action']='Resolve pending user answers for local Codex RGB fusion/web explicit-selection implementation role and official approximately10MB RGB weights. Reuse completed RGB224 adapter and5393 verified caches, do not regenerate. Follow NEXT_IMPLEMENTATION_SCOPE and FUSION_INTERFACE_REVIEW for isolated connection tests, no comparison training. Modu application remains a parallel data path, not a prerequisite for code integration checks.'
    statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    h=handoff.read_text(encoding='utf-8-sig')
    lines=h.splitlines(); lines[1]=f'갱신: {now}. 현재 웹/인식 STATE는 `research/front_keyphase_v1/ops/STATE.json`이다.'; h='\n'.join(lines)+'\n'
    new='- 최신 완료: `ops/rgb_tensor_adapter_20261010/RESULT_KO.txt`. 기존105영상/113구간에서원영상직접224입력5,393장(정상5,315+추정78)과pack/scatter구현. 합성14·별도전수사진/시각/정규화검산PASS/523입력명세파일불변. 남은185칸미관측·불확실4유지. 새추출/학습/모델추론/웹변경/다운로드0. 현재모든실행종료.\n'
    h=h.replace('## 현재 목표와 마지막 완료\n','## 현재 목표와 마지막 완료\n'+new,1)
    a=h.index('## 다음 한 작업\n'); b=h.index('## 보존한 입력과 자료 조사',a)
    h=h[:a]+'''## 다음 한 작업
새말뭉치대기중에도격리RGB결합부/웹연구모델명시선택까지진행하는사용자계획승인. 입력준비는완료했으므로반복금지.
먼저비동기질문답변확인: 이번모델·웹두구현의Codex역할예외, 공식RGB가중치약10MB다운로드. 아직답변없으며모델/웹구현시작하지않음. NEXT_IMPLEMENTATION_SCOPE.txt 및 FUSION_INTERFACE_REVIEW.txt에범위/검사준비. 기존Claude구현원칙과입력도구역할예외를혼동하지않는다.
모두의말뭉치2024한국어-한국수어병렬설명서/신청은병행. 새사람실자료비교는실제사람/어휘/시각/RGB연결과분할확인후다. 현재연결준비를정확도향상이라고하지않는다.

'''+h[b:]
    h=h.replace('- 최신STATE키rgb_comparison_contract_20261010.', '- 최신STATE키rgb_tensor_adapter_20261010. rgb_comparison_contract는이전입력색인근거.')
    handoff.write_text(h,encoding='utf-8')
    assert read(statefile)['current_stage']==result['status']
    for p in [UNIT/'RESULT_KO.txt',UNIT/'USAGE.txt',OUT/'CACHE_MANIFEST.json',OUT/'VISUAL_CHECK.png']: assert p.is_file()
    write(UNIT/'COMPLETION_CHECK.json',{'status':'PASS','updated_at':now,'documents':[str(p) for p in [context,handoff,statefile]],
                                      'result_sha256':sha(UNIT/'RESULT.json'),'remaining_questions_pending':True})
    print(json.dumps({'status':'COMPLETE_INPUT_UNIT_ONLY','cache_MiB':result['cache_bytes']/2**20}))


if __name__=='__main__': main()
