"""Research documentation only. No model/data imports, downloads or experiments."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import html
import re
import shutil

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
KEY = UNIT.name
STATE = ROOT / 'research/front_keyphase_v1/ops/STATE.json'
CONTEXT = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
HANDOFF = ROOT / 'NEW_CHAT_HANDOFF.md'
OLD = UNIT.parent / 'word100_lightweight_sign_model_review_20261028'
STATUS = 'PRETRAINED_LITERATURE_REVIEW_COMPLETE_TWO_RESEARCH_CANDIDATES_NO_EXECUTION'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jwrite(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def render(text):
    blocks=[]
    for block in text.split('\n\n'):
        safe=html.escape(block)
        safe=re.sub(r'https://[^\s<>]+', lambda m:'<a href="'+m.group(0)+'">'+m.group(0)+'</a>', safe)
        blocks.append('<p>'+safe.replace('\n','<br>')+'</p>')
    return '\n'.join(blocks)

def main():
    assert ROOT.name == 'WordsInSilence', ROOT
    source_names=['ROOT_SOURCE_LEDGER.json','SIGN_SOURCE_LEDGER.json','TEMPORAL_FEEDBACK_LEDGER.json','VIDEO_BACKBONES_LEDGER.json']
    for name in source_names:
        json.loads((UNIT/name).read_text(encoding='utf-8-sig'))
    for name in ['ROOT_REVIEW.txt','SIGN_PRETRAINED.txt','TEMPORAL_FEEDBACK.txt','VIDEO_BACKBONES.txt','FINAL_REVIEW.txt']:
        assert (UNIT/name).is_file(), name
    before=UNIT/'before'
    before.mkdir(exist_ok=True)
    for path in [STATE,CONTEXT,HANDOFF]:
        saved=before/path.name
        if not saved.exists():
            shutil.copy2(path,saved)
    old_before={str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
    summary=('사용자의 재조사 요청에 따라 수어 사전학습·시간 정렬/오류 피드백·경량 영상 모델을 '
             '3개 별도 에이전트와 넓혀 검토. 문헌상 목표 적합성 후보 SignRep와 시간별 출력 비교 후보 SEN으로 제한. '
             '둘의 KSL 교정 우열·Intel 전체 지연은 미검증이며 최종 제품 채택 아님. '
             'MoViNet-A0-Stream은 CPU 비용 근거가 있는 예비안으로 보존. '
             'SignRep의 pooled16프레임 표현/재구성 예측·stride2 중복비용·비상업 사용조건, '
             'SEN 배포 링크와 실제 파일 확인의 차이·새 속성 감독 필요를 기록. '
             '가이드+정상 참고만으로 교육적 허용경계를 확정하지 않음. 새 다운로드/설치/추론/학습/개인자료접근/운영변경0.')
    next_action=('이번 두 연구 후보의 문헌·코드 검토 결과를 사용자와 확인한다. '
                 '확인 후 기존 공개 KSL의 시간별 손모양·관계·움직임 정답과 새 수행자 평가 확보 가능성, '
                 '후보의 입력/출력·가중치 조건·Intel 실행 연결·전체 지연 비교 계약을 고정한다. '
                 '다운로드와 학습은 사용자 확인 뒤에 진행한다. 반복 열람한12영상은 개발자료로 유지하고 '
                 '독립 최종평가처럼 재사용하지 않는다. C/T3-F·운영웹/DTW·개인자료 보존.')
    text=(UNIT/'ROOT_REVIEW.txt').read_text(encoding='utf-8')
    (UNIT/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    overview='''<div class="lead"><b>현재 결론: 사전학습 모델을 가져와 발전시킬 선택지는 있다.</b><p>SignRep와 SEN을 연구 후보로 좁혔습니다. 실제 한국수어 교정 성능과 노트북 속도를 확인한 최종 채택은 아닙니다.</p></div>
<div class="cards"><div><h2>SignRep</h2><p>손 관절·굽힘·손과 몸의 관계를 배운 수어 RGB 표현.</p><b>목표 적합성의 우선 후보</b><p>속도 미확인 · 짧은 구간 요약 · 비상업 조건</p></div><div><h2>SEN</h2><p>수어 영상의 시간별 특징을 꺼내 새 판정부에 연결.</p><b>시간별 출력 구조의 비교 후보</b><p>속도 미확인 · 새 한국수어 속성 학습 필요</p></div></div>
<p class="notice">이번에는 논문·공식 코드·배포 정보만 읽었습니다. 다운로드·설치·추론·학습·기존 모델 변경은 하지 않았습니다.</p>
<div class="flow">공개 영상의 실제 특징 → 어느 손의 무엇이 어떻게 변했는지 학습 → 가이드·여러 참고와 같은 단계 비교 → 확인된 중요한 차이만 피드백</div>'''
    details=[]
    for file,title in [('SIGN_PRETRAINED.txt','수어 사전학습 모델 조사'),('TEMPORAL_FEEDBACK.txt','시간 정렬·단계·오류 분석 조사'),('VIDEO_BACKBONES.txt','경량 영상 모델 조사'),('FINAL_REVIEW.txt','별도 에이전트의 반대 검토')]:
        details.append('<details><summary>'+title+'</summary>'+render((UNIT/file).read_text(encoding='utf-8-sig'))+'</details>')
    links=' · '.join('<a href="'+name+'">'+name+'</a>' for name in source_names)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>수어 피드백 모델 — 근거 재검토</title><style>
body{font-family:system-ui,"Malgun Gothic",sans-serif;margin:0;background:#eef2f6;color:#172938;line-height:1.8}main{max-width:1040px;margin:35px auto;padding:30px;background:white;border-radius:18px}h1{font-size:30px;line-height:1.35}h2{margin-top:0}p{overflow-wrap:anywhere}.lead{background:#e8f3ee;padding:22px;border-left:5px solid #2b7358;border-radius:8px;font-size:18px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0}.cards>div{border:1px solid #bacbd8;border-radius:12px;padding:22px}.notice{background:#fff4db;padding:16px;border-radius:8px}.flow{padding:20px;background:#edf3fb;color:#284c75;border-radius:10px;margin:24px 0}a{color:#145a9b}details{border-top:1px solid #cad6df;padding:20px 0}summary{cursor:pointer;font-weight:bold;font-size:19px}.report{margin-top:35px}.meta{color:#506675;font-size:14px}footer{border-top:1px solid #cdd6df;margin-top:30px;padding-top:20px}@media(max-width:700px){main{margin:0;padding:20px}.cards{grid-template-columns:1fr}h1{font-size:25px}}
</style><main><h1>시험 전에, 어떤 모델을 고를지 다시 검토했습니다</h1>'''+ '<p class="meta">'+stamp+' · 문헌·코드 조사 / 실제 모델 성능시험 아님</p>'+overview+'<section class="report">'+render(text)+'</section>'+''.join(details)+'<footer>조사 출처와 확인 수준: '+links+'<p>공식 링크를 확인한 경우와 실제 파일을 받은 경우를 구분했습니다. 별도 에이전트 검토는 수어 전문가 평가나 모델 재현 실험이 아닙니다.</p></footer></main></html>'
    (UNIT/'RESULT_KO.html').write_text(page,encoding='utf-8')
    context=CONTEXT.read_text(encoding='utf-8-sig')
    heading='### 2026-10-28 사전학습 모델 확대 재검토 — SignRep·SEN 연구 후보, 실행 보류'
    evidence=f'근거: research/front_keyphase_v1/ops/{KEY}/RESULT_KO.txt, FINAL_REVIEW.txt, ROOT_SOURCE_LEDGER.json 및 분야별 조사 보고서.\n'
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
        assert marker in context
        start=context.index(marker)+len(marker)
        end=context.find('\n## ',start)
        section=context[start:end if end>=0 else len(context)]
        if heading not in section:
            context=context[:start]+'\n\n'+heading+'\n'+stamp+'\n'+body+'\n'+evidence+context[start:]
    CONTEXT.write_text(context,encoding='utf-8')
    state=json.loads(STATE.read_text(encoding='utf-8-sig'))
    state.update(updated_at=stamp,current_stage=STATUS,next_single_action=next_action)
    state[KEY]={'at':stamp,'status':STATUS,'unit':str(UNIT),'summary':summary,'recommendation':['SignRep: research fit candidate','SEN_CSLR: temporal-output comparator'],'cpu_fallback':'MoViNet-A0-Stream; not rejected as slow','selection_claim':'Literature-prioritized shortlist; not demonstrated KSL quality or laptop latency winner','new_downloads':0,'installations':0,'inference_runs':0,'training_runs':0,'private_access':False,'operating_changes':False,'report':str(UNIT/'RESULT_KO.html'),'independent_review':str(UNIT/'FINAL_REVIEW.txt'),'next_single_action':next_action,'running_execution':None}
    jwrite(STATE,state)
    handoff=f'''# WordsInSilence 최신 인수인계

{stamp}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: research/front_keyphase_v1/ops/{KEY}
현재 상태: {STATUS}

{summary}

최신 사용자 지시: 100편은 할당량이 아니라 충분한 탐색을 의미한다. 무턱대고 모델을 받아 시험하지 말고 논문·공식 코드·가중치 근거로 프로젝트에 맞는 후보를 먼저 찾는다. 이전 작은TCN/SLGCN만의 제한 결론을 이번 확대 조사로 갱신했다. 이번 턴 실행 승인은 없으며 이후 다운로드·학습은 사용자 확인 뒤다.

{next_action}

핵심 주의: SignRep212MB는 공식 release 표시값·hash이며 파일 다운로드/로드 검증 아님. 16프레임 pooled768에서 재구성한16시점 예측을 직접 관측이나 의미 단계 정답으로 쓰지 않는다. stride2는120프레임에서53창·848frame-visits; 실제 지연 수치 아님. CC BY-NC-SA4.0 연구 후보. SEN은 학습 완료 CSLR 링크와 framewise/visual_features 재사용 경로 확인, 파일 접근·크기·Intel속도 미확인. 오래된 모델이라 무조건 우선하거나 새 모델이라 최적이라고 단정하지 않는다. 가이드·정상자료만으로 허용경계 확정 불가. 개인자료 모델선택 금지.

유지 근거: SHuBERT15.3278→7.1595초는 cachedRGB→특징의 DINO Linear INT8 비교이며 전체 대기 아님, 품질 미확인. 특징 L2변화31.62~44.60%는 정확도하락률 아님. 공개12영상 probe는 개발자료, 추가FT없음. 운영 C/T3-F와 실제 웹은 그대로.

환경: Samsung960XHA, Ultra7 258V, RAM32GB, IntelArc140V/NPU. TorchCPU 현재, XPU/OpenVINO 현 모델 실측 없음. GPU없음으로 표현하지 않는다. 새사용자 전처리·모델·비교·UI 지연은 아직 확인하지 않음.

담당: root 종합/SignRep·UniSign·WebSMILE/문서동기화, latency_path_audit 수어 사전학습 조사, latency_quality_reviewer 시간/오류 모델 및 최종독립검토, observation_input_builder 영상백본·SignRep 코드 반대검토. 별도 에이전트 검토는 사람 수어전문가GT와 다르다. 도구 용량 오류로 끝난 path_audit의 마지막 턴 이전 보고/ledger는 저장 확인; 전체 최종검토는 quality_reviewer가 수행.

산출물: RESULT_KO.html/txt, ROOT_REVIEW.txt, FINAL_REVIEW.txt, ROOT_SOURCE_LEDGER.json, SIGN_PRETRAINED.txt/SIGN_SOURCE_LEDGER.json, TEMPORAL_FEEDBACK.txt/TEMPORAL_FEEDBACK_LEDGER.json, VIDEO_BACKBONES.txt/VIDEO_BACKBONES_LEDGER.json, FINAL_AUDIT.json.

보호: 새다운로드/설치/모델import/추론/학습/개인·멘토촬영접근/카메라/외부전송/Git/운영변경0. 이번 변경은 새 조사문서와 §8/§9·현재STATE·본인수인계만. 원본W:/sources/과거동결결과 읽기전용. 진행중모델실행·다운로드·자동후속없음.
'''
    HANDOFF.write_text(handoff,encoding='utf-8')
    old_after={str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
    assert old_before==old_after,'Previous report changed'
    assert STATUS in json.loads(STATE.read_text(encoding='utf-8'))['current_stage']
    assert heading in CONTEXT.read_text(encoding='utf-8')
    assert KEY in HANDOFF.read_text(encoding='utf-8')
    audit={'at':stamp,'status':'DOCUMENT_AND_SOURCE_LEDGER_CHECKS_PASS','checks':{'four_source_ledgers_parse':True,'independent_review_exists':True,'result_html_txt_exist':True,'context_latest_status_and_next_action_updated':True,'state_handoff_updated':True,'previous_review_unit_unchanged_during_finalize':old_before==old_after},'previous_review_files':old_after,'scope':'Documentation validation only, not model tests or source reproduction','artifacts':{p.name:sha(p) for p in UNIT.iterdir() if p.is_file() and p.name!='FINAL_AUDIT.json'},'execution_counts':{'downloads':0,'installations':0,'model_inference':0,'model_training':0}}
    jwrite(UNIT/'FINAL_AUDIT.json',audit)
    print(json.dumps({'status':audit['status'],'report':str(UNIT/'RESULT_KO.html'),'previous_review_file_count':len(old_after)},ensure_ascii=False))

if __name__=='__main__':
    main()
