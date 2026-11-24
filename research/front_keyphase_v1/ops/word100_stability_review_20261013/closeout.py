from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys
import json
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_training_20261013'))
from common import ROOT,BASE,read,write,sha

def main():
    assert not (U/'COMPLETION.json').exists()
    check=read(U/'INDEPENDENT_CHECK.json');assert check['status']=='PASS'
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    next_action='후속 재개 시 같은 TCN의 epoch별 원통계/학습자료 통계재추정 대조 계획을 먼저 고정한다. 3개 사전 초기값의 전체 결과·촬영자별 회귀·선택 안정성을 보고하며 최고seed만 고르지 않는다. 이후 오답10개 원영상/관절/표현 대조. 이번 요청은 검토 보고까지이며 새 학습은 미실행.'
    summary=dict(status='COMPLETE_VERIFIED_DIAGNOSTIC_KEEP_CURRENT_MODEL',at_kst=now,
        unit='ops/word100_stability_review_20261013',report=str(U/'RESULT_KO.txt'),review_dimensions=10,
        current_selected_before=290,current_selected_population=289,current_last_before=261,current_last_population=288,
        speed_selected_before=284,speed_selected_population=286,speed_last_before=261,speed_last_population=287,total=300,
        finding='Matched training-only normalization statistics materially change last-checkpoint inference, but do not outperform current selected model.',
        caveat='Same development300, not blind test; no proof BN is sole cause or last small batch sole cause. No demonstrated project accuracy improvement.',
        new_gradient_training_runs=0,optimizer_steps=0,new_checkpoint_files=0,production_replaced=False,
        model_sha256=check['original_web_model_sha256'],original_files_preserved=check['original_files_preserved'],
        implementer='Codex',verification='Separate arithmetic code and alternate batching; no independent human/agent review',
        running_processes_this_unit=[],next=next_action)
    write(U/'COMPLETION.json',summary)
    path=BASE/'ops/STATE.json';s=read(path);s['updated_date_kst']=now;s['current_stage']='WORD100_STABILITY_DIAGNOSTIC_COMPLETE_KEEP_CURRENT'
    s['next_single_action']=next_action;s['word100_stability_review_20261013']=summary;write(path,s)
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=path.read_text('utf-8')
    h='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
    assert h in txt
    txt=txt.replace(h,h+'\n\n### 2026-10-13 보강 실패 원인 10관점 검토·정규화 진단 완료\n현재 웹290/300(96.67%) 유지. 학습자료899개만으로 동결 가중치의 정규화통계 재계산: 현재선택290→289, 같은run마지막261→288; 속도선택284→286/마지막261→287. 마지막모델 회복은 현재웹 대비 성능향상이 아니며 정규화가 관여한다는 근거다. 교차시점 버퍼교환은 진단용, 운영후보 아님. 새 경사학습0/체크포인트0/교체0. 12예측파일×300행과4상태×8층 통계를 별도float64·다른배치17로 검산PASS. 원본/사용자파일/운영코드 등79개해시 보존. 10관점은Codex분석이며10명외부검토 아님. 근거 ops/word100_stability_review_20261013/RESULT_KO.txt, FACTORIAL/POPULATION_RESULT/INDEPENDENT_CHECK/COMPLETION.\n',1)
    h='## 9. 다음 단일 작업';assert h in txt
    txt=txt.replace(h,h+'\n\n### 2026-10-13 다음 — 새 특징 추가 전 학습·평가 안정성 대조\n'+next_action+' 현재선택에통계재계산만즉시적용하면1개회귀하므로미채택. 개선은현재290개초과·촬영자별회귀·복수초기값전체결과로판정하고,안정화만되면정확도향상과구분한다. 최종후보동결뒤보류촬영자시험;이보고에서는보류접근없음. 사용자학습/속도계수재탐색/자동모델교체없음. 중간연구보고서작성은이검토뒤별도단위로보존.\n',1)
    path.write_text(txt,'utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.

## 현재 목표와 마지막 완료
사용자는 보강 후 성능회귀의 원인과 다음 방향을 여러 번 검토하여 보고해달라고 요청했다.
`ops/word100_stability_review_20261013/RESULT_KO.txt`에 서로 다른10개 관점·논문근거·다음순서를 기록했다.
새 경사학습0/최적화0/체크포인트0/웹교체0. 저장모델 사본의통계진단만실행하고 완료.
가중치/버퍼2×2반사실검사 FACTORIAL; 학습899만의층별통계재추정 POPULATION_PLAN/RESULT.
현재선택60회 290→289; 마지막72회261→288. 속도선택58회284→286; 마지막70회261→287 (총300).
이는정규화의영향/후반하락일부회복이며 현재웹290개보다낫다는뜻아님. 즉시적용하지않음.
다른시점통계그대로복사하면크게회귀하므로금지. 마지막3개배치만이원인/BN만이원인이라고단정금지.
INDEPENDENT_CHECK:12예측파일×300행,4상태×8층통계 다른합산식/배치17검산PASS,79개원본해시보존.
Codex 구현·별도코드산술검산. 다른사람/에이전트/전문가10명검토아님.

## 다음 한 작업
{next_action}
동일가중치경로에서2가지평가통계를계산할수있으므로불필요한2배학습을피하는설계를먼저고정.
평가사본통계가다음학습가중치/버퍼/RNG를바꾸지않는것도확인필요.
개선=현재290개초과·사람별회귀·모든초기값공개. 안정성만좋으면정확도개선이라부르지않음.
후속10개오답 전수분류(관측/표현·라벨/시간/분류/판단불가), 근거있는공통수정하나만선택.
최종동결후미사용촬영자평가; 이번보류시험0. 사용자촬영튜닝/속도계수재탐색/자동교체금지.
중간연구보고서양식(상황요약·중간결론·계획변경사항)작성은요청에따라다음에; 아직문서미생성.

## 유지할 결과와 웹
현연습모델 `runs/word100_tcn_v1/whole_context/best.pt` SHA {check['original_web_model_sha256']}.
http://127.0.0.1:8765/practice.html 서버PID11368 이번턴확인; 재개때실제확인. 진행중학습/진단없음.
AIHub-only100/280채널/37,636매개변수.899학습01~09,300개발11~13;10/14~18 미사용.
99단어9개,1단어8개학습. 개발은사람3명×100이며epoch선택에도씀;새사용자일반정확도로표현금지.
전체구간277/300=92.33%→전체·핵심혼합290/300=96.67%(+4.33pp) 기존개발개선 유지.
앞선속도보강284/300=94.67%(-2pp),개선6/회귀12미채택;70epoch2030step/선택58,약130초.
합성속도0.8현재288/후보283,1.25현재290/후보285; 독립600사람표본아님.
ops/word100_robustness_20261013 완료실험반복금지. 입력1199감사에서오답만의손누락과다증거없음.
NIKL/KSL/사용자/멘토 학습·튜닝·후보선택 없음. W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013 필요.
참고영상100개ID01,연습메뉴12·분류100. UI2×2/top5막대/DTW3축막대/상위3영상.

## DTW와 사용자 기록
DTW96같은라벨 사람간비교 완료. 계산가능96≠정답96. 굽힘중앙9.91°,위치13.87/궤적9.71어깨너비%.
단어당8개,공유참고/촬영자,표현언어적정답미검증. 합격선/세부교정/비수어거절 검증아님.
그래프180°/100어깨너비%는눈금이지합격선아님. DTW는TCN순위를바꾸지않음.
ses_79977f57893d CLOSED:수어1회/감사2회,메모2개지만3회모두원본보존.
tri_22160c4af551수어1위,tri_9171304d585b감사2위,tri_cc75cf631a1f감사1위. 학습/기준튜닝금지.
촬영/카메라/장치/다운로드/운영교체/Git변경없음.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
    for path in U.glob('*.json'):read(path)
    assert read(BASE/'ops/STATE.json')['word100_stability_review_20261013']['current_last_population']==288
    assert read(BASE/'ops/STATE.json')['next_single_action']==next_action
    assert all(sha(k)==v for k,v in read(U/'PLAN.json')['source_sha256'].items())
    assert all(sha(k)==v for k,v in read(U.parent/'word100_robustness_20261013/ORIGINAL_GUARD.json').items())
    write(U/'FINAL_CHECK.json',dict(status='PASS',documents_synchronized=True,original_guard_preserved=True,
        report_sha256=sha(U/'RESULT_KO.txt'),evidence_files={p.name:sha(p) for p in U.glob('*.json') if p.name!='FINAL_CHECK.json'}))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
