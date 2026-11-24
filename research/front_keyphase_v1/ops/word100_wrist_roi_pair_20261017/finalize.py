from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,re

U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not (U/'COMPLETION.json').exists()
    independent=read(U/'INDEPENDENT_VERIFICATION.json');assert independent['status']=='PASS'
    visual=read(U/'VISUAL_REVIEW.json');assert 'PENDING' not in visual['status'] and 'RENDERED'!=visual['status']
    plan=read(U/'PLAN.json');cmp=read(U/'COMPARISON.json')
    for p,h in plan['pins'].items():assert sha(p)==h,p
    now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    summary=('公開TRAIN20영상(3촬영자)2536프레임에 손목 부근 고정 하단 경계를 같은 규칙으로 비교·독립 검증했다. '
             '전체검출 후 마스크 확대에서 손21점 관측1643→1733(+90), 기존점 손실0. '
             '실제 RGB잘림 비교는1648→1706(증가158/손실100, 순증58), 영상15개증가·3동일·2감소(수영-24/전멸-25). '
             '전체검출 처음·끝5프레임 손점 제외18/20: 잘하다 시작5점-프레임, 홍성 종료5점-프레임 잔존. '
             '새 실제crop은 잘하다 시작3점 잔존, 끝5프레임20영상 손점0. '
             '경계 확대의 정보 보존 이점은 확인했으나 휴식손 완전 제외·실제검출 전반 개선·인식률 개선은 입증하지 않았다. '
             '80조건280채널 전수·60조건배정 재생·영역밖교란·700보호파일 PASS. 새학습/수어추론/운영교체/사용자자료0.')
    summary=summary.replace('公開TRAIN','공개 TRAIN')
    nxt=('사용자와 손목 경계 비교의 개선·회귀·휴식손 잔존을 함께 검토한다. '
         '후속 후보는 초기 안정된 팔 내림 자세 확인과 경계에 걸친 휴식 손목의 공통 처리이며, 동작 중 잠깐 낮아진 손 보존 및 실제 화면 경계 조건을 함께 검토한다. '
         '단어별 예외·단순 경계 재튜닝 없이 범위를 확정하고, 사용자 검토 전 학습·899영상 재추출·운영 TCN/DTW 교체는 시작하지 않는다.')
    entry=dict(status='VERIFIED_WRIST_ROI_MIXED_RESULTS_USER_REVIEW_BEFORE_TRAINING',updated_kst=now,unit='ops/'+U.name,
               summary=summary,next_single_action=nxt,clips=20,unique_video_frames=2536,comparison_conditions=80,new_detector_conditions=20,
               new_model_views=40,new_capture_seconds=read(U/'CAPTURE_RESULT.json')['seconds'],training=0,classifier_inference=0,
               production_changes=0,actual_webcam=False,user_data=False,independent_verification='PASS',visual_review=visual['status'],
               current_model_sha256='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f',process_status='capture exit0; compare exit0; independent verification exit0; completed')
    acceptance=dict(status='ENGINEERING_VERIFIED_CANDIDATE_MIXED_NOT_TRAINED',updated_kst=now,
       independent_report_sha256=sha(U/'INDEPENDENT_VERIFICATION.json'),visual_report_sha256=sha(U/'VISUAL_REVIEW.json'),
       root_direct_visuals=['visual_review/NIA_SL_WORD1545_REAL01_F_f55_trace_sustained_new_sensor_loss.png','visual_review/NIA_SL_WORD2348_REAL02_F_f49_trace_sustained_new_sensor_loss.png','design_images/endpoint_leakage_source_details.png'],
       root_trace_audit='ROOT_TRACE_AUDIT.json',notes=['Rest exclusion counterexamples preserved, no post-result tuning.','Geometry-only gain distinguished from detector gain and recognition.','Half-second audit uses inclusive <=500000us; DESIGN_AUDIT uses <500000us, accounting for16vs15 endpoint samples.'],
       protected_pins_checked=len(plan['pins']),training_started=False)
    (U/'ROOT_ACCEPTANCE.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (U/'EXIT.json').write_text(json.dumps(dict(capture=0,compare=0,independent_verification=0,updated_kst=now),indent=2)+'\n',encoding='utf-8')
    completion=dict(entry,artifact_pins={str(p.relative_to(U)):sha(p) for p in sorted(U.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    (U/'COMPLETION.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=path.read_text('utf-8-sig')
    anchor='### 2026-10-17 휴식 손목 기준 하단 확대 사진 검토안';assert s.count(anchor)==1
    s=s.replace(anchor,'### 2026-10-17 손목 하단 경계20영상 비교 완료 — 개선·회귀 혼재\n'+summary+'\n'
         '근거 `ops/'+U.name+'/RESULT_KO.txt`, `COMPARISON.json`, `INDEPENDENT_VERIFICATION.json`, `DESIGN_AUDIT.json`, `VISUAL_REVIEW.json`. '+str(visual['images'])+'장 비교/원검출 그림과25장 초기자세 검토. 기존 동결 단위·모델 보존.\n\n'+anchor,1)
    anchor='### 2026-10-17 다음 — 손목 부근 하단 경계 검토';assert s.count(anchor)==1
    s=s.replace(anchor,'### 2026-10-17 다음 — 손목 경계 결과 검토 후 후속 보완 범위 확정\n'+nxt+'\n\n'+anchor,1)
    path.write_text(s,encoding='utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}. 사용자 승인: 손목 바로 위로 낮춘 범위로 기존 문제 영상 비교·회귀 검증. 학습은 결과 상호 검토 이후.

## 마지막 완료
{summary}
고정규칙: 처음5프레임 각 손 wrist y중앙값 중 높은 값에서 어깨너비2%만큼 위의 정수 하단. 상단·좌우전체 유지, 영상 내 고정. 20경계838~920px, 기존보다60~109px 확대. 잘하다 첫5는 초기 정착동작이라 안정된 휴식 보장 안 됨. 경계·계수 재튜닝 없이 반례 보존.

## 해석 및 남은 문제
- 전체검출 마스크 확대의 기존점 손실0은 같은 원검출의 기하학적 포함관계다. 새손 검출·정확도·TCN인식률 개선이 아니다.
- 실제crop에서 수영 일부 구간은 손이 안쪽에 보여도 원검출에 한손만 나옴. 전멸은 같은 손 근처 후보 중복으로 반대손 근거 부재. 단순 하단 확대만으로 해결되지 않음.
- 홍성군주민지원센터의 끝점 잔존은 손가락이 아닌 손목1점씩(경계안 약0.85~1.66px), 잘하다는 초기손을내려놓는구간이다. 작은 잔존이 인식에 미치는 영향은 아직 미측정.
- 처음/끝5프레임 제외는18/20(full)이다. 0.5초를포함하면full종료홍성16점·상처1점도 남음. 본 비교의<=500000us와디자인감사<500000us(16/15프레임)를구분.
- 전체영상의 휴식 손목으로 공개영상 경계를 정하는 준비와, 화면밖 손을 볼 수 없는 실제 웹캠 조건은 다름. 실제 카메라 촬영 없음.

## 다음 한 작업
{nxt}

## 근거와 담당
research/front_keyphase_v1/ops/{U.name}/RESULT_KO.txt, PLAN.json, COMPARISON.json, ROOT_ACCEPTANCE.json, COMPLETION.json.
root 고정계획/실행/결과, wrist_pair_verify 독립전수검산, wrist_pair_design 초기·종료자세감사, wrist_visual_review 기존문제·새회귀 그림검토.
visual_review/ {visual['images']}장, design_images/25장. 새추출PID11700/exec86794 종료0, compare종료0, 독립검산종료0. 진행중작업없음. 이미완료된추출재실행금지.

## 유지되는 웹·모델 상태
기존진단 URL http://127.0.0.1:8767/ 는과거joint-v3 시각판이며 이번 경계결과로 웹을 바꾸지 않음. 문서상PID45392/exec6195였으나 재개시 실제상태 확인. 현재실험자료는 위 새단위를읽는다.
운영TCN runs/word100_tcn_v1/whole_context/best.pt SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 학습9명899/개발3명300,96.67%는개발검증. 사용자12회Top1 6/12·Top5 8/12는실사용기록이며 학습·튜닝·후보선택금지.
TCN몸9점+양손42점280채널, 어깨·팔꿈치·손목이미포함. 이번검증은관측범위/검출입력연구이며운영모델/DTW변경아님. sources/와원본·과거동결산출물보존.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    path=ROOT/'research/front_keyphase_v1/ops/STATE.json';s=path.read_text('utf-8-sig');d=read(path)
    s=s.replace('"updated_date_kst": '+json.dumps(d['updated_date_kst']),'"updated_date_kst": '+json.dumps(now),1)
    s=s.replace('"current_stage": '+json.dumps(d['current_stage']),'"current_stage": "WORD100_WRIST_ROI_MIXED_RESULTS_REVIEW"',1)
    s=re.sub(r'^  "next_single_action": [^\n]*',lambda m:'  "next_single_action": '+json.dumps(nxt,ensure_ascii=False)+(',' if m.group(0).endswith(',') else ''),s,count=1,flags=re.MULTILINE)
    anchor='  "word100_wrist_boundary_preview_20261017": {';assert anchor in s
    s=s.replace(anchor,'  "'+U.name+'": '+json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')+',\n'+anchor,1)
    parsed=json.loads(s);assert parsed['next_single_action']==nxt
    path.write_text(s,encoding='utf-8')
    print(json.dumps(dict(status=entry['status'],artifacts=len(completion['artifact_pins']),protected=len(plan['pins']),visual_images=visual['images']),ensure_ascii=False))

if __name__=='__main__':main()
