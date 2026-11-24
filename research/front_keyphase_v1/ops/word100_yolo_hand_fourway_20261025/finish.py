from pathlib import Path
import json,datetime,hashlib
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=read(U/'SUMMARY.json');v=read(U/'VERIFY_SAVED.json');n=read(U/'FINAL_NOTES.json');f=read(U/'FROZEN.json')
    assert v['status']=='PASS' and read(U/'CONTRACT_TESTS.json')['status']=='PASS'
    assert not (U/'RUNNING.json').exists()
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    at=datetime.datetime.now().astimezone().isoformat();key=U.name
    entry={'status':'COMPLETED_FOURWAY_NOT_ADOPTED','at_kst':at,'unit':key,'scope':s['scope'],
           'wrist_image_consistency':s['wrist_image_consistency'],'latency_cpu_detector_components':s['latency_cpu_detector_components'],
           'paired':s['paired'],'preservation':s['preservation'],'new_native_hand_frames':s['accepted_native_hand_frames'],'new_roi_hand_frames':s['accepted_roi_hand_frames'],
           'conclusion':n['conclusion'],'remaining':n['remaining'],'next':n['next'],'validation':v,
           'protected_files':len(f['protected']),'training':0,'classifier_inference':0,'user_mentor_access':0,'runtime_changes':0,
           'checkpoint_download_bytes':7003330,'implementation':'Codex local, separate saved arithmetic verification; historical blind AI labels reused','report':str(U/'RESULT_KO.html'),'running_processes':[]}
    write(U/'FINAL_RESULT.json',entry)
    path=O/'STATE.json';st=read(path);st['updated_date_kst']=at;st['current_stage']='YOLO_HAND_FOURWAY_COMPLETED_EXISTING_MODELS_RETAINED';st['next_single_action']=n['next'];st[key]=entry;write(path,st)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig')
    assert key not in text
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
    status=f'''\n\n### 2026-10-25 YOLO 손 검출 네 조건 비교 완료 — 기존 C·T3-F 유지
- 단위 `research/front_keyphase_v1/ops/{key}`. 손21점 YOLOv8n 체크포인트1개(7,003,330bytes) 고정, 공개39영상/18표현/6명/4,781프레임. 기존TRAIN01–06 진단, 새사용자 평가 아님.
- {n['conclusion']}
- 기존 M0/M1 동결 관측 재사용, Y/MY 새 전체영상 추론 후 같은 ROI. Y좌우 확인은 공통MP몸근거 사용. MY기존관측 {s['preservation']['MY_original_joint_slots']}관절-프레임 보존. 정상/교차 보존과 정확성은 구분.
- {n['metrics_line']}
- CPU시간은 새 검출부품 순차호출 합산이며 준비/디코딩/좌우규칙/TCN/DTW/웹 제외. 미검출·판독보류 분모 유지. 전문가21관절정답/수어인식률/새사용자성능 아님.
- 자체공통계약11검사, 별도저장산술검산{v['rows_reconstructed']}행. 동일Codex 별도구현검산이며 새 독립사람/에이전트검증 아님. 이전눈가림AI영상판독범위 재사용. 보호32불변, 사용자·멘토접근0, 새학습0,TCN추론0,운영교체0.
- 구버전체크포인트 로더형식2오류는 첫공개영상 전 수정. 실패근거보존,가중치·동결비교조건불변. 실행종료. 근거 `RESULT_KO.html`, `SUMMARY.json`, `VERIFY_SAVED.json`, `FINAL_NOTES.json`.
'''
    nxt=f'\n\n### 2026-10-25 YOLO 네 조건 비교 이후\n\n{n["next"]}\n'
    text=text.replace(h8,h8+status,1).replace(h9,h9+nxt,1);p.write_text(text,encoding='utf8')
    handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 공개 원영상의 손 관측을 정확히 개선하는 보조 후보 검증. 기존C·T3-F/운영웹DTW 보존.
마지막 완료: YOLO 손21점 후보1개 고정, 기존/MP재검출/YOLO손/MP+YOLO 네 조건, 공개39영상18표현6명4781프레임 완료.
{n['conclusion']}
{n['metrics_line']}
R24와이전V18의동결AI손목범위 사용. 전문가21관절 정답 아님, 전단어/새사용자/TCN 정확도아님.
MY기존관측{s['preservation']['MY_original_joint_slots']}관절-프레임 보존. 새추가원영상{s['accepted_native_hand_frames']}손-프레임/ROI{s['accepted_roi_hand_frames']}손-프레임. 추가좌표의정확성과검출량구분.
CPU시간은검출부품 합산,실제웹캠FPS아님. M0/M1정확도동결캐시,시간새실행. 사용자·멘토 접근0/학습0/TCN추론0/운영변경0. 보호32유지.
Codex구현·11계약검사·별도산술{v['rows_reconstructed']}행검산완료. 같은작성자별도계산이며새독립에이전트/전문가검증아님.
처음모델로딩형식오류2개는공개영상이전해결. setup_failure_01/SETUP_RECOVERY/RECOVERY_PROTOCOL_CHECK에보존. 가중치·비교기준변경없음.
진행중프로세스없음. 새모델채택없음.

다음 한 작업: {n['next']}

남은문제: {' / '.join(n['remaining'])}
원본/W:/sources/과거동결결과읽기전용. 사용자촬영학습·튜닝·선택금지. Git/외부자료전송/운영교체금지.
근거: {U/'RESULT_KO.html'}
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
    execution=read(U/'CURRENT_EXECUTION.json');execution.update(status='COMPLETED_EXIT_0',at_finished=at);write(U/'CURRENT_EXECUTION.json',execution)
    assert read(O/'STATE.json')[key]['scope']==s['scope'];assert key in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf8')
    write(U/'DELIVERY_CHECK.json',{'status':'PASS','synced':['STATE current_stage/next_single_action/current unit','PROJECT_CONTEXT newest8/9','NEW_CHAT_HANDOFF'],'report_exists':(U/'RESULT_KO.html').exists(),'running':False,'protected':len(f['protected'])})
    print('SYNCED',key)
if __name__=='__main__':main()
