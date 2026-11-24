"""Close the isolated demo integration, retaining limitations and active goal."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,re,shutil
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepend(text,n,entry):
    m=re.search(r'(?m)^## '+str(n)+r'\.[^\n]*\n',text);assert m
    return text[:m.end()]+'\n'+entry+'\n\n'+text[m.end():].lstrip('\r\n')
def main():
    result=read(U/'PUBLIC_RUN_RESULT.json');v=read(U/'SAVED_RUNTIME_VERIFIED.json');browser=read(U/'BROWSER_VERIFIED.json')
    assert v['status']=='PASS_SAVED_ARITHMETIC_NOT_ACCURACY' and v['failed_count']==0 and v['check_count']==1156
    assert browser['status']=='PASS_HEADLESS_PUBLIC_BROWSER' and browser['check_count']==19
    assert sha(U/'PUBLIC_RUN_RESULT.json')==read(U/'PUBLIC_RUN_FINISHED.json')['result_sha256']
    for p,h in read(U/'EXECUTION_FROZEN.json')['pins'].items():assert sha(p)==h,p
    live=read(result['result']);launch=read(U/'DEMO_LAUNCH.json')
    assert result['local_event_forwards']==1 and len(result['actual_dtw_calls'])==1
    assert launch['pid']==18696 and launch['url']=='http://127.0.0.1:8810/'
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    summary=(
        'AM 별도 연구데모 http://127.0.0.1:8810 구현·공개새후보통합·헤드리스브라우저검증 완료. 기존 AA/AK/AL/X 파일과 C·T3F/8804/8808/8809는 보존했다. '
        'AK의 학습형손모양/관계/순서+AH시간별움직임에 AL 동일frame 병렬sensor를 연결하고, 전체analyze/warmup/close RLock·역할/입력hash/word/nativeaxis를 강제했다. '
        'public_runtime_runs는공개추론, private_sessions는명시적user_inference_only이고업로드본을public으로가장하지않는다. '
        '가짜adapter32/서버79/표시36검사·별도정적검토·544핀동결후 이미노출 AK ID10 WORD1133 동일원본에 새AM전체후보1회만실행했다(exec8881 terminal0). '
        '새분석10.9279초(초기import/준비포함28.9993초), 관측9.3572초/RGB5.5896초는병렬이라합산금지. 작은localhead12.06ms는전체지연아님. '
        '별도저장검산1156 PASS, AK대비센서5NPZ/trace/sidecar 및 입력10종·학습DTW경로exact, 입력drift0. DTW1회/원영상별참고비교0/localforward1/학습0. '
        '새UI는 이전에읽지않던local_event_feedback를 실제로표시하고 사전관련문구·원1.60~2.13초와다시보기·high5/17순간·확인보류/관측부족을분리한다. '
        '옛전체방향미확인을별도접힘영역에보존하고관련문구전체/접촉/반복/정오로승격하지않는다. '
        '저장예시는AK결과읽기만하며새모델추론버튼없음;새촬영은사용자명시업로드후추론만. 실제카메라/사용자자료접근·학습·튜닝·선택0. '
        '전용브라우저/컴퓨터도구는kernel assets 오류로못썼다. 새격리헤드리스Chrome의실제공개영상2개디코드/원시각재생/색대비·720폭/네트워크19검사와스크린샷시각확인으로대체했다. '
        '처음sandbox서버는외부브라우저연결시간초과로종료하고marker보존, 일반로컬서버 exec19318/PID18696 active·모델ready. 초기브라우저연결오류와data:내장아이콘을외부통신으로센검사오류도보존했다. '
        '헤드리스브라우저는현재사용자IAB창확인이아님. 서버79검사중TEMP/socket제한은원오류보존후승인된일반환경가짜검사로복구했다. '
        '기존12.824초보다이번10.928초측정이작지만부하차이·단회조건이므로속도인과/일반화/정답률향상을주장하지않는다. 참고영상4개는W:연결필요. '
        '정상오경고/중요한차이/누락/반복/접촉을포함한전체교육피드백완성및3초대지연은아직미달, 전체goal active.')
    next_action=(
        'AM 데모는 http://127.0.0.1:8810 실행중(PID18696, exec19318). 처음에현재process/port와DEMO_LAUNCH·PUBLIC_RUN_FINISHED·SAVED_RUNTIME_VERIFIED·BROWSER_VERIFIED를확인하고중복서버/완료추론을실행하지않는다. '
        '현재새관측·모델연결·시간문구표시는검증됐으므로이것을다시만드는작업은끝낸다. 다음은기존공개자료의정상/짧은주요변화/순서변경/관측공백을지원4표현의피드백요구사항과대조하여아직판정하지못하는중요차이를확정하는작업이다. '
        '각사례의현재픽셀/등록ID/기존노출역할과사용가능한물리정답/문구관련성/언어적미확정을분리하고,목표모델이이미낸저장결과는재사용한다. '
        'AJ처럼발생만확인하고전체누락정답이없는편집을오류정답으로바꾸지않는다. 신규평가가필요한사례만예측전정답범위/분모/한후보예산을고정해단회비교한다. '
        '결과에근거한필요한지도학습/부분미세조정은실제부족한구별에한정하고같은개발표현의임계값/예외를돌려맞추지않는다. '
        '속도는현재측정관측9.36초안의중복preflight/probe읽기1.90초·초기import/준비비용과RGB5.59초를분리해검토한다. 정보/프레임삭제나눈입제거로3초를맞추지않고,실제계산재사용/기존동일출력보존의필요한변경만검증한다. '
        'UI가부분움직임을전체성공으로안내하지않는계약과private추론전용/전체lock·원시간·관측공백을유지한다. '
        '원본/완료평가/학습·C/T3F/기존데모보존, 사용자·멘토/잠금평가자료접근·학습·튜닝·선택금지. W:원본읽기전용. 새외부모델/다운로드/Git/운영교체없음. '
        '전체목표달성은아직증명되지않았고작은통합완료로goalcomplete처리하지않는다.')
    final=dict(status='ISOLATED_LEARNED_LOCAL_EVENT_DEMO_LIVE_LIMITED_RESEARCH',at=at,url=launch['url'],server_pid=launch['pid'],server_exec_session=19318,
        one_public_candidate_job=result['job_id'],total_analysis_seconds=result['server_analysis_seconds'],cold_job_seconds=result['cold_job_seconds'],
        native_frames=118,learned_DTW_calls=1,individual_reference_video_comparisons=0,local_model_forwards=1,new_training=0,
        saved_checks=1156,adapter_checks=32,server_checks=79,display_checks=36,browser_checks=19,
        actual_user_or_camera_test=False,browser_mode='new isolated headless Chrome',visual_screenshots_inspected=True,
        source_and_old_inputs_preserved=True,accuracy_improvement_claimed=False,causal_speedup_claimed=False,
        old_demo_changed=False,whole_goal_complete=False,reference_videos_require_W=True,
        summary_ko=summary,next=next_action,
        pins={str(U/n):sha(U/n) for n in ['SCOPE.json','EXECUTION_FROZEN.json','CODE_REVIEW.json','PUBLIC_RUN_RESULT.json',
            'PUBLIC_RUN_FINISHED.json','SAVED_RUNTIME_VERIFIED.json','INPUT_DRIFT_SAVED.json','BROWSER_VERIFIED.json','check_browser.py','DEMO_LAUNCH.json']})
    write(U/'FINAL.json',final)
    report=('새 학습형 피드백 데모를 준비했습니다.\nhttp://127.0.0.1:8810/\n\n'
        '이전에는 새 시간별 움직임 판정부가 결과를 계산해도 기존 화면이 이를 읽지 않았습니다. 이번에는 새 판정부와 검증한 병렬 검출을 별도 실행부에 연결하고, 실제 결과를 화면에 표시했습니다.\n'
        '예시: “스쳐 내린다”라는 문구 옆에 1.60~2.13초를 보여 주고, 그 구간에서 팔을 따라 움직이는 근거와 확인을 보류한 순간을 나눠 표시합니다. 다시 보기 버튼은 해당 원영상 시간으로 이동합니다. 움직임 일부가 보였다는 결과를 전체 동작 성공으로 바꾸지는 않습니다.\n\n'
        '공개 원본 한 개의 새 실행부 시험을 마쳤습니다. 여러 참고 영상을 매번 비교하지 않고 학습된 기준 흐름에 DTW 한 번을 사용합니다. 센서·10종 입력 특징·DTW 경로는 기존 결과와 정확히 같았습니다. 별도 저장검산 1,156개, 실제 브라우저 검사 19개가 통과했고 화면 글자와 영상 재생도 확인했습니다.\n'
        '실제 전체 분석시간은 이번 10.93초였습니다. 기존 측정은 12.82초였지만 부하가 달라 차이를 전부 이번 변경 효과라고 주장하지 않습니다. 아직 3초 안팎의 목표에는 못 미칩니다. 최초 준비까지 포함하면 29.00초였으며, 이 수치를 매 촬영 대기시간과 구분해야 합니다.\n\n'
        '현재 화면에서: “저장된 잘하다 예시 보기”는 이전 공개 결과를 재생만 합니다. 다른 표현 또는 새 촬영은 표현을 고르고 명시적으로 분석하기를 누릅니다. 준비할 때 손은 화면 아래에 있어도 됩니다. 현재 참고 영상 재생에는 W: 연결이 필요합니다.\n'
        '남은 문제: 정상 동작에 잘못 경고하지 않는지와 중요한 손모양·순서 차이·동작 누락을 찾는지를 더 넓은 공개 사례에서 확인해야 합니다. 정확한 접촉·필수 반복 횟수·단어 전체의 합격과 불합격은 아직 판정하지 않습니다. 전체 피드백 기능은 완성되지 않았습니다.\n'
        '다음에는 지금 만든 실행부와 표시를 유지한 채, 그 중요한 차이를 구별하는 데 실제로 부족한 감독 정보를 찾아 보완하고 중복 영상 읽기 등 대기시간의 원인도 줄여갑니다.\n'
        '기존 C·T3-F와 이전 데모는 보존했습니다. 사용자·멘토 촬영은 이번 작업의 학습·튜닝·후보 선택에 쓰지 않았고 카메라도 켜지 않았습니다.\n')
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(report)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    before=U/'before';before.mkdir(exist_ok=False)
    for p in docs:shutil.copy2(p,before/p.name)
    heading='2026-11-03 시간별 움직임을 표시하는 별도 학습형 데모 연결 완료'
    text=docs[0].read_text(encoding='utf-8-sig');assert heading not in text
    text=prepend(text,8,'### '+heading+'\n\n'+summary)
    text=prepend(text,9,'### '+heading+' 이후 다음 단일 작업\n\n'+next_action)
    docs[0].write_text(text,encoding='utf-8')
    docs[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+next_action+
        '\n\n근거: '+str(U)+'\n\n직전 goal turn: progress. AL 관측보존 검증에 이어 AM 별도데모 구현·공개통합·브라우저검증 완료. 서버만 계속 실행, 후보 학습/평가는 종료. 전체goal active.\n',encoding='utf-8')
    state=read(docs[2]);state['updated_date_kst']='2026-11-03'
    state['current_stage']='LOCAL_EVENT_LEARNED_DEMO_LIVE_IMPORTANT_DIFFERENCE_VALIDATION_NEXT'
    state['next_single_action']=next_action;state[U.name]=dict(final,directory=str(U))
    docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    assert summary in docs[1].read_text(encoding='utf-8') and read(docs[2])[U.name]['whole_goal_complete'] is False
    write(U/'DOCS_SYNCED.json',dict(at=at,files={str(p):sha(p) for p in docs},whole_goal_complete=False))
    print('AM verified limited research demo live; docs/state/handoff synchronized; overall goal active')
if __name__=='__main__':main()
