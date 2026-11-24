"""Synchronize project pointers after final result inspection, without changing freezes."""
from common import *

def main():
 r=read(U/'RESULT.json');v=read(U/'FINAL_VERIFICATION.json');assert v['status']=='PASS'
 next_action='이번 고정비교 결과를 기준으로 기존8812는 보존하며 RTMW 관절전용 후보의 실제 촬영 전달·손내림 종료·시간축 연결을 별도 검증하고,4단어의 실제 중요한 오류/허용차이 정답과 대조해 교정 가능한 범위를 확인한다. 완료된 학습·평가를 재실행하거나 개발점수로시드/규칙을고르지 않는다.'
 c=r['tcn']['finetune_C'];fresh=r['tcn']['fresh'];s=r['actual_stream'];seconds=[x['post_last_capture_ms']/1000 for x in s]
 body=f'''### 2026-11-08 RTMW 공통관절 TCN·피드백 학습/비교 완료 — 운영 미교체

단위 {U.name}. 공개1199영상(899FIT ID01~09/300DEV ID11~13),150476프레임 원본감사·추출·입력85932280수치검산 완료, 입력실패0. TCN 같은구조 새학습/C복사미세조정 각3seed80epoch 완료. 전체화면 top1은 fresh{fresh['full']['top1']}/300, C미세조정{c['full']['top1']}/300; 기존상반신조건과대응하는 upper_mask는 fresh{fresh['upper_mask']['top1']},미세조정{c['upper_mask']['top1']}/300. 기존C294/T3-F295(masked,seed20261013)와비교. 모두기존DEV이며새시험이아님, 최고seed채택없음.

관절전용 시간합성곱+GRU 피드백4단어(순종/잘하다/조상/이집트),7물리지표·13설명단계 학습 완료. 최초상반신입력 편중발견→전체/상반신 공통교대 수정6회 별도학습, 첫6회보존. 수정전체화면 지표균형평균 fresh96.96%/transfer97.53%; 직접단계85·85·85/83·84·85(각92),DTW84·81·77/75·76·80로혼재. 같은상반신비교도별도보존.432공학제어/주석불변·FIT정규화검산완료. 전문가정답/실제학습자오류정답이아니며교정정확도미확정. WORD1133=잘하다,WORD2523=이집트가정확한매핑;이전인수인계문장만교정했고주석/학습변경없음.

새모델조건 공개3영상 실시간재생·녹화·RTMW추출재사용 후 두신경망+DTW 실측: 마지막예정프레임→결과 {min(seconds):.3f}~{max(seconds):.3f}초. 최초고정조합finetune_C+pose_transfer,다른3조합은별도추정. 초기준비별도,실제웹캠/Unity/브라우저렌더링포함아님. SignRep이미지계산0/참고영상재분석0. 별도저장값검산{v['checks_total']}건PASS,같은Codex검산이며독립사람검증아님. 사용자·멘토자료/카메라/새다운로드/외부전송/Git0. 원본C·T3-F/8812보존. RESULT_KO.txt/RESULT.json 참조.
'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');a=text.index('### 2026-11-08 RTMW TCN·관절전용 피드백 공동 학습 진행 중');b=text.index('\n### ',a+4);text=text[:a]+body+text[b:]
 a=text.index('### 2026-11-08 진행 중 공동 학습 이어받기');b=text.index('\n### ',a+4);text=text[:a]+'### 2026-11-08 RTMW 공동 학습 비교 이후\n\n'+next_action+' 원본C·T3-F/8812/사용자보호 유지.\n'+text[b:];p.write_text(text,encoding='utf-8')
 state=read(O/'STATE.json');status='RTMW_SHARED_NEURAL_COMPARISON_COMPLETE_NOT_DEPLOYED';state['current_stage']=status;state['next_single_action']=next_action;state['updated_date_kst']='2026-11-08'
 state[U.name].update(status=status,process_id=None,processes_running=False,tcn_training='six_fixed_runs_complete',feedback_viewfix='six_fixed_runs_complete_all_checks_pass',feedback_comparison='final_saved_output_verified',input_verification='PASS85932280scalars1199clips',report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),verification=str(U/'FINAL_VERIFICATION.json'),tcn=r['tcn'],actual_stream=r['actual_stream'],production_changed=False,educational_corrections_validated=False,next_single_action=next_action)
 write(O/'STATE.json',state)
 handoff=f'''# WordsInSilence 최신 인수인계

2026-11-08 사용자 승인 RTMW 공개 입력·TCN·관절전용 피드백 학습/비교 완료. 단위 {U.name}. RESULT_KO.txt, RESULT.json, FINAL_VERIFICATION.json 먼저 확인한다. 기존 학습과 비교를 재실행하지 않는다. 현재 실행중 학습/추출 없음.

공개1199영상899FIT/300DEV,150476프레임. 기록ID01~09 학습,11~13개발; 모든원본SHA/CRC·중복0·주석시각·입력85,932,280값검산. 실패0. TCN fresh/C복사finetune 각3seed80epoch. C는보존하는기존모델이름이며최고모델을뜻하지않음. full top1 fresh{fresh['full']['top1']},FT{c['full']['top1']}/300; upper_mask fresh{fresh['upper_mask']['top1']},FT{c['upper_mask']['top1']}/300. 기존 masked C294,T3-F295(seed20261013). full과기존sensor는화면조건이다르다. 최고시드채택없음.

피드백4단어7지표13단계. 첫상반신학습6회보존; feedback_viewfix의전체/상반신공통교대6회가최종연구후보. full 균형평균fresh96.96/transfer97.53%,직접단계fresh85·85·85/transfer83·84·85(각92),DTWfresh84·81·77/transfer75·76·80. 기존RGB+관절상반신96.30%와공정비교할수정상반신진단도저장.432개공학제어이며 실제오동작정답으로쓰지않음. 개발36프레임194물리지표/92단계지점은이전Codex시각검토;전문가정답아님. 왼손펴기긍정DEV1,오른검지긍정DEV0. WORD0739순종/1133잘하다/1567조상/2523이집트.

공개실시간재생3영상: RTMW한번으로녹화관절과두신경망/DTW재사용,마지막예정프레임→결과{min(seconds):.3f}~{max(seconds):.3f}초(고정첫pair FT+transfer 실제시간). 다른3pair공통처리+개별추정. 초기준비별도,실제웹캠/Unity/브라우저렌더링아님. 학습중및실행중SignRepRGB없음. 원본C·T3-F/8812불변. 사용자·멘토자료/카메라/다운로드/외부전송/Git0. 별도검산{v['checks_total']}PASS,동일Codex검산이지독립사람정답아님.

핵심한계: 관절존재/SimCC점수는가시성정답아님. 현재신경망은물리특징/단계/학습패턴정렬을출력하고 normal_or_wrong=None. 검증된교정문장은아직아님. 반복·짧은누락·중요한실제오류/허용차이·세밀한얼굴부위와실제촬영환경확인남음. 원래8812에는새모델미적용.

다음 단일 작업: {next_action}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status=status,processes=[],all_fixed_runs_complete=True,verification=str(U/'FINAL_VERIFICATION.json'),report=str(U/'RESULT_KO.txt'),production_changed=False,next_single_action=next_action))
 write(U/'DOCUMENT_SYNC.json',dict(status='COMPLETE',files={str(p):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']},report_sha256=sha(U/'RESULT_KO.txt')))
 event('project_documents_synchronized')
if __name__=='__main__':main()
