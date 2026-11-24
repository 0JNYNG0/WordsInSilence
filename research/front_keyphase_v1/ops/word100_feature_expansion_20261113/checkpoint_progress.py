from common import *

def main():
    c7=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization/c7_local_cause_recovery_20261113'
    graph=read(U/'GRAPH.json');graph['nodes'].update(G1_select='COMPLETE_7_EXTRA_WORDS',G2_labels='COMPLETE_RESEARCH_RGB_REVIEW',G3_inputs='COMPLETE_1273_INPUTS',G4_train='RUNNING_FIXED_9',C7_motion_hold='C7_C8_COMPARED_G2_REWORK',AV_avatar='REVIEW_COMPLETE_RAW_CAMERA_SOURCE_MISSING')
    graph['evidence']=dict(inputs=str(U/'DATA_READY.json'),labels=str(U/'ROOT_LABEL_REVIEW.json'),pretrain=str(U/'PRETRAIN_CHECK.json'),C7=str(c7/'GRAPH_PATCH.json'),avatar=str(O/'avatar_multiview_review_20261113/EVIDENCE.json'))
    write(U/'GRAPH.json',graph)
    sp=O/'STATE.json';s=read(sp);s['active_task']=U.name;s['next_single_action']='Continue fixed nine temporal finger runs from CURRENT_EXECUTION/checkpoints, then evaluate once, independent arithmetic audit and source-linked feedback gate. C7/C8 failure retained, avatar source-recovery plan separate.'
    s[U.name].update(status='TRAINING_FIXED_NINE',input_clips=1273,fit=955,development=318,common_words=100,extra_words=7,training_runs=len(list((U/'runs').glob('*/COMPLETE.json'))),C7_G2='REWORK',avatar_review_complete=True,production_changed=False,private_used=False)
    write(sp,s)
    handoff=f'''# 최신 인수인계 — 2026-11-13 선별 확장 6단계 실행 중

현재 사용자 승인은 1~6단계 전체, 기존 그래프 C7 병행, 공개5뷰 아바타 병렬 검토다. 예전 승인대기는 해제됐다.
현재 단위: {U}
목표: 기존100단어 공통 학습을 유지하고, 부족한 손가락 특징을 가르치는 추가 공개자료가 일반 수어 판정에 도움이 되는지 실제 시간합성곱+GRU로 고정 비교한다.

준비 완료: 기존1199 + 추가74 =1273영상 입력. FIT955(기존899+추가56), 개발318(기존300+숫자18). 추가7단어는 숫자1~6과사과3공개기록. 숫자4 ID05 손상은 기존실패보존, 사과ID05 뜻변이제외. 추가전체단어의RTMW재추출아님: 새1/3/5만36영상3347프레임추출, 기존캐시재사용.
정답: 원본186프레임 직접RGB검토(일반12표현+새숫자+사과). root24프레임별도확인. 옛2/4/6직접주석추가; 엄지모음/굽힘애매한2/4 엄지정답마스킹. 최종일반FIT427/DEV389 손가락항목, 추가FIT434. U는오답아님. 독립사람/전문가정답아님.
입력/코드별도검토에서가려진손목의정규화축이남는문제와캐시SHA검사누락을학습전에고쳤다. FROZEN_PREFLIGHT_V1와수정기록보존. MODEL_SANITY/code_review PASS.

진행: train.py 고정A/B/C×seed42/43/44 총9개,12epoch×38step. A100+새일반세부정답, B추가단어이름지도, C추가손가락지도. 같은공통100연구가중치에서미세조정, 기존운영TCN/C·T3-F/웹교체없음. 실제세션/프로세스는CURRENT_EXECUTION, runs/last.pt 확인하여중복금지. 완료run재학습금지. eval미실행; evaluate.py 검토후한번수행,저장예측검산. 학습과평가단계동결/gates는FROZEN.json.

기존그래프: C7 18신규모델 + C8 18신규모델 비교완료. 누락0평균/짧은전완투영증폭원인확인. C8 다른동작오인감소하지만움직임/유지/교차단서전체보존실패; 미채택/G2REWORK. 완료C6/C7/C8재실행금지. 근거 {c7}/GRAPH_PATCH.json. 다음원인11시점은추후시계열순서보존설계.
아바타: 공식5뷰동시촬영/3D·camparam제공기록확인. 조상REAL01 5뷰142frame/30fps; REAL02캐시3D는별도ID라섞지않음. 과거코드는정밀삼각측량아닌2.5D근사. 우선배포3D/카메라원본위치복구→5뷰재투영→Mixamo리타게팅→FBX저장권고. 원본keypointZIP옛경로없음. 이번은검토만,애니메이션제작/외부다운로드없음.

금지: 사용자·멘토자료학습/튜닝/선택,봉인10/14~18,외부전송,추가다운로드,Git,카메라,기존모델교체. W:/sources/과거동결읽기전용. 새DEV는아닌기존11~13개발자료; 성공률과교육적피드백완성도를혼동하지않음.
다음:9개학습→한번평가→에이전트별도검산→통과한범위만설명구간/짧은문장후보연결,미통과시웹미적용과원인보고. 완료시STATE/§8§9/이파일동기화.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
    event('checkpoint_saved',phase='fixed_training')

if __name__=='__main__':main()
