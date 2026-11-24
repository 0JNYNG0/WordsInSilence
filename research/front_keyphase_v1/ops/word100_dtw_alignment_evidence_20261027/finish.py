"""Close the explanatory audit; no training, model scoring or source edits."""
import inspect_alignment as S
import json,shutil
from pathlib import Path

U=S.U; root=S.REPO
summary='''사용자 요청에 따라 YOLO·HaMeR·MANO·최근 좌표 보정을 후속 기본 경로에서 보류하고, 실제 웹 DTW의 저장 입력과 시간 연결부터 재점검했다. 기존 연구 파일이나 운영 모델은 삭제·교체하지 않았다.

사용자가 명시적으로 요청한 이전 본인 촬영 12개와, 각 촬영 때 제시된 공개 TRAIN 참고 1개씩을 연결했다. 12개 모두 기존 저장 DTW 경로 길이·비용·관측 비율 및 위치/각도 수치를 재현했다. 별도 스칼라 동적계획법 12회와 96개 지표 산술검산 PASS. 감사·믿다·멀다는 결과를 보기 전에 상세 화면 대상으로 정했다. 사용자 자료는 이번 설명·평가에만 사용했으며 학습·튜닝·임계값·모델 선택에는 쓰지 않았다.

결과 화면에는 실제 시간 그래프, 시간 연결 경로, 길이만 맞춘 경우와 DTW 연결의 실제 장면 슬라이더, 좌우 손 높이/오른손 가로 위치/오른 검지 관절각의 전후 그래프를 넣었다. 그래프 높이·각도는 원래 값이며 시간상의 연결만 달라진다. 손이 없으면 0으로 채우지 않는다. 사용자 RGB는 최대10fps 저장 이미지여서 가까운 실제 장면을 표시하고 시각 차이를 함께 기록했다. 공개 영상은 저장 관절의 프레임 번호 및 실제 디코딩 시각으로 연결했다. 외부 요청은 차단하고 데스크톱·모바일, 모든 슬라이더·선택·자동넘김 검증을 완료했다.

확인한 근본 문제
1. 현재 웹 DTW의 동작 범위는 처음 사용 가능한 손 관측부터 마지막까지다. 의미상 핵심 동작 구간이 아니다. 참고의 내려놓은 손은 보이지만 사용자는 화면 밖이라 범위가 비대칭이다. 감사는 참고0~4.467초 전체, 사용자0.505~2.598초(2.093초)를 비교했다.
2. 실제 잘못 맞물린 동작 단계가 보인다. 멀다33번째 대응쌍은 사용자0.899926초/관절과RGB동일시각(얼굴 옆으로 손이 올라옴)과 참고1.266667초(아직 손을 내린 대기)를 연결한다. 믿다의 대응쌍에서도 사용자는 손을 올린 동안 참고가 대기 중인 장면이 있다. 이는 모든 정렬의 오류율이나 원인 전체를 확정한 결과는 아니다.
3. 전체 비용 감소가 각 항목 개선을 뜻하지 않는다. 감사의 평균 결합비용0.8229→0.7870에도 오른손15굽힘각 평균차이는11.1887→12.6263도다. 기본 경로는 손·팔·손모양 등 결합비용을 사용하고, 보고하는 굽힘각 자체의 기본 정렬 가중치는0이다. 굽힘각은 정렬 후 계산하는 항목이다. 관측 가능한 대응쌍 수도 바뀔 수 있어 감소만으로 정확도 향상이라고 할 수 없다.
4. 웹 DTW는 원래 저장 관측을 쓰고 C/T3-F는 별도 C 입력을 쓴다. 단어 인식 개선이나 연구용 재검출 개선이 DTW에 자동으로 모두 들어간 상태가 아니다.

TCN 현황과 속도
현재 C/T3-F는100단어 분류 기능을 구현·검증한 단계다. 기존 추가 공개199영상의 화면밖미관측 재현 조건에서 C193/199=96.98%, T3-F196/199=98.49%. 기존 같은 사용자12촬영은 C10/12, T3-F12/12. 모두 저장 결과를 인용했으며 이번 재추론은0이다. 새 사용자·촬영 환경·학습 밖 단어의 완성을 뜻하지 않는다.
TCN에는 DTW가 없다. 관절의 시간적 변화 패턴을 합성곱으로 분류하고 C는 전체 특징을 평균 요약한다. T3-F는 경과시간을0~1로 놓은 앞·중간·뒤 요약을 추가한다. 입력의 원래 길이와 프레임 시간 간격을 사용하며 고정 길이로 전부 재표본화하지 않는다. 약간의 속도 차이에도 인식할 수 있으나 정확히 어느 속도까지 유지되는지는 아직 입증하지 않았다. 이전30→20/15fps 자료는 관측 빈도 시험이지 행동 속도 시험이 아니다. 실제 빠른 동작의 모션블러/검출누락과 단순 재생속도 변경은 분리해야 한다.

MediaPipe 재검출의 위치
공개 문제 장면23개에서 손목 위치 근거가 기존3개→재검출12개로 늘어난 이전 기록이 있다. 이를 손가락21점 정확도나TCN인식률로 부르면 안 된다. 관측 보완의 비교 기준으로 유지할 가치가 있지만 전체 피드백이 완성됐다는 근거는 아니다. T3-F 등 단어 분류 개선은 별개로 유지한다.

다음 한 작업
공개 TRAIN에서 참고 준비·마무리와 동작 범위를 정하는 공통 규칙을 먼저 정의하고 고정한다. 원래 웹 입력/시간 정렬을 기준으로 범위 처리만 달리 비교한다. 짧은 구별 동작·중간 손내림·교차·미검출 구간을 통째로 삭제하지 않는다. 알려진 시간 대응을 가진 공개 속도 변형과 다른 공개 수행자의 정상 동작에서 단계 연결·관측 분모·위치/각도 차이·잘못된 경고를 함께 검증한다. 사용자 촬영으로 규칙을 맞추지 않는다. 통과 후 기존 MediaPipe 재검출 후보를 한 요소씩 연결할지 판단하며 YOLO/HaMeR 추가 실험은 보류한다. TCN 속도 내성 시험은 별도 공개 평가로 분리하고 현 단계에서 재학습·운영 교체하지 않는다.

구현·검증 담당: Codex 로컬. 별도 산술 구현 검산과 시각 확인이며 독립 사람의 수어 단계 정답 인증은 아니다. 최초 표시 오류는 FIRST_ERROR_render.json 및 render_before_video_fps_fix.py 보존. 멀다 영상의 컨테이너 평균FPS30.1333을 실제프레임30fps시각과 혼동한 검사만 고쳤고 관절·경로·수치 결과를 바꾸지 않았다. 새 학습0/TCN추론0/검출0/카메라0/다운로드0/외부전송0/운영변경0. 고정105출처와32보호파일 해시 유지.
'''
next_action='''공개 자료에서 웹 DTW의 비대칭 준비·마무리 범위부터 비교 검증한다. word100_dtw_alignment_evidence_20261027의 완료12재구성/VERIFIED/VIEW_DATA를 재사용하고 사용자 촬영은 설명·평가 전용으로만 유지한다. 공개TRAIN에서 공통 범위 규칙을 정해 고정하고 원래 웹입력/DTW 대비 범위만 달리 비교한다. 중간 손내림·교차·누락·짧은 동작 보존, 알려진 시간대응 및 다른 공개수행자 비교, 관측 분모 유지와 잘못된 경고를 함께 확인한다. 비용 감소만으로 채택하지 않는다. 원래DTW 관측과C/T3F 입력 경로가 다르므로 재검출이 이미 적용됐다고 오해하지 않는다. YOLO/HaMeR/MANO/최근좌표보정은 이 기본 경로에서 보류, 과거결과보존. TCN속도시험은 관측간격변경과 실제행동속도를 구별하는 별도공개평가로 계획한다. 현단계 추가학습·운영교체없음, C/T3F/E3/웹/원본보존. 사용자자료 학습·튜닝·선택금지. 활성실행없음.'''

def main():
    f=S.read(U/'FROZEN.json');S.check(f['sources']);S.check(f['protected'])
    assert S.read(U/'VERIFIED.json')['status']=='PASS'
    assert S.read(U/'HTML_RENDER_VERIFIED.json')['status']=='PASS_HEADLESS_CHROME'
    view=S.read(U/'VIEW_DATA.json');pairs=[p for r in view for p in r['pairs']['dtw']]
    assert len(pairs)==288
    for p in pairs:
        assert abs(p['reference']['time']-p['reference']['video_pts'])<.002
        assert abs(p['rgb_delta_ms']-(p['query']['time']-p['query_observation_time'])*1000)<.011
        assert (U/p['query']['file']).exists() and (U/p['reference']['file']).exists()
    S.write('VISUAL_REVIEW.json',dict(operator='Codex same operator, not independent human ground truth',
        inspected=['three paired contact sheets','감사 and 믿다 four signal rows','멀다 time path','desktop/mobile/interactive browser screenshots'],
        direct_example=dict(word='WORD0949',pair_index_zero=32,query_rgb_time=.899926,query_observation_time=.899926,reference_time=1.266667,
            visible='query hand near face while reference hands at rest'),
        presentation_pts_checks=288,max_query_rgb_time_gap_ms=max(abs(p['rgb_delta_ms']) for p in pairs),
        outcome='Input-window asymmetry and an actual stage mismatch shown. No semantic alignment accuracy or correction effect claimed.'))
    (U/'RESULT_KO.txt').write_text(summary,encoding='utf-8')
    S.write('NEXT_PLAN.json',dict(next_single_action=next_action,no_user_tuning=True,new_training=False,operating_replacement=False))
    at=S.now();stage='DTW_SAVED_USER_ALIGNMENT_VISUALIZED_BASELINE_RESTART_NO_TUNING'
    statepath=root/'research/front_keyphase_v1/ops/STATE.json';ctxpath=root/'WordsInSilence_PROJECT_CONTEXT.md';handoff=root/'NEW_CHAT_HANDOFF.md'
    before=U/'before';before.mkdir()
    for p in [statepath,ctxpath,handoff]:shutil.copy2(p,before/p.name)
    concise='''사용자 요청으로 YOLO·HaMeR·MANO·최근좌표보정 기본경로 보류, 기존 연구·C/T3F/운영은 보존. 사용자 명시 허용의 기존12촬영과 당시 시연TRAIN참고1개씩 웹DTW 재구성, 저장수치일치12/12·별도스칼라DP12/96지표검산PASS. 감사/믿다/멀다 실제시점·전후그래프·장면슬라이더 제작 및 데스크톱/모바일/3상호작용 검증. 감사 참고4.467초/사용자2.093초의 준비·마무리 범위 비대칭 확인; 멀다 대응33쌍 사용자0.900초 손얼굴옆/참고1.267초 대기장면 직접 확인(관절-RGB시각차0). 비용감소를 단계정렬정확도/교정정확도로 해석하지 않음. 감사오른굽힘차이11.19→12.63도. 현재DTW원래관측과C/T3F입력 별개. TCN에는DTW/고정길이재표본화없음,시간합성곱/풀링/T3F상대앞중뒤요약. 기존12시험속도미통제,30→20/15fps시험은속도시험아님. 사용자자료설명평가만,학습/튜닝/선택0. 105출처/32보호해시유지. 최초표시검사 평균FPS오해보존후실제frameID/PTS검사로복구,경로수치불변. 같은Codex산술검산/시각검토,독립사람정답아님. 새학습/추론/검출/운영변경0,활성실행없음.'''
    state=S.read(statepath);state['updated_date_kst']=at;state['current_stage']=stage;state['next_single_action']=next_action
    state[U.name]=dict(status=stage,at=at,summary=concise,report=str(U/'RESULT_KO.html'),sources=105,protected=32,
        saved_comparison_matches=12,independent_scalar_dp=12,metric_checks=96,details=3,
        user_data_use='Explicitly authorized existing-recording explanatory evaluation only; never training/tuning/selection',
        training=0,classifier_inference=0,detector_inference=0,production_changes=0,next_single_action=next_action)
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    ctx=ctxpath.read_text(encoding='utf-8-sig');title='### 2026-10-27 既存촬영 DTW 정렬 시각화 및 기본경로 재점검'.replace('既存','기존')
    evidence=f'근거: research/front_keyphase_v1/ops/{U.name}/RESULT_KO.html, RESULT.json, VERIFIED.json, VISUAL_REVIEW.json, HTML_RENDER_VERIFIED.json, NEXT_PLAN.json.'
    for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',concise),('## 9. 다음 단일 작업',next_action)]:
        assert ctx.count(heading)==1
        ctx=ctx.replace(heading,heading+'\n\n'+title+'\n'+at+'\n'+body+'\n'+evidence+'\n',1)
    ctxpath.write_text(ctx,encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+concise+'\n\n다음 단일 작업: '+next_action+'\n\n'+evidence+'\n\n담당: Codex 구현·별도 산술검산·시각검토. 사용자 자료 접근은 이번 명시 요청의 설명평가에 한정. 전수재구성·검산 완료 결과 재실행하지 않고 재사용. 운영 적용이나 근본문제 해결 완료는 아님.\n',encoding='utf-8')
    assert S.read(statepath)['current_stage']==stage
    assert U.name in ctxpath.read_text(encoding='utf-8') and U.name in handoff.read_text(encoding='utf-8')
    S.check(f['sources']);S.check(f['protected'])
    S.write('CLOSURE.json',dict(at=at,status='DONE_EXPLANATORY_EVIDENCE_ONLY',synced=[str(p) for p in [statepath,ctxpath,handoff]],protected_hashes_preserved=32,source_hashes_preserved=105))
    print('CLOSED',flush=True)

if __name__=='__main__':main()
