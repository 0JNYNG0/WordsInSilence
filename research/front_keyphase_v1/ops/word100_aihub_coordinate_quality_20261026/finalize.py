"""Preserve previous milestone and synchronize this source-quality audit."""
import study as S,json
from datetime import datetime,timezone,timedelta
U=S.U
def run():
    v=S.read(U/'VERIFIED.json');render=S.read(U/'HTML_RENDER_VERIFIED.json');f=S.read(U/'FROZEN.json')
    assert v['crc_rechecked']==9565 and v['clips']==100 and render['status']=='PASS_HEADLESS_CHROME'
    S.check(v['pins']);S.check(f['protected']);S.check(S.read(U/'REPORT_MANIFEST.json')['files'])
    assert (U/'VISUAL_REVIEW.json').exists()
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();unit=U.name
    summary='AI Hub 다운로드의 [라벨]02_real_word_keypoint.zip(9.70GB)에서 현재100개단어 TRAIN REAL02 정면100영상의 좌표품질 확인 완료. 영상11724프레임/제공JSON9565개 원문CRC·SHA·배열·집계 별도검산. 현재영상과프레임번호집합일치66개;뒤쪽좌표파일부재28개·중간누락1개·영상끝초과6개(중복항목). 10개는MP대비손목차이중앙값75~116px로추가점검대상이며정답오차율아님. 결과전선정12영상89장면+진단2영상8장면Codex시각검토. 후원·수영등MP누락손의제공좌표에활용가치;감사·무산은정지RGB에서도제공좌표어긋남. 영상·제공본/좌표·시간대응원인미확정. OpenPose기반추출→3D/재투영→수작업보정공식문서확인. confidence1을정확도100%로해석금지;3D크기약1000배차이와3종카메라설정있으나2D불일치원인으로확정하지않음. 전체를정답으로쓰는조건불통과,추가다운로드/학습보류. 검증된점만참고답안으로오답노트구성계획작성. 별도검산은같은Codex의산술검산이며독립사람21점정답검증아님. C/T3F/E3·운영32보호핀유지,개인/멘토접근·새검출·새학습0. 보고서15이미지/100행데스크톱·모바일검증PASS. OpenCV비연속배열최초오류보존후동일값contiguous변환으로복구. 이번완료는보유자료품질감사이며인식성능개선/실제웹적용아님.'
    nextstep='최신우선작업은AIHub제공좌표를검증된참고답안으로쓰는준비이다. word100_aihub_coordinate_quality_20261026의FROZEN/VERIFIED/DIAGNOSTICS/VISUAL_REVIEW/NEXT_PLAN과저장100배열·14영상재사용,완료9565전수CRC/100처리재실행금지. 먼저현재다운로드제공본과대응영상의프레임·카메라·3D단위정의를확인하고,뒤쪽좌표부재28개·중간누락1개·큰위치불일치10개를정답으로쓰지않는다. 단순MP맞춤이동·시간늘리기로수정정답생성금지. 보이는관절만확인/가림·불확실/불일치로구분한오답노트와작은독립위치평가표본을준비한다. 손목이확인됐어도가린손가락정답으로확대하지않는다. 현재품질조건은일괄정답활용불통과이므로추가다운로드·학습안함;적합성확인후필요한다른TRAIN ID좌표만선택확보하는사용자조건부승인유지. 기존빠른시간변환진단은후순위보존. C/T3F/E3·웹·DTW·원본보존,사용자/멘토학습·튜닝·선택금지. 현재활성실행없음.'
    root=S.O.parents[2];context=root/'WordsInSilence_PROJECT_CONTEXT.md';handoff=root/'NEW_CHAT_HANDOFF.md';state=S.O/'STATE.json'
    before=U/'before';before.mkdir(exist_ok=True)
    for p in [context,handoff,state]:
        out=before/p.name
        with out.open('xb') as b:b.write(p.read_bytes())
    title='### 2026-10-26 AI Hub 제공 관절 좌표 품질 감사 및 오답노트 활용 계획'
    evidence='근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html, VERIFIED.json, DIAGNOSTICS.json, VISUAL_REVIEW.json, NEXT_PLAN.json, HTML_RENDER_VERIFIED.json.'
    text=context.read_text(encoding='utf-8-sig');assert title not in text
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nextstep)]:
        assert marker in text;text=text.replace(marker,marker+'\n\n'+title+'\n'+at+'\n'+body+'\n'+evidence+'\n',1)
    context.write_text(text,encoding='utf-8')
    st=S.read(state);st['updated_date_kst']=at;st['current_stage']='AIHUB_COORDINATE_QUALITY_AUDITED_SUPERVISION_NOT_READY';st['next_single_action']=nextstep
    st[unit]=dict(status='AUDIT_COMPLETE_DOWNLOAD_AND_TRAINING_GATE_NOT_PASSED',at_kst=at,summary=summary,next=nextstep,report='ops/'+unit+'/RESULT_KO.html',videos=100,signer_ids=[2],split='TRAIN',json_files=9565,source_frames=11724,exact_frame_sets=66,missing_tail_clips=28,middle_gap_clips=1,extra_frame_clips=6,high_disagreement_clips=10,visual_clips=14,visual_frames=97,independent_human_groundtruth=False,training=0,detector_inference=0,personal_data_access=0,additional_download=0,operating_changed=False,active_sessions=[])
    state.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음 단일 작업: '+nextstep+'\n\n'+evidence+'\n\n작성·실행·별도산술검산·시각검토: Codex. 원본/보호모델 변경없음. exec46947 처리,30257 시각자료,57127 검산 모두EXIT0. diagnostics 첫실패는 FIRST_ERROR_diagnostics.txt 보존; 수정후EXIT0. report/check_report 종료0. 사용자최신질문은제공좌표를오답노트/검증지표로쓸수있는지이며답변·NEXT_PLAN에조건부활용구체화. 전체인식/피드백목표는미완료.\n',encoding='utf-8')
    S.write('CURRENT_EXECUTION.json',dict(at=at,status='AUDIT_COMPLETE_VERIFIED_REPORTED',active_sessions=[],training=0,additional_download=0))
    assert S.read(state)[unit]['videos']==100 and nextstep in handoff.read_text(encoding='utf-8')
    S.write('SYNC_CHECK.json',dict(status='PASS',at=at,state_key=unit,files={str(p):S.sha(p) for p in [context,handoff,state]},protected_count=32))
    print('SYNC_DONE',flush=True)
if __name__=='__main__':run()
