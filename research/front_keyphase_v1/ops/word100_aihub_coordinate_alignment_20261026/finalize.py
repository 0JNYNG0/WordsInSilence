"""Synchronize the verified diagnostic milestone; never deploy a correction."""
import json
import study as S
U=S.U
def run():
    v=S.read(U/'VERIFIED.json');render=S.read(U/'HTML_RENDER_VERIFIED.json');f=S.read(U/'FROZEN.json')
    assert v['videos']==100 and render['status']=='PASS_HEADLESS_CHROME'
    assert S.read(U/'VISUAL_REVIEW.json')['frames']==30
    S.check(f['protected']);m=S.read(U/'REPORT_MANIFEST.json');S.check(m['files']);S.check(m['images'])
    at=S.now();unit=U.name;root=U.parents[3]
    context=root/'WordsInSilence_PROJECT_CONTEXT.md';handoff=root/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
    st=S.read(state)
    assert st['current_stage']=='AIHUB_COORDINATE_QUALITY_AUDITED_SUPERVISION_NOT_READY','Latest state changed; inspect before synchronizing.'
    summary='공개 TRAIN REAL02 정면100영상의 AIHub 제공좌표 위치보정 세조건 검증 완료. A원본/B초반15프레임 어깨·팔꿈치로 구한 일정이동/C매시점 각손을 팔손목에 이동(실험C는 기존TCN C와별개). 보정계산에쓰지않은15프레임이후에서 기존큰불일치10개는B10/10 MP와가까워짐,나머지90개는27가까워짐/63멀어짐;손목맞춤C는나머지90중81멀어짐. 동작구간 감사손목68.31→8.06px,무산87.67→23.17px이나무산손끝104.83→93.20px로모양문제남음. 수치는MP와의차이이며실제정답오차·TCN인식률아님. B의손내부모양/양손관계/시간변화보존;C는양손관계·궤적변화,9534프레임중52에서가로손목순서부호변화(좌우배정오류52건확정아님). 동작구간20영상비교불가원인기존좌표/관측공백으로보존. 결과전선정6영상30시점RGB시각확인,100배열별도재구성·산술검산PASS. 같은Codex검산이며독립사람정답아님. 일정밀림선택보정가능성확인이나일괄보정/학습정답채택안함. C/T3F/E3·운영32보호핀유지,사용자/멘토접근0,새학습·새검출·다운로드·운영변경0. 보고서6이미지/100행모바일·데스크톱검증완료. 새오류없음.'
    nextstep='다음은 AIHub제공좌표의검증된참고답안준비: word100_aihub_coordinate_alignment_20261026의FROZEN/DONE/VERIFIED/VISUAL_REVIEW/NEXT_PLAN 및저장100배열·6비교화면재사용. 완료위치보정비교·이전9565JSON감사재실행금지. 실제RGB에서보이는손목·손끝만확인한작은위치평가표를만들고,초반어깨·팔꿈치에서구한일정이동이다른시간의실제보이는손에도맞는지검증한다. 단순MP일치로정답인증금지;동일Codex시각표시와독립사람정답구분. 전체좌표밀림/개별손오류/시간공백을분리하고공통적용·보류조건을결과전고정한다. 손가락가림은보류,손목확인을손모양정답으로확대금지,사용자손을참고위치로끌어맞춰실제피드백오차를지우지않는다. B전체자료적용및C매시점손목맞춤미채택. 팔방향회전·크기보정은미시험. 품질확인후필요한다른TRAIN ID자료추가확보·학습비교조건부승인유지하나아직시작안함. C/T3F/E3·웹·DTW·원본보존,사용자/멘토학습·튜닝·선택금지. 현재활성실행없음.'
    before=U/'before';before.mkdir(exist_ok=True)
    for p in [context,handoff,state]:
        with (before/p.name).open('xb') as w:w.write(p.read_bytes())
    title='### 2026-10-26 AI Hub 좌표 위치 보정 세조건 검증 — 일정 밀림 효과와 정상 회귀 분리'
    evidence='근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html, FROZEN.json, VERIFIED.json, VISUAL_REVIEW.json, NEXT_PLAN.json, HTML_RENDER_VERIFIED.json.'
    txt=context.read_text(encoding='utf-8-sig');assert title not in txt
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nextstep)]:
        assert marker in txt;txt=txt.replace(marker,marker+'\n\n'+title+'\n'+at+'\n'+body+'\n'+evidence+'\n',1)
    context.write_text(txt,encoding='utf-8')
    st['updated_date_kst']=at;st['current_stage']='AIHUB_COORDINATE_ALIGNMENT_VERIFIED_SELECTIVE_OFFSETS_NOT_ADOPTED';st['next_single_action']=nextstep
    st[unit]=dict(status='DIAGNOSTICS_COMPLETE_VERIFIED_NOT_ADOPTED',at_kst=at,summary=summary,next=nextstep,report='ops/'+unit+'/RESULT_KO.html',videos=100,recording_ids=[2],split='TRAIN',conditions=['source','fixed_global_translation','per_frame_per_hand_wrist_translation'],B_closer_previous_large=10,B_closer_other90=27,B_farther_other90=63,C_farther_other90=81,missing_action_comparison=20,visual_clips=6,visual_frames=30,independent_human_groundtruth=False,training=0,detector_inference=0,personal_data_access=0,additional_download=0,operating_changed=False,active_sessions=[])
    state.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음 단일 작업: '+nextstep+'\n\n'+evidence+'\n\n작성·실행·별도검산·시각검토: Codex. study freeze/run, visuals, verify(exec38168), report/check_report 모두EXIT0. 완료수치 재추론불필요. 원본자료와기존모델불변. 사용자최신제안의일정위치보정효과를검증한연구결과이며전체인식/피드백목표는미완료.\n',encoding='utf-8')
    S.write('CURRENT_EXECUTION.json',dict(status='COMPLETE_VERIFIED_NO_ADOPTION',at=at,active_sessions=[],training=0,operating_change=False))
    assert S.read(state)[unit]['videos']==100 and nextstep in handoff.read_text(encoding='utf-8')
    S.write('SYNC_CHECK.json',dict(status='PASS',at=at,state_key=unit,files={str(p):S.sha(p) for p in [context,handoff,state]},protected_count=32,final_tools={str(U/n):S.sha(U/n) for n in ['check_report.cjs','finalize.py','HTML_RENDER_VERIFIED.json']}))
    print('SYNC_DONE',flush=True)
if __name__=='__main__':run()
