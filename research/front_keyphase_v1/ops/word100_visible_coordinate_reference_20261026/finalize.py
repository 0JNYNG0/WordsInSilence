import json
import prepare as S
U=S.U
def run():
    f=S.read(U/'FROZEN.json');v=S.read(U/'VERIFIED.json');render=S.read(U/'HTML_RENDER_VERIFIED.json');m=S.read(U/'REPORT_MANIFEST.json')
    assert v['status']=='PASS_SOURCE_AND_RECTANGLE_ARITHMETIC' and render['status']=='PASS_HEADLESS_CHROME'
    S.check(f['protected']);S.check(m['files']);S.check(m['images'])
    S.write('VISUAL_REVIEW.json',dict(at=S.now(),status='ALL18_RGB_AND_COMPARISON_VIEWS_REVIEWED',frames=18,clips=6,source_regions=70,withheld=38,independent_human_groundtruth=False,notes='All RGB-only grids and six region sheets were inspected before annotations were frozen. All six subsequent A/B/C comparison sheets inspected. Overlapping figure legends corrected for readability; no changes to points, regions or numeric results.'))
    at=S.now();unit=U.name;root=U.parents[3];context=root/'WordsInSilence_PROJECT_CONTEXT.md';handoff=root/'NEW_CHAT_HANDOFF.md';state=U.parent/'STATE.json'
    st=S.read(state);assert st['current_stage']=='AIHUB_COORDINATE_ALIGNMENT_VERIFIED_SELECTIVE_OFFSETS_NOT_ADOPTED'
    summary='공개 TRAIN REAL02 기존6표현의 이전검토와다른동작18시점에서 RGB위치평가표 준비·검증완료. 손목/엄지끝/검지끝108항목중70개예비2D범위표시,38개가림·흐림·구분불가보류. 후보좌표없는RGB로표시후동결했으나이전클립결과는알고있고동일Codex시각검토이므로독립사람정답/완전블라인드아님. 범위내A원본28/70,B일정이동41/70,C매시점손목맞춤25/70(인식률·관절정확도%아님). B새범위내18/새범위밖5;불확실성범위까지감안한확실히가까워짐21/나머지49판정보류. 감사0/15→11/15,무산0/10→1/10,수영14/14→10/14. 원본유지검토28/보정검토12/보이는좌표불일치30/가림불확실38의REFERENCE_CATALOG 생성,학습정답승인0. 전체표본손목조건0/6통과이며부분개선과구분. 원본배열변경없음,32보호파일유지. 새학습·검출·다운로드·운영변경·사용자/멘토접근0. 별도거리/집계검산PASS,빈엄지그룹None검산오류는최초코드·로그보존후수정,동결기준/결과불변. 6비교이미지/108항목표데스크톱·모바일검증PASS.'
    nextstep='다음단일작업: word100_visible_coordinate_reference_20261026의FROZEN/ANNOTATIONS/RESULT/VERIFIED/REFERENCE_CATALOG/NEXT_PLAN을고정평가근거로재사용하고,어깨·팔꿈치이동차이의여러시점일관성과원본양호보존을확인하는공통선별규칙을정의·검증한다. 단어명/사용자촬영을조건으로쓰지않고기준을결과전에고정한다. 실제새공개시점과비교하여상수이동으로줄어드는전역밀림/개별손모양오류/시간공백을구분한다. 현재12보정점은검토후보일뿐교정완료정답아님,38불확실점보간금지. 이번표시는같은Codex예비영역이며독립사람GT로보고금지. 전체단어수작업검토로확장하지않고작은기준표로공통품질규칙검증. 이전완료100배열/9565JSON감사재실행금지,모든점일괄이동/매순간손목맞춤미채택. 추가다운로드·학습은품질조건충족후조건부승인범위로이어가되현재미실시. 기존C/T3F/E3·웹·DTW·원본보존,개인/멘토학습·튜닝·선택금지. 활성실행없음.'
    before=U/'before';before.mkdir(exist_ok=True)
    for p in [context,handoff,state]:
        with (before/p.name).open('xb') as w:w.write(p.read_bytes())
    title='### 2026-10-26 실제 RGB 위치 기준표 및 선별 보정 검토 목록 검증'
    evidence='근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html, ANNOTATIONS.json, REFERENCE_CATALOG.json, VERIFIED.json, NEXT_PLAN.json, HTML_RENDER_VERIFIED.json.'
    text=context.read_text(encoding='utf-8-sig');assert title not in text
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nextstep)]:
        assert marker in text;text=text.replace(marker,marker+'\n\n'+title+'\n'+at+'\n'+body+'\n'+evidence+'\n',1)
    context.write_text(text,encoding='utf-8')
    st['updated_date_kst']=at;st['current_stage']='VISIBLE_RGB_REFERENCE_CATALOG_VERIFIED_NOT_TRAINING_TARGETS';st['next_single_action']=nextstep
    st[unit]=dict(status='COMPLETE_VERIFIED_REVIEW_CANDIDATES_ONLY',at_kst=at,summary=summary,next=nextstep,report='ops/'+unit+'/RESULT_KO.html',clips=6,frames=18,recording_ids=[2],requested_points=108,visible_regions=70,withheld=38,inside={'A':28,'B':41,'C':25},catalog={'original_review':28,'offset_review':12,'visible_unresolved':30,'visibility_uncertain':38},approved_training_points=0,independent_human_groundtruth=False,training=0,detector_inference=0,additional_download=0,personal_data_access=0,operating_changed=False,active_sessions=[])
    state.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음 단일 작업: '+nextstep+'\n\n'+evidence+'\n\n담당: Codex 구현·시각표시·실행·별도산술검산. 독립사람정답확보아님. prepare/annotate/evaluate/수정후verify/report/check_report EXIT0. 첫verify빈집계실패는 FIRST_ERROR_verify.json 및 verify_before_empty_group_fix.py 보존. 결과 재계산·재주석화없이이어서진행한다. 전체인식/피드백목표는미완료.\n',encoding='utf-8')
    S.write('CURRENT_EXECUTION.json',dict(at=at,status='COMPLETE_VERIFIED_NOT_ADOPTED',active_sessions=[],training=0,operating_change=False))
    assert S.read(state)[unit]['visible_regions']==70 and nextstep in handoff.read_text(encoding='utf-8')
    S.write('SYNC_CHECK.json',dict(status='PASS',at=at,state_key=unit,files={str(p):S.sha(p) for p in [context,handoff,state]},protected_count=32))
    print('SYNC_DONE',flush=True)
if __name__=='__main__':run()
