"""Preserve explicit user meaning/form decision and close the unmerged source branch."""
from locate_source_frames import H,L,read,sha,write
from datetime import datetime,timezone,timedelta
import json

def main():
    now=datetime.now(timezone(timedelta(hours=9)));stamp=now.isoformat();human=now.strftime('%Y-%m-%d %H:%M KST')
    pending=read(H/'PENDING_SEMANTIC_REVIEW.json');located=read(H/'SOURCE_LOCATION_RESULT.json');audit=read(H/'SOURCE_LOCATION_AUDIT.json')
    assert audit['status']=='PASS_SOURCE_LOCATIONS_ONLY' and audit['result_sha256']==sha(H/'SOURCE_LOCATION_RESULT.json')
    decision=dict(status='DIFFERENT_WORD_AND_ACTION_CONFIRMED_EXCLUDE_FROM_INSA_TRAINING',at_kst=stamp,
      user_quotes=["너가 물어봤던 '인사'와 '안녕하세요'는 비슷한 뜻이어도 수어 동작은 다른 동작이야.","아무튼 다른 단어, 동작이야"],
      compared_videos=pending['videos'],source_label='안녕하세요',source_sentence_id='SEN0354',proposed_target_id='INSA',
      semantic_equivalence_confirmed=False,different_word_and_action_confirmed=True,
      semantic_target_id=None,training_approved=False,merge_allowed=False,question_resolved=True,ask_same_question_again=False,
      scope='Only this proposed 안녕하세요→인사 mapping. Does not revoke prior apology or KSL study/eat confirmations or prohibit verified variants in general.',
      source_metadata_retained=True,source_registered=False,new_training=0,model_changed=False,
      independent_location_result=str(H/'SOURCE_LOCATION_RESULT.json'),location_audit=str(H/'SOURCE_LOCATION_AUDIT.json'))
    write(H/'SEMANTIC_DECISION.json',decision)
    alternatives=dict(at_kst=stamp,status='BOUNDED_NEW_PRIMARY_SOURCE_REVIEW_NO_ELIGIBLE_NEW_INPUT',
      sources=[
        dict(url='https://github.com/ChelseaGH/KSL-Guide',finding='Author repository intro/download both resolve to AIHub dataset103 already used; not an independent new source.',decision='Do not duplicate source or repeat103 acquisition checks under a new name.'),
        dict(url='https://www.eis-lab.org/yonsei-ksl-dataset',finding='Author lab describes50 words,20 participants,depth-only videos; vocabulary in archiveREADME. No ready compatibleRGB/2D hand-coordinate inputs verified.',decision='Do not pivot to a new depth-input pipeline or bulk-download on this description.'),
        dict(url='https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60',finding='Producer page lists60 exact labels; none directly equals the current18 target names. Signer diversity and equivalents not verified.',decision='No media downloaded or added; not evidence for target-person coverage.'),
        dict(url='https://aclanthology.org/2022.sltat-1.9.pdf',finding='NIASL2021 authors describe signer IDs,aligned per-hand glosses andOpenPose coordinates. This paper does not provide verified locally accessible target-word training members.',decision='Potential corpus metadata only; no access application or source adoption.'),
        dict(url='https://eq4all-data.github.io/',finding='Author guide site currently says under construction; no accessible file-level guide or signer table at that page.',decision='No repeat of this unchanged page.'),
        dict(url='https://huggingface.co/PotatoPasta/ksl-sign-keypoint-embedder',finding='Uploader claims AIHub-derived3000-word signer-separated140-dimensional model. Its preprocessing description and evaluation were not independently validated; model-card metrics are not our results.',
             file_listing='https://huggingface.co/PotatoPasta/ksl-sign-keypoint-embedder/tree/main',file_listing_result='web tool401 Unauthorized; no files or code acquired and no bypass/retry',decision='No replacement or training-data adoption from a model card.')],
      known_route_reused=dict(path=str(L/'evidence/sentence_target_catalog_20260919/aihub636_access_stop.json'),finding='Existing dataset636 application/terms gate remains outside authorization; original failed sample click is not a verified acquired sample. No retry.'),
      new_media_or_model_downloads=0,inquiries=0,applications=0,agreements=0,training_runs=0,
      conclusion='No newly verified same-target signer-separated input from these inspected sources. This is bounded inspection, not a claim that no other dataset or modeling method exists.')
    write(H/'ALTERNATIVE_SOURCE_REVIEW.json',alternatives)
    report='''# 인사 관련 출처 검토 종료 — 서로 다른 단어로 유지

사용자가 두 차례 답변했으며, 마지막 답변은 **“아무튼 다른 단어, 동작이야”**다. AI Hub의 ‘안녕하세요’를 기존 ‘인사’로 묶는 제안을 종료한다. 같은 의미를 묻는 질문도 해소됐으며 다시 요청하지 않는다. 기존 사과·공부·먹다의 별도 유효 표현 확인에는 영향을 주지 않는다.

답변을 기다리는 동안 의미 판단과 독립적인 파일 위치 확인만 진행했다. 보유12개 문장 관절 ZIP에서 촬영자01–09/11–13의 SEN0354 정면12개, 총1,405개 프레임 파일 위치를 찾았다. 해당 멤버 압축 크기는7,208,330바이트, 압축 전15,778,977바이트다. 새 다운로드가 아니라 기존 ZIP 내부 위치를 확인한 것이다.

검증된 기존 중앙 디렉터리 도구를 재사용하고, 별도로1,405개 로컬 파일 헤더를 읽어 이름·촬영자·오프셋을 대조했다. 압축된 관절 본문은 읽지 않았다. 실제좌표·CRC·시간정렬·학습입력 검증 완료를 뜻하지 않는다. 위치 확인/검산 프로세스는 모두 종료0, 위치 검산65항목 PASS다. 과거선택적인덱스0건은해당원본부재가아니었음이확인됐다.

이 출처는 현재인사학습에편입하지않는다. 원천라벨안녕하세요와미승인매핑을유지하고확인한파일위치는보존한다. 입력변환·학습·추론·사용자/보류관측열람·모델교체는0이다.

대체출처의생산자문서도제한적으로확인했다. [KSL-Guide](https://github.com/ChelseaGH/KSL-Guide)는이미사용하는AIHub103으로연결된다. [연세대자료](https://www.eis-lab.org/yonsei-ksl-dataset)는depth-only라현재RGB/2D관절입력에바로쓸수있는자료로확인되지않았다. [공개60영상목록](https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60)은목표18과직접일치하는라벨이없었다. [NIASL2021논문](https://aclanthology.org/2022.sltat-1.9.pdf)의사람ID/관절주석설명은유망한출처정보지만[저자안내사이트](https://eq4all-data.github.io/)는구축중이며이번에유효입력을확보하지못했다. [다른모델배포자의설명](https://huggingface.co/PotatoPasta/ksl-sign-keypoint-embedder)도학습자료나검증된대체모델로간주하지않았다. 상세와접근한계는ALTERNATIVE_SOURCE_REVIEW.json이다.

다음추가학습에는정확한대상단어·실제사용가능한입력·출처/촬영자분리가확인된새근거가필요하다. 기존AIHub WORD자료의촬영자별관절입력은로컬미확보, KSL77 파일↔촬영자대응은미확정이다. 기존자료기반방법개선전체가이론적으로불가능하다는뜻은아니지만, 끝난비교를반복하거나유효자료없이추가학습을시작하지않는다. 의미가다른자료를합치거나목표18을축소하지않는다.

이번진전은12개파일위치확정과사용자답변에따른잘못된학습매핑차단이다. 새인식성능개선이나프로젝트완료는아니다. 문의·신청·약관동의·대용량다운로드·삭제·백업·Unity·예약없음.
'''
    with (H/'RESULT.md').open('x',encoding='utf-8') as f:f.write(report)
    state_path=L/'ops/STATE.json';s=read(state_path)
    s['greeting_semantic_gate_20260922'].update(status=decision['status'],response=decision['user_quotes'][-1],semantic_decision=str(H/'SEMANTIC_DECISION.json'),
        question_resolved=True,source_frames_resolved=True,source_directory_members_located=1405,source_payload_validated=False,
        source_location_result=str(H/'SOURCE_LOCATION_RESULT.json'),source_location_audit=str(H/'SOURCE_LOCATION_AUDIT.json'),new_training=0,semantic_target_id=None)
    s['source_route_review_20260922']=dict(record=str(H/'ALTERNATIVE_SOURCE_REVIEW.json'),new_eligible_inputs=0,media_downloads=0)
    s['updated_at']=stamp;s['updated_date_kst']=now.strftime('%Y-%m-%d')
    s['current_stage']='GREETING_DIFFERENT_WORD_CONFIRMED_SOURCE_NOT_ADOPTED_LOCATION_METADATA_PRESERVED'
    s['next_single_action']='A new same-target,actually accessible,provenance/signer-separated source input is needed for the next data-addition comparison. Greeting semantic question is resolved negatively; do not repeat it or merge안녕하세요 into인사. Do not rerun closed model comparisons,known failed access paths or author-metadata searches unchanged.'
    s['autonomous_workers_current_unit']=dict(unit=H.name,source_location_exit_code=0,location_audit_exit_code=0,all_current_unit_workers_finished=True,work_paused_by_user=False)
    s['autonomous_goal_status'].update(status='active',at_kst=stamp,project_complete=False,user_requested_pause=False,
       previous_goal_turn_classification='progress: frozen head replay completed; semantic source question prepared',
       progress_this_turn='Located12 native source clips by metadata and verified1405 headers; explicit user different-word decision recorded; excluded mapping; bounded alternative primary-source review found no eligible new input.',
       fresh_blocked_audit_count=0,remaining_dependency='Eligible new source/provenance or independent validation conditions; not a pending greeting answer')
    s['goal_status']='INCOMPLETE: greeting mapping rejected by explicit user judgment. Existing recognition regressions remain; multi-person18-target web evaluation,validated execution feedback,and validated Unity integration are still incomplete.'
    state_path.write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    root=L.parents[1]/'WordsInSilence_PROJECT_CONTEXT.md';text=root.read_text(encoding='utf-8-sig');lines=text.splitlines();lines[2]='최종 갱신: '+human;text='\n'.join(lines)+'\n'
    current=f'''### {human} 최우선 — 인사/안녕하세요는 다른 단어로 사용자 확정, 병합 제외

- 사용자최종답변“아무튼 다른 단어, 동작이야”를원문보존했다. SEN0354안녕하세요→INSA인사매핑null/학습미승인유지, 의미질문종료·재질문금지. 앞선사과/공부/먹다유효표현확인을번복하지않는다. `ops/greeting_semantic_gate_20260922/SEMANTIC_DECISION.json`.
- 답변전독립적인메타데이터위치확인으로보유ZIP12개에서01–09/11–13의정면12개·1405프레임파일을찾고로컬헤더1405개대조65PASS·두프로세스종료0. 관절본문/CRC/좌표/입력구성검증은미실행. 출처위치는보존하되이번인사학습편입0이다.
- 대체생산자문서한정검토도새유효입력0. KSL-Guide는기존AIHub103,연세depth-only와공개60영상은직접호환/대상범위미충족,NIASL2021저자안내는구축중,다른모델파일목록401로미확인이다. 동일경로재시도없음. `ops/greeting_semantic_gate_20260922/ALTERNATIVE_SOURCE_REVIEW.json`.
- 새학습/추론/모델교체/사용자·보류관측/새미디어다운로드/문의/신청/약관/카메라/화면/Unity/삭제/백업/예약0. 이번은source판정진전이며인식개선·전체완료아님. 모든worker종료, 목표active, 사용자일시중단요청아님.

'''
    nxt=f'''### {human} 다음 단일 작업 — 동일 대상 단어의 유효한 추가 입력 조건 확보

- 인사↔안녕하세요질문은부정답변으로종료다. 새학습에편입하거나같은질문/위치검색을반복하지않는다. 의미가같은유효표현은가능하다는원칙을유지하지만이번쌍은사용자가다른단어로확정했다.
- 다음자료추가비교에는실제접근가능한동일대상입력과촬영자/출처근거가필요하다. 기존KSL77사람명세미확정·AIHub WORD관절로컬미확보·AIHub636신청/약관금지경계를유지한다. `ops/greeting_semantic_gate_20260922/ALTERNATIVE_SOURCE_REVIEW.json`의완료탐색을재실행하지않고새근거/허용된접근조건이생긴경로만검토한다.
- 기존자료방법개선전체가불가능하다는주장은하지않는다. 다만이미닫힌미세조정/시간변화/head웹비교를새seed/계수/사용자오답별수정으로반복하지않으며근거없는모델탐색을진전으로세지않는다. 사용자평가전용·목표18/93출력·보류보존·Claude구현역할·기본교체전검토유지.
- 이번목표턴은실제위치확인/사용자판정반영이있어progress다. 이후같은유효입력부족으로진전불가가이어지면현재목표의blocked감사를적용하고불필요한조회/보고로활성상태를유지하지않는다. 프로젝트완료/사용자pause로오인하지않는다.

'''
    text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+current,1)
    text=text.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+nxt,1);root.write_text(text,encoding='utf-8')
    files=[q for q in H.rglob('*') if q.is_file()]
    write(H/'CLOSEOUT.json',dict(status='COMPLETE_UNMERGED_SOURCE_BRANCH',at_kst=stamp,semantic_decision=decision['status'],
       source_locations=12,frame_headers_verified=1405,payloads_read=0,training_runs=0,
       hashes={str(q):sha(q) for q in files},project_complete=False,all_current_unit_workers_finished=True))
    assert read(state_path)['greeting_semantic_gate_20260922']['question_resolved']
    print(dict(status='CLOSED',semantic_decision=decision['status'],at_kst=stamp,project_complete=False))

if __name__=='__main__':main()
