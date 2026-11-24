"""Record scope, review limits, and current project handoff. No model mutation."""
from pathlib import Path
import json,datetime
U=Path(__file__).resolve().parent; ROOT=U.parents[3];O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.datetime.now().astimezone().isoformat()
write(U/'AUTHORIZATION.json',{'at':now,'user_request':'기존 공개 수어 영상 실제 손 누락에서 HaMeR 손목/손가락 정확성을 각각 검증, 정상손/교차 유지, C·T3-F 유지, 사용자촬영 학습튜닝금지','implementation':'Codex local research evaluation and report','training':False,'deployment':False,'user_recordings':False,'originals_read_only':True,'past_predictions_reused':True})
write(U/'VISUAL_REVIEW.json',{'at':now,'reviewer':'same Codex, not independent expert','source_midpoints_reviewed':16,'source_detail_views_reviewed':16,'fourway_overlays_reviewed':16,'context_E11_frames':[28,30,32,34,36],
 'annotation_issue':'E11 initial5tip IDs uncertain, source context cannot establish exact identity. Kept first files; final all5 held. No region moved or model tuned. After-output review limitation disclosed.',
 'positive_examples':{'E01':'Separate fist/thumb estimated while existing open hand stays','E08':'Missing right wrist/thumb supported within visual range','E16':'Open hand estimated alongside existing fist; normal overlap example'},
 'failure_examples':{'E05':'Existing slot0 fingers follow visible open right hand; HaMeR side1 follows same fingers. Existing shifted wrist maps current arm0; new wrist arm1 => early different_current_arms bypasses duplicate tests. Two skeletons for same open hand, fist not correctly reconstructed.','E15':'Wrist outside visual reference region23.88px though temporal check passed. Blurred source prevents fingertip accuracy validation.','E02_E06':'Visually different open-hand/fist candidates held by ambiguous_overlap because existing current arm=-1. Potential excessive holding, not proof all should be accepted.','E12':'HT holds large pose transition; source interlocked fingers cannot independently prove temporal decision correct.'},
 'preservation_interpretation':'All prior points and20designated crossing frames unchanged; not proof old detections correct. Added duplicate E05 remains a blocker.',
 'browser':'Static HTML/image/link checks only. Prior fileURL automation denial not bypassed; browser render not verified.',
 'decision':'Complete diagnostic, not ready for deployment or fine-grained angle feedback'})
scores=read(U/'REVIEWED_SCORES.json');ver=read(U/'VERIFICATION.json')
next_action='먼저 손목/팔 연결이 다르다는 이유만으로 통과시키는 양손 통합 분기를 점검한다. 기존 관측과 새 후보가 같은 실제 손의 손가락을 따라가는 충돌은 별도 보류하되, E01/E16 정상 겹침·지정20교차·E02/E06 과도한 보류 가능성을 함께 검증할 공통 후보를 준비한다. 단어별 예외/시각점수에 맞춘 문턱조절 금지. 손목 기준점 정의와 독립적으로 확인 가능한 손가락 근거를 정리한 뒤에만 연구 DTW 연결을 검토한다. C·T3-F·운영 유지, 사용자/멘토촬영 학습·튜닝·선택금지, 추가학습 없음.'
statepath=O/'STATE.json';state=read(statepath);key=U.name
state.update(updated_date_kst=now,updated_at=now,current_stage='MANO_MISSING_HAND_AUDIT_COMPLETE_DUPLICATE_FUSION_BLOCKER_NOT_DEPLOYED',next_single_action=next_action,resume_command=next_action,activity_state='AWAITING_USER_REVIEW',latest_visual_review=str(U/'RESULT_KO.html'))
state[key]={'updated_at':now,'status':state['current_stage'],'unit':str(U),'report':str(U/'RESULT_KO.html'),'implementation':'Codex','verification':'Separate scalar arithmetic by same Codex; no independent expert annotation','sample':scores['sample'],'results':scores['summary'],'preservation':ver,'critical_finding':'E05 baseline-side conflict and same open-hand duplicated in H/HT, despite original values preserved','annotation_limit':'Approximate source-image visual rectangles, E11 5digit IDs held after review; no unbiased accuracy claim','training':0,'new_inference':0,'user_data_access':0,'production_change':False,'C_T3F_change':False,'running_processes':[],'next_single_action':next_action}
write(statepath,state)
entry8='''### 2026-10-25 실제 공개 누락 HaMeR 손목·손끝 검증 — 양손 통합 충돌 발견·미채택
- ops/word100_mano_missing_hand_accuracy_20261025: 기존7영상4명5표현812프레임, 빈손칸57프레임을16연속구간으로 묶어 중앙1장씩 RGB검토. 기존B/HaMeR/H/HT저장추론재사용, 새학습/새추론0.
- 임시시각범위 손목16곳: H14출력/4범위안/10범위밖/2보류, HT13/4/9/3. H범위밖평균6.24px/최대23.88px,중심평균23.69px. 피부-뼈중심 정의와 시각불확실성 때문에 정확GT오차/정답률 아님.
- 손끝80자리중17곳만확인가능(63보류),H15출력14범위안1약1px밖2보류;HT동일. E11초기5손끝ID불확실을앞뒤RGB재검토후전부보류. 초기22점SCORES보존·정확도근거금지. 최종REVIEWED_SCORES도동일Codex후검토편향한계,전문가독립GT아님.
- 중요 E05후원: 기존왼손칸의손가락과HaMeR오른손이같은실제펼친손을중복표현. 손목50.65px떨어져다른팔판정→relation의different_current_arms조기통과. 손끝거리는2.01~17.15px. 기존값보존과정확한양손통합은별개. E02/E06은기존팔-1로ambiguous_overlap보류,과도보류가능성도함께검토필요.
- 기존14149점/지정교차20프레임840점/정상대조94프레임불변,과거542파일/보호32불변. H33/HT31전체추정추가는정확복원수아님. 손가락각도/숨은3D/TCN·DTW성능미검증. 운영미채택.
- 원본/16비교화면시각확인,별도스칼라검산통과. HTML정적링크·문법/이미지검토,브라우저렌더미검증. C·T3-F/운영DTW웹/사용자자료변경0. 근거 REVIEWED_SCORES/ANNOTATION_REVIEW/RELATION_DIAGNOSTIC/VERIFICATION/VISUAL_REVIEW.

'''
entry9='### 2026-10-25 실제 누락 검증 이후 — 양손 통합 충돌부터 해결\n'+next_action+'\n\n'
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text('utf-8')
assert entry8.splitlines()[0] not in text
for section,entry in [('## 8.',entry8),('## 9.',entry9)]:
 start=text.index(section);insert=text.index('\n',start)+1;text=text[:insert]+'\n'+entry+text[insert:]
ctx.write_text(text,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}
현재 목표: 실제 공개 수어 누락 장면에서 HaMeR의 위치/손가락 보완 정확성과 양손 통합을 검증. C·T3-F 유지, 사용자/멘토촬영 학습·튜닝·선택금지.
마지막 완료: {U}
RESULT_KO.html, REVIEWED_SCORES.json, RGB_REFERENCE_REGIONS.json/REFERENCE_PIN.json(초기), REVIEWED_REFERENCE_REGIONS.json/REVIEWED_REFERENCE_PIN.json(최종), ANNOTATION_REVIEW.json, RELATION_DIAGNOSTIC.json, VERIFICATION.json, VISUAL_REVIEW.json.

기존7영상4명5표현812프레임,빈손칸57프레임→16연속구간중앙1장. 일부는물리적손누락뿐아니라좌우칸불일치(E05). 새추론/학습0; word100_mano_threeway_20261025 저장B/H/HT/raw_predictions 재사용. 원본W/과거결과읽기전용.
손목H14/16출력4범위안10범위밖2보류;HT13출력4안9밖3보류. H범위밖평균6.24px/최대23.88px,중심평균23.69px. 손목의시각적연결범위는정확한MANO뼈중심GT아님.
손끝80자리중17시각범위,63불확실. H15출력14안1약1px밖2보류;HT동일. E11초기5손가락번호를확정못해원본28/30/32/34/36재검토후모두보류. 초기22점SCORES삭제금지/정확도근거금지. 예측에맞춰상자이동/번호바꾸기없음. 동일Codex결과후재검토라편향한계,독립정밀정확도보장없음.

핵심통합실패 E05(후원 REAL01 f82 right): 기존slot0의손가락이실제펼친오른손을따라가며손목은왼손쪽으로치우침. 새HaMeRside1도같은펼친손. current_arm0vs1로relation이different_current_arms즉시반환하여중복검사우회. 손목거리50.65px,5손끝거리2.01~17.15px. H/HT모두중복추가. 원래값보존만으로통합성공판정금지.
정상반례 E01/E16은별개주먹+펼친손이어합침가능. E02/E06은기존current_arm=-1,ambiguous_overlap으로보류되며과도보류가능성. E12은자세변화로HT추가보류,정답판정으로오해금지. E15는시간검사통과해도손목범위밖23.88px.
전체H33/HT31추정추가(정확복원수아님). 원래14149점/지정교차20프레임840점/정상대조94프레임전배열불변. 과거542결과+32보호파일불변. C·T3-F/운영DTW웹/사용자자료변경0. 별도scalar검산(동일Codex),16원본/16overlay검토. HTML정적확인,브라우저렌더미검증(기존fileURL자동화차단우회안함).
실행완료: 준비53743종료0,검산종료0. 현재진행프로세스없음. 최초검산NaN비교false는검산기equal_nan수정으로해결,입력변경없음。

다음 한 작업: {next_action}

이전외부GT참고: word100_mano_groundtruth_check_20261025/CORRECTED_SCORES.json만최종. 공개32정지사진은MP모두검출→누락보조효과검증아님. 그외부자료초기Right오류와이번E11주석불확실은별개이며각원본기록보존.
운영교체/카메라/다운로드/Git/외부원본전송금지. MANO/HaMeR기존격리자산재사용. 원본/과거동결파일수정금지.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
print('synced latest sections8/9,STATE,NEW_CHAT_HANDOFF')

