from common import *
import collections
c=read(U/'CATALOG.json');v=read(U/'INDEPENDENT.json');assert v['status']=='PASS'
notes={
'WORD0003':'표본에서 가슴 앞 양손 위치가 번갈아 달라집니다. P1/P2는 한 반복 흐름을 나눈 것일 수 있어 필수 단계 수로 세지 않습니다.',
'WORD0106':'양손을 가슴 부근에 둔 모습 → 한손이 입 가까이 이동 → 가슴 앞으로 돌아오는 변화가 보입니다. 반대손 내림은 실제 영상에서 보이나 생략 가능 여부는 미확인입니다.',
'WORD0196':'양손 분리 → 얼굴 앞 접근 → 한손 전방 동작이 표본에 보입니다. P2/P3은 접근 동작을 중간에서 나눴을 수 있어 합치기 검토가 필요합니다.',
'WORD1139':'P1/P2 모두 가슴 앞 손 접촉 흐름으로 보입니다. 자동 2구간이 의미상 2단계라는 근거는 없으며 합치기 우선 검토 대상입니다.',
'WORD0185':'한손이 머리 옆에 있다가 가슴 앞 양손 접촉으로 바뀝니다. 중간 표본에서 실제 손이 보여도 저장 관절이 없는 장면이 있어 관측과 생략을 구분해야 합니다.',
'WORD0332':'얼굴 양옆 → 가슴 앞 → 얼굴 가까이의 위치와 손가락 모양 변화가 보입니다. 이동 과정·유지 자세 중 무엇이 필수인지 수어 검토가 필요합니다.',
'WORD1545':'가슴 앞 팔·손 겹침이 이어집니다. 같은 연속 동작이 4구간으로 나뉘었을 수 있고 일부 손 관절은 누락됩니다. 미검출을 단계 누락으로 판단하지 않습니다.',
'WORD1675':'초반 가슴 앞 양손, 중간 입 근처 한손, 후반 양손 접촉이 보입니다. 도시명 구분의 정확한 구간과 손모양은 단어 이름을 나눠 추측하지 않고 확인해야 합니다.',
'WORD1880':'초반 손 올림과 후반 손 내림도 기존 주석 안에 들어 있습니다. 중간 입 근처 한손과 후반 양손 접촉이 보이지만 성남시 표현과의 의미 구분은 별도 확인 대상입니다.',
'WORD2572':'가슴 앞 양손 움직임 뒤 마지막 손 내림이 보입니다. P2/P3 분리나 P4 마무리의 필수 여부는 자동 표시만으로 정할 수 없습니다.',
'WORD0759':'내부 자동 표시가 없어도 초반 손모양 유지와 마지막 손 내림은 표본에서 다릅니다. 1개 구간을 단일 의미 단계로 확정하지 않습니다.',
'WORD2876':'짧은 한손 표현으로 표본마다 손가락 모양이 조금 달라 보입니다. 내부 표시 없음은 변화가 없다는 뜻이 아니며 정확한 모양·반복은 원본과 수어 검토가 필요합니다.'}
reviews=[]
for r in c['items']:
 if r['word_code'] in notes:
  review=dict(word_code=r['word_code'],clip_id=r['clip_id'],sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],frames_viewed=sum(len(s['images'])for s in r['spans']),
   method='Codex visual inspection of generated start/middle/end contact-sheet frames; not full-motion expert annotation.',note=notes[r['word_code']],semantic_verified=False)
  r['new_visual_review']=review;reviews.append(review)
write(U/'VISUAL_REVIEW.json',dict(status='OBSERVATIONAL_SAMPLE_REVIEW_SEMANTICS_PENDING',at=now(),clips=len(reviews),frames_viewed=sum(r['frames_viewed']for r in reviews),reviews=reviews,all100_new_phase_boundaries_manually_verified=False,expert_reviewed=0))
out=dict(catalog_sha256=sha(U/'CATALOG.json'),items=c['items'])
(U/'catalog.js').write_text('window.REVIEW_DATA='+json.dumps(out,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+';\n',encoding='utf8')
page=(U/'review.html').read_text(encoding='utf8').replace('元영상','원영상').replace(".replace('元','')",'')
page=page.replace('충분한 관절 근거로 고른 내부 변화 표시 없음. 단어 전체 구간을 원본에서 직접 확인해 주세요.','현재 탐색·표시 간격 조건에 맞는 내부 표시 없음. 의미 단계가 하나라는 뜻은 아니며 원본에서 직접 확인해야 합니다.')
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
summary=f'''공개 100단어 동작 단계 후보·수어 검토 화면 준비 완료

완료: 고정 공개 TRAIN ID01 참고 영상 100개에서 변화 시점 215개와 연속 검토 구간 315개를 표시했다. 위치 변화 후보139개, 손가락 굽힘 정보 변화 후보76개. 95개 영상에 내부 표시가 있고5개는 단어 주석 전체를 하나의 검토 구간으로 남겼다. 같은 수어의 다른 촬영자 영상이나 실제 사용자 영상의 추가 분석은 이번에 하지 않았다.

시간대별 원본 재생, 좌우 관절 근거 부족 표시, 구간별 시작/중간/끝 945장, 100장 전체 후보 모음, 필수/선택/준비·연결·마무리/필요한 손/선후/반복/시간 수정 의견 입력 및 JSON 내보내기를 제공한다. 파일은 W: 원본을 읽고 정적 파생 이미지는 현재 작업 폴더에만 저장했다.

확실한 근거: 원본 해시, 실제 프레임·시간, 기존 단어 전체 주석, 저장 관절의 관측 여부. 100개 전체의 자료 연결·시간·누락 없는 구간 커버·945장 해시·215개 변화량을 별도 코드로 검산했다. 최대 산술차 {v['max_numeric_error']:.3g}. 동일 Codex 별도 산술검사로 독립 수어 전문가 검토가 아니다.

직접 장면 재검토: 12개 영상111개 표본 장면. 냉커피의 양손→입 근처 한손→가슴 앞 이동, 성남/전주의 후반 위치 유사 흐름 등을 관찰했다. 믿다의 동일 손 접촉 흐름이2구간으로 나뉘거나 수영의 연속 겹침 동작이4구간으로 나뉘는 과분할 가능성이 있다. 100개 모두의 새 단계 경계를 시각 확정한 것이 아니다. 이전100개 표본 관찰 메모는 출처를 표시해 재사용했다.

미확정: 필수/선택 단계, 의미 구분 손모양, 꼭 지켜야 하는 순서, 허용 반복, 적절한 단계 경계. 전문가 확인 단계 규칙0개. 숫자로 표시한 구간 수=수어의 필수 동작 수가 아니다. 손이 검출되지 않았다는 이유만으로 동작 생략으로 판정하지 않는다. 이번 자동 후보는 같은 손의 전후 관절이 모두 확보된 시간창만 써 가림을 가로질러 변화량을 만들지 않았다. 이 조건이 엄격하여 정상 변화도 빠질 수 있다.

방법: 관절의 몸 기준 손바닥 위치 변화와 손가락 굽힘 cosine 변화의 국소 최대를 각각 찾아 표시한다. 단어명, 모델 점수, 고정 3등분을 사용하지 않는다. 최소 표시 간격0.30초·최대 내부 표시5개는 초안 화면의 표시 예산일 뿐 수어 오류 기준이 아니다. 얼굴 표정·미세 손모양이나 짧은 단계를 모두 찾는다고 보장하지 않는다.

반복/속도: 반복 횟수와 절대 수행 시간을 오류 기준으로 삼지 않는다. 같은 자세 유지와 실제 동작 반복을 구분해 검토하도록 하며, 원본1명의 횟수로 허용 반복을 확정하지 않는다. 필요시 파생 영상은 원본과 동일한 데이터 분할에 둔다.

다음: 장면의 구간 합치기/나누기 검토 → 한국수어를 아는 사람의 필수·선택·필요 손·선후·허용 반복 의견 → 같은 단어의 다른 공개 TRAIN 촬영자에서 정상 변형 확인 → 이견/가림 보류 → 기준 고정 및 사용자 결과 확인 후에만 추가 학습 설계. 검토표 작성만으로 학습용 정답 승인이나 학습 실행이 되지 않는다.

C·T3-F/웹/DTW/원본 보존. 학습0, 모델추론0, 개인·멘토 촬영 접근0, ID14~18 원시접근0. 정확도 향상을 측정한 작업이 아니라 평가 근거 준비 작업이다.
'''
(U/'RESULT_KO.txt').write_text(summary,encoding='utf8')
write(U/'NEXT_PLAN.json',dict(status='USER_AND_KSL_REVIEW_BEFORE_TRAINING',training_authorized_now=False,automatic_adoption=False,
 next_single_action='검토 화면에서 관측 가능한 구간 합치기/나누기와 필요한 손을 확인하고, 한국수어를 아는 사람의 필수/선택·선후·반복 의견을 받을 소규모 라벨 검토를 진행한다.',
 steps=['현재315구간은자동책갈피로확정단계가아님.12개우선검토장면과나머지88개의경계를확인한다.',
 '공개참고영상ID01에서필수/선택/준비·연결·마무리/필요손/선후/반복을수어검토한다.한글명칭으로단계를추정하지않는다.',
 '미확정·이견을보류하고다른공개TRAIN촬영자(ID02~09)에서정상변형을검토한다.기존DEV/보류원시/사용자자료로규칙을고르지않는다.',
 '승인된정답기준을별도로고정한후정상오판·누락/순서민감도·판단보류를구분평가한다.모든파생본은원본분할에둔다.',
 '사용자와결과를확인한뒤추가학습계획을정한다.C/T3F유지,자동학습없음.']))
write(U/'CONCLUSION.json',dict(status='COMPLETE_PHASE_CANDIDATE_REVIEW_ARTIFACT_SEMANTICS_PENDING',at=now(),public_words=100,public_clips=100,source_signer_ids=[1],
 candidate_marks=215,candidate_spans=315,position_marks=139,bend_marks=76,images=945,new_visual_review_clips=12,new_visual_review_frames=111,expert_reviewed_contracts=0,
 training_runs=0,new_model_inference=0,user_recordings_read=0,heldout_raw_read=0,production_changed=False,
 limits=['Automatic bookmarks, not semantic phase truth.','All100 numeric checked;12new sampled visual checks; no expert review.','One reference signer per word; normal variations unvalidated.','No recognition improvement measured.'],implementation='Codex local',verification='Same Codex separate scalar verifier; no independent agent or KSL expert.'))
print('Review page, draft export,12visual notes and report ready.')
