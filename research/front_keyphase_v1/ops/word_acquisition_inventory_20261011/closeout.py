"""Report/document closeout only. No data processing or model execution."""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
REL = 'ops/word_acquisition_inventory_20261011'
NOW = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
REPORT = '''AI Hub 단어 자료 목록 및 소규모 영상 준비 결과 — 2026-10-11
작성·실행·별도 검산: Codex

현재 결론
단어별 주석 수, 촬영자 ID, 기존 학습·평가 분할, 보유 영상과의 연결을 정리했다.
정면 주석 54,000건 / WORD 코드 3,000개를 전수 확인했다. 이미 보유한 ZIP에는 정면 영상 1,500개가 있다.
그중 주석 조건을 통과한 1,499개가 로컬 입력 준비 후보이며, 작은 고정 표본 12개를 외장 W:에 추출해 영상·시각 검증까지 완료했다.
이번 작업은 자료 준비다. 새 모델을 학습하거나 정확도 향상을 확인한 것은 아니다.

1. 단어별 자료량
- Training 정면 주석 48,000건 + 공식 Validation 정면 주석 6,000건 = 54,000건.
- WORD 코드는 3,000개이며 코드당 정면 주석 18건이다. 같은 수행의 다른 4개 각도는 독립 수행이나 별도 사람으로 더하지 않았다.
- 원문 단어 표기는 2,965종이다. 공백·숫자 접미어·표현 차이를 포함하므로 곧바로 서로 다른 의미 클래스 2,965개라는 뜻은 아니다.
- 학습 후보 ID 01~09가 모두 있는 원문 표기는 2,594종이다. WORD 코드별로 9개 학습 ID가 모두 있고 그들의 표기가 일치하는 코드는 2,713개다.
- 이는 주석 ID의 다양성이다. 실제 사람 신원, 수어 형태의 일치, 관절 추적 품질 검증을 대신하지 않는다.
- 기존 개발 ID 11~13, 보류 14~16, 공식 Validation 17~18을 학습으로 옮기지 않았다. ID 10은 미배정을 유지했다.
- 과거 Training 집계(앞뒤 공백 정리 후 2,946표기)와 목표 단어 주석 248건을 재사용·대조했다.

2. TOP100을 바로 확정하지 않은 이유
'꺼지다'는 학습 후보 ID 9개, 정면 주석 54개, WORD 코드 6개다. 54명이 촬영한 것이 아니다.
'모르다'는 ID 9개, 주석 18개, WORD 코드 2개다. 같은 이름의 다른 코드를 같은 수형이나 동의어로 자동 합치지 않았다.
많은 후보가 ID 9개로 동률이므로 최종 어휘는 교육적 용도·표현의 일관성·영상품질도 함께 보고 선정해야 한다.
단어자료_현황.html / 단어별_자료목록.txt는 자료량 비교표이며 최종 학습 어휘 선정표가 아니다.

3. 기존 18단어와의 연결
- 정확한 같은 표기가 있는 7종: 알다, 자다, 좋다, 모르다, 나, 너, 싫다.
- '감사 → 감사합니다', '죄송 → 죄송하다'는 기존 의미 연결 후보로 보존했으며 자동 라벨 변경하지 않았다.
- 공부·먹다·감·맥주·물·빵·사랑·운동·인사는 이번 WORD 주석에서 정확한 같은 표기를 찾지 못했다.
- AI Hub의 다른 자료나 동의어에도 없다는 뜻은 아니다. WORD 전체 확보만으로 기존 18단어의 공백이 모두 채워진다고 보장할 수 없다.
- 국립국어원 참조 18개, 현재 모델의 어휘와 운영 설정은 그대로다.

4. 주석 이상은 원본 변경 없이 보류
정면 83건을 학습 준비 통과 집계에서 보류했다. 아래 문제 유형은 중복될 수 있다.
- 폴더의 사람 ID와 파일명 ID 충돌 55건. 예: 02 폴더 안 REAL25 파일. 5개 각도 합계는 이전 조사 275파일과 같다.
- 파일명 추가 토큰 8건. 전체 각도에서는 이전 조사 40파일과 같다.
- 시간 구간 조건 실패 21건. 빈 주석 11건을 포함한다.
- 단일 비어 있지 않은 라벨 조건 실패 11건.
파일명과 JSON의 name·URL이 같아도 실제 사람을 확정할 수 없다. REAL25를 폴더만 믿고 REAL02로 고치지 않았다.
이 문제를 전체 자료가 사용 불가하다는 뜻으로 확대하지 않는다.

5. 기존 영상에서 준비한 것
보유 [원천]01_real_word_video.zip은 32,547,266,828바이트이며 REAL01 / WORD1501~3000 / 5개 각도 총 7,500개 영상이 들어 있다.
정면은 1,500개, 약 6.97GB다. 주석 구간이 주석상 길이를 넘는 WORD2152(외할아버지) 1개를 보류하고 1,499개 / 6,961,290,231바이트를 입력 준비 후보로 정리했다.
전체 ZIP을 다시 다운로드하거나 모두 압축 해제하지 않았다.
후속 표본 12개는 선택 규칙을 먼저 고정한 뒤 W:/WordsInSilence/data/AIHub_103/pilot12_20261011에 추출했다.
- 총 54,321,598바이트(약 54.32MB). 12개 모두 정상 해제·전체 영상 디코딩 통과.
- 총 1,444프레임, 모두 1920×1080. 프레임 시각이 순서대로 증가함을 확인했다.
- 12개 모두 주석 구간이 실제 영상 길이 안에 있고, 주석상 길이 차이는 1프레임+1ms 이내였다.
- ZIP 항목 CRC, 파일 SHA-256, 계획과 실제 파일 일치, 별도 PowerShell 해시·집계 검산 통과.
- 기존에 준비한 2개 목표 WORD 영상을 제외하고 고정 해시 순서로 골랐다. 사용자·모델 점수를 보고 고르지 않았다.
숫자·기관 이름 등도 포함된다. 입력 처리 경로를 점검하기 위한 표본이며 게임에 넣을 추천 12단어가 아니다.
모두 촬영자 01이다. 현재 TCN의 문장 자료 학습에도 등장한 사람이므로 새 사람 성능 시험으로 쓰지 않는다.
MediaPipe 관절 추출, 손 RGB 입력 준비, 모델 추론·학습은 이 단위에서 하지 않았다.

6. 추가 다운로드의 남은 조건
공식 WORD 영상 ZIP은 Training 32개 + Validation 1개 = 33개다. 내부 확인을 마친 것은 기존 Training 01 한 개다.
나머지 32개 내부의 사람·단어 배치는 공개 파일 목록만으로 확정하지 못했다. 정면 주석 52,500건의 영상 ZIP 연결은 미확정이다.
ZIP 번호와 사람 ID가 같다고 추정하지 않았다. 이전 원격 부분 수신 점검에서도 실제 ZIP 내부 선택 지원을 확인하지 못했으므로 같은 실패를 반복하지 않았다.
VIDEO_ACQUISITION_QUEUE.json에 공식 파일 ID·크기·경로와 확인 상태를 보존했다. 자동 다운로드 지시가 아니다.
대용량 영상·키포인트를 새로 받지 않았고 진행 중 전송이나 20시 이후 자동 재개 작업도 없다.

7. 다음 단일 작업
준비한 12개 영상으로 기존 MediaPipe 추출기와 RGB 준비 도구를 재사용해 같은 동작 구간·프레임 시각의 관절/손 이미지 입력을 만든다.
완료 조건은 영상·관절·RGB의 시각 대응, 검출 누락 표시, 원본·계획 해시 보존이다. 새 모델 학습이나 정확도 비교로 확대하지 않는다.
다인 학습에는 추가 영상 ZIP 내부 연결 확인과 단계적 확보가 별도로 필요하다. 학습 전에 어휘·사람 분할·중복·비교 조건을 고정한다.

검증과 근거
Codex가 구현·실행했고 별도 검산 코드로 원본 주석 54,000개의 해시·라벨·시간 구간을 출력과 전수 대조했다. 별도 에이전트 검토로 표현하지 않는다.
코드 3,000개 집계, 기존 사람 역할, 33개 영상 ZIP 용량 합계, 로컬 연결 1,499개를 확인했다. 원본 주석 ZIP 해시 불변.
첫 실행은 과거 집계의 앞뒤 공백 처리 차이를 발견해 중단했다. 원문을 보존하고 과거 결과와 비교할 때만 같은 공백 처리를 적용했다(FIRST_FAILURE.json).
목록 구축에는 영상 본문을 읽지 않았고 이후 고정 12개만 전체 디코딩했다. 사용자 자료·보류 좌표·체크포인트는 읽지 않았다.
새 다운로드 0, 학습 0, 모델 추론 0, 운영·카메라·Git 변경 0.
근거: SUMMARY.json, VERIFIED_RESULT.json, PILOT_RESULT.json, PILOT_INDEPENDENT_CHECK.json, FRONT_MANIFEST.jsonl, WORD_FAMILIES.json, LABEL_COUNTS.json, TARGET18_COVERAGE.json.
'''

if sys.argv[1] == 'report':
    (UNIT / 'RESULT_KO.txt').write_text(REPORT, encoding='utf-8')
    print('report updated')
elif sys.argv[1] == 'docs':
    delivery = json.loads((UNIT / 'DELIVERY.json').read_text(encoding='utf-8-sig'))
    assert delivery['status'] == 'PASS'
    summary = json.loads((UNIT / 'SUMMARY.json').read_text(encoding='utf-8'))
    pilot = json.loads((UNIT / 'PILOT_RESULT.json').read_text(encoding='utf-8'))
    assert pilot['status'] == 'PASS'
    paths = [ROOT / 'WordsInSilence_PROJECT_CONTEXT.md', ROOT / 'NEW_CHAT_HANDOFF.md', ROOT / 'research/front_keyphase_v1/ops/STATE.json']
    before = UNIT / 'before'
    before.mkdir(exist_ok=True)
    for p in paths:
        backup = before / p.name
        if not backup.exists():
            shutil.copy2(p, backup)
    next_action = 'Prepare paired MediaPipe/RGB inputs for the fixed12 already decoded local WORD clips using existing tools and timestamps; verify masks/source hashes, no training or model scoring. Further signer-video ZIP mappings remain unknown; no automatic large download.'
    state = json.loads(paths[2].read_text(encoding='utf-8-sig'))
    state['updated_date_kst'] = NOW
    state['current_stage'] = 'WORD_METADATA_INVENTORY_AND_LOCAL_PILOT_COMPLETE'
    state['next_single_action'] = next_action
    entry = {
        'status': 'COMPLETE_VERIFIED_METADATA_AND_PILOT_MEDIA', 'at_kst': NOW,
        'report': REL+'/RESULT_KO.txt', 'verification': REL+'/VERIFIED_RESULT.json',
        'pilot_result': REL+'/PILOT_RESULT.json', 'pilot_verification': REL+'/PILOT_INDEPENDENT_CHECK.json',
        'delivery': REL+'/DELIVERY.json', 'front_annotation_rows': 54000, 'word_codes': 3000,
        'raw_label_strings': 2965, 'raw_labels_with9_clean_training_ids': 2594,
        'quarantined_front_rows': 83, 'local_ready_front_videos': 1499,
        'pilot_clips': 12, 'pilot_bytes': 54321598, 'pilot_decoded_frames': 1444,
        'pilot_signer_ids': [1], 'new_person_evaluation': False,
        'unknown_video_archive_mappings': 32, 'new_downloads': 0, 'training': 0, 'model_inference': 0,
        'running_task_processes': [], 'next': next_action,
        'source_preservation': 'Original files read-only; old signer roles preserved; no label merging or automatic ID correction.',
        'implementer_verifier': 'Codex implementation + separate raw source and PowerShell checks; not a separate agent review.'
    }
    # Keep the latest unit near the top without altering historical unit contents.
    state.pop('word_acquisition_inventory_20261011', None)
    ordered = {}
    for k, v in state.items():
        ordered[k] = v
        if k == 'current_stage':
            ordered['word_acquisition_inventory_20261011'] = entry
    paths[2].write_text(json.dumps(ordered, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    context = paths[0].read_text(encoding='utf-8-sig')
    status = '''### 2026-10-11 WORD 전수 목록·로컬 영상12개 준비 완료

`ops/word_acquisition_inventory_20261011/RESULT_KO.txt`. 정면 주석54,000건/WORD코드3,000개 전수 집계·원문 검산 완료. 원문표기2,965종 중 학습ID01~09가 모두 있는 표기2,594종, 9ID/표기일치 WORD코드2,713개. 의미 클래스나 최종TOP100 확정 아님. 정면83건은 ID/시간/빈라벨 등으로 보류; 원본/이전 사람 분할 보존. 나머지32영상ZIP 내부 연결 미확정, 추정 다운로드0.
기존32.55GB ZIP에서 정면1,500개 확인, 주석통과1,499개/6.96GB 목록 작성. 그중 점수와 무관한 고정12개/54.32MB만 W:에 추출, CRC/해시/전체1,444프레임 디코딩/1920×1080/시각단조/주석구간12/12 통과. 별도PowerShell 해시·집계 검산 완료. 모두REAL01로 새사람시험 아님. 관절/RGB 추출·학습·모델추론0. 목록·보고서 W:/WordsInSilence/data/AIHub_103/inventory_20261011, 영상 pilot12_20261011에 저장·해시확인.
담당 Codex 구현·별도 검산(에이전트 교차검토 아님). 최초 집계의 공백 정의 차이 기록·수정. 새다운로드/카메라/운영/Git변경0, 진행전송/자동재개없음. 기존18 중 정확표기7종+기존의미연결후보2종; 공부/먹다 등을 전체AIHub에 없다고 확대해석하지 않는다.

'''
    next_block = '''### 2026-10-11 다음 — 고정 로컬12개 관절·RGB 입력 준비

W:에 이미 추출·디코딩 검증된12개에 기존 MediaPipe/RGB 도구를 재사용한다. 주석 동작구간과 원본 PTS를 함께 보존하고 관절·손 이미지의 동일시각 대응, 누락마스크, 원본/계획해시를 검산한다. 입력준비 단위이며 학습/새정확도평가/사용자자료/운영교체 없음. 한 사람의 새 클립을 새사람 평가로 재분류하지 않는다. 다인영상 추가확보는 나머지32ZIP 내부대응/용량/수신시간 확인 후 별도 단계로 진행하며 대용량 자동수신·20시 이후 자동재개하지 않는다. 완료된 주석전수검산·12개디코딩·과거모델실험 반복 없음.

'''
    for heading, body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', status), ('## 9. 다음 단일 작업', next_block)]:
        assert context.count(heading) == 1
        context = context.replace(heading+'\n\n', heading+'\n\n'+body, 1)
    paths[0].write_text(context, encoding='utf-8')
    handoff = f'''# WordsInSilence 최신 인수인계
갱신: {NOW}. 웹 STATE: `research/front_keyphase_v1/ops/STATE.json`.
시작: AGENTS.md, WordsInSilence_PROJECT_CONTEXT.md 최신 §3·§8·§9, WORKFLOW_HANDOFF.md.

## 현재 목표·마지막 완료
새 학습 자료를 많이 받기 전에 단어·사람·원본·분할 대응을 검증하고 기존 영상으로 입력 준비 경로를 확인한다.
최신 단위 `{REL}`. RESULT_KO.txt / VERIFIED_RESULT.json / PILOT_RESULT.json / PILOT_INDEPENDENT_CHECK.json / DELIVERY.json 완료.
- WORD코드3,000개, 정면주석54,000건 전수 검산. 원문표기2,965종 중 학습후보ID01~09 모두 있는 표기2,594종. 9ID·표기일치 코드는2,713개.
- 표본 수 순 목록은 최종TOP100/의미클래스 확정 아님. 같은 수행5각도는 별도사람/독립수행으로 세지 않는다. 기존 개발11~13/보류14~16/공식Validation17~18 역할 유지,10미배정.
- 정면83건 보류: ID폴더/파일명 충돌55, 추가토큰8, 시간조건실패21(빈주석11포함), 단일라벨조건실패11. 유형중복 있음. 원본 수정/ID추정변경 없음.
- 기존18 정확표기7종(알다·자다·좋다·모르다·나·너·싫다), 감사/죄송은 기존 의미연결후보. 공부/먹다 등은 이번WORD정확표기에 없다는 뜻이며 전체AIHub 부재 주장금지.
- 기존32.55GB영상ZIP에 REAL01 / WORD1501~3000 / 5각도7,500개. 정면1,500개 중 주석조건1,499개/6.96GB 준비가능.
- 고정12개만 W:에54.32MB 추출: CRC/SHA256/전체1,444프레임디코딩/시각단조/주석구간12/12 통과. 전부1920×1080. 별도PowerShell 파일해시/계획/집계검산 통과.
- 이12개는 고정해시순 기술입력표본이며 추천게임어휘나 성능선택표본이 아니다. 모두사람01/기존학습사람이므로 새사람평가 아님.
- 관절/RGB 추출은 아직 안 함. 학습/모델추론/새다운로드/운영/카메라/Git변경0. 진행작업/전송/20시 이후 자동재개 없음.
- W: 보고서/목록 `W:/WordsInSilence/data/AIHub_103/inventory_20261011`, 영상 `.../pilot12_20261011`.
- 나머지 영상ZIP32개 내부배치는 미확정. ZIP번호=사람ID로 추정하지 않음. 이전원격부분수신실패를 반복하지 않음.

## 다음 한 작업
준비된12개에 기존MediaPipe와RGB도구를 재사용해 동일동작구간/원본PTS의 관절·손이미지 입력을 준비한다.
검출누락마스크·시각정렬·원본/계획해시 보존 확인이 완료조건. 학습/새모델평가로 확대하지 않는다.
완료된54,000주석검산·12개영상디코딩·과거모델실험을 반복하지 않는다. 다인영상은32ZIP연결/용량/수신시간을 확인한 뒤 단계적으로 확보한다.

## 저장소·다운로드
- WIS_DATA(W:) GPT/NTFS3TB 준비 완료. **재포맷 금지**. 두 번째HDD 사용보류/교체고려.
- `ops/hdd_dock_diagnosis_20261011`:16MiB파일쓰기/읽기해시통과, 내부Disk0파티션불변.32도/20,829시간/읽기오류0; 전체건강·표면검증 아님.
- `ops/metadata_acquisition_20261011`:주석ZIP2개129,786,759바이트, CRC270,000항목·별도해시검산 완료. 공식파일목록/구축가이드/말뭉치공개설명서 포함 총약134MB.
- WORD영상+주석1.181TB는 압축본 수용가능, 전체AIHub2.888TB는112GB정도만남아 일괄수신비추천. 실제해제/캐시크기는미확인. 현재소용량승인을전체다운로드승인으로해석하지 않는다.
- 모두의말뭉치 신청·승인·실제코퍼스수신은 미확인. 공개설명서만확보. 예전NIKL18사전시범과혼동금지.

## 기준 모델과 완료 실험
- 기준TCN832: `runs/partial_encoder_pair_v1/checkpoint_control_final.pt`, SHA83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91. 모델불변.
- RGB A/B `ops/rgb_tcn_development_pair_20261010`:개발699 모두691/699(98.86%). KSL34는83222/A23/B22(64.71/67.65/64.71%). RGB미채택.
- RGB진단 `ops/rgb_input_contribution_20261010`:사진을켜도22/34,1순위변경1개는오답→오답. 사진통로가무시된것은아니지만 정확도개선0. 흑백은진단변환/운영변경아님,수형·외형불변성미검증.
- 학습사진131구간/82단어는기존학습원본노출. KSL34/이전2 분리,사용자학습0. 완료자료재추출/모델재학습금지.
- 기하bed는691/699·KSL23/34이나 대조군도같은먹다1건개선하여기하만효과아님. ST-GCNlocalshape662/699·8/34 미채택. 웹미교체.
- DTW/task_match 구현,수행합격/거절은UNCONFIGURED. top2/3이수형경미오류를뜻하지않는다.

## 역할·제한
최신AGENTS:승인범위Codex로컬구현·실행·검증·문서가능. Claude불가시담당자예외반복질문없음. 이번담당Codex/별도검산이며에이전트교차검토주장없음.
원본·동결결과 읽기전용, 사용자자료 학습/튜닝/후보선택금지. 카메라/장치/외부명세전송/운영기본교체/Git변경없음.
웹8765/카메라현재상태는이자료단위에서조회하지않음. st-gcn heartbeat 과거완료후PAUSED. 대용량수신/20시이후자동재개미설정.
'''
    paths[1].write_text(handoff, encoding='utf-8')
    assert json.loads(paths[2].read_text(encoding='utf-8'))['current_stage'] == 'WORD_METADATA_INVENTORY_AND_LOCAL_PILOT_COMPLETE'
    for p in paths[:2]:
        assert 'word_acquisition_inventory_20261011' in p.read_text(encoding='utf-8')
    print(json.dumps({'status':'PASS', 'at_kst':NOW, 'updated':[str(p) for p in paths]}, ensure_ascii=False))
else:
    raise ValueError('report or docs required')
