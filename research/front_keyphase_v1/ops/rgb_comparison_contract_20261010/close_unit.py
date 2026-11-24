"""Record verified input-contract work and update the existing project handoff."""
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

unit = Path(__file__).resolve().parent
base = unit.parents[1]
root = base.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
check = read(unit/'INDEPENDENT_CHECK.json')
build = read(unit/'BUILD_CHECK.json')
encoder = read(unit/'ENCODER_CONTRACT.json')
assert check['status'] == build['status'] == 'PASS'
assert check['counts'] == build['counts']
assert encoder['local_full_sha256'] is None and not encoder['weight_download_performed']
stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
context, handoff, state = root/'WordsInSilence_PROJECT_CONTEXT.md', root/'NEW_CHAT_HANDOFF.md', base/'ops/STATE.json'
backup = unit/'before'
backup.mkdir(exist_ok=False)
for p in (context, handoff, state):
    shutil.copy2(p, backup/p.name)
next_action = ('Acquire Modu KSL annotation/guide/file list, then verify gloss/person/time/video links and required raw-video list. '
               'Reuse completed INPUT_INDEX and COMPARISON_CONTRACT; no repeated ROI experiment. '
               'RGB weights and independent-person paired evaluation are still missing. No model training or automatic downloads.')
result = {'status':'INPUT_LINK_AND_ENCODER_RECIPE_COMPLETE_COMPARISON_AWAITS_DATA',
          'artifact_root':'research/front_keyphase_v1/ops/rgb_comparison_contract_20261010',
          'completed_at':stamp,'segments':113, 'counts':check['counts'],
          'source_hashes_unchanged':check['source_files_unchanged'],
          'verification':'producer checks plus separate-code exhaustive source audit; same Codex author, not external review',
          'encoder_candidate':'MobileNetV3-Small IMAGENET1K_V1; URL and installed recipe verified, weights not downloaded',
          'remaining':['paired independent-person annotated RGB data','actual weights/full hash/use terms','model implementation and prespecified training/evaluation contract'],
          'training':0,'inference':0,'new_crops':0,'word_accuracy_claim':False,
          'role':'Approved offline RGB preparation continuation only; no extension to model implementation',
          'next':next_action,'active_processes':[]}
with (unit/'RESULT.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
new8 = '''### 2026-10-10 RGB 비교 입력 연결·이미지 부품 명세 완료 — 학습은 대응 자료 확보 후

`ops/rgb_comparison_contract_20261010/RESULT_KO.txt`·`INPUT_INDEX.json`이 최신이다. 승인된 오프라인 RGB 준비 후속으로 기존113구간/2,789시간행/5,578손칸의 관절·정상RGB·추정영역RGB를 연결했다. A관절만/B정상RGB5,315칸/C동일정상RGB+추정78칸(합계5,393)을 같은시각·좌우로 참조한다. 원관절/사진을 재생성하지 않았고추정사진을관절관측으로바꾸지않았다. 불확실4건은유지.

별도검산코드로113구간/5,578손칸/78사진연결전수PASS,309파일해시불변. 실제모델구현·학습·추론0. MobileNetV3-Small의정확한공식가중치URL/설치전처리/소스해시를확인했지만가중치는미확보다. 기본중심크롭대신원래정사각영역전체224입력을제안했으며공식평가전처리와다름을명시/사진변환미실행. 타입표기검사오류만기록후AST읽기로수정,기존입력/실험미재실행.

핵심미충족:새사람의주석·대응영상/실제RGB가중치/학습계약. 현재01·02는기존TCN학습사람이므로113미학습구간과새사람평가를혼동하지않는다. A/B/C는입력비교설계이며새정확도아님.자료확보계획을재사용한다. 사용자사진으로ST3000DM001 3TB·3.5SATA두개를확인했으나연결/남은공간/상태/포트는미확인,장치접근없음.

'''
new9 = '''### 2026-10-10 다음 — 한국수어 주석·대응 영상 조건 확보 후 RGB 비교 실행

`ops/rgb_comparison_contract_20261010/COMPARISON_CONTRACT.txt`와 기존 `ops/data_pretraining_inventory_20261010/DATA_ACQUISITION_PLAN.txt`를 따른다. 입력연결/전처리검토는완료했으므로반복하지않는다. 사용자가모두의말뭉치주석·설명서·파일목록을확보하면어휘/사람/시간/원영상연결과필요한영상목록을검증한다. 실제자료전에는분할·정확도·top100·학습예산을추정으로채우지않는다.

이미지부품가중치는약10MB별도확보대상이어서외장하드가필수는아니지만자동다운로드하지않는다. 실제파일/해시/사용조건확인,사람분리자료준비후A관절만/B정상RGB결합/C공백실제RGB추가의학습·성공·회귀기준을사전에동결한다. 기존832는별도기준. 입력도구예외를모델구현으로확대하지않는다. 현재학습·실험프로세스없음. 카메라/장치/사용자자료학습/운영교체/Git변경금지유지.

'''
ctx = context.read_text(encoding='utf-8')
for header, addition in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',new8),('## 9. 다음 단일 작업',new9)]:
    assert ctx.count(header) == 1
    ctx = ctx.replace(header, header+'\n\n'+addition,1)
role = ('- 2026-10-10 후속 재개: 사용자 “아까 했던 작업에 이어서 다음 작업” 요청으로 `ops/rgb_comparison_contract_20261010`의 오프라인 입력 연결 색인·명세 검산을 Codex가 완료했다. 기존 RGB 입력 준비 역할의 연속이며 모델 구현·학습·추론·다운로드는 포함하지 않는다. SCOPE.json 참조.\n\n')
ctx = ctx.replace('## 3. 운영 역할','## 3. 운영 역할\n\n'+role,1)
context.write_text(ctx,encoding='utf-8')
s = read(state)
s['updated_at'] = stamp
s['updated_date_kst'] = stamp[:10]
s['current_stage'] = 'RGB_COMPARISON_INPUT_LINK_COMPLETE_AWAITING_PAIRED_DATA'
s['next_single_action'] = next_action
s['rgb_comparison_contract_20261010'] = result
s['storage_photo_20261010'] = {'model':'Seagate ST3000DM001','nominal_gb_each':3000,'count_user_reported':2,
                              'interface':'3.5-inch SATA','label_manufacture_date':'2012-11',
                              'connection_or_device_access_performed':False,'free_space_health_laptop_ports':'unknown',
                              'source':'user photo and report; purchase/connection not confirmed'}
state.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
h = handoff.read_text(encoding='utf-8')
h = re.sub(r'^갱신: [^.]+\.', '갱신: '+stamp+'.', h, count=1, flags=re.M)
h = h.replace('- 최신 완료: `ops/rgb_temporal_roi_20261010/', '- 직전 완료: `ops/rgb_temporal_roi_20261010/',1)
latest = ('- 최신 완료: `ops/rgb_comparison_contract_20261010/RESULT_KO.txt`·`INPUT_INDEX.json`.113구간/2789행/5578손칸에같은관절을사용하는A/B/C입력연결완료. B정상사진5315/C정상+추정사진5393(추가78),별도전수검산PASS/309파일불변. 불확실4유지. 모델·영상재실행없음.\n'
          '- MobileNetV3-Small정확가중치URL/설치소스·전처리확인,가중치미다운로드. 원영상정사각영역전체224입력을제안/공식중심크롭과다름/224변환미실행. 약10MB가중치는외장하드필수아님. ENCODER_CONTRACT·COMPARISON_CONTRACT참조.\n')
h = h.replace('## 현재 목표와 마지막 완료\n','## 현재 목표와 마지막 완료\n'+latest,1)
start = h.index('## 다음 한 작업')
end = h.index('## 보존한 입력과 자료 조사',start)
h = h[:start]+'''## 다음 한 작업
모두의말뭉치주석·설명서·파일목록확보후어휘/사람/시간/영상연결과받을영상목록검증. 기존DATA_ACQUISITION_PLAN을재사용한다. 주석은사용자미신청·미확보상태이며하드연결전에도작은자료부터신청할수있는지확인한다.
RGB입력연결완료이므로재실행하지않는다. 정확가중치파일/전체해시/이용조건및새사람대응영상이필요하다. 자료준비후COMPARISON_CONTRACT의A/B/C학습층·예산·성공기준을결과전에동결한다. 현재모델학습준비완료/새사람성능향상아님. 모델구현은입력도구역할예외밖이다.

'''+h[end:]
h = h.replace('장치규격/포트미확인,구매/장치접근/다운로드자동승인아님.',
              '사진으로Seagate ST3000DM001 3TB/3.5SATA/DOM11-2012두개확인. 연결/건강/남은공간/노트북포트미확인,구매/장치접근/다운로드자동승인아님.')
h = h.replace('- 최신STATE키rgb_temporal_roi_20261010. 이전rgb_fusion_feasibility와rgb_pose_readiness는완료근거로재사용한다.',
              '- 최신STATE키rgb_comparison_contract_20261010. rgb_temporal_roi/rgb_fusion_feasibility/rgb_pose_readiness는완료근거로재사용한다.')
handoff.write_text(h,encoding='utf-8')
evidence = ['SCOPE.json','INPUT_INDEX.json','BUILD_CHECK.json','INDEPENDENT_CHECK.json','ENCODER_CONTRACT.json',
            'COMPARISON_CONTRACT.txt','RESULT_KO.txt','RESULT.json','FIRST_SPEC_CHECK_FAILURE.json']
for file in evidence:
    p = unit/file
    assert p.is_file()
    if p.suffix == '.json': read(p)
assert read(state)['rgb_comparison_contract_20261010']['counts']['C_available_rgb'] == 5393
assert 'rgb_comparison_contract_20261010' in handoff.read_text(encoding='utf-8')
assert 'rgb_comparison_contract_20261010' in context.read_text(encoding='utf-8')
completion = {'status':result['status'],'completed_at':stamp,
              'evidence_sha256':{file:hashlib.sha256((unit/file).read_bytes()).hexdigest() for file in evidence},
              'documents_updated':[str(p) for p in (context,handoff,state)],
              'roles':'Codex offline input preparation and separate-code verification; no external dispatch',
              'training_or_inference':False,'active_processes':[]}
with (unit/'COMPLETION_CHECK.json').open('x',encoding='utf-8') as f:
    json.dump(completion,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'documents':3,'completed_at':stamp}))
