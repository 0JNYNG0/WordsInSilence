"""Serialize agent A's directly viewed RGB phase draft; no coordinates or models."""
import hashlib
import json
from collections import Counter
from pathlib import Path

UNIT = Path(__file__).resolve().parents[1]
VIS = UNIT.parent / 'word100_semantic_episode_pilot_20261031' / 'visuals'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def seg(role, bounds, description, features, evidence):
    a,b,c,d = bounds
    return dict(role=role, start_bracket=[a,b], end_bracket=[c,d], core=[b,c],
                description_ko=description, visible_features=features,
                evidence_frames=evidence, support='sampled_visible_RGB',
                linguistic_core=None, obligatory_linguistic_order=None)

REST0 = '양손을 몸 옆 아래에 둔 채 큰 손·팔 동작 없이 서 있다.'
REST1 = '손을 몸 옆으로 내린 뒤 큰 손·팔 동작 없이 서 있다.'
RESTFEAT = ['양손 위치: 몸통 아래/몸 옆', '세부 손가락 모양과 손바닥 방향은 판정하지 않음']
FACEPREP = '오른팔을 몸 옆에서 얼굴 쪽으로 들어 올린다. 펼친 손을 얼굴 앞에 두는 다음 자세로 진입하는 시각적 준비 후보이다.'
FACE = '오른손의 긴 손가락을 펴서 모은 상태로 코·입 앞에 세워 잠시 둔다. 왼손은 아래 몸 옆에 있다.'
FACECLOSE = '얼굴 앞 오른손을 닫으면서 가슴 쪽으로 내린다. 후반에는 닫힌 손이 가슴 앞에 있고 오른팔꿈치는 같은 쪽 바깥에 있어 전완이 가로로 놓인다.'
FACEFIST = '가슴 앞에 가로로 놓였던 오른전완을 세우며 닫힌 오른손을 같은 쪽 어깨 옆으로 옮긴 뒤 잠시 둔다. 얼굴 앞 펼친 손 단계와 구별되는 후반 자세이다.'
FACERETURN = '어깨 옆의 오른손을 가슴과 허리 높이를 지나 몸 옆으로 내리며 자세를 푼다.'

clips=[]
face_specs = {
 'P033': [[0,0,33,38],[35,38,43,46],[43,46,58,61],[58,61,71,73],[71,73,89,94],[91,94,104,106],[106,109,119,119]],
 'P034': [[0,0,30,33],[30,33,38,41],[38,41,60,63],[60,63,74,77],[74,77,96,99],[96,99,110,113],[110,113,129,129]],
 'P035': [[0,0,38,41],[38,41,48,51],[48,51,67,70],[67,70,82,86],[82,86,101,105],[101,105,114,117],[114,117,149,149]]}
face_evidence = {
 'P033': [[0,18,33],[38,41,43],[46,53,58],[61,66,71],[73,78,84,89],[94,99,104],[109,114,119]],
 'P034': [[0,16,30],[33,36,38],[41,49,55,60],[63,69,74],[77,82,88,96],[99,104,110],[113,121,129]],
 'P035': [[0,19,38],[41,44,48],[51,57,63,67],[70,76,79,82],[86,92,95,101],[105,108,114],[117,133,149]]}
for alias,b in face_specs.items():
    fs=[RESTFEAT,
        ['오른손/오른전완의 상방 이동', '왼손은 몸 옆', '진입 중 세부 손모양은 전 프레임 확정하지 않음'],
        ['오른손: 긴 손가락 펼침·모음', '위치: 코·입 앞', '왼손: 몸 옆', '얼굴 접촉 여부 미확정'],
        ['오른손: 펼침에서 닫힘으로 변화', '위치: 얼굴 앞에서 가슴 앞', '후반 오른전완: 가로 방향', '오른팔꿈치: 몸통 같은 쪽 바깥'],
        ['오른손: 닫힌 주먹 형태', '위치: 오른쪽 어깨 옆/윗가슴', '오른전완: 가로에서 세운 방향으로 변화', '그 자세에서 짧게 머묾', '정확한 손바닥 방향 미확정'],
        ['오른손의 하방 이동', '왼손은 계속 몸 옆', '복귀 중 세부 손가락 형태 미확정'],RESTFEAT]
    clips.append(dict(alias=alias, segments=[seg(r,x,t,f,e) for r,x,t,f,e in zip(
        ['rest','preparation','core_candidate','core_candidate','core_candidate','return','rest'],b,
        [REST0,FACEPREP,FACE,FACECLOSE,FACEFIST,FACERETURN,REST1],fs,face_evidence[alias])],
        notes=['얼굴 앞 펼침과 이후 주먹/전완 재배치를 모두 남겼다. 출처 문장의 일부에 맞추어 어느 단계도 삭제하지 않았다.',
               '시각적 preparation/return 역할은 수어의 비필수 또는 무의미 구간 판정이 아니다.',
               '얼굴 접촉·주먹의 정확한 손바닥 방향·언어적 필수 손모양은 unknown이다.']))

ARM_PREP='양팔을 몸 옆에서 올린다. 왼손은 닫힌 형태로 앞에 나오고 오른손은 펼쳐 반대편 팔 쪽으로 접근한다.'
ARM_MOVE='왼팔을 몸 앞에 두고 닫힌 왼손을 유지한다. 펼친 오른손이 왼쪽 위팔/팔꿈치 부근 위에서 전완을 따라 왼손목 쪽으로 옮겨 간다. 영상 평면에서 오른손의 하방 이동과 옆/깊이 방향 성분이 함께 보이며 얼굴을 기준으로 한 동작은 아니다.'
ARM_HOLD='오른손이 왼손목/전완 끝 부근 위에 머문다. 왼팔은 앞에 둔 상태이고 왼손은 닫힌 형태를 유지한다.'
ARM_RETURN='양손의 가까운 배치를 풀고 두 팔을 몸 옆 아래로 내린다.'
arm_specs={
 'P045': [[0,0,38,41],[38,41,44,46],[44,46,63,65],[63,65,79,82],[79,82,93,95],[93,95,128,128]],
 'P046': [[0,0,34,37],[34,37,43,45],[43,45,64,67],[64,67,75,77],[75,77,88,90],[88,90,101,101]],
 'P047': [[0,0,38,40],[38,40,48,50],[48,50,65,68],[65,68,80,83],[80,83,95,98],[95,98,118,118]]}
arm_ev={
 'P045':[[0,19,38],[41,44],[46,49,52,57,60,63],[65,71,79],[82,87,93],[95,112,128]],
 'P046':[[0,17,34],[37,41,43],[45,49,54,60,64],[67,71,75],[77,82,88],[90,95,101]],
 'P047':[[0,20,38],[40,43,48],[50,53,58,63,65],[68,73,80],[83,88,95],[98,108,118]]}
for alias,b in arm_specs.items():
    feats=[RESTFEAT,['양팔 상방 진입','오른손 펼침','왼손 닫힘'],
      ['오른손: 긴 손가락을 펴서 모은 납작한 형태','왼손: 주먹 형태','오른손과 반대쪽 팔의 관계','오른손 위치가 왼팔꿈치 쪽에서 왼손목 쪽으로 이동','2D RGB에서 가까움/겹침 관측; 실제 피부 접촉/압력 미확정'],
      ['오른손: 펼친 형태','왼손: 닫힌 형태','오른손이 왼손목/전완 끝 위에 가까이 머묾','세부 접촉면/압력 미확정'],
      ['양팔 하방 복귀','손 사이/팔 위 배치 풀림'],RESTFEAT]
    clips.append(dict(alias=alias,segments=[seg(r,x,t,f,e) for r,x,t,f,e in zip(
      ['rest','preparation','core_candidate','core_candidate','return','rest'],b,
      [REST0,ARM_PREP,ARM_MOVE,ARM_HOLD,ARM_RETURN,REST1],feats,arm_ev[alias])],
      notes=['팔을 따라가는 관계와 이동 경로의 관찰 초안이며 정밀 팔축 좌표·피부 접촉·올바른 수어 수행을 판정한 것이 아니다.',
             '오른손의 화면 상하만으로 이 동작 전체를 설명하지 않는다. 왼팔 자체도 진입 과정에서 움직인다.']))

TH_PREP='양손을 몸 옆에서 몸 앞 아래쪽으로 들며 닫힌 손과 세운 엄지 형태를 만들기 시작한다. 첫 교대 동작과의 언어적 경계는 알 수 없다.'
TH_1='엄지를 위로 세운 닫힌 오른손이 몸 앞에서 얼굴 높이 쪽으로 높아지고, 왼손은 더 낮은 몸 앞에 있다. 양손의 첫 높낮이 배치를 보존한다.'
TH_2='오른손이 낮아지는 동안 왼손이 얼굴 높이 쪽으로 올라와 양손의 위아래 배치가 바뀐다. 닫힌 손에 엄지를 세운 형태가 보인다.'
TH_3='왼손이 낮아지고 오른손이 다시 얼굴 높이 쪽으로 올라오며 양손의 위아래 배치가 다시 바뀐다.'
TH_4='오른손이 가슴 쪽으로 낮아지고 왼손이 이마 위쪽까지 올라오며 마지막으로 양손의 위아래 배치가 바뀐다.'
TH_HOLD='왼손이 이마 앞/위쪽에, 오른손이 가슴 앞에 놓인 높낮이 배치를 잠시 유지한다. 양손 모두 엄지를 세운 닫힌 손 형태이다.'
TH_RETURN='높이 차가 있던 양손을 몸 앞을 거쳐 몸 옆으로 내리고 닫힌 손 자세를 푼다.'
th_specs={
 'P063':[[0,0,42,45],[42,45,48,51],[48,51,54,57],[54,57,60,63],[60,63,69,72],[69,72,78,81],[78,81,96,99],[96,99,114,117],[114,117,141,141]],
 'P064':[[0,0,32,35],[32,35,40,43],[40,43,48,51],[48,51,56,59],[56,59,67,70],[67,70,78,80],[78,80,99,102],[99,102,115,118],[115,118,126,126]],
 'P065':[[0,0,28,30],[28,30,35,38],[35,38,43,46],[43,46,51,53],[51,53,61,63],[61,63,71,73],[71,73,81,84],[81,84,96,99],[96,99,119,119]]}
th_ev={
 'P063':[[0,21,42],[45,48],[51,54],[57,60],[63,66,69],[72,75,78],[81,90,96],[99,105,111,114],[117,129,141]],
 'P064':[[0,16,32],[35,38,40],[43,46,48],[51,54,56],[59,62,64,67],[70,72,75,78],[80,88,97,99],[102,107,113,115],[118,123,126]],
 'P065':[[0,13,28],[30,33,35],[38,41,43],[46,48,51],[53,56,58,61],[63,66,68,71],[73,78,81],[84,89,94,96],[99,109,119]]}
for alias,b in th_specs.items():
    common=['양손: 엄지가 위로 나와 있고 나머지 손가락은 접힌 형태','양팔/손의 상대 높낮이 변화','정확한 손바닥 방향·손가락별 관절 각도 미확정']
    fs=[RESTFEAT,['양손 상방 진입','닫힌 손/세운 엄지 형태로 변화 중'],
        common+['오른손이 왼손보다 높아짐'],common+['왼손이 오른손보다 높아짐'],
        common+['오른손이 다시 더 높아짐'],common+['왼손이 이마 위쪽으로 높아짐'],
        ['왼손: 이마 앞/위쪽','오른손: 가슴 앞','양손: 엄지를 세운 닫힌 손','마지막 높낮이 배치에서 머묾'],
        ['양손 하방 복귀','마지막 높낮이 배치 풀림'],RESTFEAT]
    clips.append(dict(alias=alias,segments=[seg(r,x,t,f,e) for r,x,t,f,e in zip(
      ['rest','preparation','core_candidate','core_candidate','core_candidate','core_candidate','core_candidate','return','rest'],b,
      [REST0,TH_PREP,TH_1,TH_2,TH_3,TH_4,TH_HOLD,TH_RETURN,REST1],fs,th_ev[alias])],
      notes=['전체 교대와 사이의 내려감, 마지막 유지를 함께 기록했다. 과거 최대 두 up 창을 단계 라벨로 복사하지 않았다.',
             '관찰된 오른쪽 높음→왼쪽 높음→오른쪽 높음→왼쪽 높음의 순서를 이 영상의 물리 흐름으로만 기술한다. 언어적 필수 반복 횟수/순서 주장은 하지 않는다.',
             '두 손이 교차해 보이는 표본의 세부 접촉/손가락 숨은 면은 unknown이다.']))

ROOF_PREP='양손을 몸 옆에서 가슴 앞으로 올리며 벌린 손을 서로 향하게 한다. 중앙 모양을 만들기 전의 시각적 진입 구간이다.'
ROOF_MEET='양손을 얼굴 아래/코·입 앞 중앙으로 모은다. 긴 손가락 쪽은 위에서 가까워지고 엄지 쪽은 아래쪽에 놓여 두 손 사이에 삼각형에 가까운 빈 공간을 만든 뒤 잠시 둔다.'
ROOF_APART='중앙에서 가까웠던 양손을 좌우 어깨 쪽으로 벌리면서 조금 낮춘다. 손은 비교적 납작한 펼침 형태를 유지한다.'
ROOF_HOLD='두 손을 서로 떨어뜨린 채 어깨 안쪽/옆 높이에 대략 대칭으로 둔다. 긴 손가락은 모여 있고 손은 납작하게 보인다.'
ROOF_RETURN='양손을 가슴과 허리 앞을 지나 몸 옆으로 내리며 앞의 배치를 푼다.'
roof_specs={
 'P069':[[0,0,21,23],[21,23,27,29],[27,29,39,41],[39,41,43,45],[43,45,55,57],[55,57,66,68],[66,68,84,84]],
 'P070':[[0,0,37,40],[37,40,44,47],[44,47,54,56],[54,56,59,61],[59,61,80,82],[80,82,94,96],[94,96,110,110]],
 'P071':[[0,0,45,47],[45,47,53,55],[53,55,66,69],[66,69,74,77],[74,77,84,87],[84,87,98,100],[98,100,124,124]]}
roof_ev={
 'P069':[[0,11,21],[23,25,27],[29,32,36,39],[41,43],[45,46,50,55],[57,61,64,66],[68,77,84]],
 'P070':[[0,19,37],[40,42,44],[47,49,51,54],[56,59],[61,68,75,80],[82,87,91,94],[96,103,110]],
 'P071':[[0,24,45],[47,50,53],[55,58,61,66],[69,71,74],[77,79,82,84],[87,92,95,98],[100,113,124]]}
for alias,b in roof_specs.items():
    fs=[RESTFEAT,['양손 상방 진입','두 손이 몸 중앙으로 접근','진입 중 세부 손가락 자세 변화는 미확정'],
        ['양손: 긴 손가락을 펴서 모음','엄지는 긴 손가락에서 벌어짐','위치: 코·입 앞/얼굴 아래 중앙','두 손 사이 삼각형에 가까운 빈 공간','손끝과 엄지의 실제 접촉 범위 미확정'],
        ['양손의 좌우 분리','상방이 아닌 작은 하방 성분 동반','손의 납작한 펼침 형태'],
        ['양손 위치: 어깨 안쪽/옆 높이','양손 간격 유지','긴 손가락을 모은 납작한 형태','손바닥이 얇게 보이는 시점의 정확한 방향 미확정'],
        ['양손 하방 복귀','중앙/어깨 배치 해제'],RESTFEAT]
    clips.append(dict(alias=alias,segments=[seg(r,x,t,f,e) for r,x,t,f,e in zip(
      ['rest','preparation','core_candidate','core_candidate','core_candidate','return','rest'],b,
      [REST0,ROOF_PREP,ROOF_MEET,ROOF_APART,ROOF_HOLD,ROOF_RETURN,REST1],fs,roof_ev[alias])],
      notes=['중앙 모임, 바깥으로 벌림, 떨어진 자세 유지를 각각 남겼다. 초기 팔 올리기만 핵심으로 바꾸지 않았다.',
             '삼각형은 화면에서 관찰되는 형상 묘사이며 언어 의미·사전 의미를 부여한 것이 아니다.',
             '손가락의 겹친 면과 실제 접촉 범위는 보이는 표본만으로 확정하지 않는다.']))

EXTRA={
 'P033':[53,84], 'P034':[], 'P035':[79,95],
 'P045':[52], 'P046':[60], 'P047':[53],
 'P063':[78], 'P064':[46], 'P065':[68],
 'P069':[36,46], 'P070':[51], 'P071':[77]}
evidence_files=[]
for clip in clips:
    alias=clip['alias']
    mp=VIS/alias/'MANIFEST.json'
    manifest=json.loads(mp.read_text(encoding='utf-8-sig'))
    selected={x['frame_id']:x for x in manifest['selected']}
    ts=lambda ids:[selected[f]['timestamp_us'] for f in ids]
    clip.update(native_frames=manifest['frames'], source_path=manifest['source'],
                source_sha256_as_recorded_in_manifest=manifest['source_sha256'],
                manifest_path=str(mp),manifest_sha256=sha(mp),
                full_clip_frame_scope=[0,manifest['frames']-1],
                source_frames_trimmed=False, sampled_frame_ids=list(selected),
                individually_viewed_original_frame_ids=EXTRA[alias],
                sampled_rgb_reviewed_in_full_temporal_order=True,
                all_native_frames_visually_reviewed=False,
                linguistic_core=None, source_clause_alignment=None)
    reviewed=[VIS/alias/'sequence_full.jpg']+[VIS/alias/f'f{f:04d}.jpg' for f in EXTRA[alias]]
    clip['directly_viewed_files']=[str(x) for x in reviewed]
    for x in reviewed:
        evidence_files.append(dict(path=str(x),sha256=sha(x),actually_viewed=True,
          kind='48-sample whole-clip contact sheet, central view crop' if x.name=='sequence_full.jpg' else 'original full-frame RGB still'))
    clip['unknown']={
      'native_frames_not_individually_seen_in_any_reviewed_image':[f for f in range(manifest['frames']) if f not in selected],
      'boundary_brackets_are_uncertain':True,
      'fine_finger_joint_angles':None, 'physical_contact_or_pressure':None,
      'linguistic_necessity_or_correctness':None,
      'preparation_and_return_can_contain_linguistic_information':True,
      'unannotated_gaps_are_not_negative_labels':True}
    last=-1
    for i,s in enumerate(clip['segments']):
        s['segment_id']=f'{alias}_s{i:02d}'
        for k in ['start_bracket','end_bracket','core']:
            assert all(f in selected for f in s[k]),(alias,k,s[k])
            assert s[k][0]<=s[k][1]
            s[k+'_pts_us']=ts(s[k])
        assert last<s['core'][0]<=s['core'][1]<manifest['frames']
        assert s['start_bracket'][0]<=s['core'][0]<=s['start_bracket'][1]
        assert s['end_bracket'][0]<=s['core'][1]<=s['end_bracket'][1]
        assert all(s['core'][0]<=f<=s['core'][1] for f in s['evidence_frames'])
        last=s['core'][1]
        s['evidence_pts_us']=ts(s['evidence_frames'])
        s['core_interpretation']='sample-supported physical span draft, not framewise dense GT or mandatory linguistic core'
    clip['uncertain_boundary_intervals']=sorted({tuple(s[k]) for s in clip['segments'] for k in ['start_bracket','end_bracket'] if s[k][0]!=s[k][1]})

out=dict(schema='RGB_VISIBLE_MOTION_PHASE_DRAFT_V1',status='PROVISIONAL_RGB_PHASE_DRAFT_NOT_LINGUISTIC_GT',
  reviewer='Codex agent A; same Codex family; not an independent human or KSL expert annotation',
  basis='Actual visual inspection of all 12 existing 48-sample whole-clip RGB sheets and 14 original full-frame stills; native frame/PTS from manifests',
  scope='authorized public fit12 only; ID1/2/3; no dev or held images in this task',
  role_contract={
   'rest':'sampled low-motion posture before/after the displayed activity; not semantic negative truth',
   'preparation':'visible entry toward a later recognizable posture; linguistic importance remains unknown',
   'core_candidate':'visually distinct handshape/body-relation/motion/hold worth retaining for later source linkage; linguistic necessity remains unknown',
   'return':'visible release/lowering toward initial posture; linguistic importance remains unknown',
   'uncertain':'role cannot be resolved from visible evidence'},
  temporal_contract={
   'frame_ids':'zero-based original native frame IDs; inclusive endpoints',
   'pts_us':'copied exactly from existing extraction MANIFEST selected entries, not recomputed from rounded sheet seconds',
   'core':'inner sample-supported descriptive span; NOT every native frame observed, NOT a crop for automatic training, NOT linguistic core',
   'brackets':'possible transition interval between viewed samples; do not turn every bracket frame into either adjacent phase',
   'missing_or_unannotated':'unknown; cannot become other/rest/negative or inferred correct sequence',
   'whole_source':'full native source scope retained; no clip or action segment was removed'},
  blinding_limitations={
   'word_names_known_from_prior_work':True,
   'required_project_handoff_exposed_historical_summary_metrics':True,
   'historical_summary_metrics_used_as_annotation_evidence':False,
   'new_model_prediction_or_score_files_read_for_phase_task':False,
   'coordinates_used_as_RGB_truth':False,
   'old_up_windows_automatically_relabelled_core':False,
   'source_word_clauses_read_to_force_phase_boundaries':False,
   'source_clause_id_omitted_until_separate_alignment_review':True},
  limitations=[
   'This is same-Codex provisional sampled RGB evidence, not blind independent human ground truth.',
   'Phase names describe visible physical organization; they do not certify valid sign language or required linguistic stages.',
   'No models, training, inference, new scores, private inputs, production edits or old-output edits were performed.',
   'Full-frame RGB can show normal hand lowering outside a model ROI; ROI support must be assessed separately and is not a physical-motion failure.',
   'Source descriptions and model masks may be compared only downstream; absent/occluded hand detail must remain unknown.'],
  clips=clips,direct_visual_evidence=evidence_files)
(UNIT/'PHASE_FIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
counts=Counter(s['role'] for c in clips for s in c['segments'])
check=dict(status='PASS_STRUCTURE_AND_MANIFEST_PTS_ONLY',clips=len(clips),segments=sum(counts.values()),
           roles=dict(counts),sampled_images_reviewed=48*len(clips),sheets_reviewed=len(clips),
           original_full_frames_reviewed=sum(map(len,EXTRA.values())),
           phase_fit_sha256=sha(UNIT/'PHASE_FIT.json'),
           assertion_scope='selected frame existence, bracket/core ordering, evidence within span, exact manifest PTS, unique fit12; does not certify semantic correctness')
assert len(clips)==12 and len({c['alias'] for c in clips})==12
(UNIT/'review'/'PHASE_FIT_STRUCTURE.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['공개 fit 12개 RGB 시각 단계 초안',
       '48표본 시트 12장(576표본) 및 원본 전체화면 14장을 직접 확인했다. 전체 원본 프레임을 본 dense GT가 아니다.',
       '준비/핵심 후보/복귀는 시각적 역할이며 언어적 필수 여부, 수어 정오, 실제 접촉은 판정하지 않았다.',
       '기존 up 창을 핵심으로 복사하지 않았고 원본 전체 시간 범위를 보존했다.', '']
for c in clips:
    lines.append(c['alias'])
    for s in c['segments']:
        sec=[v/1e6 for v in s['core_pts_us']]
        lines.append(f"  {s['role']} f{s['core'][0]}–{s['core'][1]} ({sec[0]:.3f}–{sec[1]:.3f}s): {s['description_ko']}")
    lines.append('')
(UNIT/'review'/'PHASE_FIT_SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False))
