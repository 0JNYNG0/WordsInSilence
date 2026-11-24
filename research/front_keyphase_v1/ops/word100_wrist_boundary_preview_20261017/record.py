from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import re

u=Path(__file__).resolve().parent;root=u.parents[3]
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
summary=('사용자가 하단을 휴식 자세의 손목 부근까지 내리는 방안을 요청해 공개 TRAIN 냉커피·수영 2영상에 같은 제안 규칙을 사진으로 표시했다. '
         '처음5프레임 양손 hand wrist y중앙값 중 높은 손목에서 어깨너비2% 여유를 뺀 고정선: 냉커피788→864, 수영822→905. '
         '상단·좌우 보존. 기존 native 저장좌표에서 두 영상 처음·끝5프레임 손 입력0 유지, 포함 손관절점-프레임 +193/+143. '
         '이는 2예시의 기하학적 범위 확인으로 새 검출 회복·인식률 개선이 아니다. 전체20/899영상 적용·새 검출·학습·운영 변경 없음. '
         '최종 일반 규칙과 실제 웹캠 경계 적합성은 미확정이다.')
nxt='손목 부근 고정 경계 검토안을 사용자와 확인한 뒤 공개 TRAIN 범위에서 휴식손 제외·동작손 보존·실제 잘린 입력 조건을 검증하고 학습 구성을 확정한다. 검토 전 학습하지 않는다.'
entry=dict(status='REST_WRIST_BOUNDARY_PREVIEW_ONLY_REVIEW_BEFORE_TRAINING',updated_kst=now,unit='ops/'+u.name,summary=summary,next_single_action=nxt,
           original_source_preserved=True,new_detection=0,new_training=0,production_changes=0)
(u/'COMPLETION.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
path=root/'WordsInSilence_PROJECT_CONTEXT.md';s=path.read_text('utf-8-sig')
old='### 2026-10-17 입력 범위 원본 사진 확인 — 학습 전'
assert s.count(old)==1
s=s.replace(old,'### 2026-10-17 휴식 손목 기준 하단 확대 사진 검토안\n'+summary+'\n근거 `ops/'+u.name+'/PREVIEW.json`, `SCOPE.json`, 비교사진2장. 구현·사진 root, 기존 관측/규칙 검토 wrist_boundary_review.\n\n'+old,1)
old='### 2026-10-17 다음 — 입력 분리 결과 상호 검토 후 학습 구성 확정'
assert s.count(old)==1
s=s.replace(old,'### 2026-10-17 다음 — 손목 부근 하단 경계 검토\n'+nxt+'\n\n'+old,1)
path.write_text(s,encoding='utf-8')
path=root/'NEW_CHAT_HANDOFF.md';s=path.read_text('utf-8-sig')
a=s.index('갱신 ');b=s.index('. 사용자 승인:',a)
s=s[:a]+'갱신 '+now+s[b:]
s=s.replace('## 마지막 완료\n','## 마지막 완료\n'+summary+'\n근거/사진 `ops/'+u.name+'/PREVIEW.json`, `coldcoffee_wrist_boundary.png`, `swimming_wrist_boundary.png`. 기존 전체검출 짝 비교 결과와 별도의 새 경계 검토안이다.\n\n',1)
a=s.index('## 다음 한 작업\n');b=s.index('\n## ',a+1)
s=s[:a]+'## 다음 한 작업\n'+nxt+'\n'+s[b:]
path.write_text(s,encoding='utf-8')
path=root/'research/front_keyphase_v1/ops/STATE.json';s=path.read_text('utf-8-sig');data=json.loads(s)
s=s.replace('"updated_date_kst": '+json.dumps(data['updated_date_kst']), '"updated_date_kst": '+json.dumps(now),1)
s=s.replace('"current_stage": '+json.dumps(data['current_stage']), '"current_stage": "WORD100_REST_WRIST_BOUNDARY_PREVIEW_REVIEW"',1)
s=re.sub(r'^  "next_single_action": [^\n]*', lambda m: '  "next_single_action": '+json.dumps(nxt,ensure_ascii=False)+(',' if m.group(0).endswith(',') else ''), s, count=1, flags=re.MULTILINE)
anchor='  "word100_roi_extent_view_20261017": {';assert anchor in s
s=s.replace(anchor,'  "'+u.name+'": '+json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')+',\n'+anchor,1)
json.loads(s);path.write_text(s,encoding='utf-8')
print('Preview recorded; training remains off.')
