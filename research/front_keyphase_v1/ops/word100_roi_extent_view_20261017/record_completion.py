from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
summary = ('전체 원본에서 시작/동작중 프레임을 읽어 냉커피·수영의 하단전용 입력 범위를 사진에 표시했다. '
           '냉커피1920×788(y<788), 수영1920×822(y<822), 원본1920×1080. '
           '상단·좌우 보존, 영상 내 경계 고정, 20영상 하단740~836. 정확한 배꼽 측정이나 손을 따라가는 경계가 아니다. '
           '하단도 MediaPipe 전체 검출에는 사용하지만 후보 모델 입력에서는 제외한다. 정상 영역 이탈과 영역 안 검출 누락은 구분한다. '
           '원본 해시 확인·2장 시각 검토·별도 에이전트 범위/코드 검증 완료. 새 검출·학습·운영 변경 없음.')
next_action = '사진의 실제 범위와 입력 분리 결과를 사용자와 확인한 후 학습 구성 확정. 사용자 검토 전 학습은 시작하지 않는다.'
verification = dict(status='VERIFIED_VISUALIZATION', updated_kst=now,
    implementation='root: exact source frames and frozen PLAN geometry; source SHA checked before decode',
    independent_review='roi_scope_check: boxes, 20-clip range, formula, pointwise mask and no-training status read-only checked',
    visual_review='root viewed both complete PNGs; original full extent, labels and exact boundary visible; no clipping of figures',
    summary=summary, next_single_action=next_action, training_started=False)
(U/'COMPLETION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context = ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
s = context.read_text(encoding='utf-8-sig')
old = '### 2026-10-17 전체 검출·상반신 입력 분리 구현 및 짝 비교 완료'
new = ('### 2026-10-17 입력 범위 원본 사진 확인 — 학습 전\n'+summary+'\n'
       '근거 `ops/word100_roi_extent_view_20261017/PROVENANCE.json`, `COMPLETION.json`, 냉커피·수영 PNG 2장.\n\n'+old)
assert s.count(old)==1
s=s.replace(old,new,1)
needle='### 2026-10-17 다음 — 입력 분리 결과 상호 검토 후 학습 구성 확정\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'입력 범위 사진 2장을 추가 제공했다. '+next_action+'\n',1)
context.write_text(s,encoding='utf-8')
handoff=ROOT/'NEW_CHAT_HANDOFF.md'
s=handoff.read_text(encoding='utf-8-sig')
start=s.index('갱신 ')
end=s.index('. 사용자 승인:',start)
s=s[:start]+'갱신 '+now+s[end:]
s=s.replace('## 마지막 완료\n','## 마지막 완료\n'+summary+'\n사진: `ops/word100_roi_extent_view_20261017/coldcoffee_exact_extent.png`, `swimming_exact_extent.png`. 원본 범위를 유지한 설명용 사진이며 검출 결과를 새로 생성하지 않았다.\n\n',1)
old='현재crop은상단y=0을유지하지만좌우도잘림. 향후자료준비에서사용자의상단/좌우동작공간보존요구를확인한다.'
assert old in s
s=s.replace(old,'기존 8767 진단 페이지의 좁은 crop은 상단y=0을 유지하지만 좌우도 잘린 과거 비교조건이다. 최신 후보 full_then_bottom_mask는 상단·좌우 전체를 보존한다. 두 범위를 혼동하지 않는다.')
handoff.write_text(s,encoding='utf-8')
state=ROOT/'research/front_keyphase_v1/ops/STATE.json'
s=state.read_text(encoding='utf-8-sig')
data=json.loads(s)
old_date=data['updated_date_kst']
s=s.replace('"updated_date_kst": '+json.dumps(old_date), '"updated_date_kst": '+json.dumps(now),1)
key='word100_roi_extent_view_20261017'
assert key not in data
entry=dict(status='VERIFIED_SOURCE_FRAME_BOUNDARY_VISUALIZATION_NO_TRAINING',updated_kst=now,unit='ops/'+key,summary=summary,next_single_action=next_action)
anchor='  "word100_fullframe_roi_pair_20261016": {'
assert anchor in s
entry_text=json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')
s=s.replace(anchor,'  "'+key+'": '+entry_text+',\n'+anchor,1)
json.loads(s)
state.write_text(s,encoding='utf-8')
assert all((U/n).is_file() for n in ['PROVENANCE.json','COMPLETION.json','coldcoffee_exact_extent.png','swimming_exact_extent.png'])
print('Recorded source-boundary clarification; STATE parses; next task remains user review before training.')
