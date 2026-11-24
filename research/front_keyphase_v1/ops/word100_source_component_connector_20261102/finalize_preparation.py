from pathlib import Path
import json,sys,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 r=read(U/'INDEPENDENT_PREPARATION_REVIEW.json')
 assert not r['failures'] and all(r['checks'].values())
 for n,h in r['artifacts'].items():assert sha(U/n)==h,n
 for p in U.glob('*.json'):read(p)
 for p in U.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 review=U/'CONNECTOR_DESIGN_REVIEW.json'
 assert review.exists(),'Wait for the separately requested design review before closing preparation'
 read(review)
 final=dict(status='SOURCE_COMPONENT_COMPARISON_PREPARED',
  independent_code_arithmetic_checks=len(r['checks']),failures=[],
  fit_records=550,development_records=186,development_clips=12,
  new_model_training=0,new_model_inference=0,whole_goal_complete=False,
  authors='root plus separate Codex agents; not independent human/KSL-expert truth',
  pins={p.name:sha(p) for p in [U/'DIRECT_TARGETS.json',U/'TEMPORAL_CONTROLS.json',
       U/'INDEPENDENT_PREPARATION_REVIEW.json',review,U/'READINESS_AND_NEXT.json',U/'RESULT_KO.txt']})
 with (U/'FINAL_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(final,f,ensure_ascii=False,indent=2)
 sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
 import sync_flow
 sync_flow.U=U
 summary=('공개 개발12영상/381표본 장면에서 설명 부분행동42근거 준비. '+
  'FIT 추가18(반례16/미확인2)로 직접정답 FIT550/DEV186 편성. 시간변환180명세와 기존532 보존/48입력 원시간의 별도70검산 불일치0. '+
  'SignRep16프레임 문맥은 단순역순 재배열로 실제역재생을 대신할 수 없어 재추출 또는 현재시점기하전용 계약 보완. '+
  '엄지복합은 단어배경지름길/DEV음성부족으로 진단전용. 새학습·성능개선·웹적용 아님, 전체goal active.')
 nxt=('READINESS_AND_NEXT/CONNECTOR_DESIGN_REVIEW에 따라 실제시간변환 입력과 학습조건 고정 후 작은 source-component 신경망 비교. '+
  '5직접비교항목/2구성motion대조/엄지진단전용/2unknown제외. cached RGB·BiGRU latent 단순재배열을 실제영상시험으로 금지; '+
  '원영상변환후16프레임 RGB 및문맥 재추출 또는 명확한현재시점기하전용대조로 분리. 기존16물리출력/원시간/누락/C/T3F/웹 보존. '+
  'DEV반복노출명시, H14/15·사용자·멘토 학습/튜닝/선택금지. 완료학습·비교 재실행금지. 새훈련프로세스 미시작, 전체goal 미완료.')
 sync_flow.sync('설명 행동 연결부 평가정답·시간대조 준비 검산 완료',summary,nxt,final['status'])
 assert read(O/'STATE.json')['current_stage']==final['status']
 assert str(U) in (U.parents[3]/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 print(final['status'],'checks',len(r['checks']),'new training0, goal incomplete')
if __name__=='__main__':main()
