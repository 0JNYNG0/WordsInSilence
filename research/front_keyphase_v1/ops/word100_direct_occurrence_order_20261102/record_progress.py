from pathlib import Path
import sys,json
Y=Path(__file__).resolve().parent;O=Y.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=Y
if __name__=='__main__':
 active=json.loads(sys.argv[1]) if len(sys.argv)>1 else {'training':False,'reviewer':'/root/retention_direction_review'}
 training=bool(active.get('training'))
 sync_flow.sync('공통 순서 판독부 고정 비교 '+('학습 실행' if training else '입력·코드 검산'),
  '이전goalturn은8808 실제demo12저장/1fresh 검증으로progress. Y에서기존FIT252입력756해시/축·630질문 검산PASS. '
  '576known/54held는45원관계27FIT영상계보. 학습전score없는평가동결:FIT315unswapped(288known27held),DEV110(103known7held)/15원관계. '
  '동일13824param새head두조건,base전체고정, tied roleprior+physicalproduct vs direct둘제거. '
  'generic model/train import충돌2건학습전기록·새파일고유명으로수정. rawphysicalsidecar/gap/nativePTS/numericgate/closurepins보완. '
  'C/T3F/기존16관측/prototype/RGBMotion/8804·8808유지,개인/멘토/H14·15미사용. 전체goal미완료.',
  ('실행handle/PID TRAIN_PROGRESS실제확인후중복금지. 완료후evaluate_readout.py 새2후보FIT252+DEV88각1회,검산agent의verify_saved_readout.py로저장값만검산. '
   if training else 'CODE_REVIEW.json 최종training_allowed확인후 train_readout.py 새두조건80epoch동일예산실행. 현재학습미시작. ')+
  '학습/평가대형출력은W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102만. '
  '새조건한번비교뒤epoch/seed/DEVthreshold탐색금지. wholegraph완료/필수반복/누락정답새로만들지않음. '
  '구현·검산은Codex,별도code/inputreview≠수어전문가정답. X새jobfd785d.../12저장 및완료old학습평가재실행금지. '
  '통과시부분pair순서근거만새격리demo경로검증,미달시FIT/DEV차이와오류원인저장값으로진단.전체goalactive.',
  'DIRECT_OCCURRENCE_ORDER_TRAINING' if training else 'DIRECT_OCCURRENCE_ORDER_PREFLIGHT',active)
