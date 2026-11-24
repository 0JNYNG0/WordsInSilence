from pathlib import Path
from datetime import datetime
import argparse,json
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--session',default='');a=p.parse_args()
now=datetime.now().astimezone().isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
done=list((U/'runs').glob('*/RESULT.json'))
progress=read(U/'PROGRESS.json');progress.update(stage=a.stage,at=now,execution_session=a.session,training_runs_completed=len(done),features_completed=61)
(U/'PROGRESS.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
path=U.parent/'STATE.json';s=read(path);s['current_stage']='TEMPORAL_REFINER_AUTONOMOUS_'+a.stage;s['updated_at']=now
s[U.name].update(status=a.stage,current_execution=a.session,training_runs_completed=len(done),features_completed=61)
path.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
h=ROOT/'NEW_CHAT_HANDOFF.md';text=h.read_text(encoding='utf-8');text=text.replace('진행: RGB 특징 exec 88014 확인 후 이어받기.',f'진행: RGB61 완료. 현재 {a.stage}, 실행세션 {a.session}, 학습 {len(done)}/6 완료. CURRENT_PROGRESS는 PROGRESS.json/RUNNING.json 확인.')
h.write_text(text,encoding='utf-8')
print(json.dumps(progress,ensure_ascii=False))
