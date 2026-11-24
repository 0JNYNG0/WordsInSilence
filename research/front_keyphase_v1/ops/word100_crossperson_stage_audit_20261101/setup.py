from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def main():
 prev=O/'word100_core_context_split_20261101'; c=read(prev/'COMPLETION.json')
 assert c['active'] is None and all(sha(prev/p)==h for p,h in c['pins'].items())
 d=read(prev/'AUTHORIZATION.json');d.update(unit=U.name,request='2026-11-01 user: 다음 작업 빠르고 정확하게 진행해줘 그럼.',scope='RGB cross-person physical-stage correspondence audit; reuse saved paths, no new inference/training',new_neural_training=False,old_scores_unchanged=True,new_annotation_metric=True)
 write(U/'AUTHORIZATION.json',d)
 write(U/'CURRENT_EXECUTION.json',dict(status='RGB_CORRESPONDENCE_REVIEW',active='root+rgb_labels_a+rgb_labels_b+guide_stage_map',prior_completed=prev.name,no_model_process=True))
 s=read(O/'STATE.json');s['current_stage']='CROSSPERSON_PHYSICAL_STAGE_REVIEW';s['active']=U.name;s['next_single_action']='Prepare RGB-supported physical stage correspondence without reading pair predictions; freeze then audit saved 36 person-pair results. No new matching/model or tuning.';s[U.name]=dict(status='RGB_CORRESPONDENCE_REVIEW',unit=str(U),previous=prev.name)
 (O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
 print('READY',U.name)
if __name__=='__main__':main()
