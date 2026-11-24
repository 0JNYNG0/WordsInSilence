"""Evidence-gated web completion graph. No implicit experiment execution."""
from pathlib import Path
import json, datetime, hashlib
U=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def initialize():
 p=U/'TASK_GRAPH.json'
 if p.exists(): return
 nodes=[
 ('G0','범위·자료·기준 고정',[],['public_only','protected_hashes','predeclared_gates']),
 ('G1','설명 요구와 직접 주석 정리',['G0'],['provenance','guide_variant_consistency','unknown_not_negative']),
 ('G2','행동 구간과 설명 연결 학습',['G1'],['normal_preserved','false_support_reduced','not_abstention_only','three_seeds']),
 ('G3','모양·방향·순서·누락 통합',['G2'],['component_preserved','combined_errors','missing_not_pass','original_time_evidence']),
 ('G4','미사용 공개자료·별도 검산',['G3'],['exposure_audit','frozen_final_test','verification']),
 ('G5','웹 시각 피드백 연결',['G4'],['evidence_text_visual_agree','no_unqualified_claim','rollback']),
 ('G6','촬영·자동종료·전체 지연 확인',['G5'],['camera_user_started','capture_regression','end_to_end_latency','user_flow']),
 ('DONE','웹 테스트 완료',['G6'],['all_required_gates_pass'])]
 write(p,dict(version=1,created=stamp(),objective='현재4단어 웹촬영→TCN→DTW→학습근거 피드백→재시도 완성. 공통100 보존. 게임 병합 제외.',status='ACTIVE',nodes=[dict(id=i,title=t,depends=d,required_checks=c,status='PENDING',attempts=[],evidence=[]) for i,t,d,c in nodes],loop_policy='실패 보존→원인 분리→FIT 근거 보완→새 후보 고정→개발 비교. 최종평가로 튜닝 금지. 보류만 증가하면 성공 아님.',constraints=['No user/mentor training/tuning/selection','Preserve C/T3-F and common100','No word-specific fixes','No repeated completed experiment','Human/KSL review distinct from automated audit'],deferred=['Unity/game integration','100/3000 fine-feedback expansion']))
def record(node,status,checks,evidence,note):
 g=read(U/'TASK_GRAPH.json');n=next(n for n in g['nodes'] if n['id']==node)
 assert status in ['RUNNING','PASS','REWORK','NEEDS_INPUT','PENDING']
 if status=='PASS':
  assert all(next(a for a in g['nodes'] if a['id']==d)['status']=='PASS' for d in n['depends'])
  assert all(checks.get(k) is True for k in n['required_checks']), (node,checks)
  assert evidence and all(Path(e).is_file() for e in evidence)
 entry=dict(at=stamp(),status=status,checks=checks,evidence=[dict(path=str(Path(e).resolve()),sha256=hashlib.sha256(Path(e).read_bytes()).hexdigest()) for e in evidence],note=note)
 n['attempts'].append(entry);n.update(status=status,evidence=entry['evidence']);g['updated']=stamp();write(U/'TASK_GRAPH.json',g)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(node=node,**entry),ensure_ascii=False)+'\n')
if __name__=='__main__':initialize()
