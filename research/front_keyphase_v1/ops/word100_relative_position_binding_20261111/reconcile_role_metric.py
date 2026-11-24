"""Fix a comparison-schema mismatch, preserving all first outputs. No inference."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')
def main():
 assert not (U/'METRIC_CORRECTION.json').exists()
 first=U/'first_verification';first.mkdir(exist_ok=True)
 for name in ['FINAL_RESULT.json','VERIFICATION.json','verify_saved.py']:(first/name).write_bytes((U/name).read_bytes())
 rows=read(U/'FRAME_ROLE_SCORES.json');out=[]
 for r in rows:out.append(dict(r,new_raw_after=r['after'],after=r['after'] if r['after']<8 else 0))
 write(U/'FRAME_ROLE_COMPARABLE.json',out)
 write(U/'METRIC_CORRECTION.json',dict(error='Raw11class new role prediction compared against legacy8class score; classes8/9/10 must map to legacy0 in both arms.',source='word100_candidate_role_learning_20261111/evaluate.py: after=pred if pred<8 else0',fixed_rule='Map new predicted roles>=8 to0 only for legacy aggregate comparison. Preserve raw role output and all annotations.',first_results=str(first),first_frame_scores_sha256=sha(U/'FRAME_ROLE_SCORES.json'),corrected_sha256=sha(U/'FRAME_ROLE_COMPARABLE.json'),network_output_changes=0,threshold_changes=0,label_changes=0,inference_repeats=0,post_score_arithmetic_correction=True))
 print('Comparison schema corrected; raw predictions and first verification preserved.')
if __name__=='__main__':main()
