"""Actual public-derived RGB+MediaPipe controls; one fixed runtime seed."""
from pathlib import Path
from collections import Counter
import json,time
from audit import U,read,save
from retained_live import analyze,warmup
def main():
 assert not (U/'VIDEO_CONTROL_RESULTS.json').exists()
 manifest=read(U/'VIDEO_CONTROLS.json');warmup();rows=[]
 for r in manifest['rows']:
  dest=U/'actual_controls'/f"{r['word_id']}_{r['kind']}"
  if (dest/'RESULT.json').exists():result=read(dest/'RESULT.json')
  else:result=analyze(r['path'],dest,r['word_id'])
  summary=dict(word_id=r['word_id'],kind=r['kind'],result_path=str(dest/'RESULT.json'),
   statuses=dict(Counter(s['status'] for s in result['stages'])),hard_difference=sum(s['status']=='FEATURE_DIFFERENCE_CANDIDATE' for s in result['stages']),
   evidence_gaps=len(result['evidence_gaps']),fully_similar=not result['evidence_gaps'] and all(s['status']=='SUPPORTED_FEATURES_SIMILAR' for s in result['stages']),
   latency=result['latency'],linguistic_error_truth=False)
  if r['kind']=='main_blackout':assert summary['evidence_gaps']>0 and not summary['fully_similar']
  rows.append(summary);(U/'CONTROL_PROGRESS.json').write_text(json.dumps(dict(done=len(rows),total=len(manifest['rows']),last=summary),ensure_ascii=False),encoding='utf8')
  print(json.dumps(summary,ensure_ascii=False),flush=True)
 save('VIDEO_CONTROL_RESULTS.json',dict(status='ACTUAL_PUBLIC_DERIVED_VIDEO_CONTROLS_COMPLETE',rows=rows,private_inputs=0,new_training=0,
  no_new_model_or_threshold_selected_using_results=True,claim='Real extraction after controlled public video edits; not natural learner-error educational accuracy'))
if __name__=='__main__':main()
