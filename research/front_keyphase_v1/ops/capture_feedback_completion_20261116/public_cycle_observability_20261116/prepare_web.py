from pathlib import Path
M=Path(__file__).resolve().parent;L=M.parent/'public_temporal_visibility_20261116'
def main():
 (M/'runtime').mkdir(exist_ok=True);(M/'web').mkdir(exist_ok=True)
 edits={'runtime/worker.py':[('from final_temporal import TemporalVisibilityLesson','from final_cycle import LearnedCycleLesson'),('net=TemporalVisibilityLesson()','net=LearnedCycleLesson()')],
 'serve.py':[('from temporal_web_adapter import CaptureService','from cycle_web_adapter import CaptureService')],
 'check_integrated.py':[('from temporal_web_adapter import','from cycle_web_adapter import')],
 'check_endpoint_cards.py':[('assert net.temporal_admitted','assert net.cycle_admitted')],
 'check_http.py':[('temporal_visibility_active','cycle_observability_active'),('TEMPORAL_VISIBILITY_ACTIVE_CAMERA_OFF','CYCLE_OBSERVABILITY_ACTIVE_CAMERA_OFF')],
 'activate.ps1':[('public_temporal_visibility_20261116','public_cycle_observability_20261116'),('public_view_arrival_support_20261116\\serve.py','public_temporal_visibility_20261116\\serve.py')]}
 for name,pairs in edits.items():
  t=(L/name).read_text(encoding='utf8')
  for a,b in pairs:
   assert a in t,(name,a);t=t.replace(a,b)
  (M/name).write_text(t,encoding='utf8')
if __name__=='__main__':main()
