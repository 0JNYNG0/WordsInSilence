from pathlib import Path
L=Path(__file__).resolve().parent;K=L.parent/'public_view_arrival_support_20261116'
def main():
 (L/'runtime').mkdir(exist_ok=True)
 text=(K/'runtime/worker.py').read_text(encoding='utf8').replace('from preserved_arrival import PreservedArrivalLesson','from final_temporal import TemporalVisibilityLesson').replace('net=PreservedArrivalLesson()','net=TemporalVisibilityLesson()').replace('from form_presentation import present','from temporal_presentation import present');(L/'runtime/worker.py').write_text(text,encoding='utf8')
 for name in ['serve.py','check_integrated.py']:
  text=(K/name).read_text(encoding='utf8').replace('from arrival_web_adapter','from temporal_web_adapter');(L/name).write_text(text,encoding='utf8')
 text=(K/'check_http.py').read_text(encoding='utf8').replace('arrival_support_active','temporal_visibility_active').replace('ARRIVAL_SUPPORT_ACTIVE','TEMPORAL_VISIBILITY_ACTIVE');(L/'check_http.py').write_text(text,encoding='utf8')
 text=(K/'activate.ps1').read_text(encoding='utf8').replace("public_view_arrival_support_20261116'","public_temporal_visibility_20261116'").replace('public_view_observation_repair_20261116\\serve.py','public_view_arrival_support_20261116\\serve.py');(L/'activate.ps1').write_text(text,encoding='utf8')
if __name__=='__main__':main()
