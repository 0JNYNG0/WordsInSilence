from pathlib import Path
K=Path(__file__).resolve().parent;J=K.parent/'public_view_observation_repair_20261116'
def main():
 (K/'runtime').mkdir(exist_ok=True)
 text=(J/'runtime/worker.py').read_text(encoding='utf-8').replace('from final_candidate import PlaneViewLesson','from preserved_arrival import PreservedArrivalLesson').replace('net=PlaneViewLesson()','net=PreservedArrivalLesson()')
 (K/'runtime/worker.py').write_text(text,encoding='utf-8')
 text=(J/'serve.py').read_text(encoding='utf-8').replace('from view_web_adapter','from arrival_web_adapter')
 (K/'serve.py').write_text(text,encoding='utf-8')
 text=(J/'check_integrated.py').read_text(encoding='utf-8').replace('from view_web_adapter','from arrival_web_adapter')
 (K/'check_integrated.py').write_text(text,encoding='utf-8')
 text=(J/'check_http.py').read_text(encoding='utf-8').replace('view_posture_active','arrival_support_active').replace('VIEW_POSTURE_ACTIVE','ARRIVAL_SUPPORT_ACTIVE')
 (K/'check_http.py').write_text(text,encoding='utf-8')
 text=(J/'activate.ps1').read_text(encoding='utf-8').replace("public_view_observation_repair_20261116'","public_view_arrival_support_20261116'").replace('public_motion_decoupling_20261116\\serve.py','public_view_observation_repair_20261116\\serve.py')
 (K/'activate.ps1').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
