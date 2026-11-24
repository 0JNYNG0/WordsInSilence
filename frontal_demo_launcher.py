"""Launch the verified frontal-camera demo; five-view data is avatar-only."""
import demo_launcher
import sys

demo_launcher.SERVER=demo_launcher.OPS/'capture_feedback_completion_20261116/moa_coaching_room_20261117/motion_dock_20261117/serve.py'
previous_get=demo_launcher.get
def get(path,token=None):
 q=previous_get(path,token)
 if path=='/api/status'and q.get('state')!='preparing':
  if not q.get('frontal_scope_active'):raise RuntimeError('정면용 판정부 서버가 아닙니다. 촬영을 닫고 서버를 갱신해 주세요.')
  if not q.get('evidence_repair_active'):raise RuntimeError('손 겹침·손가락·자동 종료 보완이 적용된 서버가 아닙니다. 촬영을 닫고 서버를 갱신해 주세요.')
  if not q.get('selective_retry_active'):raise RuntimeError('중요 구간 재관측 기능이 준비되지 않았습니다. 촬영을 닫고 서버를 갱신해 주세요.')
  if q.get('selective_retry_version')!='selective_front_rtmw_retry_v3':raise RuntimeError('재관측 시간·겹침 보완이 적용된 최신 서버로 다시 실행해 주세요.')
  if not q.get('stop_short_cycle_active') or q.get('stop_short_cycle_version')!='seated_rest_dual_window_v1':raise RuntimeError('자동 종료·짧은 반복 보완이 적용된 최신 서버로 다시 실행해 주세요.')
  if not q.get('rest_stability_active') or q.get('rest_stability_version')!='bounded_weak_rest_pause_v2':raise RuntimeError('순간 손점 흔들림 종료 보완이 적용된 최신 서버로 다시 실행해 주세요.')
  if not q.get('endpoint_role_active') or q.get('companion_ui_version')!='moa_coaching_room_v3' or not q.get('layout_admitted'):raise RuntimeError('최신 MOA 연습 화면·끝 상태 판정이 적용된 서버로 다시 실행해 주세요.')
  # Compatibility with the original launcher's hand-form readiness check.
  q['hand_form_active']=True
 return q
demo_launcher.get=get
if __name__=='__main__':
 try:demo_launcher.main()
 except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
