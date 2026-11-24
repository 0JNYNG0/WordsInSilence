"""Launch the verified frontal-camera demo; five-view data is avatar-only."""
import demo_launcher
import sys

demo_launcher.SERVER=demo_launcher.OPS/'capture_feedback_completion_20261116/frontal_scope_recovery_20261117/serve.py'
previous_get=demo_launcher.get
def get(path,token=None):
 q=previous_get(path,token)
 if path=='/api/status'and q.get('state')!='preparing':
  if not q.get('frontal_scope_active'):raise RuntimeError('정면용 판정부 서버가 아닙니다. 촬영을 닫고 서버를 갱신해 주세요.')
  # Compatibility with the original launcher's hand-form readiness check.
  q['hand_form_active']=True
 return q
demo_launcher.get=get
if __name__=='__main__':
 try:demo_launcher.main()
 except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
