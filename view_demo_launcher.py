"""Latest local demo launcher; retain previously admitted launcher files."""
import motion_demo_launcher
import sys
demo_launcher=motion_demo_launcher.demo_launcher
demo_launcher.SERVER=demo_launcher.OPS/'capture_feedback_completion_20261116/public_view_observation_repair_20261116/serve.py'
original_get=demo_launcher.get
def get(path,token=None):
 q=original_get(path,token)
 if path=='/api/status'and q.get('state')!='preparing'and not q.get('view_posture_active'):
  raise RuntimeError('이전 판정부 서버가 실행 중입니다. 촬영을 닫고 최신 서버로 다시 시작해 주세요.')
 return q
demo_launcher.get=get
if __name__=='__main__':
 try:demo_launcher.main()
 except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
