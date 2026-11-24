from pathlib import Path
import sys,json,numpy as np,traceback,time
UNIT=Path(__file__).resolve().parent;OPS=UNIT.parent
sys.path.insert(0,str(OPS/'word100_action_feedback_demo_20261108'))
try:
 from consumer import FeedbackConsumer
 c=FeedbackConsumer();p=OPS/'word100_positive_direction_20261109/predictions/NIA_SL_WORD2523_REAL11_F_crop_normal.npz';z=np.load(p)
 r=c.channels(z['x'],z['timestamp_us'],z['frame_id'],'WORD2523');last=r['cards'][-1]
 assert last['display_status']=='uncertain' and len(last['unverified_components'])==2
 assert r['component_coverage_version']=='scoped_components_v1' and r['overall_pass'] is False
 report=dict(passed=True,scope='One public cached coordinate integration, not accuracy rerun',status=last['status'],message=last['message'],models_replaced=False,camera_opened=False,timing=r['timing'])
 (UNIT/'RUNTIME_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(last['message'])
except BaseException:
 (UNIT/f'RUNTIME_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
