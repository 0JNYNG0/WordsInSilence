"""Exercise deployed raw->channels/display wiring with a spy, not model re-evaluation."""
from prepare import *
from guard import quarantine
import importlib.util,copy
D=O/'word100_action_feedback_demo_20261108'
sys.path.insert(0,str(D))
spec=importlib.util.spec_from_file_location('guard_consumer_check',D/'consumer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(name,v):checks.append(dict(name=name,passed=bool(v)))
def main():
    row=next(r for r in read(U/'INPUT_PLAN.json')['rows'] if r['clip_id']=='NIA_SL_WORD0739_REAL11_F')
    z=np.load(row['cache']);raw={k:z[k].copy() for k in z.files};g=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],640,480)
    expected=m.A.convert(g['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],640,480)
    c=m.FeedbackConsumer.__new__(m.FeedbackConsumer);seen=[]
    def spy(x,ts,fid,word):
        seen.append(dict(x=x.copy(),ts=ts.copy(),fid=fid.copy(),word=word))
        return dict(cards=[dict(hands=['L'],query_indices=[60,65],status='supported',display_status='similar',message='supported'),dict(hands=['R'],query_indices=[60,65],status='supported',display_status='similar',message='supported')],overall_pass=False,normal_or_wrong=None)
    c.channels=spy
    answer=c.raw(raw,'WORD0739')
    check('same quarantined coordinates reach TCN and feedback channel entry',np.array_equal(seen[0]['x'],expected['tcn280']))
    check('no timeline shrink',np.array_equal(seen[0]['ts'],raw['timestamp_us']) and np.array_equal(seen[0]['fid'],raw['frame_id']))
    shown=np.asarray(answer['query_xy']);bad=~np.isfinite(g['xy_px']).all(2)
    check('uncertain coordinates not rendered as normal points',np.all(shown[bad]==-1))
    check('unsupported left card cannot retain similar status',answer['cards'][0]['status']=='unknown' and answer['cards'][0]['display_status']=='uncertain' and not answer['cards'][0]['educational_wrong'])
    check('visible right evidence retained',answer['cards'][1]['status']=='supported')
    check('raw originals unchanged',all(np.array_equal(raw[k],z[k],equal_nan=True) for k in raw))
    check('valid JSON no NaN',bool(json.dumps(answer,allow_nan=False)))
    answer2=c.raw(raw,'WORD1133')
    check('target cannot influence observation mask',np.array_equal(answer['query_xy'],answer2['query_xy']) and np.array_equal(seen[0]['x'],seen[1]['x']))
    check('no whole-word correctness enabled',answer['normal_or_wrong'] is None and answer['overall_pass'] is False)
    result=dict(status='PASS' if all(r['passed'] for r in checks) else 'FAIL',checks=checks,model_inference_calls=0,camera=False,raw_source='public cropped REAL11')
    write(U/'INTEGRATION_CHECKS.json',result);print(json.dumps(result,ensure_ascii=False));assert result['status']=='PASS'
if __name__=='__main__':main()
