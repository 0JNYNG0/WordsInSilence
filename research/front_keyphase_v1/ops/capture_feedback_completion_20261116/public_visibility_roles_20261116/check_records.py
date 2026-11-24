from pathlib import Path
import sys, json, tempfile
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent.parent/'web_tutorial_live_integration_20261113'))
from attempt_records import save_once,save_result

def main():
    with tempfile.TemporaryDirectory(dir=D) as folder:
        root=Path(folder);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
        attempt=dict(id='a'*24,target='WORD0099',mode='practice')
        x=save_result(a,attempt,dict(query_xy=[1,2],cards=['normal'],_stop_clock=9,stop_reason='arms_returned'),{})
        assert 'query_xy'not in x['result'] and '_stop_clock'not in x['result']
        y=save_once(a,'browser_ack.json',attempt['id'],dict(stop_to_browser_ack_seconds=.7))
        z=save_once(a,'browser_ack.json',attempt['id'],dict(stop_to_browser_ack_seconds=9))
        assert y==z and z['stop_to_browser_ack_seconds']==.7
        save_result(b,dict(attempt,id='b'*24),dict(cards=['unknown']),{})
        assert json.loads((a/'inference_result.json').read_text())['result']['cards']==['normal']
        assert json.loads((b/'inference_result.json').read_text())['result']['cards']==['unknown']
        for f,id_ in [('browser_ack.json','c'*24),('../x.json','a'*24)]:
            try:save_once(a,f,id_,{})
            except ValueError:pass
            else:raise AssertionError('Identity/path guard failed')
        assert not x['training_use'] and not x['tuning_use'] and not x['candidate_selection_use']
    (D/'RECORD_CHECK.json').write_text(json.dumps(dict(passed=True,checks=['separate attempts','first acknowledgement retained','dense landmarks excluded','identity conflict rejected','path name constrained','non-training metadata'],physical_camera=False),indent=2))
    print('RECORD_CHECK_PASS')
if __name__=='__main__':main()
