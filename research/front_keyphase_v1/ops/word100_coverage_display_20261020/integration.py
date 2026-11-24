from pathlib import Path
import sys,json,hashlib,copy,types,traceback
from unittest.mock import patch
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1];O=U.parent/'word100_multiref_observation_compare_20261019';M=U.parent/'word100_missing_phase_validation_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100_web_analysis as W,word100_observation_coverage as C,word100_role_dtw as R,word100_multiref as MR
from wis_lab.core.observation import Observation
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def run():
    assert read(U/'PUBLIC_VALIDATION.json')['status']=='PASS'
    index={r['clip_id']:r for r in read(O/'FROZEN.json')['inputs']}
    rows=read(M/'ROWS.json');count=0;removed=0
    cache={}
    for r in rows:
        if r['condition'] not in ('whole','peak10','peak20'):continue
        cid=r['clip_id'];side=r['side']
        if cid not in cache:
            obs=Observation.load(index[cid]['inputs']['sensor']);cache[cid]=(obs,C.metric_flags(obs))
        obs,flags=cache[cid];ts=obs.timestamp_us;mask=np.ones(len(ts),bool) if r['condition']=='whole' else ((ts/1e6>=r['window']['start_s'])&(ts/1e6<=r['window']['end_s']))
        for key in C.AXES:
            original=C.timeline(ts,flags[side][key]);f=flags[side][key]&~mask;v=C.timeline(ts,f)
            assert v['observed_percent']<=original['observed_percent']+1e-8
            if r['condition']=='whole':assert v['observed_percent']==0 and len(v['missing_intervals'])==1
            if (flags[side][key]&mask).any():removed+=1
            count+=1
    # Compare old and new transport on public evidence, never on stored user sessions.
    old=types.ModuleType('wis_lab.service._coverage_before');old.__package__='wis_lab.service'
    exec(compile((U/'before/word100_web_analysis.py').read_text('utf-8'),'<before>', 'exec'),old.__dict__)
    cid='NIA_SL_WORD1139_REAL12_F';path=Path(index[cid]['inputs']['legacy']);word=read(O/'words/legacy/WORD1139.json');q=next(x for x in word['queries'] if x['clip_id']==cid)
    # Read immutable TRAIN bank metadata, not new scoring or classifier inference.
    bank,pin=R.bank();legacy,lp=MR.bank();item=bank['words']['WORD1139']
    metrics=R.relative(copy.deepcopy(q['metrics']),item['baseline'])
    role={'word_id':'WORD1139','label':'믿다','reference_count':9,'required_sides':['L','R'],'role_label':'양손 비교','metrics':metrics,'notes':[],'comparisons':[]}
    fixture={'requested_word_id':'WORD1139','session_id':'ses_000000000000','trial_id':'tri_000000000000','practice100':{'model_key':'word100_tcn_roi_c_v1','checkpoint_sha256':'PUBLIC_UI_FIXTURE_NOT_A_MODEL_RESULT','ranking':[{'word_id':'WORD1139','label':'믿다','rank':1,'model_score':.7}]},'analysis':{'dtw':{'status':'INSUFFICIENT'},'training_comparison':copy.deepcopy(role)}}
    with patch.object(R,'saved_comparison',return_value=role):
        for model in ('word100_tcn_context_v1','word100_tcn_roi_c_v1','word100_tcn_t3f_v1'):
            fixture['practice100']['model_key']=model
            previous=old._with_hand_roles(copy.deepcopy(fixture),path.with_suffix(''),sha(path))
            new=W._with_hand_roles(copy.deepcopy(fixture),path.with_suffix(''),sha(path))
            coverage=new['analysis'].pop('observation_coverage');assert new==previous
            assert coverage['schema']==C.SCHEMA and coverage['query_observation_sha256']==sha(path)
    new['analysis']['observation_coverage']=coverage;new['fixture_note']='Public DEV source; illustrative classifier ranking for UI only, no inference/accuracy claim.'
    write('PUBLIC_UI_FIXTURE.json',new)
    # Sensor display fixture shares the public cached descriptive values from the research path.
    sensor=copy.deepcopy(new);sx=read(O/'words/sensor/WORD1139.json');sq=next(x for x in sx['queries'] if x['clip_id']==cid)
    sensor['analysis']['training_comparison']['metrics']=R.relative(sq['metrics'],item['baseline'])
    sensor['analysis']['observation_coverage']=C.summarize(cache[cid][0],['L','R']);sensor['fixture_note']='Public sensor-path coverage and descriptive values; illustrative display only. Runtime continues to use its pinned legacy DTW path.'
    write('PUBLIC_SENSOR_FIXTURE.json',sensor)
    write('INTEGRATION.json',{'status':'PASS','masked_metric_checks':count,'effective_masks':removed,'model_transports_unchanged':3,'legacy_and_role_bank_pins_valid':True,'http_public_fixture':False,'http_harness_note':'In-process import/TestClient stalled; runtime health and static delivery checked separately. Same backend adapter tested on public observation.','user_sessions_read':0,'ranking_fixture_not_inference':True})
    print('INTEGRATION PASS')

if __name__=='__main__':
    try:run()
    except Exception:
        p=U/'FIRST_ERROR_integration.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
