"""User-requested saved-recording diagnosis only; no inference/training or runtime edits."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np,cv2,av
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent
sys.path.insert(0,str(O/'word100_action_feedback_demo_20261108'))
from auto_end import ReturnToRest
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
names=['br3bkdu4','rhqbhjnq','txfmr3vq'];rows=[]
pins=read(G/'selective_rtmw_anchor_repair_20261117/web/ADMISSION.json')['pins']
assert all(sha(f)==h for f,h in pins.items())
for ordinal,name in enumerate(names,1):
    source=Path('C:/Users/local-user/AppData/Local/Temp')/('wis_action_demo_'+name)
    record=read(source/'inference_result.json');q=record['result'];ack=read(source/'browser_ack.json');session=read(source/'rtmw/SESSION.json');raw=dict(np.load(source/'rtmw/observations.npz'))
    assert q['selective_retry']['version']=='selective_front_rtmw_retry_v3'
    assert all(not record[k] for k in ['training_use','tuning_use','candidate_selection_use'])
    hashes={str(source/p):sha(source/p) for p in ['recorded.mp4','inference_result.json','browser_ack.json','rtmw/observations.npz','rtmw/SESSION.json','selective_retry.json']}
    end=ReturnToRest();facts=[]
    for i,xy in enumerate(raw['xy_px']):
        f=end.geometry(xy,int(raw['source_width']),int(raw['source_height']))
        facts.append(dict(frame=i,time=float(raw['timestamp_us'][i])/1e6,geometry_valid=f is not None,high=f[1].tolist()if f else None,arm_delta=f[0].tolist()if f else None,elbows_below_shoulders=bool((f[0][:,1]>.12).all())if f else None))
    selections={'action':list(range(15,76,4)) if ordinal>1 else list(range(22,83,4)), 'ending':np.linspace(max(0,len(facts)-65),len(facts)-1,12).round().astype(int).tolist()}
    needed=set(sum(selections.values(),[]));images={}
    with av.open(str(source/'recorded.mp4'))as c:
        for i,f in enumerate(c.decode(video=0)):
            if i in needed:images[i]=f.to_ndarray(format='bgr24')
    sheets={}
    for part,ids in selections.items():
        sheet=np.full((((len(ids)+3)//4)*270,4*360,3),245,np.uint8)
        for n,i in enumerate(ids):
            if i not in images:continue
            im=cv2.resize(images[i],(360,240));x=(n%4)*360;y=(n//4)*270
            sheet[y+30:y+270,x:x+360]=im
            cv2.putText(sheet,f'#{ordinal} f{i} {facts[i]["time"]:.2f}s',(x+8,y+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),1,cv2.LINE_AA)
        path=D/f'attempt_{ordinal}_{part}.jpg';cv2.imwrite(str(path),sheet);sheets[part]=str(path)
    last=facts[-1]['time'];tail=[f for f in facts if last-f['time']<=2]
    rows.append(dict(attempt=ordinal,source=str(source),recorded_kst=datetime.fromisoformat(record['recorded_at']).astimezone(timezone(timedelta(hours=9))).isoformat(),word=q['word'],label=q['label'],overall_pass=q['overall_pass'],criteria=q['lesson_acceptance']['criteria'],cards=[{k:c.get(k)for k in ['id','status','message','hands','query_spans']}for c in q['cards']],events=q['lesson_acceptance'].get('events'),movement_evidence=q['lesson_acceptance'].get('movement_evidence'),tcn_top3=q['tcn_top3'],stop_reason=q['stop_reason'],auto_end_calibrated=q['auto_end_calibrated'],stop_to_result=q['stop_to_result_seconds'],stop_to_browser_ack=ack['stop_to_browser_ack_seconds'],selective_retry=q['selective_retry'],frames=q['recorded_frames'],received=session['received'],completed=session['completed'],dropped=session['dropped'],errors=session['errors'],missing_ranges=session['missing_input_id_ranges'],duration=last,tail2s=dict(frames=len(tail),geometry_valid=sum(f['geometry_valid']for f in tail),either_hand_high=sum(bool(f['high']and any(f['high']))for f in tail),both_elbows_below=sum(bool(f['elbows_below_shoulders'])for f in tail)),geometry=facts,source_hashes=hashes,sheets=sheets))
    print('SAVED',ordinal,q['label'],q['overall_pass'],q['stop_reason'],round(ack['stop_to_browser_ack_seconds'],3),rows[-1]['tail2s'])
assert all(sha(f)==h for r in rows for f,h in r['source_hashes'].items())
write(D/'DIAGNOSIS.json',dict(rows=rows,private_diagnosis_only=True,training=False,tuning=False,candidate_selection=False,new_inference=False,runtime_changed=False,pins_unchanged=len(pins),recorded_rest_baseline_available=False,user_intent_pending=True))
