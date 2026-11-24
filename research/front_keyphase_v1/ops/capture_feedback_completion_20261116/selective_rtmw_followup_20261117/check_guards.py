"""Software contracts with a fake pose engine; not evidence of pose accuracy."""
from pathlib import Path
import json,copy
from types import SimpleNamespace
import numpy as np
from selective_retry import retry,requests
D=Path(__file__).resolve().parent;G=D.parent;N=G/'frontal_scope_recovery_20261117'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    r=next(r for r in read(N/'INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL01_F')
    truth=dict(np.load(r['raw']));z={k:v.copy()for k,v in truth.items()};z['xy_px'][45:50,125:129]=np.nan
    initial=read(G/'selective_rtmw_retry_20261117/probe2/NIA_SL_WORD1290_REAL01_F_missing_one_finger.json')
    initial['lesson_acceptance']['criteria'][-1]['status']='unknown'
    original=read(G/'selective_rtmw_retry_20261117/probe2/NIA_SL_WORD1290_REAL01_F_normal.json')
    # Use the actual deficient output, because the successful probe saved its
    # recovered cards. Only public card IDs and indices are fixture metadata.
    for c in initial['cards']:
        if c['id']=='repeat_order':c['status']='unknown'
    rows=[]
    class Reader:
        decoded=0
        def read(self,ids,deadline):
            self.ids=ids;self.decoded=len(ids)
            w=int(z['source_width']);h=int(z['source_height'])
            pattern=np.broadcast_to((np.arange(w,dtype=np.uint16)%128+64).astype('uint8')[None,:,None],(h,w,3)).copy()
            return {i:pattern for i in ids}
    class Pose:
        def __init__(self,t,rd):self.t=t;self.rd=rd;self.calls=0
        def __call__(self,im,bboxes):
            i=self.rd.ids[self.calls//2];self.calls+=1
            return self.t['xy_px'][i:i+1].copy(),np.ones((1,133),np.float32)
    def test(name,raw,base,new,expected):
        rd=Reader();pose=Pose(truth,rd);answer,fixed=retry(raw,base,rd,SimpleNamespace(pose=pose),lambda *args:copy.deepcopy(new),r['word'])
        passed=answer['selective_retry']['reason']==expected
        rows.append(dict(case=name,passed=passed,reason=answer['selective_retry']['reason'],pose_calls=answer['selective_retry']['pose_calls']))
        assert passed,rows[-1]
    changed=copy.deepcopy(original);changed['lesson_acceptance']['criteria'][0]['status']='difference'
    test('established_normal_cannot_be_changed',z,initial,changed,'existing_verdict_changed')
    wrong=copy.deepcopy(initial);wrong['lesson_acceptance']['criteria'][0]['status']='difference';wrong['lesson_acceptance']['criteria'][0]['difference_spans']=[[43,44]]
    test('established_error_cannot_be_erased',z,wrong,original,'existing_verdict_changed')
    changed=copy.deepcopy(original);changed['lesson_acceptance']['criteria'][0]=copy.deepcopy(wrong['lesson_acceptance']['criteria'][0]);changed['lesson_acceptance']['criteria'][0]['difference_spans']=[[55,56]]
    test('established_error_cannot_be_relocated',z,wrong,changed,'existing_error_location_changed')
    test('complete_observations_no_retry',truth,original,original,'not_needed')
    model_unknown=copy.deepcopy(original)
    for c in model_unknown['cards']:c['status']='unknown'
    test('classification_uncertainty_without_missing_points_no_retry',truth,model_unknown,original,'not_needed')
    # Common left/right handling: swapping hand streams and card hands yields
    # the same proposal for the left hand, without a word-specific condition.
    left={k:v.copy()for k,v in z.items()};left['xy_px'][:,91:112]=z['xy_px'][:,112:133];left['xy_px'][:,112:133]=z['xy_px'][:,91:112]
    li=copy.deepcopy(initial)
    for c in li['cards']:c['hands']=[{'L':'R','R':'L'}[h]for h in c['hands']]
    a,_=requests(z,initial);b,_=requests(left,li)
    passed=bool(a and b)and a[0]['span']==b[0]['span']and a[0]['hand']=='R'and b[0]['hand']=='L'
    rows.append(dict(case='symmetric_hand_proposal',passed=passed));assert passed
    class MissingReader:
        decoded=0
        def read(self,*args):return {}
    result,_=retry(z,initial,MissingReader(),None,None,r['word'])
    rows.append(dict(case='missing_video_frame_safe_fallback',passed=result['selective_retry']['reason']=='missing_frame_or_read_budget'))
    class BrokenReader:
        decoded=0
        def read(self,*args):raise ValueError('fixture PTS mismatch')
    result,_=retry(z,initial,BrokenReader(),None,None,r['word'])
    rows.append(dict(case='decoder_failure_safe_fallback',passed=result['selective_retry']['reason']=='retry_failed_keep_original'))
    q=dict(passed=all(r['passed']for r in rows),rows=rows,scope='Fake-engine software invariants, not pose performance',private_data=False)
    (D/'GUARD_CHECK.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(q,ensure_ascii=False));assert q['passed']
if __name__=='__main__':main()
