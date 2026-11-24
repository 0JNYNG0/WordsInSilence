"""Synthetic VFR/queue/assignment plumbing tests; no source video or model inference."""
import json
import time
import numpy as np
import stream_capture as S


class FakeLandmarker:
    def __init__(self,contract,fail_at=None):
        self.contract=contract;self.raw_rows=[];self.fail_at=fail_at
        self.handedness_disagreements=0;self.handedness_comparisons=0;self.unassigned_hands=0
    def extractor_id(self):return 'synthetic-no-model'
    def process(self,rgb,ts,out,row):
        if len(self.raw_rows)==self.fail_at:raise ValueError('synthetic consumer failure')
        time.sleep(.006)
        out['xy_px'][row,0]=[float(rgb[0,0,0]),2.]
        out['observed'][row,0]=True;out['usable'][row,0]=True;out['conf'][row,0]=1.
        self.raw_rows.append(dict(raw_count=np.int16(0),width=np.int32(4),height=np.int32(4),hand_xy_px=np.full((2,21,2),np.nan),raw_label=np.full(2,-1,np.int8),raw_score=np.full(2,np.nan),pose_xy_px=out['xy_px'][row,:33].copy(),pose_usable=out['usable'][row,:33].copy()))


class FakeAssigner:
    def __init__(self):self.frames=0
    def identity(self):return 'synthetic-no-coefficients'
    def assign(self,*args):self.frames+=1;return [],dict(calls=self.frames)


def main():
    contract={'usable_rule':{'bounds_margin_ratio':.05,'hand_score_threshold':.5},'hand_assignment':{'policy':'synthetic'}}
    us=[0,5000,20000,40000]
    def fixtures(path):
        for i,t in enumerate(us):
            yield i,100+t//1000,t,np.full((4,4,3),i,np.uint8),dict(width=4,height=4,time_base='1/1000',origin_pts=100)
    actual_decode=S.decode_once;S.decode_once=fixtures;results=[];checks=[]
    try:
        for mode in ['single_decode_batch','single_decode_paced']:
            processor=S.FrameProcessor(FakeLandmarker(contract),FakeAssigner(),contract)
            epoch=time.perf_counter_ns();feed=S.feed_single_decode('fixture',processor,mode,epoch)
            arrays=S.concatenate_rows(processor.baseline_rows)
            assert arrays['frame_id'].tolist()==[0,1,2,3]
            assert arrays['timestamp_us'].tolist()==us
            assert len(processor.timing)==4 and processor.assigner.frames==4
            assert all(r['queue_wait_seconds']>=0 for r in processor.timing)
            if mode.endswith('paced'):
                assert feed['last_frame_available_ns']>=epoch+us[-1]*1000
                assert all(r['available_ns']>=r['due_ns'] for r in feed['arrivals'])
            results.append(arrays);checks.append(dict(name=mode+'_every_vfr_frame_once',passed=True))
        assert all(S.equal_array(results[0][k],results[1][k]) for k in S.ARRAY_NAMES)
        checks.append(dict(name='batch_paced_exact_synthetic_observation',passed=True))
        processor=S.FrameProcessor(FakeLandmarker(contract,fail_at=1),FakeAssigner(),contract)
        try:S.feed_single_decode('fixture',processor,'single_decode_paced',time.perf_counter_ns())
        except ValueError as error:assert str(error)=='synthetic consumer failure'
        else:raise AssertionError('Consumer failure was swallowed')
        checks.append(dict(name='consumer_failure_propagates_without_deadlock',passed=True))
    finally:S.decode_once=actual_decode
    # Verify fixed hand-assignment implementation agrees for a synthetic empty-hand sequence.
    capture,assignment,base=S.dependencies()
    processor=S.FrameProcessor(FakeLandmarker(contract),assignment.HandAssigner(),contract)
    for i,t in enumerate(us):processor.process_frame(np.zeros((4,4,3),np.uint8),t,i,time.perf_counter_ns(),0,100+t//1000)
    baseline=S.Observation(**S.concatenate_rows(processor.baseline_rows));baseline.sidecar={}
    raw={key:np.stack([r[key] for r in processor.landmarker.raw_rows]) for key in processor.landmarker.raw_rows[0]}
    expected,sides,trace=base.associate(raw,baseline,assignment.HandAssigner,contract)
    actual=S.Observation(**S.concatenate_rows(processor.assigned_rows))
    assert all(S.equal_array(getattr(actual,k),getattr(expected,k)) for k in S.ARRAY_NAMES)
    assert S.equal_array(np.stack(processor.sides),sides) and processor.diagnostics==trace
    checks.append(dict(name='unchanged_frozen_assigner_causal_replay_exact',passed=True))
    S.write_new(S.HERE/'SELFTEST.json',dict(status='PASS',checks=checks,model_inference=0,public_video_decode=0))
    print(json.dumps(dict(status='PASS',checks=len(checks),model_inference=0)))


if __name__=='__main__':main()
