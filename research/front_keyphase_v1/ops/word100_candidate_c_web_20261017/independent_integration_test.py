"""Independent synthetic integration review. No camera, detector, forward or training.

Uses generated coordinates/RGB and stub recognisers. Keeps temporary evidence in
this approved unit only. Existing recordings, weights and frozen reports are read-only.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import numpy as np

UNIT = Path(__file__).resolve().parent
LAB = UNIT.parents[1]
sys.path.insert(0, str(LAB / 'src'))
from wis_lab.core.observation import Observation, empty_frames, sha256_file
from wis_lab.service import camera, neural, word100, word100_analysis as A, word100_multiref as D

RESULTS = []
EVIDENCE = UNIT / ('independent_synthetic_' + uuid.uuid4().hex[:12])
EVIDENCE.mkdir()


def check(name, fun):
    try:
        detail = fun()
        RESULTS.append({'name': name, 'status': 'PASS', 'detail': detail})
    except Exception as exc:
        RESULTS.append({'name': name, 'status': 'FAIL', 'error': str(exc), 'traceback': traceback.format_exc()})


def obs(t=15):
    a = empty_frames(t)
    a['frame_id'] = np.arange(t, dtype=np.int64) + 101
    a['timestamp_us'] = np.arange(t, dtype=np.int64) * 40000
    for i in range(t):
        for j, xy in {11:(35,25),12:(65,25),13:(30,40),14:(70,40),15:(30,50),16:(70,50),23:(40,75),24:(60,75)}.items():
            a['xy_px'][i,j] = xy
            a['observed'][i,j] = a['usable'][i,j] = True
            a['conf'][i,j] = .9
    for i in (2,3,4,8,9,12):
        if i >= t: continue
        for j in range(33,54):
            a['xy_px'][i,j] = (30+(j-33)%5,35+(j-33)//5)
            a['observed'][i,j] = a['usable'][i,j] = True
    out = Observation(**a)
    out.sidecar = {'width':100,'height':80,'origin':'synthetic','mirrored':False,'rotation_deg':0}
    out.validate()
    return out


class StubDetector:
    def __init__(self, offset=0, error=False, action=None):
        self.calls=[]; self.offset=offset; self.error=error; self.action=action
    def process(self, rgb, ts, out, i):
        self.calls.append({'object':id(rgb),'sha':hashlib.sha256(rgb.tobytes()).hexdigest(),'ts':int(ts)})
        if self.action: self.action()
        if self.error: raise RuntimeError('synthetic detector failure')
        if int(rgb[0,0,0]) == 0: return
        for j,xy in {11:(35,25),12:(65,25),33:(30+self.offset,35),34:(31+self.offset,36)}.items():
            out['xy_px'][i,j] = xy
            out['observed'][i,j] = out['usable'][i,j] = True
            out['conf'][i,j] = .9
    def close(self): pass


def worker(candidate=True):
    w = camera.CameraWorker({'hand_assignment':{'policy':'synthetic_legacy'}})
    w._actual=(100,80); w.extractor_id='synthetic_legacy'; w._recording=True
    w._candidate_c=candidate; w._rec_started_ns=1000000000; w._rec_generation=1
    w._candidate_lm=StubDetector(offset=8)
    w._candidate_extractor_id='synthetic_C_detector'
    return w


def reading(key, pin, winner):
    logits = np.zeros(100).tolist(); logits[winner]=4.
    return {'model_key':key,'checkpoint_sha256':pin,'model_status':'READY','logits':logits,
            'ranking':word100.rank_logits(logits),'is_a_verdict':False,'accepted':None}


def trial(C, sid='ses_000000000001', tid='tri_000000000001'):
    return {'session_id':sid,'trial_id':tid,'requested_word_id':word100.config()['word_codes'][0],
            'research_model':{'model_key':C.KEY,'checkpoint_sha256':C.PIN},
            'reference_binding':{'status':'BOUND'},'timing':{},'clock_id':'synthetic'}


def main():
    from wis_lab.service import word100_candidate as C
    from wis_lab.service import word100_web_analysis as W
    changed = [camera.__file__,word100.__file__,A.__file__,D.__file__,C.__file__,W.__file__]
    pins = {p:sha256_file(p) for p in changed}

    def same_frame_capture():
        w=worker(); m=StubDetector()
        for i,value in enumerate((1,0,1)):
            rgb=np.full((80,100,3),value,np.uint8)
            w._record_frame(rgb,1000000000+i*100000000,70+i,m)
        legacy,meta=w.take(); candidate=meta['_candidate_c_obs']
        assert len(m.calls)==len(w._candidate_lm.calls)==3
        assert [x['sha'] for x in m.calls]==[x['sha'] for x in w._candidate_lm.calls]
        assert [x['object'] for x in m.calls]==[x['object'] for x in w._candidate_lm.calls]
        assert [x['ts'] for x in w._candidate_lm.calls]==[0,100000,200000]
        assert [x['ts'] for x in m.calls]==[1000000,1100000,1200000]
        assert np.array_equal(legacy.frame_id,candidate.frame_id)
        assert np.array_equal(legacy.timestamp_us,candidate.timestamp_us)
        assert legacy.n_frames==candidate.n_frames==3
        assert not legacy.usable[1].any() and not candidate.usable[1].any()
        assert legacy.xy_px[0,33,0]==30 and candidate.xy_px[0,33,0]==38
        assert candidate.sidecar['extractor_id']=='synthetic_C_detector'
        a=legacy.save(str(EVIDENCE/'pair_m0.npz')); b=candidate.save(str(EVIDENCE/'pair_c.npz'))
        assert a!=b
        return {'rgb_frames':3,'same_full_rgb_and_clock':True,'separate_hashes':[a,b]}
    check('same_RGB_frame_ID_PTS_and_separate_observation_hashes',same_frame_capture)

    def detector_failure():
        w=worker(); w._candidate_lm=StubDetector(error=True)
        w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,StubDetector())
        legacy,meta=w.take()
        assert legacy.n_frames==1 and legacy.usable.any()
        assert '_candidate_c_obs' not in meta and 'C_DETECTOR_FRAME_ERROR' in meta['candidate_c_error']
        return 'C failure does not discard M0 observation'
    check('C_detector_failure_preserves_M0',detector_failure)

    def finalize_failure():
        w=worker()
        w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,StubDetector())
        with patch.object(C,'finalize_observation',side_effect=ValueError('synthetic finalization failure')):
            legacy,meta=w.take()
        assert legacy.n_frames==1 and legacy.usable.any()
        assert '_candidate_c_obs' not in meta
        assert 'C_OBSERVATION_FINALIZE_ERROR' in meta['candidate_c_error']
        return 'C-only finalization error leaves original M0 observation intact'
    check('C_finalize_error_does_not_discard_M0',finalize_failure)

    def stale_failure_guard():
        w=worker()
        w._candidate_lm=StubDetector(error=True,action=lambda:setattr(w,'_rec_generation',2))
        w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,StubDetector())
        assert not w._rec_rows
        assert w._candidate_error is None,'Prior take exception leaked to the next recording generation'
        return 'An exception from a discarded in-flight old-generation frame cannot poison the next take'
    check('stale_C_failure_is_generation_scoped',stale_failure_guard)

    def generation_guard():
        w=worker(); m=StubDetector(action=lambda:setattr(w,'_rec_generation',2))
        w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,m)
        assert not w._rec_rows
        return 'No pending row from a prior take was appended'
    check('take_generation_guard',generation_guard)

    def m0_only():
        w=worker(False); w._candidate_lm=StubDetector(error=True)
        w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,StubDetector())
        o,meta=w.take()
        assert o.n_frames==1 and not w._candidate_lm.calls and '_candidate_c_obs' not in meta
        return 'M0-only opt-in absence avoids candidate detector'
    check('M0_path_does_not_touch_C_detector',m0_only)

    def internal_gap_stop():
        findings=[]
        for selected in (True,False):
            w=worker(selected); m=StubDetector()
            w._record_frame(np.ones((80,100,3),np.uint8),1000000000,1,m)
            w._record_frame(np.zeros((80,100,3),np.uint8),1900000000,2,m)
            findings.append(w._end_reason)
        assert findings==[None,'auto_hands_lowered']
        return {'C_gap_preserved':True,'M0_legacy_stop_unchanged':True}
    check('C_internal_missing_gap_not_auto_ended',internal_gap_stop)

    def roi_channel_order():
        original=obs(); original.xy_px[0,33]=[30,81]; original.observed[0,33]=original.usable[0,33]=True
        original.xy_px[-1,33]=[-1,30]; original.observed[-1,33]=original.usable[-1,33]=True
        original.xy_px[3,34]=[100,30]
        original.xy_px[4,35]=[10,80]
        before=copy.deepcopy(original)
        visible=C.finalize_observation(original)
        for i,j in ((0,33),(14,33),(3,34),(4,35)):
            assert not visible.observed[i,j] and not visible.usable[i,j]
            assert np.isnan(visible.xy_px[i,j]).all() and np.isnan(visible.conf[i,j])
        assert np.array_equal(visible.frame_id,before.frame_id)
        assert np.array_equal(visible.timestamp_us,before.timestamp_us)
        x,why,stats=C.build_candidate_channels(visible)
        assert x is not None,(why,stats)
        assert x['x'].shape==(11,280),x['x'].shape
        assert not visible.usable[5:8,33:75].any()
        assert np.isfinite(x['x']).all()
        return {'active_span_frames':11,'all_recorded_frames':15,'channels':280,'internal_gap_preserved':True}
    check('strict_mask_before_active_span_preserves_real_missing_frames',roi_channel_order)

    def partial_hand_and_hidden_perturbation():
        q=obs()
        q.xy_px[3,33]=np.nan;q.observed[3,33]=q.usable[3,33]=False
        # Body wrist remains visible: it must not substitute for the hand wrist.
        assert q.usable[3,15]
        q.xy_px[4,54:75]=[110,90];q.observed[4,54:75]=q.usable[4,54:75]=True
        q2=copy.deepcopy(q);q2.xy_px[4,54:75]=[10000,10000]
        a=C.finalize_observation(q);b=C.finalize_observation(q2)
        x,_,_=C.build_candidate_channels(a);y,_,_=C.build_candidate_channels(b)
        assert np.array_equal(x['x'],y['x'])
        # Frame3 maps to active-span index1; hand-local mask begins at237.
        assert np.all(x['x'][1,237:258]==0)
        assert x['x'][1,186+9]==0 and x['x'][1,186+10]==1
        np.testing.assert_allclose(x['x'][1,20:22],[(31-50)/30,(35-25)/30],atol=1e-6)
        assert np.all(x['x'][3:6,195:237]==0) # original5..7 internal missing hands
        np.testing.assert_allclose(x['x'][1:,279],.04,atol=1e-7)
        return 'Outside coordinate perturbations invariant; visible fingers retain body-relative XY without invented hand wrist/local XY; real gap timestamps retained'
    check('partial_hand_no_body_wrist_substitution_hidden_coordinate_invariance',partial_hand_and_hidden_perturbation)

    def forward_adapter_uses_all_real_frames():
        import torch
        capture={}
        class StubModel:
            def __call__(self,x,pad):
                capture['x']=x.detach().cpu().numpy();capture['pad']=pad.detach().cpu().numpy()
                return torch.zeros((1,100),dtype=torch.float32)
        with patch.object(C,'_load',return_value=StubModel()):
            r=C.recognise(C.finalize_observation(obs()))
        assert r['model_status']=='READY'
        assert capture['x'].shape==(1,11,280)
        assert np.array_equal(capture['pad'],np.ones((1,11),dtype=np.float32))
        assert r['candidate_count']==100 and len(r['logits'])==100
        return 'Classifier adapter tested using stub callable only: pad=1 on all real frames including internal missing frames, 100 logits'
    check('C_classifier_adapter_real_missing_frames_are_not_batch_padding',forward_adapter_uses_all_real_frames)

    def candidate_evidence_identity():
        stored=json.loads((LAB/'ops/word100_body_boundary_20261017/training_C_20261013_RESULT.json').read_text(encoding='utf-8'))
        assert C.KEY==word100.CANDIDATE_KEY
        assert C.PIN==stored['checkpoint_sha256'] and C.EPOCH==stored['selected_epoch']==77
        assert C.SEED==stored['seed']==20261013
        assert C.TRAIN_CONTEXT_PIN==stored['context_sha256']
        assert sha256_file(C.CHECKPOINT)==C.PIN
        return {'seed':C.SEED,'epoch':C.EPOCH,'checkpoint_sha256':C.PIN,'read_only_checkpoint_hash_only':True}
    check('frozen_C_seed_epoch_checkpoint_identity',candidate_evidence_identity)

    def cfailure_no_fallback():
        t=trial(C)
        with patch.object(C,'recognise',side_effect=AssertionError('no C input means no C recognise')),patch.object(word100,'recognise',return_value=reading(word100.KEY,word100.PIN,1)):
            r=word100.result(obs(),t,candidate_obs=None,candidate_error='SYNTHETIC_C_MISSING')
        assert r['practice100']['model_key']==C.KEY
        assert r['practice100']['model_status']!='READY' and not r['practice100']['ranking']
        assert r['comparison']['baseline']['model_key']==word100.KEY
        assert r['top1_word_id'] is None
        return 'C stays unavailable while separately labelled M0 score remains'
    check('C_unavailable_has_no_hidden_M0_fallback',cfailure_no_fallback)

    def mismatched_pair_rejected():
        q=obs();cq=copy.deepcopy(q);cq.frame_id+=1
        with patch.object(C,'recognise',side_effect=AssertionError('mismatched C input must not be scored')),patch.object(word100,'recognise',return_value=reading(word100.KEY,word100.PIN,1)):
            r=word100.result(q,trial(C),candidate_obs=cq)
        assert r['practice100']['model_status']!='READY'
        assert r['practice100']['reason']=='CANDIDATE_FRAME_IDENTITY_MISMATCH'
        assert not r['comparison']['same_frames']
        assert r['comparison']['baseline']['model_status']=='READY'
        return 'Mismatched frame IDs reject candidate inference without substituting its M0 comparison'
    check('mismatched_C_M0_frame_identity_rejected',mismatched_pair_rejected)

    def label_blind():
        q=obs(); cq=copy.deepcopy(q); calls=[]
        def classify(x): calls.append(x); return reading(C.KEY,C.PIN,2)
        with patch.object(C,'recognise',side_effect=classify),patch.object(word100,'recognise',side_effect=lambda x:reading(word100.KEY,word100.PIN,1)):
            t1=trial(C); t2=copy.deepcopy(t1);t2['requested_word_id']=word100.config()['word_codes'][99]
            a=word100.result(q,t1,cq);b=word100.result(q,t2,cq)
        assert calls==[cq,cq]
        assert a['practice100']['logits']==b['practice100']['logits']
        assert a['comparison']['same_frames'] and b['comparison']['same_frames']
        assert a['data_use']=='evaluation_only_never_training_tuning_or_model_selection'
        return 'Requested word only affects explanatory ranks after blind inference'
    check('requested_word_does_not_enter_classifier',label_blind)

    def dtw_saved_binding():
        base=EVIDENCE/'dtw_test'; q=obs();qh=q.save(str(base.with_suffix('.npz')))
        word=word100.config()['word_codes'][0]
        ref={'word_id':word,'clip_id':'synthetic_ref','source_sha256':'synthetic_video',
             'observation':str(base.with_suffix('.npz')),'observation_sha256':qh}
        r={'requested_word_id':word, 'practice100':reading(C.KEY,C.PIN,1),'session_id':'synthetic',
           'trial_id':'synthetic','observation_sha256':qh,
           'reference_binding':{'demonstration_clip_id':'synthetic_ref','source_sha256':'synthetic_video'}}
        original=copy.deepcopy(r)
        calls=[]
        def compare(x,word): calls.append((x,word));return {'status':'PARTIAL','model_sha256':word100.PIN,'metrics':[]}
        with patch.object(W,'saved_result',return_value=(r,base)),patch.object(A,'references',return_value={word:ref}),patch.object(A,'_details',return_value={'dtw':{},'observation_quality':{}}),patch.object(D,'bank',return_value=({},'syntheticbank')),patch.object(D,'compare_query',side_effect=compare):
            got=W.analysis('synthetic','synthetic')
            again=W.analysis('synthetic','synthetic')
        assert len(calls)==1
        assert np.array_equal(calls[0][0].xy_px,q.xy_px,equal_nan=True)
        detail=got['analysis']; dtw=detail['training_comparison']
        assert dtw['model_sha256']==word100.PIN and detail['classifier_checkpoint_sha256']==C.PIN
        assert detail['dtw_reference_model_sha256']==word100.PIN
        assert got==again and dtw['query_observation_sha256']==qh
        assert r==original and got['practice100']['checkpoint_sha256']==C.PIN
        return 'Original observation and old DTW bank model identity remain separate from C classifier'
    check('DTW_original_observation_and_bank_identity',dtw_saved_binding)

    def saved_result_identity_and_tamper():
        sid='ses_000000000002';tid='tri_000000000002';base=EVIDENCE/sid/'trials'/tid
        base.parent.mkdir(parents=True)
        q=obs(); cq=copy.deepcopy(q);cq.xy_px[2,33,0]+=2
        qh=q.save(str(base.with_suffix('.npz')));ch=cq.save(str(base.with_name(base.name+'.candidate_c.npz')))
        r={'session_id':sid,'trial_id':tid,'practice100':reading(C.KEY,C.PIN,1),
           'research_model':{'model_key':C.KEY,'checkpoint_sha256':C.PIN},
           'candidate_observation':{'observation_sha256':ch},'observation_sha256':qh,
           'comparison':{'baseline':reading(word100.KEY,word100.PIN,1)}}
        path=base.with_suffix('.result.json');path.write_text(json.dumps(r),encoding='utf-8')
        with patch.object(A,'RUNS',EVIDENCE):
            got,_=W.saved_result(sid,tid);assert got['practice100']['checkpoint_sha256']==C.PIN
            r['candidate_observation']['observation_sha256']='0'*64;path.write_text(json.dumps(r),encoding='utf-8')
            try:W.saved_result(sid,tid)
            except ValueError as exc:assert 'HASH_MISMATCH' in str(exc)
            else:raise AssertionError('Tampered candidate observation accepted')
        return 'Candidate identity allowlist and separate observation hash checked'
    check('saved_C_result_binding_and_tamper_rejection',saved_result_identity_and_tamper)

    def backend_finish_saved_pair():
        from wis_lab.service import app as api
        sid='ses_000000000003';tid='tri_000000000003';run=EVIDENCE/sid
        (run/'trials').mkdir(parents=True)
        q=obs();cq=C.finalize_observation(copy.deepcopy(q));cq.xy_px[2,33,0]+=2
        meta={'end_candidate_ns':time.perf_counter_ns(),'end_reason':'manual','_candidate_c_obs':cq,'candidate_c_error':None,'_capture_buffer':None}
        st=SimpleNamespace(camera=SimpleNamespace(take=lambda:(q,meta)),contract={'contract_id':'synthetic'})
        s={'session_id':sid,'run_dir':str(run),'store_video':False,'shadow_assignment':False,'purpose':'synthetic','participant_id':'SYNTHETIC_INDEPENDENT'}
        t=trial(C,sid,tid)
        with patch.object(C,'recognise',return_value=reading(C.KEY,C.PIN,2)),patch.object(word100,'recognise',return_value=reading(word100.KEY,word100.PIN,1)):
            got=api._finish(t,st,s)
        assert got['state']=='FEEDBACK'
        saved=json.loads((run/'trials'/f'{tid}.result.json').read_text(encoding='utf-8'))
        assert saved['practice100']['checkpoint_sha256']==C.PIN
        assert saved['comparison']['baseline']['checkpoint_sha256']==word100.PIN
        assert saved['observation_sha256']!=saved['candidate_observation']['observation_sha256']
        assert saved['comparison']['same_frames'] and saved['candidate_observation']['same_frames_as_primary']
        assert saved['accepted'] is None and not saved['is_a_verdict']
        assert saved['diagnostic_capture']['status']=='DISABLED'
        with patch.object(A,'RUNS',EVIDENCE),patch.object(W,'RUNS',EVIDENCE):
            loaded,_=W.saved_result(sid,tid)
        assert loaded['result_id']==saved['result_id']
        return 'Backend finish→separate NPZ/hash→atomic result JSON→C-aware saved_result roundtrip verified with stub scorers'
    check('backend_paired_finish_persistence_roundtrip',backend_finish_saved_pair)

    def dtw_function_equivalence():
        groups={A.__file__:('_sample','compare','compare_samples','_reference_obs','_details'),D.__file__:('aggregate','distribution','relative','sample','compare_query')}
        results=[]
        for current,names in groups.items():
            before=UNIT/'before'/('src__wis_lab__service__'+Path(current).name)
            a={n.name:n for n in ast.parse(before.read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            b={n.name:n for n in ast.parse(Path(current).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            for name in names:
                assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
                results.append(name)
        return {'unchanged_numeric_functions':results}
    check('DTW_numerical_functions_unchanged_AST',dtw_function_equivalence)

    def dtw_bank_ready():
        D.bank.cache_clear()
        bank,pin=D.bank()
        assert pin=='0af5e66be355ab0a781fc0b7bfcbcee551fd92f9f1cf3bf2097f4c87d34440cf'
        assert bank['model_sha256']==word100.PIN
        return {'bank_pin':pin,'words':len(bank['words']),'loads_only_no_comparisons':True}
    check('unchanged_DTW_bank_available_under_validated_adapter',dtw_bank_ready)

    changed_during_test=[p for p,h in pins.items() if sha256_file(p)!=h]
    result={'schema':'independent_candidate_C_web_integration_v1','status':'PASS' if all(x['status']=='PASS' for x in RESULTS) and not changed_during_test else 'FAIL_OR_SOURCE_CHANGED',
        'checks':RESULTS,'passed':sum(x['status']=='PASS' for x in RESULTS),'failed':sum(x['status']=='FAIL' for x in RESULTS),
        'source_pins':pins,'source_changed_during_test':changed_during_test,'evidence':str(EVIDENCE),
        'limits':['Synthetic coordinates/RGB and stub detectors/classifiers only.','No actual camera or MediaPipe construction; no model forward, DEV inference or training.','Not a measurement of real webcam accuracy, performance, or detector quality.']}
    path=UNIT/('INDEPENDENT_INTEGRATION_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'passed':result['passed'],'failed':result['failed'],'report':str(path),'failures':[x for x in RESULTS if x['status']=='FAIL']},ensure_ascii=False))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
