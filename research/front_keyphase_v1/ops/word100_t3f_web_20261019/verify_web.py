"""Public TRAIN arithmetic + synthetic API persistence. No camera or user records."""
from pathlib import Path
import copy, hashlib, json, sys, threading, time, traceback
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

U = Path(__file__).resolve().parent
LAB = U.parents[1]
sys.path.insert(0, str(LAB / 'src'))
from wis_lab.service import word100 as W, word100_candidate as C, word100_t3f as T
from wis_lab.service import word100_web_analysis as A, word100_analysis as OLD, app
from wis_lab.core.observation import Observation, sha256_file

checks = []
def check(name, fn):
    try:
        checks.append(dict(name=name, status='PASS', detail=fn()))
    except Exception:
        checks.append(dict(name=name, status='FAIL', traceback=traceback.format_exc()))

def main():
    import torch
    preserved = [C.CHECKPOINT, T.CHECKPOINT, Path(W.config()['checkpoint']), Path(C.__file__),
                 LAB/'src/wis_lab/service/camera.py', LAB/'src/wis_lab/service/word100_role_dtw.py',
                 LAB/'src/wis_lab/service/word100_multiref.py', LAB/'src/wis_lab/service/word100_analysis.py']
    before = {str(p):sha256_file(p) for p in preserved}
    record = json.loads((C.UNIT/'data/records/NIA_SL_WORD0106_REAL01_F.json').read_text('utf-8'))
    assert record['role'] == 'train_candidate' and record['signer_id'] == 1
    views = record.get('views', record.get('modes'))
    co = C.finalize_observation(Observation.load(views['sensor']['observation']))
    mo = Observation.load(views['m0_sensor']['observation'])
    identity = T.identity()
    pair = T.recognise_pair(co)
    trial = dict(session_id='ses_000000000001', trial_id='tri_000000000001',
        requested_word_id=W.config()['word_codes'][0], research_model=identity,
        reference_binding=dict(status='BOUND'), timing={}, clock_id='synthetic')

    def runtime():
        assert identity['model_status'] == 'READY', identity
        assert all(x['model_status'] == 'READY' for x in pair), pair
        old = C.recognise(co)
        np.testing.assert_array_equal(old['logits'], pair[1]['logits'])
        assert pair[0]['input_sha256'] == pair[1]['input_sha256']
        assert pair[0]['input_shape'] == pair[1]['input_shape']
        return dict(source=record['clip_id'], role='TRAIN', C_adapter_logits_exact=True,
                    shared_input_sha256=pair[0]['input_sha256'], shape=pair[0]['input_shape'])
    check('fixed_models_and_exact_existing_C_output', runtime)

    def arithmetic():
        # Separate explicit reduction of early/middle/late summaries; no phase helper use.
        clip,_,_ = C.build_candidate_channels(co)
        x = torch.from_numpy(clip['x'][None].astype(np.float32)); pad = torch.ones(x.shape[:2])
        m = T._load(); base = C._load()
        oldstate = {k:v.clone() for k,v in m.state_dict().items()}
        with torch.inference_mode():
            h = base.proj(x.transpose(1,2))
            for block in base.blocks: h = block(h, pad.unsqueeze(1))
            g = h.mean(dim=2)
            times = x[:,:,279].cumsum(1)
            span = times[:,-1:] - times[:,:1]
            u = (times-times[:,:1])/span if span.item()>0 else torch.full_like(times,.5)
            weights = [(1-2*u).clamp_min(0), 1-(2*u-1).abs(), (2*u-1).clamp_min(0)]
            pieces = []
            for w in weights:
                mass = w.sum(1,keepdim=True)
                summary = (h*w.unsqueeze(1)).sum(2)/mass if mass.item()>0 else g
                pieces.append(summary-g)
            expected = base.head(g) + torch.cat(pieces,1) @ m.phase_weight.T
        err = float(np.max(np.abs(expected.numpy()[0]-pair[0]['logits'])))
        np.testing.assert_allclose(expected.numpy()[0], pair[0]['logits'], rtol=2e-5, atol=2e-5)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in oldstate.items())
        assert not m.training
        return dict(independent_formula_max_absolute_error=err, buffers_and_weights_unchanged=True)
    check('independent_temporal_arithmetic_and_frozen_state', arithmetic)

    def build_once():
        with patch.object(C,'build_candidate_channels', wraps=C.build_candidate_channels) as build:
            result = W.result(mo,trial,co)
        assert build.call_count == 1
        assert result['comparison']['same_input'] and result['comparison']['same_frames']
        assert result['data_use']=='evaluation_only_never_training_tuning_or_model_selection'
        changed = copy.deepcopy(trial); changed['requested_word_id'] = W.config()['word_codes'][99]
        with patch.object(T,'recognise_pair',return_value=copy.deepcopy(pair)) as blind:
            other = W.result(mo,changed,co)
        assert blind.call_args.args == (co,)
        assert result['practice100']['logits'] == other['practice100']['logits']
        assert result['comparison']['baseline']['logits'] == other['comparison']['baseline']['logits']
        return 'One channel build; target word only annotates ranks after inference'
    check('same_input_single_build_and_label_blind', build_once)

    def failures():
        mismatched = copy.deepcopy(co); mismatched.frame_id += 1
        with patch.object(T,'recognise_pair',side_effect=AssertionError('must not infer')):
            out = W.result(mo,trial,mismatched)
        assert out['practice100']['reason'] == 'CANDIDATE_FRAME_IDENTITY_MISMATCH'
        assert not out['comparison']['same_frames'] and not out['comparison']['same_input']
        with patch.object(T,'_load',side_effect=ValueError('synthetic pin failure')):
            a,b = T.recognise_pair(co)
        assert a['model_status']=='MODEL_ERROR' and not a['ranking'] and b['model_status']=='READY'
        assert a['model_key']==T.KEY and b['model_key']==C.KEY
        none = W.result(mo,trial,None)
        assert none['practice100']['model_status']=='CANDIDATE_OBSERVATION_UNAVAILABLE'
        return 'Mismatched/missing inputs rejected; failed T3-F never borrows C prediction'
    check('fail_closed_no_fallback', failures)

    def creation():
        calls = []
        s = dict(state='OPEN',request_ids={},trial_ids=[],client_id=None,store_video=False)
        st = SimpleNamespace(lock=threading.RLock(), sessions={trial['session_id']:s},trials={},
            camera_session=trial['session_id'],reap_expired=lambda:None,
            camera=SimpleNamespace(clock_id='synthetic',begin_recording=lambda **kw:calls.append(kw)))
        body = app.TrialCreate(request_id='synthetic1',requested_word_id=trial['requested_word_id'],
            research_model_key=T.KEY,demonstration_clip_id='synthetic')
        with patch.object(app,'S',return_value=st),patch.object(W,'binding',return_value={'status':'BOUND'}):
            a = app.create_trial(trial['session_id'],body)
            b = app.create_trial(trial['session_id'],body)
        assert calls==[dict(store_video=False,candidate_c=True)] and b['deduplicated']
        assert st.trials[a['trial_id']]['research_model']['checkpoint_sha256']==T.PIN
        return 'T3-F selects existing fresh C detector, duplicate request records once; camera mocked'
    check('trial_opt_in_capture_routing_and_deduplication', creation)

    def persistence():
        evidence = U/'synthetic_transport'
        run = evidence/trial['session_id']; (run/'trials').mkdir(parents=True,exist_ok=True)
        q,cq = copy.deepcopy(mo),copy.deepcopy(co)
        meta = dict(end_candidate_ns=time.perf_counter_ns(),end_reason='manual',_candidate_c_obs=cq,
                    candidate_c_error=None,_capture_buffer=None)
        st = SimpleNamespace(camera=SimpleNamespace(take=lambda:(q,meta)),contract={'contract_id':'synthetic'})
        s = dict(session_id=trial['session_id'],run_dir=str(run),store_video=False,shadow_assignment=False,
                 purpose='synthetic',participant_id='SYNTHETIC_T3F_INTEGRATION')
        t = copy.deepcopy(trial)
        with patch.object(T,'recognise_pair',return_value=copy.deepcopy(pair)):
            done = app._finish(t,st,s)
        assert done['state']=='FEEDBACK', done
        path = run/'trials'/f"{trial['trial_id']}.result.json"
        saved = json.loads(path.read_text('utf-8'))
        assert saved['candidate_observation']['same_frames_as_primary']
        assert saved['comparison']['same_input']
        with patch.object(OLD,'RUNS',evidence),patch.object(A,'RUNS',evidence), \
             patch.object(T,'recognise_pair',side_effect=AssertionError('no re-inference')):
            loaded,_ = A.saved_result(trial['session_id'],trial['trial_id'])
            assert loaded==saved
            bad = copy.deepcopy(saved);bad['comparison']['baseline']['input_sha256']='0'*64
            path.write_text(json.dumps(bad),'utf-8')
            try: A.saved_result(trial['session_id'],trial['trial_id'])
            except ValueError as exc: assert str(exc)=='T3F_PAIRED_INPUT_MISMATCH'
            else: raise AssertionError('bad hash accepted')
            path.write_text(json.dumps(saved,ensure_ascii=False,indent=2),'utf-8')
            # Existing C/M0 saved layout is still accepted, using synthetic scores only.
            prior=copy.deepcopy(saved)
            prior['practice100']=dict(pair[1]);prior['research_model']=C.identity()
            prior['comparison']['baseline']=dict(pair[1],model_key=W.KEY,checkpoint_sha256=W.PIN)
            path.write_text(json.dumps(prior),'utf-8')
            A.saved_result(trial['session_id'],trial['trial_id'])
            path.write_text(json.dumps(saved,ensure_ascii=False,indent=2),'utf-8')
        return dict(path=str(path.relative_to(U)), both_logits=200, candidate_NPZ_bound=True,
                    old_C_history_compatible=True, replay_without_inference=True)
    check('save_load_pair_and_tamper_rejection', persistence)

    def unchanged():
        assert {str(p):sha256_file(p) for p in preserved} == before
        assert before[str(Path(C.__file__))]=='e043f662fb926438e5ef6fbd40713f8b08f9f676a8973dfca742a24a7125ff2e'
        cat = W.catalog()
        assert cat['model']['model_key']==W.KEY
        assert [m['model_key'] for m in cat['models']]==[W.KEY,C.KEY,T.KEY]
        return dict(preserved_hashes=before, default_M0=True)
    check('existing_C_M0_DTW_capture_files_preserved', unchanged)
    output = dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
        checks=checks,camera_access=0,user_recording_reads=0,heldout_reads=0,training=0,
        accuracy_claim='None. Public TRAIN plumbing checks only.',
        verification='Codex local independent arithmetic and synthetic transport, not independent reviewer')
    out=U/'VERIFICATION.json'
    if out.exists(): out=U/f'VERIFICATION_{time.time_ns()}.json'
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return 0 if output['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
