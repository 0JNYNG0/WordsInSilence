"""No real SignRep: mock outputs plus local synthetic lossless video decoding."""
import ast
import hashlib
import importlib.util
import json
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np

A=Path(__file__).resolve().parent
O=A.parent
RESULT=A/'PROCESS_SHORT_INPUT_REVIEW.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run():
    assert not RESULT.exists(), 'Preserve first completed test report'
    bridge=module('_process_short_bridge',A/'process_rgb_features.py')
    base_path=O/'word100_seated_web_framing_20261102/adapter.py'
    base=module('_process_short_base',base_path)
    shared_path=O/'word100_learned_guide_feedback_20261102/live_adapter.py'
    # Execute only these existing functions; no engine/checkpoint imports.
    tree=ast.parse(shared_path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('source_axis','extract_rgb')]
    assert len(selected)==2
    namespace={'np':np,'time':time,'base':base}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(shared_path),'exec'),namespace)
    shared=SimpleNamespace(__file__=str(shared_path),base=base,
                           extract_rgb=namespace['extract_rgb'],source_axis=namespace['source_axis'])
    tests=[]
    pins={str(p):sha(p) for p in [base_path,shared_path,base.SIGNREP/'common.py']}
    directory=A/'short_rgb_synthetic_tests'
    directory.mkdir(exist_ok=False)

    def video(n,vfr=False):
        path=directory/f'synthetic_{n}_{"vfr" if vfr else "cfr"}.nut'
        with av.open(str(path),'w') as out:
            stream=out.add_stream('ffv1',rate=30)
            stream.width=64;stream.height=48;stream.pix_fmt='bgr0'
            t=0
            for i in range(n):
                y,x=np.indices((48,64))
                array=np.stack(((x+i*13)%256,(y*3+i*7)%256,((x+y)*2+i*19)%256),axis=-1).astype(np.uint8)
                frame=av.VideoFrame.from_ndarray(array,format='rgb24')
                frame.pts=90+t;frame.time_base=Fraction(1,90)
                t+=3+(i%2 if vfr else 0)
                for packet in stream.encode(frame):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
        return path

    def independently_decode(path):
        images=[];stamps=[];origin=None
        with av.open(str(path)) as source:
            for frame in source.decode(video=0):
                if origin is None:origin=frame.pts*frame.time_base
                stamps.append(int(round(float(frame.pts*frame.time_base-origin)*1e6)))
                images.append(frame.to_ndarray(format='rgb24'))
        return images,np.asarray(stamps,np.int64)

    def normalize_expected(image):
        image=cv2.resize(image,(224,224),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mean=np.asarray([.485,.456,.406],np.float32)*255
        scale=1/(np.asarray([.229,.224,.225],np.float32)*255)
        return ((image-mean)*scale).transpose(2,0,1)

    class MockCompiled:
        def __init__(self):self.inputs=[]
        def __call__(self,args):
            value=np.array(args[0],copy=True)
            assert value.shape==(1,3,16,224,224)
            self.inputs.append(value)
            z=np.arange(768,dtype=np.float32)[None,:]+np.float32(value.mean())
            return {0:z,1:z+np.float32(1000)}

    # All possible short supported lengths, plus exact boundary and long path.
    for n in list(range(2,16))+[16,17,35]:
        source=video(n,vfr=(n==7))
        images,expected_ts=independently_decode(source)
        ids,ts,w,h=shared.source_axis(source)
        assert len(images)==n and np.array_equal(ids,np.arange(n)) and np.array_equal(ts,expected_ts)
        old_ids=ids.copy();old_ts=ts.copy();mock=MockCompiled();events=[]
        original_shared=shared.extract_rgb;delegated=[]
        def spy(*args):
            delegated.append(args)
            return original_shared(*args)
        shared.extract_rgb=spy
        rgb,pre,seconds,meta=bridge.extract_rgb(shared,source,ids,ts,w,h,mock,events.append)
        shared.extract_rgb=original_shared
        assert rgb.shape==pre.shape==(n,768) and seconds>=0
        assert np.array_equal(ids,old_ids) and np.array_equal(ts,old_ts)
        assert meta['real_frame_id']==ids.tolist() and meta['real_timestamp_us']==ts.tolist()
        assert meta['returned_rows']==n and not meta['frame_ids_extended'] and not meta['timestamps_extended']
        assert meta['exact_decode_pts_verified'] and meta['source_sha256']==sha(source)
        assert meta['labels_modified'] is False and meta['caller_image_available_mask_unchanged'] is True
        if n<16:
            assert not delegated and len(mock.inputs)==1 and len(events)==1
            indices=list(range(n))+[n-1]*(16-n)
            assert meta['model_window_indices']==[indices]
            assert meta['source_mask']==[[True]*n+[False]*(16-n)]
            assert meta['padding_count']==16-n
            for j,i in enumerate(indices):
                assert np.array_equal(mock.inputs[0][0,:,j],normalize_expected(images[i]))
            assert np.array_equal(rgb,np.repeat(rgb[:1],n,axis=0))
            assert np.array_equal(pre,np.repeat(pre[:1],n,axis=0))
        else:
            assert len(delegated)==1 and meta['padding_count']==0
            starts=list(range(0,n-15,2))
            if starts[-1]!=n-16:starts.append(n-16)
            assert meta['model_window_indices']==[list(range(s,s+16)) for s in starts]
            assert len(mock.inputs)==len(starts)
            for k,s in enumerate(starts):
                for j in range(16):
                    assert np.array_equal(mock.inputs[k][0,:,j],normalize_expected(images[s+j]))
            # Independently compute interpolation of the mock's window outputs.
            centers=np.asarray([(int(ts[s])+int(ts[s+15]))/2 for s in starts])
            z=np.stack([np.arange(768,dtype=np.float32)+np.float32(x.mean()) for x in mock.inputs])
            expected=np.stack([np.interp(ts,centers,z[:,j]) for j in range(768)],axis=1).astype(np.float32)
            expected_pre=np.stack([np.interp(ts,centers,(z+np.float32(1000))[:,j]) for j in range(768)],axis=1).astype(np.float32)
            assert np.array_equal(rgb,expected) and np.array_equal(pre,expected_pre)
        tests.append({'real_frames':n,'vfr':n==7,'source':str(source),'source_sha256':sha(source),
                      'mock_calls':len(mock.inputs),'padding_count':meta['padding_count'],
                      'exact_pixels_and_pts':True,'delegated':bool(delegated)})

    short=directory/'synthetic_7_vfr.nut'
    ids,ts,w,h=shared.source_axis(short)
    for name,bad_ids,bad_ts,bad_w,bad_h in [
        ('incorrect_pts',ids,ts+np.asarray([0,0,1,0,0,0,0]),w,h),
        ('missing_real_frame',ids[:-1],ts[:-1],w,h),
        ('source_frame_id_mismatch',ids+1,ts,w,h),
        ('wrong_whole_frame_dimensions',ids,ts,w+1,h),
        ('nonmonotone_time',ids,ts[::-1],w,h),
    ]:
        mock=MockCompiled()
        try:bridge.extract_rgb(shared,short,bad_ids,bad_ts,bad_w,bad_h,mock)
        except ValueError as exc:
            assert len(mock.inputs)==0
            tests.append({'rejection':name,'message':str(exc),'mock_calls':0})
        else:raise AssertionError(f'Invalid input accepted: {name}')
    for n in [0,1]:
        source=video(1) if n==1 else short
        mock=MockCompiled()
        rgb,pre,_,meta=bridge.extract_rgb(shared,source,np.arange(n,dtype=np.int64),np.zeros(n,np.int64),64,48,mock)
        assert rgb.shape==pre.shape==(n,768) and np.isnan(rgb).all() and np.isnan(pre).all()
        assert not meta['supported'] and not meta['exact_decode_pts_verified'] and len(mock.inputs)==0
        tests.append({'unsupported_frames':n,'unknown_features':True,'mock_calls':0})
    assert all(sha(p)==value for p,value in pins.items())
    report={
      'at':datetime.now(timezone.utc).isoformat(),'status':'PASS_MOCK_AND_SYNTHETIC_INPUT_CONTRACT',
      'reviewer':'Codex predicate_contract_review; implementation and own structural test, not independent model validation',
      'decision':'Use an explicit short-input branch with final-real-frame edge padding only inside the existing frozen 16-frame encoder; keep N real rows and all original time/labels.',
      'source_code_pins':pins,
      'new_code_pins':{str(A/name):sha(A/name) for name in ['process_rgb_features.py','check_process_rgb_features.py']},
      'runtime_versions':{'numpy':np.__version__,'PyAV':av.__version__,'opencv':cv2.__version__},
      'tests':tests,'tests_passed':len(tests),
      'verified':['All 2–15 real frames preserved exactly once before padding; last pixels only repeated',
        '224 RGB normalization and resizing equal the original common.norm preprocessing',
        'Compiled outputs0/1 retained as RGB/prelinear768; one short-window result only broadcast at real N times',
        'Exact decoder PTS and frame IDs; mismatch rejects before any mock model call',
        'N>=16 invokes original shared extraction unchanged; exact boundary, odd tail and longer interpolation checked',
        'N<2 returns unsupported unknown features without inference; no invented observations',
        'Full-frame dimensions checked on short path; no body/hand cropping',
        'Existing adapter and common source hashes unchanged'],
      'limitations':['Edge padding changes temporal context and may bias a short clip toward apparent holding; accuracy is not proven.',
        'N broadcast feature rows are one contextual observation, not N independent motion or phase predictions.',
        'source_mask describes real vs padded slots only; caller must preserve its actual image/visibility masks, including blackouts.',
        'Synthetic raw-video test validates plumbing only; real short KSL movement and speed validation remain separate.',
        'Keep original-to-edited frame provenance outside this adapter; ids and timestamps here address the supplied edited video.',
        'Current original16-frame route remains unchanged; this short extension is a new explicitly marked input condition.'],
      'actual_signrep_inferences':0,'actual_model_training':0,'public_or_private_videos_read':0,
      'whole_word_correctness_claim':None,'whole_goal_complete':False,
      'first_inspection_issue':{'chunk_id':'f53579','error':'rg shell wildcard path unsupported on Windows',
        'correction':'Read exact adapter path identified by existing PROCESS_INPUT_REVIEW; no data/model changes.'}
    }
    RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'tests':len(tests),'report':str(RESULT),'sha256':sha(RESULT)}))

if __name__=='__main__':
    try:run()
    except BaseException:
        failure=A/'PROCESS_SHORT_INPUT_FIRST_FAILURE.txt'
        if not failure.exists():failure.write_text(traceback.format_exc(),encoding='utf-8')
        raise
