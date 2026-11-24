"""Synthetic RGB and stub landmarks only. Opening a real camera is forbidden here."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from wis_lab.service.camera import CameraWorker
from wis_lab.service import diagnostic_capture as dc

CONTRACT={'contract_id':'synthetic-test','hand_assignment':{'policy':'synthetic-test'}}
RGB=np.zeros((48,64,3),dtype=np.uint8)
RGB[:,:,0]=230

class Landmarks:
    def process(self,rgb,ts,out,i):
        out['xy_px'][i,11]=[12,20]
        out['observed'][i,11]=True
        out['usable'][i,11]=True
        out['conf'][i,11]=1

def worker(cls=CameraWorker):
    cam=cls(CONTRACT)
    cam._actual=(64,48)
    cam.extractor_id='synthetic'
    return cam

def feed(cam):
    for index, us in enumerate([0,50000,100000,240000]):
        cam._record_frame(RGB,1_000_000_000+us*1000,index+10,Landmarks())

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.no_camera=patch('cv2.VideoCapture',side_effect=AssertionError('REAL CAMERA FORBIDDEN'))
        self.no_camera.start()
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup();self.no_camera.stop()
    def recorded(self):
        cam=worker();cam.begin_recording(store_video=True);feed(cam)
        obs,meta=cam.take();obs.save(str(self.path/'sample.npz'))
        return obs,meta['_capture_buffer']
    def save(self,obs,buf):
        return dc.persist_capture(buf,obs,self.path/'sample.capture.zip',session_id='s',trial_id='t')

    def test_disabled_does_not_encode_or_save(self):
        cam=worker();cam.begin_recording()
        with patch.object(dc.cv2,'imencode',side_effect=AssertionError('MUST NOT ENCODE')):
            feed(cam);obs,meta=cam.take()
            self.assertIsNone(meta['_capture_buffer'])
            self.assertEqual(self.save(obs,None)['status'],'DISABLED')
        self.assertFalse((self.path/'sample.capture.zip').exists())

    def test_opt_in_pairs_exact_frames_and_decodable_color(self):
        obs,buf=self.recorded();summary=self.save(obs,buf)
        self.assertEqual(summary['status'],'AVAILABLE')
        self.assertEqual(summary['frame_count'],3)
        m=dc.read_manifest(self.path/'sample.capture.zip')
        self.assertEqual([f['frame_id'] for f in m['frames']],[10,12,13])
        self.assertEqual([f['timestamp_us'] for f in m['frames']],[0,100000,240000])
        self.assertEqual([f['observation_row'] for f in m['frames']],[0,2,3])
        self.assertEqual(m['observation_sha256'],obs.sidecar['observation_sha256'])
        np.testing.assert_allclose(m['frames'][0]['points_xy_normalized'][11],[12/64,20/48],atol=1e-7)
        self.assertIsNone(m['frames'][0]['points_xy_normalized'][12])
        decoded=cv2.imdecode(np.frombuffer(dc.read_frame(self.path/'sample.capture.zip',0),np.uint8),1)
        self.assertGreater(float(decoded[:,:,2].mean()),220)  # RGB red preserved through BGR codec
        self.assertFalse(m['training_use']);self.assertFalse(m['audio'])

    def test_original_and_new_opt_in_off_observations_identical(self):
        # Synthetic golden arrays captured from the preceding camera.py, SHA pinned in evidence.
        cam=worker();cam.begin_recording();feed(cam);obs,_=cam.take()
        with np.load(ROOT/'tests/fixtures/r26_default_capture.npz') as prior:
            for name in ('xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score'):
                np.testing.assert_array_equal(prior[name],getattr(obs,name))

    def test_opt_in_does_not_change_observation(self):
        observations=[]
        for enabled in (False,True):
            cam=worker();cam.begin_recording(store_video=enabled);feed(cam);observations.append(cam.take()[0])
        for name in ('xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score'):
            np.testing.assert_array_equal(getattr(observations[0],name),getattr(observations[1],name))

    def test_limits_and_encode_failure_are_explicit(self):
        b=dc.CaptureBuffer()
        with patch.object(dc,'MAX_BYTES',1):b.add(RGB,1,0)
        self.assertTrue(b.limit_reached);self.assertEqual(b.bytes,0)
        b=dc.CaptureBuffer()
        with patch.object(dc.cv2,'imencode',return_value=(False,None)):b.add(RGB,1,0)
        self.assertEqual(b.failures,1)
        with patch.object(dc,'MAX_FRAMES',1):
            b.add(RGB,2,100000);b.add(RGB,3,200000)
        self.assertTrue(b.limit_reached);self.assertEqual(len(b.frames),1)

    def test_bad_identity_and_duplicate_file_rejected(self):
        obs,b=self.recorded();b.frames[0]['timestamp_us']=1
        with self.assertRaisesRegex(ValueError,'identity'):self.save(obs,b)
        self.assertFalse((self.path/'sample.capture.zip').exists())
        b.frames[0]['timestamp_us']=0;self.save(obs,b)
        with self.assertRaises(FileExistsError):self.save(obs,b)
        with self.assertRaises(IndexError):dc.read_frame(self.path/'sample.capture.zip',-1)

    def test_cancel_then_new_take_cannot_receive_inflight_frame(self):
        cam=worker();cam.begin_recording(store_video=True)
        class CancelDuringExtraction(Landmarks):
            def process(self,*args):
                super().process(*args)
                cam.request_stop('cancel');cam.take();cam.begin_recording(store_video=True)
        cam._record_frame(RGB,1_000_000_000,1,CancelDuringExtraction())
        self.assertEqual(cam.status()['recorded_frames'],0)
        self.assertEqual(len(cam._capture_buffer.frames),0)
        feed(cam);obs,meta=cam.take()
        self.assertEqual(obs.frame_id.tolist(),[10,11,12,13])
        self.assertEqual(len(meta['_capture_buffer'].frames),3)
        cam.stop();self.assertIsNone(cam._capture_buffer)

    def test_empty_and_partial_capture_honest(self):
        obs,b=self.recorded();b.limit_reached=True
        self.assertEqual(self.save(obs,b)['status'],'PARTIAL')
        empty=dc.persist_capture(dc.CaptureBuffer(),obs,self.path/'empty.zip',session_id='s',trial_id='t')
        self.assertEqual(empty['status'],'UNAVAILABLE');self.assertFalse((self.path/'empty.zip').exists())

    def test_resolution_mismatch_and_tampered_rgb_rejected(self):
        import zipfile
        obs,b=self.recorded();b.frames[0]['source_width']=65
        with self.assertRaisesRegex(ValueError,'resolution'):self.save(obs,b)
        b.frames[0]['source_width']=64;self.save(obs,b)
        p=self.path/'sample.capture.zip'
        with zipfile.ZipFile(p) as z:entries={n:z.read(n) for n in z.namelist()}
        entries['frames/0000.jpg']=b'corrupt'
        with zipfile.ZipFile(p,'w') as z:
            for n,data in entries.items():z.writestr(n,data)
        with self.assertRaisesRegex(ValueError,'hash'):dc.read_frame(p,0)

class ApiTests(unittest.TestCase):
    def test_full_lifecycle_off_on_cancel_and_rgb_endpoints(self):
        from fastapi.testclient import TestClient
        import wis_lab.service.app as api
        with tempfile.TemporaryDirectory() as folder:
            cam=worker()
            st=SimpleNamespace(lock=threading.RLock(),sessions={},trials={},camera=cam,
                camera_session=None,contract=CONTRACT,lineage={'set_id':'synthetic','extractor_id':'synthetic'},
                refs=[],reap_expired=lambda:[])
            # mirrors Store.acquire_camera(session_id, device_index, shadow_assignment=False); the
            # synthetic worker is never started. These sessions never opt in, so the default must
            # arrive unchanged.
            def acquire(sid,device,shadow_assignment=False):
                self.assertFalse(shadow_assignment)
                st.camera_session=sid
            st.acquire_camera=acquire
            with patch.object(api,'store',st),patch.object(api,'ROOT',folder), \
                 patch.object(api,'predict',side_effect=lambda *args:{'observation_status':'OK'}), \
                 patch.object(api.neural,'recognise',return_value={'schema':'stub'}), \
                 patch.object(api.neural,'word_coverage',return_value={}), \
                 patch.object(api,'_split_result'),patch.object(api,'_research_comparison',return_value=None), \
                 patch.object(api,'_append_decision'),patch.object(api,'_timing_event',return_value=(True,None)), \
                 patch('cv2.VideoCapture',side_effect=AssertionError('REAL CAMERA FORBIDDEN')):
                client=TestClient(api.app)
                for enabled in (False,True):
                    session=client.post(api.API+'/sessions',json={'store_video':enabled}).json()
                    sid=session['session_id'];self.assertEqual(session['store_video'],enabled)
                    start=client.post(api.API+f'/sessions/{sid}/trials',json={'request_id':'one','sample_label':'synthetic'}).json()
                    tid=start['trial_id'];feed(cam)
                    done=client.post(api.API+f'/trials/{tid}/stop').json()
                    self.assertEqual(done['state'],'FEEDBACK',done)
                    cap=done['result']['diagnostic_capture']
                    self.assertEqual(cap['status'],'AVAILABLE' if enabled else 'DISABLED')
                    get=client.get(api.API+f'/trials/{tid}/diagnostic',params={'session_id':sid})
                    self.assertEqual(get.status_code,200 if enabled else 404)
                    self.assertEqual(client.get(api.API+f'/trials/{tid}/diagnostic',params={'session_id':'wrong'}).status_code,404)
                    if enabled:
                        self.assertEqual(len(get.json()['frames']),3)
                        img=client.get(api.API+f'/trials/{tid}/diagnostic/frames/1',params={'session_id':sid})
                        self.assertEqual(img.status_code,200);self.assertEqual(img.content[:2],b'\xff\xd8')
                        self.assertEqual(img.headers['cache-control'],'no-store')
                        self.assertEqual(client.get(api.API+f'/trials/{tid}/diagnostic/frames/99',params={'session_id':sid}).status_code,404)
                    cancelled=client.post(api.API+f'/sessions/{sid}/trials',json={'request_id':'two','sample_label':'synthetic'}).json()['trial_id']
                    feed(cam);client.post(api.API+f'/trials/{cancelled}/cancel')
                    self.assertFalse((Path(folder)/'runs'/sid/'trials'/(cancelled+'.capture.zip')).exists())
                # An opt-in disk failure is visible but cannot turn recognition into an error.
                tid=client.post(api.API+f'/sessions/{sid}/trials',json={'request_id':'disk_error','sample_label':'synthetic'}).json()['trial_id']
                feed(cam)
                with patch.object(api.diagnostic_capture,'persist_capture',side_effect=OSError('synthetic disk full')):
                    done=client.post(api.API+f'/trials/{tid}/stop').json()
                self.assertEqual(done['state'],'FEEDBACK')
                self.assertEqual(done['result']['diagnostic_capture']['status'],'ERROR')
                self.assertEqual(client.get(api.API+f'/trials/{tid}/diagnostic',params={'session_id':sid}).status_code,404)

if __name__=='__main__':unittest.main(verbosity=2)
