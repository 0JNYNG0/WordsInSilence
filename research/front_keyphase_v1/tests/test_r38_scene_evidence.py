"""R38 synthetic tests: python -m unittest tests/test_r38_scene_evidence.py

No dataset, network, camera or model. Every clip, observation and video here is generated in a
temporary directory; the real-video case is skipped if PyAV/cv2 cannot encode one locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from wis_lab.core.observation import Observation, empty_frames  # noqa: E402
from wis_lab.service import scene_evidence as se  # noqa: E402


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# --------------------------------------------------------------- fake PyAV container
class FakeFrame:
    def __init__(self, pts, value):
        self.pts, self.value, self.converted = pts, value, 0

    def to_ndarray(self, format):          # noqa: A002 - PyAV's own keyword
        assert format == "rgb24"
        self.converted += 1
        return np.full((4, 6, 3), self.value, np.uint8)


class FakePacket:
    def __init__(self, frames):
        self.frames = frames

    def decode(self):
        return list(self.frames)


class FakeContainer:
    def __init__(self, frames, time_base=Fraction(1, 10)):
        self.frames = frames
        self.streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=time_base)])
        self.closed = False

    def demux(self, stream):
        assert stream is self.streams.video[0]
        return [FakePacket([f]) for f in self.frames]

    def close(self):
        self.closed = True


# PTS None is skipped and NOT numbered; pts 2 after pts 3 is numbered (id 3) and dropped.
# Numbered: pts0->0, pts1->1, pts3->2, pts2->3 (dropped), pts4->4.
FAKE_SEQUENCE = [(None, 250), (0, 10), (1, 20), (None, 251), (3, 30), (2, 252), (4, 40)]
KEPT_IDS = [0, 1, 2, 4]
KEPT_TS = [0, 100_000, 300_000, 400_000]
VALUE_BY_ID = {0: 10, 1: 20, 2: 30, 4: 40}


def write_observation(npz, ids, ts, sidecar):
    out = empty_frames(len(ids))
    out["frame_id"][:] = ids
    out["timestamp_us"][:] = ts
    obs = Observation(**out)
    obs.sidecar = dict(sidecar)
    return obs.save(npz)


class Fixture:
    """A registered clip: a source file, an NPZ + sidecar, and a manifest record."""

    def __init__(self, root, clip_id="clip_a", word_id="word_a", ids=KEPT_IDS, ts=KEPT_TS,
                 source_bytes=b"synthetic source bytes, decoded by a fake container"):
        self.root = root
        self.src = os.path.join(root, clip_id + ".mp4")
        if source_bytes is not None:
            with open(self.src, "wb") as fh:
                fh.write(source_bytes)
        self.src_sha = sha(self.src)
        self.npz_rel = os.path.join("obs", clip_id + ".npz").replace("\\", "/")
        self.npz = os.path.join(root, self.npz_rel)
        self.obs_sha = write_observation(self.npz, ids, ts, {
            "schema": "wis_observation_2d_v1", "origin": "clip", "clip_id": clip_id,
            "source_sha256": self.src_sha, "mirrored": False})
        self.clip = {"clip_id": clip_id, "word_id": word_id, "absolute_path": self.src,
                     "sha256": self.src_sha, "observation_npz": self.npz_rel,
                     "observation_sha256": self.obs_sha}

    def sidecar_path(self):
        return os.path.splitext(self.npz)[0] + ".sidecar.json"

    def edit_sidecar(self, **changes):
        with open(self.sidecar_path(), encoding="utf-8") as fh:
            doc = json.load(fh)
        doc.update(changes)
        with open(self.sidecar_path(), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)


class RaisingModule(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"scene evidence touched {self.__name__}.{name}")


class SceneEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.fx = Fixture(self.root)
        self.containers = []

    def tearDown(self):
        self.tmp.cleanup()

    def opener(self, frames=FAKE_SEQUENCE, time_base=Fraction(1, 10)):
        def open_(path):
            c = FakeContainer([FakeFrame(p, v) for p, v in frames], time_base)
            self.containers.append(c)
            return c
        return open_

    def call(self, frame_id, clips=None, src=None, obs=None, clip_id="clip_a", opener=None,
             encode=None):
        return se.reference_frame(
            [self.fx.clip] if clips is None else clips, self.root, clip_id, frame_id,
            self.fx.src_sha if src is None else src, self.fx.obs_sha if obs is None else obs,
            opener=opener or self.opener(), encode=encode or (lambda rgb: rgb.tobytes()))

    def refused(self, status, reason, **kw):
        with self.assertRaises(se.EvidenceError) as cm:
            self.call(**kw)
        self.assertEqual((cm.exception.status, cm.exception.reason), (status, reason))
        self.assertNotIn(self.root, str(cm.exception))
        return cm.exception

    # ------------------------------------------------------------- exact frame
    def test_exact_frame_time_and_content_with_valid_pts_numbering(self):
        for fid, ts in zip(KEPT_IDS, KEPT_TS):
            out = self.call(fid)
            pixels = np.frombuffer(out["jpeg"], np.uint8)
            self.assertTrue((pixels == VALUE_BY_ID[fid]).all(), fid)
            self.assertEqual(out["timestamp_us"], ts)
            h = out["headers"]
            self.assertEqual(h["X-Evidence-Clip-Id"], "clip_a")
            self.assertEqual(h["X-Evidence-Frame-Id"], str(fid))
            self.assertEqual(h["X-Evidence-Timestamp-Us"], str(ts))
            self.assertEqual(h["X-Evidence-Source-Sha256"], self.fx.src_sha)
            self.assertEqual(h["X-Evidence-Observation-Sha256"], self.fx.obs_sha)
            self.assertEqual(h["X-Evidence-Mirrored"], "false")
        self.assertTrue(all(c.closed for c in self.containers))

    def test_only_requested_frame_converted_and_decoding_stops_there(self):
        frames = [FakeFrame(p, v) for p, v in FAKE_SEQUENCE]
        c = FakeContainer(frames)
        rgb = se.decode_frame("ignored", 2, 300_000, opener=lambda _p: c)
        self.assertTrue((rgb == 30).all())
        self.assertEqual([f.converted for f in frames], [0, 0, 0, 0, 1, 0, 0])
        self.assertTrue(c.closed)

    def test_non_monotonic_frame_is_numbered_but_refused(self):
        c = FakeContainer([FakeFrame(p, v) for p, v in FAKE_SEQUENCE])
        with self.assertRaises(se.EvidenceError) as cm:
            se.decode_frame("ignored", 3, 200_000, opener=lambda _p: c)
        self.assertEqual(cm.exception.reason, "FRAME_DROPPED_AS_NON_MONOTONIC")
        self.assertTrue(c.closed)
        # id 3 is not in the observation, so the route refuses before decoding at all
        n = len(self.containers)
        self.refused(404, "FRAME_NOT_IN_OBSERVATION", frame_id=3)
        self.assertEqual(len(self.containers), n)

    def test_timestamp_mismatch_and_short_stream_never_substitute(self):
        # the observation says id 2 is 300000us; this source puts it at 200000us
        shifted = [(None, 1), (0, 10), (1, 20), (2, 30), (4, 40)]
        self.refused(409, "TIMESTAMP_MISMATCH", frame_id=2, opener=self.opener(shifted))
        self.refused(404, "FRAME_NOT_DECODED", frame_id=4, opener=self.opener(FAKE_SEQUENCE[:3]))
        self.assertTrue(all(c.closed for c in self.containers))

    def test_decoder_failure_closes_container_and_hides_detail(self):
        class Boom(FakeFrame):
            def to_ndarray(self, format):   # noqa: A002
                raise RuntimeError("/secret/path/clip.mp4 broke")
        c = FakeContainer([Boom(0, 1)])
        with self.assertRaises(se.EvidenceError) as cm:
            self.call(0, opener=lambda _p: c)
        self.assertEqual(cm.exception.reason, "DECODE_FAILED")
        self.assertNotIn("secret", str(cm.exception))
        self.assertTrue(c.closed)

    # ------------------------------------------------------------- hashes
    def test_hash_mismatches_requested_registered_actual(self):
        other = "f" * 64
        self.refused(409, "REQUEST_HASH_DIFFERS_FROM_REGISTERED", frame_id=0, src=other)
        self.refused(409, "REQUEST_HASH_DIFFERS_FROM_REGISTERED", frame_id=0, obs=other)
        reg = dict(self.fx.clip, sha256=other)                # registered differs from actual
        self.refused(409, "SOURCE_HASH_MISMATCH", frame_id=0, clips=[reg], src=other)
        reg = dict(self.fx.clip, observation_sha256=other)
        self.refused(409, "OBSERVATION_HASH_MISMATCH", frame_id=0, clips=[reg], obs=other)
        with open(self.fx.src, "ab") as fh:                    # actual file changed afterwards
            fh.write(b"x")
        self.refused(409, "SOURCE_HASH_MISMATCH", frame_id=0)
        self.assertEqual(self.containers, [])

    def test_sidecar_must_agree(self):
        for change in ({"clip_id": "clip_b"}, {"source_sha256": "0" * 64},
                       {"observation_sha256": "0" * 64}, {"mirrored": True}):
            fx = Fixture(self.root)
            self.fx = fx
            fx.edit_sidecar(**change)                          # sidecar edits leave the NPZ hash
            self.refused(409, "OBSERVATION_SIDECAR_MISMATCH", frame_id=0)
        os.remove(self.fx.sidecar_path())
        self.refused(409, "OBSERVATION_SIDECAR_MISSING", frame_id=0)

    def test_invalid_requests(self):
        for bad in ("", "ABC", "a" * 63, "A" * 64, None, "g" * 64):
            self.refused(400, "INVALID_HASH", frame_id=0, src=bad if bad is not None else 5)
        for fid in (-1, True, 1.0, "1", None):
            self.refused(400, "INVALID_FRAME_ID", frame_id=fid)
        self.refused(400, "INVALID_CLIP_ID", frame_id=0, clip_id="")
        self.refused(404, "NO_SUCH_CLIP", frame_id=0, clip_id="clip_zzz")
        self.refused(409, "CLIP_ID_NOT_UNIQUE", frame_id=0, clips=[self.fx.clip, dict(self.fx.clip)])
        self.refused(404, "FRAME_NOT_IN_OBSERVATION", frame_id=99)
        no_hash = {k: v for k, v in self.fx.clip.items() if k != "observation_sha256"}
        self.refused(409, "CLIP_HAS_NO_REGISTERED_HASHES", frame_id=0, clips=[no_hash])
        self.assertEqual(self.containers, [])

    def test_not_a_file_and_missing_file(self):
        reg = dict(self.fx.clip, absolute_path=self.root)        # a directory
        self.refused(404, "REGISTERED_FILE_MISSING", frame_id=0, clips=[reg])
        reg = dict(self.fx.clip, observation_npz="obs")
        self.refused(404, "REGISTERED_FILE_MISSING", frame_id=0, clips=[reg])
        reg = dict(self.fx.clip, absolute_path=os.path.join(self.root, "gone.mp4"))
        self.refused(404, "REGISTERED_FILE_MISSING", frame_id=0, clips=[reg])

    def test_duplicate_registered_frame_ids_refused(self):
        self.fx = Fixture(self.root, clip_id="clip_d", ids=[0, 1, 1], ts=[0, 100_000, 200_000])
        self.refused(409, "DUPLICATE_REGISTERED_FRAME_IDS", frame_id=0, clip_id="clip_d")

    def test_no_camera_landmarker_or_capture_and_files_unchanged(self):
        before = {p: (sha(p), os.stat(p).st_mtime_ns) for p in
                  (self.fx.src, self.fx.npz, self.fx.sidecar_path())}
        guards = {name: RaisingModule(name) for name in
                  ("wis_lab.capture.landmarker", "wis_lab.service.camera", "mediapipe")}
        patches = [mock.patch.dict(sys.modules, guards)]
        try:
            import cv2
            patches.append(mock.patch.object(cv2, "VideoCapture",
                                              side_effect=AssertionError("VideoCapture opened")))
        except ImportError:
            pass
        for p in patches:
            p.start()
        try:
            out = self.call(4)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertTrue((np.frombuffer(out["jpeg"], np.uint8) == 40).all())
        after = {p: (sha(p), os.stat(p).st_mtime_ns) for p in before}
        self.assertEqual(before, after)

    def test_replay_source_identity_is_the_registered_input(self):
        ident = se.source_identity(self.fx.clip)
        self.assertEqual((ident["clip_id"], ident["word_id"], ident["source_sha256"],
                          ident["observation_sha256"]),
                         ("clip_a", "word_a", self.fx.src_sha, self.fx.obs_sha))
        self.assertEqual(ident["selected_by"], "observation_id")

    # ------------------------------------------------------------- real PyAV video
    def test_real_pyav_video_exact_frame(self):
        try:
            import av
            import cv2
        except ImportError as exc:
            self.skipTest(f"PyAV/cv2 unavailable: {exc}")
        path = os.path.join(self.root, "clip_real.mp4")      # the Fixture's source path
        levels = [30, 90, 150, 210, 250]
        try:
            with av.open(path, "w") as out:
                st = out.add_stream("mpeg4", rate=10)
                st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
                for v in levels:
                    fr = av.VideoFrame.from_ndarray(np.full((32, 32, 3), v, np.uint8), format="rgb24")
                    for p in st.encode(fr):
                        out.mux(p)
                for p in st.encode():
                    out.mux(p)
        except Exception as exc:                          # noqa: BLE001
            self.skipTest(f"no local encoder: {exc}")
        # timestamps read with the documented rule, independently of the module under test
        ids, ts = [], []
        with av.open(path) as c:
            s = c.streams.video[0]
            tb = s.time_base or Fraction(1, 1000)
            origin, last, n = None, -1, 0
            for pkt in c.demux(s):
                for f in pkt.decode():
                    if f.pts is None:
                        continue
                    origin = f.pts if origin is None else origin
                    t = int(round(float((f.pts - origin) * tb) * 1e6))
                    if t > last:
                        ids.append(n)
                        ts.append(t)
                        last = t
                    n += 1
        self.assertEqual(len(ids), len(levels))
        fx = Fixture(self.root, clip_id="clip_real", source_bytes=None, ids=ids, ts=ts)
        for k, fid in enumerate(ids):
            out = se.reference_frame([fx.clip], self.root, "clip_real", fid, fx.src_sha, fx.obs_sha)
            img = cv2.imdecode(np.frombuffer(out["jpeg"], np.uint8), cv2.IMREAD_COLOR)
            self.assertLess(abs(float(img.mean()) - levels[k]), 12.0, fid)
            self.assertEqual(out["headers"]["X-Evidence-Timestamp-Us"], str(ts[k]))
        with self.assertRaises(se.EvidenceError) as cm:
            se.decode_frame(path, ids[1], ts[1] + 1)
        self.assertEqual(cm.exception.reason, "TIMESTAMP_MISMATCH")


class InspectPathTimestampTests(unittest.TestCase):
    def test_inspect_path_adds_exact_timestamps_and_keeps_existing_fields(self):
        try:
            from wis_lab.core import phase_feedback
            from wis_lab.core.dtw import _SLICES
            from wis_lab.core.features import GROUP_ORDER
        except Exception as exc:                          # noqa: BLE001
            self.skipTest(f"core feature modules unavailable here: {exc}")
        width = max(s.stop if isinstance(s, slice) else int(np.max(s)) + 1
                    for s in _SLICES.values())

        def sample(ids, ts, clip=None, word=None):
            n = len(ids)
            return {"features": np.zeros((n, width), np.float32),
                    "present": np.zeros((n, len(GROUP_ORDER)), bool),
                    "timestamp_us": np.asarray(ts, np.int64), "frame_id": np.asarray(ids, np.int64),
                    "clip_id": clip, "word_id": word}
        q = sample([5, 7, 9], [1_500_000, 1_700_000, 1_900_000])
        r = sample([0, 3], [0, 100_100], "clip_a", "word_a")
        snapshot = {k: v.copy() for k, v in q.items() if isinstance(v, np.ndarray)}
        rep = phase_feedback.inspect_path(q, r, [(0, 0), (1, 0), (2, 1)])
        self.assertEqual(rep["inputs"]["query"]["timestamp_us"], [1_500_000, 1_700_000, 1_900_000])
        self.assertEqual(rep["inputs"]["reference"]["timestamp_us"], [0, 100_100])
        self.assertTrue(all(type(v) is int for v in rep["inputs"]["query"]["timestamp_us"]))
        self.assertEqual(rep["inputs"]["query"]["frame_ids"], [5, 7, 9])
        self.assertEqual(rep["inputs"]["query"]["rows"], 3)
        self.assertEqual([(s["i"], s["j"]) for s in rep["steps"]], [(0, 0), (1, 0), (2, 1)])
        fp = hashlib.sha256()
        for key in ("features", "present", "timestamp_us", "frame_id"):
            arr = np.ascontiguousarray(q[key])
            fp.update(f"{key}:{arr.dtype.str}:{arr.shape}".encode())
            fp.update(arr.tobytes())
        self.assertEqual(rep["inputs"]["query"]["sampled_feature_sha256"], fp.hexdigest())
        for k, v in snapshot.items():
            np.testing.assert_array_equal(q[k], v)
        json.dumps(rep, allow_nan=False)


class RouteTests(unittest.TestCase):
    """Thin transport over the module, with a fake Store. Skipped where the app cannot import."""

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from wis_lab.service import app as app_mod
        except Exception as exc:                          # noqa: BLE001
            self.skipTest(f"service app unavailable here: {exc}")
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(self.tmp.name)
        fake_store = types.SimpleNamespace(manifest={"clips": [self.fx.clip]}, camera=None)
        self.patches = [
            mock.patch.object(app_mod, "store", fake_store),
            mock.patch.object(app_mod, "ROOT", self.tmp.name),
            mock.patch.object(app_mod, "CameraWorker", side_effect=AssertionError("camera")),
            mock.patch.object(se, "_open_container",
                              lambda _p: FakeContainer([FakeFrame(p, v) for p, v in FAKE_SEQUENCE])),
            mock.patch.object(se, "encode_jpeg", lambda rgb: b"\xff\xd8" + rgb.tobytes()),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app_mod.app)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def url(self, fid, clip="clip_a", src=None, obs=None):
        return (f"/api/v1/evidence/references/{clip}/frames/{fid}"
                f"?source_sha256={src or self.fx.src_sha}&observation_sha256={obs or self.fx.obs_sha}")

    def test_route_returns_jpeg_with_identity_headers(self):
        res = self.client.get(self.url(4))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "image/jpeg")
        self.assertEqual(res.headers["x-evidence-frame-id"], "4")
        self.assertEqual(res.headers["x-evidence-timestamp-us"], "400000")
        self.assertEqual(res.headers["x-evidence-mirrored"], "false")
        self.assertEqual(res.headers["x-evidence-source-sha256"], self.fx.src_sha)
        self.assertTrue(res.content.endswith(bytes([40]) * 72))

    def test_route_errors_are_explicit_and_pathless(self):
        cases = [(self.url(3), 404), (self.url(0, clip="nope"), 404),
                 (self.url(0, src="e" * 64), 409), (self.url(-1), 400)]
        for url, code in cases:
            res = self.client.get(url)
            self.assertEqual(res.status_code, code, url)
            self.assertEqual(res.json()["error"], "EVIDENCE_UNAVAILABLE")
            self.assertNotIn(self.tmp.name.replace("\\", "/"), res.text.replace("\\\\", "/"))
        missing = self.client.get("/api/v1/evidence/references/clip_a/frames/0")
        self.assertEqual(missing.status_code, 422)        # both hashes are required


if __name__ == "__main__":
    unittest.main()
