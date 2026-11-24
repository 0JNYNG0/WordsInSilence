"""Optional, separate face observations; does not alter the 75-point TCN input.

Face presence is a model estimate, not per-landmark visibility or contact truth.
Full decoded RGB frames are passed to MediaPipe, before any observation ROI.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..core.observation import sha256_file

SCHEMA = "wis_face_landmarks_sidecar_v1"


class FaceLandmarks:
    def __init__(self, model_path):
        self.engine = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False))
        self.last_ms = -1

    def process(self, rgb, timestamp_us):
        ms = int(timestamp_us // 1000)
        if ms <= self.last_ms:
            raise ValueError("Face VIDEO timestamps must increase by at least 1ms")
        self.last_ms = ms
        result = self.engine.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
        xyz = np.full((478, 3), np.nan, dtype=np.float32)
        if result.face_landmarks:
            if len(result.face_landmarks[0]) != 478:
                raise ValueError("Unexpected face topology")
            xyz[:] = [(p.x, p.y, p.z) for p in result.face_landmarks[0]]
        return xyz

    def close(self):
        self.engine.close()


def extract_face_video(video, model, output):
    """Save every decoded frame, with actual decoder PTS, no inferred missing frames."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {video}")
    engine = FaceLandmarks(model)
    xyz, pts, elapsed = [], [], []
    dimensions = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000))
            if pts and t <= pts[-1]:
                raise ValueError("Decoder did not provide increasing PTS")
            dimensions = [bgr.shape[1], bgr.shape[0]]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tic = perf_counter()
            xyz.append(engine.process(rgb, t))
            elapsed.append(perf_counter() - tic)
            pts.append(t)
    finally:
        engine.close()
        cap.release()
    if not xyz:
        raise ValueError("No decoded frames")
    xyz = np.asarray(xyz, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, schema=np.array(SCHEMA), xyz_normalized=xyz,
                        timestamp_us=np.array(pts, dtype=np.int64),
                        frame_id=np.arange(len(pts), dtype=np.int64),
                        found=np.isfinite(xyz).all(axis=(1, 2)),
                        image_size=np.array(dimensions), inference_seconds=np.array(elapsed))
    meta = dict(schema=SCHEMA, video=str(video), video_sha256=sha256_file(str(video)),
                model=str(model), model_sha256=sha256_file(str(model)),
                output_sha256=sha256_file(str(output)), mediapipe_version=mp.__version__,
                input="full original RGB, every decoded frame, actual PTS",
                identity_recognition=False, expression_scoring=False,
                per_landmark_visibility_available=False,
                z_is_metric_depth=False, options="VIDEO, 1 face, default confidence 0.5")
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def join_face_observation(face_path, observation, pts_tolerance_us=1000):
    with np.load(face_path, allow_pickle=False) as z:
        if str(z['schema']) != SCHEMA:
            raise ValueError("Wrong face schema")
        ids = z['frame_id']
        picks = np.searchsorted(ids, observation.frame_id)
        if (picks >= len(ids)).any() or not np.array_equal(ids[picks], observation.frame_id):
            raise ValueError("Face/body frame-ID mismatch")
        delta = np.abs(z['timestamp_us'][picks] - observation.timestamp_us)
        if delta.max(initial=0) > pts_tolerance_us:
            raise ValueError(f"Face/body PTS mismatch: {delta.max()} us")
        xy = z['xyz_normalized'][picks, :, :2] * z['image_size']
        return dict(xy_px=xy, found=z['found'][picks], frame_id=ids[picks],
                    timestamp_us=z['timestamp_us'][picks], max_pts_delta_us=int(delta.max(initial=0)))


def head_box(pose_xy, usable, width, height):
    """Head ROI from current full-image pose. Does not change body/hand extraction ROI."""
    if not usable[11:13].all():
        return None
    shoulder_width = np.linalg.norm(pose_xy[11]-pose_xy[12])
    visible = usable[:11] & np.isfinite(pose_xy[:11]).all(axis=1)
    if visible.sum() < 3 or not np.isfinite(shoulder_width) or shoulder_width < 16:
        return None
    center = np.median(pose_xy[:11][visible],axis=0)
    lo = np.floor(center - shoulder_width*np.array((.65,.70))).astype(int)
    hi = np.ceil(center + shoulder_width*np.array((.65,.70))).astype(int)
    x0,y0 = np.maximum(lo,(0,0)); x1,y1 = np.minimum(hi,(width,height))
    return (int(x0),int(y0),int(x1),int(y1)) if x1-x0>=32 and y1-y0>=32 else None


def recover_face_video(video, model, original_face, observation, output):
    """Use current head RGB only when the original full-image face result is absent.

    Original full-frame outputs are preserved. ROI coordinates are mapped to original
    image coordinates, including z's normalized-x convention (not metric depth).
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(original_face, allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    if not np.array_equal(data['frame_id'], observation.frame_id):
        raise ValueError('Recovery requires complete matching frame IDs')
    cap=cv2.VideoCapture(str(video));engine=FaceLandmarks(model)
    source=np.where(data['found'],1,0).astype(np.int8)
    boxes=np.full((len(source),4),-1,dtype=np.int32)
    extra_seconds=np.zeros(len(source),dtype=np.float64)
    try:
        for i in range(len(source)):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Recovery decode is shorter than source')
            pts=int(round(cap.get(cv2.CAP_PROP_POS_MSEC)*1000))
            if abs(pts-int(data['timestamp_us'][i]))>1000:raise ValueError('Recovery PTS drift')
            if source[i]==1:continue
            h,w=bgr.shape[:2]
            box=head_box(observation.xy_px[i],observation.usable[i],w,h)
            if box is None:continue
            boxes[i]=box;x0,y0,x1,y1=box
            tic=perf_counter()
            xyz=engine.process(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),pts)
            extra_seconds[i]=perf_counter()-tic
            if np.isfinite(xyz).all():
                xyz[:,0]=(xyz[:,0]*(x1-x0)+x0)/w
                xyz[:,1]=(xyz[:,1]*(y1-y0)+y0)/h
                xyz[:,2]*=(x1-x0)/w
                data['xyz_normalized'][i]=xyz;data['found'][i]=True;source[i]=2
    finally:
        engine.close();cap.release()
    data['inference_seconds']+=extra_seconds
    np.savez_compressed(output,**data,source_kind=source,head_box=boxes,recovery_seconds=extra_seconds)
    meta=dict(schema=SCHEMA, original_sha256=sha256_file(str(original_face)),
              video_sha256=sha256_file(str(video)),model_sha256=sha256_file(str(model)),
              output_sha256=sha256_file(str(output)),
              source_kind={'0':'no result','1':'original full-frame','2':'current pose-guided head ROI'},
              body_hand_observations_modified=False,expression_scoring=False,
              head_roi_rule='current pose face center; +-0.65 shoulder width x, +-0.70 shoulder width y')
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta
