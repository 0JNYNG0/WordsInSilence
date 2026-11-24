"""Memory-only capture aid, unrelated to feedback/model input."""
from pathlib import Path
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions,vision
ROOT=Path(__file__).resolve().parents[4]
class Preview:
 def __init__(self):
  self.pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=str(ROOT/'Tools/models/pose_landmarker_lite.task')),
   running_mode=vision.RunningMode.IMAGE,num_poses=1))
  self.hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
   base_options=BaseOptions(model_asset_path=str(ROOT/'Tools/models/hand_landmarker.task')),
   running_mode=vision.RunningMode.IMAGE,num_hands=2))
 def process(self,body):
  im=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
  if im is None or max(im.shape[:2])>800:raise ValueError('Bounded JPEG preview required')
  rgb=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  p=self.pose.detect(rgb);h=self.hand.detect(rgb)
  pts=p.pose_landmarks[0] if p.pose_landmarks else []
  def pt(i):
   if not pts or min(pts[i].visibility,pts[i].presence)<.5:return None
   return [float(pts[i].x),float(pts[i].y)]
  return dict(shoulders=[pt(11),pt(12)],elbows=[pt(13),pt(14)],wrists=[pt(15),pt(16)],
    hands=[[float(v[0].x),float(v[0].y)] for v in h.hand_landmarks],
    source='preview_only_not_feedback_input',saved=False)
 def close(self):self.pose.close();self.hand.close()
