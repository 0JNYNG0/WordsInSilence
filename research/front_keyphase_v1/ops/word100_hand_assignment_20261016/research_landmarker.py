"""Opt-in research extractor. No production import or registration is changed."""
from pathlib import Path
import hashlib
import sys
import numpy as np

U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parents[1]/'src'))
from wis_lab.capture.landmarker import Landmarker
from assignment import HandAssigner


class ResearchLandmarker(Landmarker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.assigner=HandAssigner()
        self.assignment_trace=[]

    def extractor_id(self):
        payload=super().extractor_id()+HandAssigner.identity()+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return 'handassoc2_'+hashlib.sha256(payload.encode()).hexdigest()[:16]

    def reset_assignment_state(self):
        self.assigner.reset()

    def process(self,rgb,timestamp_us,out,row):
        self._original_us=int(timestamp_us)
        self._assigned_this_frame=False
        super().process(rgb,timestamp_us,out,row)
        if not self._assigned_this_frame:
            h,w=rgb.shape[:2]
            _,diag=self.assigner.assign(np.empty((0,21,2)),[],[],out['xy_px'][row,:33],
                out['usable'][row,:33],timestamp_us,w,h)
            self.assignment_trace.append(diag)

    def _assign_sides(self,hand_res,pose_xy,pose_usable,w,h):
        self._assigned_this_frame=True
        hands=np.array([[(p.x*w,p.y*h) for p in pts] for pts in hand_res.hand_landmarks],dtype=np.float64)
        labels=[{'Left':0,'Right':1}.get(v[0].category_name,-1) for v in hand_res.handedness]
        scores=[float(v[0].score) for v in hand_res.handedness]
        sides,diag=self.assigner.assign(hands,labels,scores,pose_xy,pose_usable,self._original_us,w,h)
        self.assignment_trace.append(diag)
        for i,s in sides.items():
            if s is not None:
                self.handedness_comparisons+=1
                self.handedness_disagreements+=int(labels[i]!=s)
        return {i:(s,scores[i]) for i,s in sides.items()}
