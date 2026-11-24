"""Observation support is separate from a classifier's generated probability.

This module cannot return 'correct sign' or 'wrong sign'. It never shrinks an
expected comparison interval after a query hand disappears.
"""
import numpy as np
def shape_support(angles_mask):
 m=np.asarray(angles_mask,bool)
 assert m.shape[-1]==14
 count=m.sum(-1)
 # Thumb2 + index/middle/ring/little3 each. Both physical targets require
 # evidence about all four non-thumb fingers. A generic10/14 count could pass
 # when every index angle is missing, so it is only diagnostic, never a gate.
 required=m[...,2:14].all(-1)
 return dict(observed_angle_count=count,expected_angle_count=np.full(count.shape,14),required_non_thumb_angle_count=m[...,2:14].sum(-1),comparison_supported=required,scope='All required projected angle coordinates exist; not a certificate of 3D pose correctness or actual pixel visibility')
def interval_support(wrist_mask,velocity_mask):
 w=np.asarray(wrist_mask,bool);v=np.asarray(velocity_mask,bool)
 assert w.shape==v.shape and w.ndim==2 and w.shape[1]==2 and len(w)>0
 known=w.all(-1);continuous=v.all(-1);continuous[0]=True
 valid=known&continuous
 return dict(expected_frames=len(w),observed_frames=int(valid.sum()),observation_ratio=float(valid.mean()),comparison_supported=bool(valid.all()),status='관측 근거 있음' if valid.all() else '비교 근거 부족',educational_verdict=None)
def selftest():
 assert not shape_support(np.zeros((5,14),bool))['comparison_supported'].any()
 assert shape_support(np.ones((5,14),bool))['comparison_supported'].all()
 missing_index=np.ones((5,14),bool);missing_index[:,2:5]=False
 assert not shape_support(missing_index)['comparison_supported'].any()
 w=np.ones((19,2),bool);v=w.copy();full=interval_support(w,v)
 w[6:8]=False;v[6:9]=False;miss=interval_support(w,v)
 assert full['expected_frames']==miss['expected_frames']==19 and not miss['comparison_supported']
 assert miss['educational_verdict'] is None and miss['observed_frames']<19
 assert interval_support(np.zeros((19,2),bool),np.zeros((19,2),bool))['observed_frames']==0
 return dict(all_missing_not_pass=True,missing_index_not_pass=True,interval_denominator_unchanged=True,unobserved_not_educational_error=True)
