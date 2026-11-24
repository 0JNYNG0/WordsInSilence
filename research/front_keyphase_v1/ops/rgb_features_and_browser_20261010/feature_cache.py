"""Load fixed-encoder RGB features without changing time, handedness or observation provenance."""
from pathlib import Path
import hashlib
import numpy as np
import torch

FIELDS = {'pose_x','frame_id','timestamp_us','features','available','observed_rgb',
          'region_estimated','landmark_observed_count','bbox_xyxy'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(a):
    if set(a) != FIELDS: raise ValueError('Unexpected cache fields (labels/IDs are metadata only)')
    n = len(a['pose_x'])
    if n <= 0 or a['pose_x'].shape != (n,280): raise ValueError('Invalid pose shape')
    for k in ['pose_x','features']:
        if a[k].dtype != np.float32 or not np.isfinite(a[k]).all(): raise ValueError('Invalid '+k)
    if a['features'].shape != (n,2,576): raise ValueError('Invalid feature shape')
    for k in ['frame_id','timestamp_us']:
        if a[k].shape != (n,) or a[k].dtype.kind not in 'iu' or a[k][0] < 0 or not (np.diff(a[k])>0).all():
            raise ValueError('Invalid '+k)
    for k in ['available','observed_rgb','region_estimated']:
        if a[k].shape != (n,2) or a[k].dtype != bool: raise ValueError('Invalid '+k)
    av,ob,est = a['available'],a['observed_rgb'],a['region_estimated']
    if (ob & est).any() or not np.array_equal(av,ob|est): raise ValueError('Invalid provenance')
    if a['features'][~av].any(): raise ValueError('Missing slot contains a feature')
    counts=a['landmark_observed_count']
    if counts.shape != (n,2) or counts.dtype.kind not in 'iu' or not ((counts>=0)&(counts<=21)).all():
        raise ValueError('Invalid observation counts')
    if (counts[est]!=0).any() or (counts[ob]!=21).any(): raise ValueError('Observation provenance mismatch')
    boxes=a['bbox_xyxy']
    if boxes.shape != (n,2,4) or boxes.dtype.kind not in 'iu' or not (boxes[~av]==-1).all(): raise ValueError('Invalid boxes')
    valid=boxes[av]
    if not (((valid[:,2]-valid[:,0])==(valid[:,3]-valid[:,1])) & ((valid[:,2]-valid[:,0])>=2)).all():
        raise ValueError('Invalid box dimensions')
    return n


def load(path, expected_sha256):
    if sha(path) != expected_sha256: raise ValueError('Feature cache hash mismatch')
    with np.load(path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    validate(a)
    return a


def model_inputs(a, arm):
    n=validate(a)
    if arm not in ('A','B','C'): raise ValueError('Unknown arm')
    av=(np.zeros((n,2),bool) if arm=='A' else a['observed_rgb'].copy() if arm=='B' else a['available'].copy())
    f=a['features'].copy(); f[~av]=0
    return {'pose_x':torch.from_numpy(a['pose_x'].copy()), 'pad':torch.ones(n,dtype=torch.float32),
            'features':torch.from_numpy(f), 'available':torch.from_numpy(av),
            'estimated':torch.from_numpy(a['region_estimated'].copy()&av),
            'frame_id':torch.from_numpy(a['frame_id'].copy()),'timestamp_us':torch.from_numpy(a['timestamp_us'].copy())}
