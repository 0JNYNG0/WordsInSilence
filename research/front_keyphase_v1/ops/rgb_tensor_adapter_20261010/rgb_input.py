"""Offline RGB224 tensor input; no model, network, camera or label inputs."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def crop_original(frame_rgb, box):
    require(frame_rgb.dtype == np.uint8 and frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3,
            'Expected uint8 RGB H,W,3')
    b = np.asarray(box)
    require(b.shape == (4,) and b.dtype.kind in 'iu', 'Box must contain four integers')
    x0, y0, x1, y1 = map(int, b)
    h, w = frame_rgb.shape[:2]
    side = x1-x0
    require(side == y1-y0 and 2 <= side <= 3*max(h, w), 'Invalid square crop')
    lx, ty, rx, by = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    require(rx > lx and by > ty, 'Crop outside image')
    canvas = np.zeros((side, side, 3), np.uint8)
    valid = np.zeros((side, side), np.uint8)
    canvas[ty-y0:by-y0, lx-x0:rx-x0] = frame_rgb[ty:by, lx:rx]
    valid[ty-y0:by-y0, lx-x0:rx-x0] = 255
    image = np.array(Image.fromarray(canvas).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    mask = np.array(Image.fromarray(valid).resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0
    return image, mask


def validate(a):
    x = a['pose_x']
    n = len(x)
    require(n > 0 and x.shape == (n, 280) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid pose')
    for k in ('frame_id', 'timestamp_us'):
        require(a[k].shape == (n,) and a[k].dtype.kind in 'iu' and (np.diff(a[k]) > 0).all(), 'Invalid '+k)
    require(a['frame_id'][0] >= 0 and a['timestamp_us'][0] >= 0, 'Negative frame/time')
    require(a['rgb'].shape == (n, 2, SIZE, SIZE, 3) and a['rgb'].dtype == np.uint8, 'Invalid RGB')
    for k in ('available', 'observed_rgb', 'region_estimated'):
        require(a[k].shape == (n, 2) and a[k].dtype == bool, 'Invalid '+k)
    av, observed, estimated = a['available'], a['observed_rgb'], a['region_estimated']
    require(not (observed & estimated).any() and np.array_equal(av, observed | estimated), 'Invalid provenance')
    require(a['pixel_valid'].shape == (n, 2, SIZE, SIZE) and a['pixel_valid'].dtype == bool, 'Invalid pixel mask')
    require(not a['rgb'][~av].any() and not a['pixel_valid'][~av].any(), 'Missing slot contains pixels')
    require(a['landmark_observed_count'].shape == (n, 2) and a['landmark_observed_count'].dtype.kind in 'iu', 'Invalid landmark counts')
    require(((a['landmark_observed_count'] >= 0) & (a['landmark_observed_count'] <= 21)).all(), 'Invalid landmark count range')
    require((a['landmark_observed_count'][estimated] == 0).all(), 'Estimated region is not a detected hand')
    require((a['landmark_observed_count'][observed] == 21).all(), 'Observed source must have21 points')
    require(a['bbox_xyxy'].shape == (n, 2, 4) and a['bbox_xyxy'].dtype.kind in 'iu', 'Invalid boxes')
    boxes = a['bbox_xyxy'][av]
    require(((boxes[:, 2]-boxes[:, 0]) == (boxes[:, 3]-boxes[:, 1])).all()
            and ((boxes[:, 2]-boxes[:, 0]) >= 2).all(), 'Invalid available boxes')
    require((a['bbox_xyxy'][~av] == -1).all(), 'Missing box must be sentinel')
    return n


def load_cache(path, expected_sha256):
    require(sha(path) == expected_sha256, 'Input hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    validate(a)
    return a


def pack(a, arm='C'):
    """Only available images become encoder inputs. Model labels never enter here."""
    n = validate(a)
    require(arm in ('A', 'B', 'C'), 'Unknown arm')
    available = (np.zeros((n, 2), bool) if arm == 'A' else
                 a['observed_rgb'].copy() if arm == 'B' else a['available'].copy())
    slots = np.argwhere(available).astype(np.int64)
    images = torch.from_numpy(np.ascontiguousarray(a['rgb'][available])).permute(0, 3, 1, 2).float()/255.0
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    images = (images-mean)/std
    return {
        'pose_x': torch.from_numpy(a['pose_x'].copy()),
        'frame_id': torch.from_numpy(a['frame_id'].astype(np.int64, copy=True)),
        'timestamp_us': torch.from_numpy(a['timestamp_us'].astype(np.int64, copy=True)),
        'images': images.contiguous(), 'slots': torch.from_numpy(slots),
        'available': torch.from_numpy(available),
        'pixel_valid': torch.from_numpy(a['pixel_valid'][available].copy()),
        'region_estimated': torch.from_numpy(a['region_estimated'].copy() & available),
        'landmark_observed_count': torch.from_numpy(a['landmark_observed_count'].astype(np.int64, copy=True)),
    }


def scatter_features(features, slots, frames):
    """Scatter external encoder outputs without filling absent hand/time slots."""
    require(isinstance(frames, int) and frames > 0, 'Invalid frame count')
    require(features.ndim == 2 and features.is_floating_point() and torch.isfinite(features).all().item(), 'Invalid features')
    require(slots.ndim == 2 and slots.shape == (len(features), 2) and slots.dtype == torch.int64, 'Invalid slots')
    require(slots.device == features.device, 'Slot/feature device mismatch')
    require(bool(((slots[:, 0] >= 0) & (slots[:, 0] < frames) & (slots[:, 1] >= 0) & (slots[:, 1] < 2)).all()), 'Out-of-range slots')
    require(len(torch.unique(slots, dim=0)) == len(slots), 'Duplicate slots')
    out = features.new_zeros((frames, 2, features.shape[1]))
    out[slots[:, 0], slots[:, 1]] = features
    return out
