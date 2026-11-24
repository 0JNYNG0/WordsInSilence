"""Pure, word-independent image-plane orientation factorization for hand63.

Input is already wrist-centered/hand-scaled and anatomically reflected by the
existing data preparation (left x reflection happens ONCE, outside this helper).
This transform removes image-plane rotation only; it cannot reconstruct 3D
articulation, recover hidden landmarks, or make inaccurate points correct.
No model, label, source ID, frame time or expected gesture is consulted.
"""
import numpy as np

# Numerical degeneracy threshold only, in the existing hand-scale coordinate
# units. This is not a learned angle, pose or feedback acceptance threshold.
BASIS_EPSILON = 1e-6

def _validate(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.shape[1] != 63:
        raise ValueError('Expected [frames,63]:42local coordinates +21availability values')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite input is malformed; do not silently impute coordinates')
    if ((a[:,42:]<0)|(a[:,42:]>1)).any():
        raise ValueError('Availability values must be in [0,1]')
    return a

def rotate_hand63(x, degrees):
    """Rigid image-plane counterfactual, not a new normal KSL recording.

    One constant angle per supplied window or one explicitly supplied angle per
    frame. Missing points stay zero. Distances, scale and masks are preserved.
    """
    a = _validate(x)
    p = a[:,:42].reshape(-1,21,2).astype(np.float64)
    theta = np.asarray(degrees,dtype=np.float64)
    if theta.ndim == 0:
        theta = np.full(len(a),float(theta))
    if theta.shape != (len(a),) or not np.isfinite(theta).all():
        raise ValueError('Angle must be a finite scalar or exactly one angle per frame')
    theta = np.deg2rad(theta)
    c,s = np.cos(theta),np.sin(theta)
    q = np.empty_like(p)
    q[:,:,0] = p[:,:,0]*c[:,None]-p[:,:,1]*s[:,None]
    q[:,:,1] = p[:,:,0]*s[:,None]+p[:,:,1]*c[:,None]
    q = np.where((a[:,42:]> .5)[:,:,None],q,0.)
    return np.c_[q.reshape(-1,42),a[:,42:]].astype(np.float32)

def canonicalize_hand63(x):
    """Rotate wrist->middle MCP9 to upward y, without changing hand scale.

    Returns (features63, basis_valid, audit). An unavailable or degenerate basis
    produces zero coordinates and basis_valid=False, never an invented hand.
    Original21availability values are retained exactly for audit; consumers MUST
    separately apply basis_valid before loss or evidence emission. Orientation
    outputs for other tasks must use the original stream, not these coordinates.
    """
    a = _validate(x)
    p = a[:,:42].reshape(-1,21,2).astype(np.float64)
    masks = a[:,42:] > .5
    v = p[:,9]-p[:,0]
    norm = np.linalg.norm(v,axis=1)
    valid = masks[:,0]&masks[:,9]&(norm>BASIS_EPSILON)
    unit = np.zeros_like(v)
    unit[valid] = v[valid]/norm[valid,None]
    # right=(-uy,ux); up=(ux,uy). Standard image x-right,y-down means
    # canonical y is -dot(point,up). Determinant+1: this is a rotation,
    # not a second anatomical reflection.
    delta = p-p[:,0:1]
    q = np.zeros_like(p)
    q[:,:,0] = -delta[:,:,0]*unit[:,1,None]+delta[:,:,1]*unit[:,0,None]
    q[:,:,1] = -delta[:,:,0]*unit[:,0,None]-delta[:,:,1]*unit[:,1,None]
    q = np.where((masks&valid[:,None])[:,:,None],q,0.)
    out = np.c_[q.reshape(-1,42),a[:,42:]].astype(np.float32)
    return out,valid,dict(basis_length=norm.astype(np.float32),
                          original_basis_unit=unit.astype(np.float32),
                          numerical_epsilon=BASIS_EPSILON,
                          anatomical_reflection_applied=False,
                          scale_changed=False)
