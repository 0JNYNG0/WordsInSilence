"""Research candidate. No class name, label, signer or query visibility rules."""
import numpy as np


def existing_correspondence(full, old, sides, pose, usable, w, h, merge):
    """One-to-one same-observation corroboration, NOT new anatomical identity.

    Current arm contradiction vetoes; absent arm evidence cannot veto an otherwise
    unique physical match. All raw candidates (including unassigned) compete.
    Coordinates, existing side assignments, and new-hand admission stay unchanged.
    """
    good, scale = merge.context(pose, usable, w, h)
    result = []
    if scale <= 1:
        return result
    oa = [merge.arm(p, pose, good, scale) for p in old]
    fa = [merge.arm(np.asarray(d["xy"]), pose, good, scale) for d in full]
    matches = np.zeros((len(full), len(old)), bool)
    ambiguous = np.zeros_like(matches)
    for i, d in enumerate(full):
        p = np.asarray(d["xy"])
        if not merge.inside(p[merge.PALM], w, h).all():
            continue
        for j, b in enumerate(old):
            # Geometry match is independent of handedness label and current arm.
            rel = merge.relation(p, b, -1, -1, scale)
            matches[i, j] = rel == "same_geometry"
            ambiguous[i, j] = rel == "ambiguous_overlap"
    for i, j in zip(*np.nonzero(matches)):
        side = int(sides[j])
        reason = "unique_existing_geometry"
        if side not in (0, 1):
            reason = "old_identity_unassigned"
        elif matches[i].sum() != 1 or matches[:, j].sum() != 1 or ambiguous[i].any():
            reason = "ambiguous_geometry"
        elif sum(s == side for s in sides) != 1:
            reason = "duplicate_old_side"
        elif (fa[i] >= 0 and fa[i] != side) or (oa[j] >= 0 and oa[j] != side):
            reason = "current_arm_identity_conflict"
        result.append(dict(full_index=int(i), old_index=int(j), side=side,
                           accepted=reason == "unique_existing_geometry", reason=reason,
                           current_arm=fa[i], old_arm=oa[j],
                           identity_newly_verified=False))
    return result


def reference_schedule(target, peers, paths):
    """Positive, multi-person rest only. Unknown never means resting.

    Each reference target contributes one vote. Other reference videos contribute
    one only if ALL mapped frames are stable lowered frames inside their core.
    Any visible in-ROI hand point in any reference vetoes rest. Query is absent.
    """
    core = target["core"]
    rest_votes = (target["rest"] & core[:, None]).astype(np.int16)
    active_votes = (target["active"] & core[:, None]).astype(np.int16)
    for peer, path in zip(peers, paths):
        qi, ri = path.T
        for i in np.unique(qi):
            js = ri[qi == i]
            for side in (0, 1):
                rest_votes[i, side] += bool(peer["core"][js].all() and peer["rest"][js, side].all())
                active_votes[i, side] += bool((peer["core"][js] & peer["active"][js, side]).any())
    rest = (core[:, None] & target["rest"] & (rest_votes >= 5) & (active_votes == 0))
    return dict(rest=rest, expected=core[:, None] & ~rest,
                rest_votes=rest_votes, active_votes=active_votes)


def pair(q, r, path, required, alignment_ok, expected_hand, adapter):
    """Identical saved DTW path; fixed reference schedule is separate from evidence."""
    qi, ri = np.asarray(path, int).T
    expected = np.broadcast_to(expected_hand[:, :, None], r["angle_state"].shape).copy()
    for h, side in enumerate(adapter.SIDES):
        if side not in required:
            expected[:, h] = False
    raw = expected[ri] & np.isfinite(q["angle_deg"][qi]) & np.isfinite(r["angle_deg"][ri])
    strict = raw & (q["angle_state"][qi] == 4) & (r["angle_state"][ri] == 4)
    delta = np.abs(q["angle_deg"][qi] - r["angle_deg"][ri])
    w = adapter.cells(r["timestamp_us"])
    rows = []
    for h, side in enumerate(adapter.SIDES):
        for a, name in enumerate(adapter.NAMES):
            denominator = float(w @ expected[:, h, a])
            row = dict(side=side, angle=a, name=name, required=side in required,
                       expected_frames=int(expected[:, h, a].sum()),
                       expected_seconds=denominator, alignment_ok=bool(alignment_ok))
            for label, mask in (("raw", raw), ("strict", strict)):
                ok = mask[:, h, a]
                refs = np.unique(ri[ok]); nq = len(np.unique(qi[ok]))
                coverage = float(w[refs].sum() / denominator) if denominator else None
                values = [float(delta[ok & (ri == j), h, a].mean()) for j in refs]
                mean = float(np.average(values, weights=w[refs])) if len(refs) and w[refs].sum() else None
                ready = bool(side in required and alignment_ok and coverage is not None
                             and coverage >= adapter.MIN_SUPPORT and min(nq, len(refs)) >= adapter.MIN_FRAMES)
                row[label] = dict(mean_deg=mean, coverage=coverage, query_frames=nq,
                                  reference_frames=len(refs), ready=ready)
            rows.append(row)
    return dict(rows=rows, expected=expected, raw=raw, strict=strict,
                difference_deg=np.where(strict, delta, np.nan), path=np.asarray(path, int))


def summarize(pairs, required, adapter):
    """Reference-grid coverage; do not promise that a projected query time is fixed."""
    rows = []
    for h, side in enumerate(adapter.SIDES):
        for a, name in enumerate(adapter.NAMES):
            rr = [p["rows"][h * 15 + a] for p in pairs]
            valid = [r["strict"]["mean_deg"] for r in rr if r["strict"]["ready"]]
            rows.append(dict(side=side, angle=a, name=name,
                status="NOT_REQUIRED" if side not in required else
                       "PARTIAL_COMPARISON" if len(valid) >= adapter.MIN_REFS else "INSUFFICIENT",
                compared_median_deg=float(np.median(valid)) if len(valid) >= adapter.MIN_REFS else None,
                reference_count=len(valid), accepted=None, wrong_action=None))
    return rows
