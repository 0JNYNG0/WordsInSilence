"""Research-only angle evidence after DTW; never edits coordinates or judges signs."""
import numpy as np

MIN_FRAMES = 8
MIN_SUPPORT = .5
MIN_REFS = 5
SIDES = ("L", "R")
NAMES = tuple(f"{f}{j}" for f in ("엄지", "검지", "중지", "약지", "소지") for j in (1, 2, 3))


def cells(ts):
    t = np.asarray(ts, float) / 1e6
    if len(t) < 2:
        return np.ones(len(t))
    return np.diff(np.r_[t[0], (t[:-1] + t[1:]) / 2, t[-1]])


def intervals(ts, mask):
    """Exact saved-frame cell intervals, seconds from the original clip start."""
    t = (np.asarray(ts, float) - ts[0]) / 1e6
    edge = np.r_[t[0], (t[:-1] + t[1:]) / 2, t[-1]]
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [dict(start_s=float(edge[g[0]]), end_s=float(edge[g[-1] + 1]),
                 first_row=int(g[0]), last_row=int(g[-1]), frames=len(g)) for g in groups]


def pair(q, r, path, required, alignment_ok):
    """Fixed reference core denominator; each reference frame has equal time weight.

    Raw-only differences remain diagnostic. Strict differences require state4 at
    both ends. Repeated DTW path entries do not inflate time coverage.
    """
    path = np.asarray(path, int).reshape(-1, 2)
    qi, ri = path.T
    expected = np.broadcast_to(r["core"][:, None, None], r["angle_state"].shape).copy()
    for h, side in enumerate(SIDES):
        if side not in required:
            expected[:, h] = False
    e = expected[ri]
    raw = e & np.isfinite(q["angle_deg"][qi]) & np.isfinite(r["angle_deg"][ri])
    strict = raw & (q["angle_state"][qi] == 4) & (r["angle_state"][ri] == 4)
    delta = np.abs(q["angle_deg"][qi] - r["angle_deg"][ri])
    w = cells(r["timestamp_us"])
    rows = []
    for h, side in enumerate(SIDES):
        for a, name in enumerate(NAMES):
            row = dict(side=side, angle=a, name=name, required=side in required,
                       expected_frames=int(expected[:, h, a].sum()),
                       expected_seconds=float(w @ expected[:, h, a]), alignment_ok=bool(alignment_ok))
            for label, mask in (("raw", raw), ("strict", strict)):
                good = mask[:, h, a]
                ref_ids = np.unique(ri[good])
                n_q = len(np.unique(qi[good]))
                coverage = float(w[ref_ids].sum() / row["expected_seconds"]) if row["expected_seconds"] else None
                by_ref = [float(delta[good & (ri == j), h, a].mean()) for j in ref_ids]
                mean = float(np.average(by_ref, weights=w[ref_ids])) if len(ref_ids) and w[ref_ids].sum() else None
                ready = bool(side in required and alignment_ok and coverage is not None
                             and coverage >= MIN_SUPPORT and min(n_q, len(ref_ids)) >= MIN_FRAMES)
                row[label] = dict(mean_deg=mean, coverage=coverage, query_frames=n_q,
                                  reference_frames=len(ref_ids), ready=ready)
            rows.append(row)
    return dict(rows=rows, expected=expected, raw=raw, strict=strict,
                difference_deg=np.where(strict, delta, np.nan), path=path)


def aggregate(q, pairs, required, people):
    """Return every angle plus per-query-frame reference votes and missing intervals.

    The query annotation core is a PUBLIC OFFLINE audit anchor. Runtime has no
    such oracle. No signer is repeated and query signer must be excluded by caller.
    """
    if len(people) != len(set(people)):
        raise ValueError("Duplicate reference person")
    n = len(q["timestamp_us"])
    votes = np.zeros((n, 2, 15), np.int16)
    expected_votes = np.zeros_like(votes)
    for p in pairs:
        qi, ri = p["path"].T
        one = np.zeros_like(votes, bool)
        ex = np.zeros_like(votes, bool)
        if p["rows"][0]["alignment_ok"]:
            np.logical_or.at(one, qi, p["strict"])
        np.logical_or.at(ex, qi, p["expected"][ri])
        votes += one
        expected_votes += ex
    rows = []
    for h, side in enumerate(SIDES):
        for a, name in enumerate(NAMES):
            rr = [p["rows"][h * 15 + a] for p in pairs]
            raw_values = [r["raw"]["mean_deg"] for r in rr if r["raw"]["ready"]]
            strict_values = [r["strict"]["mean_deg"] for r in rr if r["strict"]["ready"]]
            enough = len(strict_values) >= MIN_REFS
            optional = side not in required
            core = q["core"] if not optional else np.zeros(n, bool)
            enough_here = (votes[:, h, a] >= MIN_REFS) & core
            gap = core & ~enough_here
            reasons = {
                "query_missing": core & (q["angle_state"][:, h, a] == 0),
                "query_degenerate": core & (q["angle_state"][:, h, a] == 1),
                "query_unassessed": core & (q["angle_state"][:, h, a] == 2),
                "query_conflict": core & (q["angle_state"][:, h, a] == 3),
                "reference_or_alignment_insufficient": gap & (q["angle_state"][:, h, a] == 4),
            }
            status = ("NOT_REQUIRED" if optional else "INSUFFICIENT" if not enough else
                      "PARTIAL" if gap.any() else "OBSERVED")
            rows.append(dict(side=side, angle=a, name=name, status=status,
                raw_only_median_deg=float(np.median(raw_values)) if len(raw_values) >= MIN_REFS else None,
                compared_median_deg=float(np.median(strict_values)) if enough and not optional else None,
                strict_reference_count=len(strict_values), raw_reference_count=len(raw_values),
                reference_people=people, actual_compared_reference_people=[
                    person for person, r in zip(people, rr) if r["strict"]["ready"]],
                core_frames=int(core.sum()), five_reference_frames=int(enough_here.sum()),
                missing_intervals=intervals(q["timestamp_us"], gap),
                reasons={k: intervals(q["timestamp_us"], v) for k, v in reasons.items()},
                interpretation="observed portions only; not correctness, not percent match",
                accepted=None, sign_error=None))
    return dict(rows=rows, votes=votes, expected_votes=expected_votes,
                accepted=None, sign_error=None, whole_hand_score=None)
