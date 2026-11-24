"""Exact structural RGB time support, without encoder/model/media calls.

Support means pixels presented to an encoder window with a positive interpolation
coefficient. It is not an attention map, per-pixel causal contribution or evidence
that a physical process happened. All coordinates below are cache-row indices.
"""
from bisect import bisect_left
from fractions import Fraction

WINDOW = 16
MAX_TIMESTAMP_US = 20_000_000


def _integer_axis(values, name, increasing=True):
    out = list(values)
    if any(isinstance(v, bool) or int(v) != v for v in out):
        raise ValueError(name + ' must contain integers')
    out = [int(v) for v in out]
    if any(v < 0 for v in out) or (increasing and any(b <= a for a,b in zip(out,out[1:]))):
        raise ValueError(name + ' must be nonnegative and strictly increasing')
    return out


def window_starts(n):
    """Same long-input rule as frozen SignRep common.starts; no imports of it."""
    if n < WINDOW:
        raise ValueError('Long-input starts require at least16 actual rows')
    starts = list(range(0, n-15, 2))
    if starts[-1] != n-16:
        starts.append(n-16)
    return starts


def interpolation_terms(timestamp_us, center_twice_us):
    """[window_index, numerator, denominator], excluding EXACT zero coefficients.

    Integer/half-integer centers below20e6 are exactly represented by float64 in
    the frozen np.interp route. Integer arithmetic avoids a tolerance-based halo.
    Fractions describe coefficient dependence, not uniform weights on RGB pixels.
    """
    if not center_twice_us:
        return []
    z = 2*int(timestamp_us)
    if z <= center_twice_us[0]:
        return [[0,1,1]]
    if z >= center_twice_us[-1]:
        return [[len(center_twice_us)-1,1,1]]
    right = bisect_left(center_twice_us,z)
    if center_twice_us[right] == z:
        return [[right,1,1]]
    left = right-1
    width = center_twice_us[right]-center_twice_us[left]
    terms = []
    for w,num in ((left,center_twice_us[right]-z),(right,z-center_twice_us[left])):
        f = Fraction(num,width)
        if f > 0:
            terms.append([w,f.numerator,f.denominator])
    assert sum(Fraction(n,d) for _,n,d in terms) == 1
    return terms


def build_rgb_support(timestamp_us, image_available, frame_ids=None,
                      source_frame_ids=None, source_timestamp_us=None, rgb_finite=None):
    ts = _integer_axis(timestamp_us,'timestamp_us')
    n = len(ts)
    if ts and ts[-1] > MAX_TIMESTAMP_US:
        raise ValueError('Timestamp outside the frozen bounded source contract')
    ids = _integer_axis(range(n) if frame_ids is None else frame_ids,'frame_ids')
    sf = _integer_axis(ids if source_frame_ids is None else source_frame_ids,'source_frame_ids')
    st = _integer_axis(ts if source_timestamp_us is None else source_timestamp_us,'source_timestamp_us')
    av = [bool(x) for x in image_available]
    finite = [True]*n if rgb_finite is None else [bool(x) for x in rgb_finite]
    if not all(len(v)==n for v in (ids,sf,st,av,finite)):
        raise ValueError('Frame/time/availability axis lengths disagree')
    if n < 2:
        windows=[];centers=[];weights=[[] for _ in ts];status='UNSUPPORTED_N_LT2'
    elif n < WINDOW:
        windows=[dict(slot_rows=list(range(n))+[n-1]*(WINDOW-n),
                      real_slot_mask=[True]*n+[False]*(WINDOW-n))]
        centers=[];weights=[[[0,1,1]] for _ in ts];status='SHORT_ONE_PADDED_WINDOW_BROADCAST'
    else:
        starts=window_starts(n)
        windows=[dict(slot_rows=list(range(s,s+WINDOW)),real_slot_mask=[True]*WINDOW) for s in starts]
        centers=[ts[s]+ts[s+WINDOW-1] for s in starts]
        weights=[interpolation_terms(t,centers) for t in ts];status='LONG_CENTER_LINEAR_INTERPOLATION'
    raw=[];effective=[]
    for i,terms in enumerate(weights):
        support=sorted({j for w,num,den in terms if num>0 for j in windows[w]['slot_rows']})
        raw.append(support)
        # Mirrors process_head.make_input: unavailable or nonfinite RGB row is
        # masked before the process model. Neighboring available rows can still
        # depend on blackout pixels encoded in the same16-frame window.
        effective.append(support if av[i] and finite[i] else [])
    return dict(schema='exact_structural_rgb_support_v1',status=status,rows=n,
        frame_ids=ids,timestamp_us=ts,source_frame_ids=sf,source_timestamp_us=st,
        image_available=av,rgb_finite=finite,windows=windows,center_twice_us=centers,
        row_window_terms=weights,row_encoded_support=raw,row_consumed_support=effective,
        padding_slots=sum(not b for w in windows for b in w['real_slot_mask']),
        interpolation_exact_zero_excluded=True,model_or_pixel_values_read=False)


def span_support(support, nominal_rows):
    """Separate nominal physical target extent from actual auxiliary RGB support."""
    ii=_integer_axis(nominal_rows,'nominal_rows')
    if not ii or ii[-1]>=support['rows']:
        raise ValueError('Nominal span outside cache')
    if ii != list(range(ii[0],ii[-1]+1)):
        raise ValueError('Nominal span must be contiguous cache rows')
    raw=sorted({j for i in ii for j in support['row_encoded_support'][i]})
    consumed=sorted({j for i in ii for j in support['row_consumed_support'][i]})
    av=support['image_available'];sf=support['source_frame_ids'];st=support['source_timestamp_us']
    observed=[i for i in consumed if av[i]]
    synthetic=[i for i in consumed if not av[i]]
    active_windows=sorted({w for i in ii if av[i] and support['rgb_finite'][i]
                           for w,num,den in support['row_window_terms'][i] if num>0})
    return dict(nominal_cache_rows=ii,nominal_source_frames=[sf[i] for i in ii],
        nominal_source_timestamps_us=[st[i] for i in ii],nominal_unavailable_rows=[i for i in ii if not av[i]],
        encoded_context_cache_rows=raw,consumed_context_cache_rows=consumed,
        observed_context_source_frames=[sf[i] for i in observed],
        observed_context_source_timestamps_us=[st[i] for i in observed],
        synthetic_context_cache_rows=synthetic,synthetic_context_source_coordinates=[sf[i] for i in synthetic],
        outside_nominal_consumed_rows=[i for i in consumed if i not in set(ii)],
        outside_nominal_observed_source_frames=[sf[i] for i in observed if i not in set(ii)],
        contributing_window_indices=active_windows,
        internal_padding_slots=sum(not b for w in active_windows for b in support['windows'][w]['real_slot_mask']),
        structural_context_is_direct_nominal_evidence=False,
        local_process_truth=None,exact_completion_boundary=None)
