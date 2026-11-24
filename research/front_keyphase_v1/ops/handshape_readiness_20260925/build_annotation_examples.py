"""Build traceable CANDIDATE handshape annotations for the nine targets that have appendix rows.

Every field records WHERE it came from. Nothing here is a training label:
  * the dictionary row is a HEADWORD-level entry, not a statement about this recording
  * the measurements are read from the already-cached observation, not from any model
  * no field asserts that the recording performs the listed form

No model, no training, no download, no new extraction, no user data.
"""
import hashlib, json, pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation           # noqa: E402
from wis_lab.core import features as F                     # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')

# Rows transcribed from LEXICAL_ROWS.json (text extraction) and confirmed on the saved page images
# nikl_page008.png (seq 399/400) and nikl_page271.png (seq 13511/13533/13546/13548).
APPENDIX = {
    'GAMSAHAMNIDA': {'seq': 399, 'headword': '감사', 'form_name': '고맙다', 'sh': '0900', 'loc': '12',
                     'hands': 2, 'same_diff': 0, 'page_image_confirmed': 'nikl_page008.png',
                     'sense_caveat': "the headword 감사 also carries a 조사 sense at 0204/04/1 (seq 400)"},
    'NA': {'seq': 2230, 'headword': '나', 'form_name': '나', 'sh': '0900', 'loc': '12',
           'hands': 1, 'same_diff': 0, 'page_image_confirmed': None},
    'NEO': {'seq': 2543, 'headword': '너', 'form_name': '지시 너', 'sh': '0100', 'loc': '12',
            'hands': 1, 'same_diff': 0, 'page_image_confirmed': None},
    'MOREUDA': {'seq': 4280, 'headword': '모르다', 'form_name': '모르다', 'sh': '0900', 'loc': '12',
                'hands': 1, 'same_diff': 0, 'page_image_confirmed': None},
    'SILTA': {'seq': 7424, 'headword': '싫다', 'form_name': '싫다2', 'sh': '0603', 'loc': '12',
              'hands': 1, 'same_diff': 0, 'page_image_confirmed': 'nikl_page271.png (seq 13533 싫어하다/싫다2)',
              'alternate_form': '싫다1 at 0603/07/1 (seq 7423): same handshape code, different location'},
    'ALDA': {'seq': 7709, 'headword': '알다', 'form_name': '알다', 'sh': '0900', 'loc': '12',
             'hands': 1, 'same_diff': 0, 'page_image_confirmed': None},
    'JADA': {'seq': 9399, 'headword': '자다', 'form_name': '자다2', 'sh': '0200', 'loc': '04',
             'hands': 1, 'same_diff': 0, 'page_image_confirmed': None,
             'alternate_form': '잠자다 (seq 9560) maps to the same 자다2 row; 자다1/자다3 have no row here'},
    'JOTA': {'seq': 13546, 'headword': '좋다', 'form_name': '좋다', 'sh': '1000', 'loc': '05',
             'hands': 1, 'same_diff': 0, 'page_image_confirmed': 'nikl_page271.png'},
    'JOESONGHADA': {'seq': 13548, 'headword': '죄송하다', 'form_name': '미안 양손', 'sh': '0005', 'loc': '03',
                    'hands': 2, 'same_diff': 1, 'page_image_confirmed': 'nikl_page271.png',
                    'alternate_form': '미안 한손 at 0005/03/1 (seq 13525): same handshape and location code, one hand'},
}


def _slice(seq, name):
    """(column slice, group index) of one feature group inside the flat feature vector."""
    cursor = 0
    for i, g in enumerate(seq['group_order']):
        w = F.GROUPS[g][0]
        if g == name:
            return slice(cursor, cursor + w), i
        cursor += w
    raise KeyError(name)


def flex(seq, side, lo, hi):
    """Per-finger interior flexion (mean of the 3 joint angles, arccos/pi in [0,1]) and support.

    Rows come from sequence_features(trim=False), so row index == frame index.
    """
    cols, gi = _slice(seq, f'bend_{side}')
    g = seq['features'][:, cols]
    present = seq['present'][:, gi]
    rows = np.arange(len(g))
    sel = rows[(rows >= lo) & (rows <= hi) & present]
    if sel.size == 0:
        return None
    block = g[sel]
    out = {}
    for i, name in enumerate(FINGERS):
        vals = block[:, i * 3:(i + 1) * 3]
        out[name] = round(float(np.mean(vals)), 4)
    spread = block[:, 15:19]
    return {'observed_frames': int(sel.size), 'per_finger_mean_flexion': out,
            'adjacent_spread_mean': round(float(np.mean(spread)), 4),
            'flexion_range_over_window': {n: round(float(
                np.max(np.mean(block[:, i * 3:(i + 1) * 3], axis=1))
                - np.min(np.mean(block[:, i * 3:(i + 1) * 3], axis=1))), 4)
                for i, n in enumerate(FINGERS)}}


def main():
    manifest = read(L / 'config' / 'source_manifest.json')
    windows = read(L / 'config' / 'reference_review_windows_20260915.json')
    clips = {c['word_id']: c for c in manifest['clips']}
    out = []
    for wid, row in APPENDIX.items():
        c = clips[wid]
        npz = L / c['observation_npz']
        obs = Observation.load(str(npz))
        seq = F.sequence_features(obs, trim=False)
        lo, hi = c['quality']['active_span_first_frame'], c['quality']['active_span_last_frame']
        win = windows['entries'].get(c['clip_id'])
        thirds = [(lo, lo + (hi - lo) // 3), (lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3),
                  (lo + 2 * (hi - lo) // 3, hi)]
        entry = {
            'word_id': wid, 'display_name': c['display_name'],
            'candidate_status': 'CANDIDATE_NOT_TRAINING_TRUTH',

            'dictionary_row': {
                'source': 'NIKL handshape-based KSL dictionary final report, appendix 2',
                'pdf_sha256': 'de6077e2142c514c84e34503432511a7ad506e8fe423448253ecd72a2864b593',
                'columns': '순|대분류|중분류|표제어|분류명|수형|수위|한 손/두 손|동형/이형',
                'column_header_evidence': 'page 149 extracted text and the saved page images',
                **{k: v for k, v in row.items()},
                'granularity': 'HEADWORD level. This row does not state what this recording did.',
                'code_form_definition': 'NOT AVAILABLE LOCALLY. Appendix 2 is the lexical table only; '
                                        'the 수형 chart that names the shape of code '
                                        + row['sh'] + ' is not in the held PDF.',
            },

            'recording': {
                'clip_id': c['clip_id'], 'source_kind': c['source_kind'],
                'video_path': c['absolute_path'], 'video_sha256': c['sha256'],
                'observation_npz': c['observation_npz'], 'observation_sha256': c['observation_sha256'],
                'frames': c['frame_count'], 'duration_ms': c['duration_ms'],
                'signer_identity': 'NOT RECORDED for the NIKL demonstration set',
                'crosswalk_to_dictionary_row': 'NOT_ESTABLISHED; the pairing below is by Korean '
                                               'headword spelling only',
            },

            'hands_observed': {
                'left_frame_rate': c['quality']['left_hand_frame_rate'],
                'right_frame_rate': c['quality']['right_hand_frame_rate'],
                'both_frame_rate': c['quality']['both_hands_frame_rate'],
                'meaning': 'detection rate only. A one-handed sign can still show the resting hand, '
                           'so this NEITHER confirms NOR contradicts the 한 손/두 손 column.',
            },

            'observed_form': {
                'source': 'cached observation + existing bend_L/bend_R feature path; no model was run',
                'window': {'active_span': [lo, hi]},
                'left': flex(seq, 'L', lo, hi),
                'right': flex(seq, 'R', lo, hi),
                'thirds_right': [flex(seq, 'R', a, b) for a, b in thirds],
                'thirds_left': [flex(seq, 'L', a, b) for a, b in thirds],
                'units': ('INTERIOR joint angle, arccos/pi in [0,1]. features.bend_angles computes '
                          '_angle(a-b, c-b) at the middle joint b, so a STRAIGHT finger gives ~1.0 '
                          '(the two bone vectors oppose) and a fully BENT finger approaches 0.0. '
                          'Measured on the 2-D image projection.'),
                'reading': '1.0 = straight / extended, 0.0 = fully bent',
                'is_not': ('not a 3-D flexion angle, not a handshape label, not a correctness judgement. '
                           'A change in this value can come from a real shape change OR from hand '
                           'rotation, foreshortening toward the camera, motion blur or detection noise, '
                           'so the range alone does not establish that the handshape changed.'),
            },

            'review_basis': ({
                'reference_review_window': win['frame_id_range'],
                'status': win['status'], 'scope': win['scope'],
                'boundary_sensitivity_frames': win['boundary_sensitivity_frames'],
                'limit': windows['selection_limit'],
            } if win else {'reference_review_window': None,
                           'status': 'NO_VISUAL_REVIEW_RECORD_FOR_THIS_CLIP'}),

            'unconfirmed': [
                'what hand shape the code ' + row['sh'] + ' actually names (chart not held)',
                'whether this recording performs the form named by this dictionary row',
                'which hand carries the sign (detection rate cannot decide it)',
                'whether the observed flexion change is linguistic or preparation/recovery',
            ],
        }
        if 'alternate_form' in row:
            entry['unconfirmed'].append('which listed form this recording corresponds to: ' + row['alternate_form'])
        if 'sense_caveat' in row:
            entry['unconfirmed'].append(row['sense_caveat'])
        out.append(entry)

    groups = {}
    for e in out:
        groups.setdefault(e['dictionary_row']['sh'], []).append(e['word_id'])
    result = {
        'schema': 'wis_handshape_candidate_annotation_v1',
        'at_kst': '2026-09-25',
        'status': 'SAMPLE_LINKED_CANDIDATE_ANNOTATION_PREPARED_NOT_VERIFIED',
        'purpose': 'traceable candidate annotation example; never a training label',
        'targets_with_dictionary_row': len(out),
        'handshape_code_groups_among_these_targets': groups,
        'entries': out,
        'never_asserted': [
            'that a recording performs the form its headword row lists',
            'that words sharing a 수형 code make the same hand shape in these recordings',
            'that the measured flexion verifies or refutes any code',
        ],
        'models_run': 0, 'training_runs': 0, 'downloads': 0, 'new_extraction': 0,
        'user_arrays_read': 0,
    }
    with (H / 'CANDIDATE_ANNOTATIONS.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('entries', len(out))
    print('handshape code groups:', json.dumps(groups, ensure_ascii=False))
    for e in out:
        r = e['observed_form']['right']
        l = e['observed_form']['left']
        print(f"  {e['word_id']:<13} 수형 {e['dictionary_row']['sh']} 수위 {e['dictionary_row']['loc']} "
              f"{e['dictionary_row']['hands']}손 | R obs {r['observed_frames'] if r else 0:>3} "
              f"L obs {l['observed_frames'] if l else 0:>3} | review "
              f"{'YES' if e['review_basis'].get('reference_review_window') else 'no'}")
    print('written CANDIDATE_ANNOTATIONS.json', sha(H / 'CANDIDATE_ANNOTATIONS.json')[:16])


if __name__ == '__main__':
    main()
