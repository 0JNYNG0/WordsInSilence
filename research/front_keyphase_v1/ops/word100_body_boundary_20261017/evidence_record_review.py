"""Persist the ten original-frame contact sheets actually reviewed by Codex."""
import hashlib
import json
from pathlib import Path

U = Path(__file__).resolve().parent


def main():
    notes = {
        'NIA_SL_WORD2732_REAL02_F': '백: sampled core f4–31 shows the right hand near the face and left hand down at the side. Left palm removal concerns the down hand in those samples; the raised right action is above the boundary.',
        'NIA_SL_WORD0500_REAL03_F': '마음에들다: f28 left hand remains down while right is near the face; f33–61 both raised hands remain above the line. Initial left-down removal is visible; raised-hand action clipping is not evident in these samples.',
        'NIA_SL_WORD2732_REAL04_F': '백: core f19–48 shows right finger action near the face, left hand down at the side. Most left hand points lie below the line, with wrist/palm-edge fragments remaining. Right raised action remains above the line.',
        'NIA_SL_WORD2867_REAL05_F': '이천오백: core f16–63 shows right-hand number changes at chest/face height, while left stays down. The line excludes much of the left hand but leaves wrist fragments. Raised right hand remains above the line.',
        'NIA_SL_WORD2700_REAL07_F': '육십팔: core f21–48 shows right hand finger changes around chest/face height; left remains at the side. Most left hand is below the line, with wrist fragments retained. No raised right-hand clipping is evident in the samples.',
        'NIA_SL_WORD0651_REAL08_F': '소복: f0/f22 shows right hand already raised and left down. From f30 through f79 both hands move in front of the chest/abdomen and remain above the line in sampled frames. The lowest sampled right-hand posture f79 is close to but above the line.',
        'NIA_SL_WORD0708_REAL08_F': '얌전하다: f25/f32 left hand hangs at the side below the line while right acts near face; f39 left begins entering. Both raised hands f46–75 remain above the line while moving from chest toward abdomen. The lowering/raising transition can be partly hidden; its linguistic importance is not established.',
        'NIA_SL_WORD0795_REAL08_F': '장애인복지관: f39/f47/f56 left hand is lowered or down beside the body and partially excluded; right remains active in front of the body. Both hands later rise above the line. At f75 both RGB hands are visibly above the boundary, but legacy hand overlays appear duplicated on one side and absent on the other: this is an inherited detector/assignment concern, not evidence of ROI clipping. The linguistic importance of the intervening left lowering is not established.',
        'NIA_SL_WORD0920_REAL08_F': '추출: f27–63 both hands act in front of face/chest and then abdomen above the line. An overlay is absent/overlapped at f27 despite raised RGB hands, which is inherited observation uncertainty. No new ROI palm removal in the sampled core is evident.',
        'NIA_SL_WORD2309_REAL09_F': '놀리다: core f27–63 right hand acts around face/chin while left remains down at side below the line. The exclusion concerns down left hand in sampled frames; active right hand remains above the line.',
    }
    rows = []
    for cid, note in notes.items():
        p = U / f'evidence_{cid}.json'
        item = json.loads(p.read_text(encoding='utf-8'))
        for image_path, expected in item['images'].items():
            assert hashlib.sha256(Path(image_path).read_bytes()).hexdigest() == expected
        rows.append({'clip_id': cid, 'label': item['label'], 'bottom': item['bottom'],
            'reviewed_contact_sheet': next(k for k in item['images'] if k.endswith('_contacts.png')),
            'image_sha256': item['images'], 'source_sha256': item['source_sha256'],
            'observation_sha256': item['observation_sha256'], 'selected_frames': item['selected_frames'],
            'review_note': note, 'evidence_json': str(p),
            'evidence_json_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    result = {'status': 'TEN_SPARSE_ORIGINAL_RGB_CONTACT_SHEETS_REVIEWED', 'reviewer': 'Codex /root/roi_data_budget',
        'coefficient_sha256': '4df61776c9aa97859b146707edb7acfa5bed4da4e2d2e4e0c91120f66eec76a1',
        'summary': 'No obvious removal of the sampled raised action hand was seen. Several clips lose a down resting hand as intended. Some resting wrist fragments remain; an extra wrist-removal filter remains disabled. Two multi-hand clips have transition or inherited detection uncertainty, so this is not a blanket semantic-preservation pass.',
        'limits': ['Contact sheets are sparse frames, not a continuous review of every source frame.',
            'Legacy joint observations are used; these images do not validate new joint-v3 or actual cropped-sensor detections.',
            'The presence of any hand or the absence of palm5-loss does not establish preservation of the required sign action.',
            'No linguistic expert adjudication, classifier evaluation, new detector execution or training was performed.',
            'Hands outside the chosen boundary are intentionally unobserved; hidden finger shape and unsampled transitions cannot be certified.'],
        'rows': rows, 'new_detector_conditions': 0, 'training': 0}
    with (U / 'evidence_VISUAL_REVIEW.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'reviewed': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
