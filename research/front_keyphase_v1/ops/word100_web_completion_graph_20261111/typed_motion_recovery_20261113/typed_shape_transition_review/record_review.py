"""Serialize actual visual inspection; does not label from model outputs or run models."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
from collections import Counter

ROOT = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

# These labels were recorded after viewing every contact sheet in frame order.
# Ranges only compress contiguous individually inspected frames, not interpolation.
R = {
 2: {
   'hand':[(58,61,'open'),(62,62,'unknown'),(63,64,'flexing'),(65,86,'closed')],
   'body':[(58,61,'hold'),(62,69,'wrist_lowering'),(70,72,'hold'),(73,73,'unknown'),(74,80,'forearm_raising'),(81,83,'unknown'),(84,86,'hold')],
   'hand_unknown': {62:'손이 기울어지는 순간이라 투영상 짧아짐과 실제 굽힘 시작을 분리하기 어렵다.'},
   'body_unknown':[(73,73,'가슴 앞 유지에서 팔뚝 세움으로 전환하는 정확한 시작 경계가 모호하다.'),(81,83,'세움이 끝나며 안정되는 미세 변화로 진행과 유지의 정확한 경계를 보류한다.')],
   'summary':'편 손이 얼굴 앞에 있다가 손가락을 접으며 아래로 이동한다. 주먹이 된 뒤 가슴 앞에 잠시 머물고, 주먹을 유지하면서 팔꿈치가 내려가 팔뚝이 더 서는 모습이다.'
  },
 3: {
   'hand':[(62,69,'open'),(70,71,'flexing'),(72,96,'closed')],
   'body':[(62,68,'hold'),(69,78,'wrist_lowering'),(79,81,'hold'),(82,90,'forearm_raising'),(91,92,'unknown'),(93,96,'hold')],
   'hand_unknown':{},
   'body_unknown':[(91,92,'세움이 끝나는 작은 위치 변화와 유지 경계를 한 프레임으로 확정하지 않는다.')],
   'summary':'열린 손의 손가락이 내려오는 중에 접힌다. 가슴 앞 주먹을 잠깐 유지한 다음, 손가락을 다시 펴지 않고 팔뚝을 세운다.'
  },
 4: {
   'hand':[(82,87,'open'),(88,89,'unknown'),(90,90,'flexing'),(91,91,'unknown'),(92,120,'closed')],
   'body':[(82,86,'hold'),(87,97,'wrist_lowering'),(98,101,'hold'),(102,102,'unknown'),(103,112,'forearm_raising'),(113,113,'unknown'),(114,120,'hold')],
   'hand_unknown':{88:'손의 기울어짐과 초기 손가락 굽힘을 정면에서 분리하기 어렵다.',89:'손가락이 옆으로 보이며 짧아져 초기 굽힘 여부를 확정하지 않는다.',91:'닫힘으로 넘어가는 경계로, 더 접히는 중인지 이미 닫힌 것인지 확정하지 않는다.'},
   'body_unknown':[(102,102,'잠깐의 유지에서 세움으로 넘어가는 경계이다.'),(113,113,'팔뚝 세움이 끝나 안정되는 경계이다.')],
   'summary':'직접 보이는 편 손, 굽히는 손, 닫힌 손이 시간순으로 이어진다. 후속 팔뚝 세움 동안은 닫힌 손을 유지한다. 초기 기울어짐은 굽힘 확정 라벨에서 제외했다.'
  },
 5: {
   'hand':[(50,55,'open'),(56,56,'unknown'),(57,58,'flexing'),(59,86,'closed')],
   'body':[(50,53,'hold'),(54,63,'wrist_lowering'),(64,66,'hold'),(67,68,'unknown'),(69,79,'forearm_raising'),(80,80,'unknown'),(81,86,'hold')],
   'hand_unknown':{56:'비스듬한 손에서 굽힘 시작과 투영 축소를 구분하기 어렵다.'},
   'body_unknown':[(67,68,'짧은 유지에서 팔뚝이 서는 방향으로 바뀌는 경계가 모호하다.'),(80,80,'세움에서 안정된 유지로 넘어가는 경계이다.')],
   'summary':'손가락을 편 상태에서 내려오며 주먹을 만들고, 이후 주먹 모양을 유지한 채 팔뚝을 세운다. 손의 방향도 달라지지만 새 손모양 변화로 라벨링하지 않았다.'
  },
 6: {
   'hand':[(66,70,'open'),(71,71,'unknown'),(72,74,'flexing'),(75,100,'closed')],
   'body':[(66,69,'hold'),(70,80,'wrist_lowering'),(81,83,'hold'),(84,84,'unknown'),(85,93,'forearm_raising'),(94,95,'unknown'),(96,100,'hold')],
   'hand_unknown':{71:'손이 기울어지는 동시에 아래로 움직여 초기 굽힘 여부를 확정하지 않는다.'},
   'body_unknown':[(84,84,'가슴 앞 유지에서 팔뚝 세움으로 넘어가는 경계이다.'),(94,95,'세움 종료와 안정된 유지의 경계를 보류한다.')],
   'summary':'열린 손이 내려오며 닫힌다. 가슴 앞에서 이미 닫힌 주먹을 유지한 뒤 팔뚝이 서는 움직임을 한다. 확대 화면의 아래가 잘린 80~84는 원본 전체 PNG도 각각 열어 닫힌 손을 확인했다.'
  },
 7: {
   'hand':[(46,51,'open'),(52,52,'unknown'),(53,55,'flexing'),(56,56,'unknown'),(57,76,'closed')],
   'body':[(46,51,'hold'),(52,60,'wrist_lowering'),(61,63,'hold'),(64,64,'unknown'),(65,71,'forearm_raising'),(72,72,'unknown'),(73,76,'hold')],
   'hand_unknown':{52:'손의 기울어짐 때문에 펴진 손과 굽힘 시작을 분리하기 어렵다.',56:'움직임 흐림과 닫힘 경계 때문에 굽히는 중인지 이미 닫혔는지 확정하지 않는다.'},
   'body_unknown':[(64,64,'유지와 팔뚝 세움의 경계를 보류한다.'),(72,72,'세움이 끝나 유지로 넘어가는 경계를 보류한다.')],
   'summary':'열린 손에서 손가락이 접히며 아래로 이동하고, 이후 닫힌 손을 유지하면서 팔뚝이 선다. 닫힘 경계의 흐린 한 프레임은 추정으로 채우지 않았다.'
  }
}

def expand(ranges):
    out={}
    for a,b,label in ranges:
        for f in range(a,b+1):
            assert f not in out
            out[f]=label
    return out

def reason_body(spec, f):
    for a,b,note in spec['body_unknown']:
        if a <= f <= b:
            return note
    return None

if __name__ == '__main__':
    outpath=ROOT/'REVIEW_COMPLETE.json'
    assert not outpath.exists(), 'Preserve completed labels; create a separately named revision if required.'
    prep=read(ROOT/'PREPARED.json')
    frozen=read(ROOT/'FROZEN.json')
    counts_hand=Counter(); counts_body=Counter(); rows=[]
    checks=[]
    for raw in prep['rows']:
        person=raw['person']; spec=R[person]
        lo,hi=raw['proposed_inspection_range']; expected=list(range(lo,hi+1))
        hand=expand(spec['hand']); body=expand(spec['body'])
        assert sorted(hand)==expected==sorted(body)
        assert sha(raw['source'])==raw['source_sha256']
        pages=[]; coverage=[]
        for pg in raw['pages']:
            assert sha(pg['path'])==pg['sha256']
            coverage.extend(pg['frames'])
            pages.append({**pg,'viewed':True,'method':'tools.view_image; each of all six (or final fewer) body-and-hand panels inspected in original order'})
        assert coverage==expected
        frame_records=[]
        for rec in raw['frame_records']:
            f=rec['frame']; assert sha(rec['path'])==rec['sha256']
            hn=spec['hand_unknown'].get(f); bn=reason_body(spec,f)
            assert (hand[f]=='unknown')==(hn is not None)
            assert (body[f]=='unknown')==(bn is not None)
            counts_hand[hand[f]]+=1; counts_body[body[f]]+=1
            page=next(pg for pg in pages if f in pg['frames'])
            frame_records.append({
                **rec, 'observed_rgb':True, 'hand_side':'performer_right',
                'viewed_page':page['path'], 'viewed_page_sha256':page['sha256'],
                'hand_state':hand[f], 'hand_evidence':'direct_rgb' if hn is None else 'unknown',
                'hand_unknown_reason':hn,
                'body_motion':body[f], 'body_evidence':'direct_consecutive_rgb' if bn is None else 'unknown',
                'body_unknown_reason':bn,
                'temporal_inference_not_training_truth': (
                    'This lies between directly observed open/flexing/closed states; exact hand state remains unknown.' if hn is not None else None),
                'semantic_completion':None, 'educational_correctness':None
            })
        assert [x['frame'] for x in frame_records]==expected
        extras=[]
        if person==6:
            extras=[next(x for x in raw['frame_records'] if x['frame']==f) for f in [80,81,82,83,84]]
        rows.append({
            'clip_id':raw['clip_id'],'person':person,'split':'public_fit',
            'source':raw['source'],'source_sha256':raw['source_sha256'],
            'original_frame_count':raw['original_frame_count'],'decoder_fps':raw['decoder_fps'],
            'inspection_range':raw['proposed_inspection_range'],'observed_frames':expected,
            'all_original_frames_in_range_inspected':True,'contact_sheets':pages,
            'additional_full_frame_view_image':extras,
            'hand_state_runs':spec['hand'],'body_motion_runs':spec['body'],
            'observations_ko':spec['summary'],
            'existing_motion_events_for_provenance_only':raw['existing_motion_events'],
            'frame_records':frame_records,
            'direct_observed_sequence':{
              'open_frames':[f for f in expected if hand[f]=='open'],
              'flexing_frames':[f for f in expected if hand[f]=='flexing'],
              'closed_during_lowering_frames':[f for f in expected if hand[f]=='closed' and body[f]=='wrist_lowering'],
              'closed_during_forearm_raising_frames':[f for f in expected if hand[f]=='closed' and body[f]=='forearm_raising'],
              'hand_transition_happens_before_later_closed_forearm_raising':True,
              'semantic_completion':None
            }
        })
        checks.append({'clip_id':raw['clip_id'],'source_hash_matches':True,'frame_and_page_hashes_match':True,'exact_contiguous_coverage':True,'frames':len(expected),'pages':len(pages)})
    total=sum(len(r['observed_frames']) for r in rows)
    assert total==206 and len(rows)==6 and sorted(R)==list(range(2,8))
    result={
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'ALL_206_ORIGINAL_FRAMES_RGB_REVIEWED_ROOT_REVIEW_PENDING',
      'reviewer':'Codex delegated absence_contract_review_1113; not an independent human or KSL expert',
      'contract':str(ROOT/'FROZEN.json'),'contract_sha256':sha(ROOT/'FROZEN.json'),
      'prepared':str(ROOT/'PREPARED.json'),'prepared_sha256':sha(ROOT/'PREPARED.json'),
      'record_script_sha256':sha(Path(__file__)),
      'label_definitions':{
        'open':'Directly visible extended fingers; not a numerical joint-angle claim.',
        'closed':'Directly visible folded/compact fist-like finger configuration; no contact/force claim.',
        'flexing':'Directly visible intermediate finger bending within the inspected open-to-closed sequence.',
        'unknown':'Exact state or movement boundary not confidently visible; excluded from supervised truth.',
        'wrist_lowering':'Wrist/hand moves down from face area toward chest relative to the same performer.',
        'forearm_raising':'Forearm becomes more upright through elbow/wrist relation; wrist need not move upward in absolute pixels.',
        'hold':'No distinct progression at the inspected local interval; natural settling is allowed.',
        'transition':'Reserved for directly supported movement not assigned to either named movement; none forced here.'
      },
      'counts':{'sources':6,'original_frames':206,'contact_sheets':sum(len(r['contact_sheets']) for r in rows),'hand_state':dict(counts_hand),'body_motion':dict(counts_body)},
      'findings_ko':[
        '6개 원본 모두 편 손→손가락 굽힘→닫힌 손이 관찰되고, 뒤의 팔뚝 세움에서는 닫힌 손을 유지한다.',
        '따라서 손목이 아래로 움직였고 끝에 주먹이었다는 근거만으로 손을 닫는 변화를 배웠다고 볼 수 없다. 손 상태 변화와 팔 이동은 분리해 가르칠 수 있는 관측이다.',
        '이는 1개 표현의 6명 공개 FIT 관측 보강이며, 다른 표현으로의 일반화나 모델 개선 성능을 입증한 결과가 아니다.',
        'UNKNOWN은 앞뒤 맥락으로 강제 채우지 않았다. 모든 프레임은 실제로 보았지만 모든 프레임에 확정 라벨을 붙인 것은 아니다.'
      ],
      'safeguards':{'source_hashes_unchanged':True,'no_model_scores_used':True,'no_training_or_inference':True,'no_dev_user_mentor_final8_access':True,'no_numerical_angle_or_contact_thresholds':True,'no_interpolated_truth':True,'semantic_completion':None,'educational_correctness':None,'all_six_clips_public_fit_not_independent_test':True},
      'root_independent_review':'pending; especially conservative open/flexing and flexing/closed boundary unknowns',
      'rows':rows
    }
    dump(outpath,result)
    verification={'created_utc':datetime.now(timezone.utc).isoformat(),'review_sha256':sha(outpath),'checks':checks,'total_frames':total,'total_pages':sum(c['pages'] for c in checks),'all_checks_pass':True,'note':'Mechanical coverage/hash verification only; independent visual review is pending.'}
    dump(ROOT/'VERIFICATION.json',verification)
    lines=[
      '공개 FIT 6명, 206개 원본 프레임을 전부 직접 RGB 화면으로 확인했습니다.',
      '모델 점수를 보지 않고, 기존 구간에서 정한 범위를 끝까지 확인했습니다.',
      '',
      '확인한 내용: 편 손이 내려오며 닫힘 → 이미 닫힌 손을 유지한 채 팔뚝을 세움.',
      '손모양 변화와 팔 이동은 같은 정답으로 섞지 않고 별도로 기록했습니다.',
      '다른 표현의 성능이나 교육적 정오를 판정한 것은 아닙니다. 학습은 실행하지 않았습니다.',
      '',
      '손 상태: '+str(dict(counts_hand)),
      '팔 움직임: '+str(dict(counts_body)),
      '',
      '불확실한 프레임은 unknown으로 보존했고, 앞뒤 흐름으로 정답을 채우지 않았습니다.',
      '손의 기울어짐, 흐림, 진행/유지 경계는 손과 팔 각각 따로 보류할 수 있습니다.',
      '모든 원본/프레임/접촉시트 해시와 원래 프레임 번호를 JSON에 보존했습니다.',
      '이 검토자는 Codex이며 독립 사람 정답 또는 한국수어 전문가 검수가 아닙니다.',
      '루트 에이전트의 별도 시각 재검수를 기다립니다.',
      ''
    ]
    for r in rows:
        lines.extend([r['clip_id']+' / '+str(r['inspection_range']),r['observations_ko'],'손: '+str(r['hand_state_runs']),'팔: '+str(r['body_motion_runs']),''])
    (ROOT/'REVIEW_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':result['status'],'counts':result['counts'],'review_sha256':sha(outpath)},ensure_ascii=False))
