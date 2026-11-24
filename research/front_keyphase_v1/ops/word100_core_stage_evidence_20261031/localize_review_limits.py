from pathlib import Path
U=Path(__file__).resolve().parent;p=U/'index.html';s=p.read_text(encoding='utf8')
old="for(let raw of (Array.isArray(q.rgb_review?.unknown)?q.rgb_review.unknown:Object.entries(q.rgb_review?.unknown||{}).map(([k,v])=>({reason_ko:k+': '+(Array.isArray(v)?v.join(' / '):String(v))}))))"
new="for(let raw of reviewLimits(q.rgb_review?.unknown))"
assert old in s;s=s.replace(old,new)
helper="""function reviewLimits(u){if(Array.isArray(u))return u;if(!u)return [];return [{reason_ko:`개별 장면으로 직접 확인하지 않은 원본 시점은 ${(u.native_frames_not_individually_seen_in_any_reviewed_image||[]).length}개입니다. 해당 시점의 손 모양을 정답으로 채우지 않았습니다.`},{reason_ko:'단계 경계에는 불확실한 범위가 있습니다. 세부 손가락 각도, 실제 접촉·압력, 언어적으로 필수인지 여부는 확정하지 않았습니다.'},{reason_ko:'준비·복귀에도 의미 있는 움직임이 포함될 수 있습니다. 미주석 구간을 잘못된 행동이나 무의미한 구간으로 처리하지 않습니다.'}];} """
s=s.replace("const el=id=>",helper+"const el=id=>");p.write_text(s,encoding='utf8')
