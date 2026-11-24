"""Compact presentation only. All numeric data are read from verified summaries."""
from pathlib import Path
from pipeline import UNIT,read,sha

EXAMPLES=(('R_hand_axis2d',0,'오른손 화면 방향'),
          ('L_hand_axis2d',0,'왼손 화면 방향'),
          ('R_finger_angles14',2,'오른손 검지 시작마디의 화면상 각도'),
          ('L_finger_angles14',2,'왼손 검지 시작마디의 화면상 각도'))

def short(value):
    if isinstance(value,str):return value
    if isinstance(value,dict):
        if 'summary_ko' in value:return str(value['summary_ko'])
        return '; '.join(str(k)+': '+short(v) for k,v in value.items())
    if isinstance(value,list):return '; '.join(short(v) for v in value)
    return str(value)

def render():
    inp=read(UNIT/'INPUTS.json'); v=read(UNIT/'review/VISUAL_REVIEW.json')
    ev=read(UNIT/'EVALUATION.json'); cb=read(UNIT/'CASEBOOK.json')
    rows={(r['query'],r['reference'],r['parent_id']):r for r in ev['natural_rows']}
    lines=['동작 비교 사례 4개 — 원본 장면과 숫자를 함께 읽는 방법','',
        '목표: 숫자 차이, 저장 관측 부족, 시점 대응, 실제 보이는 모습, 설명의 지원 범위를 따로 읽습니다.',
        '교육적 정오를 판정한 자료가 아닙니다. 새로 본 원본은 사전 고정10창이며 나머지29범위에는 원본 확인 완료 표시를 하지 않았습니다.',
        '아래 숫자 예시는 미리 정한 공통4특징 순서입니다. 좋은 점수만 고른 것이 아닙니다.',
        '평균/최대값은 저장된 주요 범위 전체의 요약이고, 원본 RGB 확인 창과 범위가 다를 수 있습니다.',
        '3차원 관절각·손바닥 법선·손목 굽힘각을 뜻하지 않습니다. 최대 차이가 난 순간도 같은 동작 단계인지 확정된 것은 아닙니다.',
        '전체14각도·상대 방향·양쪽 시간축의 값과 모든 동률 최대 시점은 CASEBOOK.json이 연결하는 results JSON에 보존되어 있습니다.','']
    for c in cb['cases']:
        cv=next(x for x in v['cases'] if x['query']==c['query'])
        lines += ['='*48,c['label']+' / 비교영상 '+c['query'],
                  '서로 다른 참고3개 유지: '+', '.join(c['references']),
                  short(cv['summary_ko']),'한계: '+short(cv['limits_ko']),'']
        for wi,w in enumerate(v['windows']):
            if w['query']!=c['query']:continue
            row=rows[(w['query'],w['reference'],w['parent_id'])]
            result=read(row['result']['path']);s=result['summary']
            lines += [w['window_id']+' / '+w['query']+' ↔ '+w['reference']+' / '+w['parent_id']]
            for key,label in [('query_axis','비교영상'),('reference_axis','참고영상')]:
                a=w[key]; ids=a['reviewed_frame_ids'];pts=a['reviewed_timestamp_us']
                lines += [f"{label}: 원프레임 {ids[0]}–{ids[-1]} ({len(ids)}장), {pts[0]/1e6:.3f}–{pts[-1]/1e6:.3f}초",
                          '원본 관측: '+short(a['observations'])]
                pages=a['contact_pages']
                for p in pages:
                    path=p if isinstance(p,str) else p.get('file',p.get('path'))
                    if path:lines.append('시간순 이미지: '+str(path))
            for key,label in [('visible_difference_review','보이는 차이'),('observation_missing','저장 관측 부족'),('correspondence_review','시간 대응')]:
                lines += [label+': '+short(w[key])]
            lines += [f"숫자 예시의 저장 범위: 비교 {row['query_range']} / 참고 {row['reference_range']}"]
            for group,dim,label in EXAMPLES:
                g=s['groups'][group];a=g['summary_by_dimension'][dim]
                suffix=f" (지원 {a['shared_links']}/{a['possible_links']} 대응)"
                if a['shared_links']:
                    peak=a['first_peak'];q=peak['query'];r=peak['reference']
                    text=f"평균차 {a['mean_abs_deg']:.2f}°, 최대차 {a['max_abs_deg']:.2f}°"
                    text+=f"; 첫 최대 시점: 비교 f{q['frame_id']}({q['timestamp_us']/1e6:.3f}초) ↔ 참고 f{r['frame_id']}({r['timestamp_us']/1e6:.3f}초)"
                else:text='비교할 공통 관측 없음(0° 차이가 아님)'
                lines += [label+': '+text+suffix]
            lines += ['관측/값/최대 시점 출처: '+row['result']['path'],
                      '원본 검토 출처: '+str(UNIT/'review/VISUAL_REVIEW.json')+f' #/windows/{wi}','']
        lines += ['이 표현의 전체 주요 비교 범위 — 각 참고를 모두 유지']
        for r in ev['natural_rows']:
            if r['query']!=c['query']:continue
            n=r['counts']
            lines += [f"{r['reference']} / {r['parent_id']}: 비교 {r['query_range']} ↔ 참고 {r['reference_range']}; 설명부분연결 {r['source_cards']}개; 새RGB일부검토 {'있음' if r['scoped_RGB_review_attached'] else '없음'}; 특징기회 {n['possible']} 중 직접 {n['direct_complete']}, 부분 {n['partial_shared']}, 불가 {n['not_comparable']}"]
        lines+=['']
    lines += ['마지막 확인','손이 보이는데 저장값이 비어 있어도 원검출 실패라고 단정하지 않습니다. 영역 제한이나 후처리 제외일 수도 있으며 이번에 원인 단계까지 확인하지 않았습니다.',
              '같은 Codex가 본 정성 검토이고 독립 수어 전문가 정답이 아닙니다. 여기서 교육적 오답 정답이나 새 학습 표적을 만들지 않았습니다.']
    path=UNIT/'CASEBOOK_KO.txt'
    if path.exists():
        before=UNIT/'before/PRESENTATION_INITIAL_CASEBOOK_KO.txt'
        if before.exists():raise RuntimeError('Compact presentation already rendered; inspect rather than repeat')
        before.write_bytes(path.read_bytes())
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'presentation_lines':len(lines),'sha256':sha(path),'numeric_recomputation':False})

if __name__=='__main__':render()
