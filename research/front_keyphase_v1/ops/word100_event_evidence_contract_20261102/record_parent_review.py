"""Codex visual observations of displayed native windows, not dense gold labels."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'PARENT_TRAVERSAL_REVIEW.json').exists();m=read(U/'REVIEW_MANIFEST.json');by={r['alias']:r for r in m['rows']}
    observations=[
      ('A04',[53,58],[61,65],[59,62],'오른손이 왼팔의 몸쪽에서 손목쪽으로 이어서 옮겨가고, 뒤쪽 장면에는 손목 부근의 관계가 남는다. 몸/왼팔 자체의 작은 이동도 함께 있어 도착 직후의 미세이동 종료를 한 프레임으로 확정하지 않는다.'),
      ('A05',[54,59],[63,66],[60,63],'손바닥을 편 오른손이 왼팔을 따라 이동한다. 뒤쪽에는 손목 부근에 놓여 있다. 정지했음을 언어 필수 조건으로 해석할 근거는 없다.'),
      ('A06',[55,63],[67,69],[63,67],'이동이 비교적 점진적이다. 왼팔의 몸쪽과 먼쪽 관계는 구분되지만 이동→끝관계의 소속을 바꾸는 단일 프레임은 불확실하다.'),
      ('A07',[79,87],[91,93],[87,91],'오른손과 왼팔의 상대 위치가 점진적으로 달라진다. 끝 근처에서도 작은 이동이 이어져 보인다. 한 프레임의 형태만으로 이동 완료나 필수 멈춤을 확정하지 않는다.'),
      ('A08',[52,58],[62,65],[59,62],'오른손이 팔을 따라 이동한 뒤 손목 부근 관계가 유지된다. 뒤쪽의 몸/팔 전체 이동과 팔을 따라가는 추가 이동을 동일하게 세지 않는다.'),
      ('A09',[44,50],[54,59],[50,54],'앞부분에서 왼팔을 따르는 상대 이동이 보이고 뒤쪽은 손목 근처 모습이다. 중간 경계를 정확한 수어 규칙의 시점으로 단정할 수 없다.'),
      ('A10',[55,61],[65,67],[61,65],'왼팔을 따라 이동하는 한 흐름 안에서 손목 부근 관계로 이어진다. 끝 위치에 더 가까워지는 과정과 별도의 필수 동작 시작을 구별할 근거는 이 창만으로 부족하다.'),
      ('A11',[82,88],[92,94],[88,92],'오른손을 편 모습으로 왼팔의 먼쪽에 접근하고 손목 부근에 놓인다. 이동의 마지막 부분과 끝관계가 나타나는 경계는 범위로 남긴다.'),
      ('A12',[52,57],[61,64],[57,61],'오른손이 왼팔 쪽으로 내려오며 팔을 따라 이동하고 손목 부근 모습이 남는다. 이 창에서 두 개의 별도 의무 행동 또는 의무 정지를 확인한 것은 아니다.')]
    rows=[]
    for alias,change,relation,uncertain,note in observations:
        r=by[alias];assert r['word_id']=='WORD1133' and r['person'] in range(1,10)
        for bounds in [change,relation,uncertain]:assert set(bounds)<=set(r['frames'])
        for sheet in r['sheets']:assert sha(sheet['path'])==sheet['sha256']
        rows.append(dict(alias=alias,clip_id=r['clip_id'],person=r['person'],sheets=r['sheets'],reviewed_native_frames=r['frames'],
            observed_change=dict(value='yes',evidence_pair=change,meaning='right hand changes its projected relation along left arm'),
            body_relative_relation=dict(value='yes',evidence_window=relation,meaning='right hand presented near distal left forearm/wrist'),
            main_or_return_context=dict(value='main_event_evidence',scope='selected window only; no separate return is evident here'),
            simultaneous_evidence=dict(value='unknown',meaning='A moving hand can enter an end region, but exact simultaneous predicate ownership is not labeled from these sheets.'),
            boundary_uncertainty=uncertain,required_stop='unsupported',exact_contact_or_pressure='unknown',note=note))
    out=dict(reviewer='root Codex visual inspection of all displayed sheets',rows=rows,
        display_blinding='No prediction/confidence/DEV error overlay. Reviewer knows general hypothesis; not a blinded independent human study.',
        interpretation='The source describes one arm-sweeping clause; early movement and later body relation are visible. Exact exclusive event boundary is not established by these observations. This does not prove all current event IDs invalid.',
        no_dense_targets=True,no_existing_label_change=True,no_existing_result_rescore=True,no_semantic_correctness_claim=True,whole_goal_complete=False)
    with (U/'PARENT_TRAVERSAL_REVIEW.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('9publicFITwindows reviewed; no dense labels or model changes')
if __name__=='__main__':main()
