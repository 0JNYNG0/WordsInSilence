"""Explain saved mismatches without changing paths, labels, or scores."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def region(rs,f):
 hits=[r for r in rs if r['core'] and r['core'][0]<=f<=r['core'][1]]
 assert len(hits)<=1
 return hits[0] if hits else None
def main():
 data=read(U/'REPORT_DATA.json');out=[]
 for c in data['cases']:
  if c['role']=='fit':continue
  for e in c['events']:
   for name,m in e['methods'].items():
    wrong=[]
    for q,r in m['path'] or []:
     qr=region(c['query_regions'],q);rr=region(c['reference_regions'],r)
     if qr and rr and qr['kind']!=rr['kind']:
      wrong.append(dict(query=q,reference=r,query_kind=qr['kind'],reference_kind=rr['kind']))
    if wrong or m['occurrence']=='different_documented_ordinal':
     out.append(dict(case=c['id'],segment=e['segment_id'],method=name,
       center=m['center'],reference_center=e['reference_center'],
       occurrence=m['occurrence'],different_kind_links=wrong,
       note='Saved physical-stage mismatch; not a sign correctness label.'))
 assert any(x['case']=='B_same_path_P066_P063' and x['segment']=='P063_s05' and x['method']=='core_first' and x['center']==73 and x['occurrence']=='different_documented_ordinal' for x in out)
 p=next(x for x in out if x['case']=='B_same_path_P036_P034' and x['segment']=='P034_s04' and x['method']=='core_first')
 result=dict(status='SAVED_PATH_CAUSE_REVIEW',report_data_sha256=hashlib.sha256((U/'REPORT_DATA.json').read_bytes()).hexdigest(),rows=out,
  findings_ko=[
   '조상 P066→P063 s05: 참고 두 번째 왼손 상승에 기존 전체 DTW는 비교96프레임(두 번째), 핵심 우선은73프레임(첫 번째)을 연결했다. 같은 모양·상승 종류를 찾았어도 반복 회차가 맞는 것은 아니다.',
   '순종 P036→P034 s04: 핵심 우선 경로에는 참고의 손 재지향 이동과 비교 영상의 이미 멈춘 유지가 연결된 구간이 있다. 대표시점만 보면 유지끼리 맞으므로 전체 경로 확인이 필요하다.',
   '원인 해석: 구간별 유사성 검색만으로 전체 단계·반복 회차의 일대일 대응이 보장되지 않는다는 저장 결과다. 특정 손 특징이나 가중치가 원인이라는 인과 결론은 별도 시험 없이는 내리지 않는다.'
  ],new_search=0,new_inference=0,educational_truth=False)
 (U/'CAUSE_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(dict(mismatch_rows=len(out),move_hold_example=p),ensure_ascii=False))
if __name__=='__main__':main()
