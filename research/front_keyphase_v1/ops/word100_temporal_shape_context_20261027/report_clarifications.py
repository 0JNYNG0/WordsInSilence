from pathlib import Path
import json,hashlib,runpy
U=Path(__file__).resolve().parent
p=U/'build_report.py';s=p.read_text(encoding='utf-8')
pairs=[
 ('직전 0.2초 안에서 최소 두 번 같은 상태가 보이면, 불확실한 현재를 그 상태로 <b>추정</b>한다.',
  '직전 0.2초 안에서 최소 두 번 확인되고, 그 범위의 모든 확인된 상태가 같으면 현재를 그 상태로 <b>추정</b>한다.'),
 ('서로 다르면 보류한다.</p></div></div>', '범위 안에 반대 상태가 하나라도 보이면 보류한다.</p></div></div>'),
 ('앞뒤 정보는 안정된 모양을 유지하는 짧은 누락을 보완하는 데 쓸 수 있다.',
  '이번 인위적 마스킹 시험에서 앞뒤 정보가 안정된 모양을 되찾는 경우를 확인했다.'),
 ('<p class="notice">관측 모델의 학습은',
  '<p>이번 결과는 확인된 상태를 앞뒤로 연결하는 고정 규칙의 한계다. 현재 RGB의 변화까지 함께 배우는 시간 모델 전체가 효과 없다는 뜻은 아니다.</p><p class="notice">관측 모델의 학습은'),
 ('수행\n6표현','이번 고정 상태 일치 규칙의 한계이며, RGB 변화까지 배우는 시간 모델 전체가 불가능하다는 뜻이 아니다.\n\n수행\n6표현')]
for a,b in pairs:
 assert a in s,a
 s=s.replace(a,b,1)
p.write_text(s,encoding='utf-8')
runpy.run_path(str(p))
record={'status':'APPLIED','review':'review/REPORT_REVIEW.json','changes':[b for a,b in pairs],
 'first_attempt_error':'PowerShell inline python quoting caused SyntaxError before any file mutation. Recovered via saved Python script; comparison, labels, conditions untouched.',
 'comparison_sha256':hashlib.sha256((U/'COMPARISON.json').read_bytes()).hexdigest(),
 'html_sha256':hashlib.sha256((U/'RESULT_KO.html').read_bytes()).hexdigest()}
(U/'REPORT_CLARIFICATIONS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
