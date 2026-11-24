"""Render root-approved SUMMARY.json as an offline research report.

No model execution, network access, camera, user uploads or operating activation.
Final generation is a separate root action; --self-test writes only research fixtures.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path

U = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def h(value):
    return html.escape(str(value), quote=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pct(value):
    if value is None:
        return "미확정"
    if number(value):
        return f"{value * 100 if abs(value) <= 1 else value:.2f}%"
    return str(value)


def span(value, percentage=True):
    if value is None:
        return "미확정"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        fn = pct if percentage else str
        return f"{fn(value[0])} ~ {fn(value[1])}"
    return str(value)


def plain(value):
    if value is None:
        return "미확정"
    if value is True:
        return "예"
    if value is False:
        return "아니요"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return str(value)


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def local_uri(value, base=U):
    if not value:
        return ""
    text = str(value)
    # This viewer only references local, saved public images.
    if text.startswith("file:"):
        return text
    if "://" in text or text.startswith("data:"):
        return ""
    p = Path(text)
    if not p.is_absolute():
        p = base / p
    return p.resolve().as_uri()


def list_html(values, fallback="아직 확정된 내용이 없습니다."):
    if not values:
        return f'<p class="muted">{h(fallback)}</p>'
    return "<ul>" + "".join(f"<li>{h(plain(v))}</li>" for v in values) + "</ul>"


def table(headers, rows, empty="아직 집계되지 않았습니다."):
    if not rows:
        return f'<div class="empty">{h(empty)}</div>'
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{h(x)}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{h(x)}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>"


FEATURE_NAMES = {"index_only": "검지만 폄", "open_four": "네 손가락 폄", "four_closed": "네 손가락 닫힘"}
FAMILY_NAMES = {"handshape": "손모양", "orientation": "손 방향", "movement": "이동", "repetition_description": "반복 설명", "body_relation": "신체 위치 관계", "contact_description": "접촉 설명", "order": "순서", "hand_count": "사용 손", "two_hand_relation": "양손 관계"}
FACT_NAMES = {"source_profiles": "정리한 단어 프로필", "source_supported": "출처 연결 프로필", "source_claims": "출처 기반 특징 설명", "manual_images": "사람이 수형을 주석한 외부 사진", "classes": "외부 수형 종류", "fit": "학습 자료", "validation": "개발 검증 자료", "test": "최종 평가 자료", "seeds": "반복 학습 seed", "training_runs": "학습 실행 수", "protected_files_unchanged": "보호 파일 보존", "user_mentor_data_used": "사용자·멘토 자료 사용", "operating_changed": "운영 기본 변경", "educational_verdict_enabled": "교육적 정오 판정 활성화"}
STATUS_NAMES = {
    "completed": "완료", "complete": "완료", "passed": "검증 통과", "done": "완료",
    "in_progress": "진행 중", "running": "진행 중", "pending": "아직 실행 전",
    "partial": "일부 완료", "gated": "적용 근거 확인 필요", "blocked": "필수 근거 부족",
    "not_activated": "운영 미적용", "not_validated": "검증 미완료", "research_only": "연구 범위 완료",
}


def status_name(value):
    text = str(value or "pending")
    return STATUS_NAMES.get(text.lower(), text)


def stage_cards(stages):
    defaults = ["출처 있는 특징 프로필", "학습·평가 정답 자료", "작은 관측 모델", "네 조건의 효과 비교", "다중 참고 DTW 연결", "별도 웹 후보 검토"]
    supplied = {int(s["n"]): s for s in stages if str(s.get("n", "")).isdigit()}
    result = []
    for n, title in enumerate(defaults, 1):
        s = supplied.get(n, {"n": n, "title": title, "status": "pending", "detail": "최종 집계에 아직 기록되지 않았습니다."})
        raw = str(s.get("status", "pending")).lower()
        tone = "done" if raw in ("completed", "complete", "passed", "done") else "pending"
        result.append(f'<article class="stage {tone}"><div class="stage-top"><b>{n:02}</b><span>{h(status_name(s.get("status")))}</span></div><h3>{h(s.get("title", title))}</h3><p>{h(plain(s.get("detail")))}</p></article>')
    return '<div class="stage-grid">' + "".join(result) + "</div>"


def source_profiles(cards):
    profiles = cards.get("profiles", [])
    supported = [p for p in profiles if p.get("source_variants")]
    entry_count = sum(len(p.get("source_variants", [])) for p in profiles)
    claim_count = sum(len(v.get("feature_claims", [])) for p in profiles for v in p.get("source_variants", []))
    blocks = []
    for p in profiles:
        variants = p.get("source_variants", [])
        citations = []
        for v in variants:
            url = str(v.get("source_url", ""))
            families = sorted({FAMILY_NAMES.get(c.get("family"), c.get("family", "기타")) for c in v.get("feature_claims", [])})
            anchor = f'<a href="{h(url)}" target="_blank" rel="noopener noreferrer">{h(v.get("source_id", "출처"))} · {h(v.get("entry_id", ""))}</a>' if url.startswith("https://") else "출처 URL 없음"
            citations.append(f'<div>{anchor}<span class="source-families">{h(" / ".join(families))}</span></div>')
        blocks.append(f'<tr><td><b>{h(p.get("display_label", ""))}</b><small>{h(p.get("word_id", ""))}</small></td><td>{len(variants)}</td><td>{"".join(citations) or "이번 범위에서 연결된 항목 없음"}</td></tr>')
    metrics = '<div class="metrics">' + "".join(f'<div><strong>{v}</strong><span>{h(t)}</span></div>' for v, t in [(len(profiles), "단어 프로필"), (len(supported), "출처 연결 프로필"), (entry_count, "사전 항목 후보"), (claim_count, "출처 기반 특징 설명")]) + "</div>"
    details = '<details><summary>단어별 출처와 특징 종류 보기</summary><div class="table-wrap"><table><thead><tr><th>단어</th><th>항목 수</th><th>원문 연결 · 설명 종류</th></tr></thead><tbody>' + "".join(blocks) + "</tbody></table></div></details>" if profiles else '<div class="empty">출처 프로필 파일이 아직 없습니다.</div>'
    return metrics + '<p class="explain">사전의 설명을 출처와 함께 정리했습니다. 같은 한국어 이름이 같은 수어 변이를 보장하지 않으므로, 영상의 필수 동작이나 허용 오차로 자동 확정하지 않습니다. 연결되지 않은 항목도 사전에 없다는 뜻은 아닙니다.</p>' + details


def viewer_payload(predictions, labels):
    rows = []
    for r in (predictions or {}).get("rows", []):
        rows.append({k: r.get(k) for k in ["case_id", "word_id", "side", "person", "frame_id", "pts_us", "features", "top3", "draft_only"]} | {
            "image": local_uri(r.get("image")), "full_image": local_uri(r.get("full_image")),
            "label": labels.get(r.get("word_id"), r.get("word_id", "공개 표본"))})
    return {"rows": rows, "summary": (predictions or {}).get("summary", {})}


CSS = r"""
:root{--bg:#f4f7fa;--ink:#18313e;--muted:#5c6e78;--line:#dce5e9;--brand:#177d7c;--soft:#e4f3ef;--warn:#a96513;--warnbg:#fff4de;--blue:#3977bb}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 'Segoe UI','Malgun Gothic',sans-serif}a{color:#146a91;text-underline-offset:3px}button,select,input{font:inherit}button,select{border:1px solid #b6c8d1;border-radius:8px;background:#fff;color:var(--ink);padding:8px 12px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #4ca3d7;outline-offset:3px}.wrap{max-width:1260px;margin:auto;padding:28px 30px 64px}.hero{background:#173844;color:#fff;padding:42px;border-radius:20px;position:relative;overflow:hidden}.hero .eyebrow{color:#a1e1d9;font-size:13px;font-weight:700;letter-spacing:.08em}.hero h1{font-size:36px;line-height:1.3;margin:10px 0 14px;max-width:820px}.hero p{max-width:850px;color:#dce7eb;margin:0}.hero-meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}.chip{display:inline-block;padding:4px 11px;border-radius:99px;font-size:13px;background:#315361;color:#e8f5f5}.critical{margin:22px 0;background:#fff;border:2px solid #237b91;border-left:8px solid #237b91;padding:22px 25px;border-radius:13px}.critical strong{display:block;font-size:23px;line-height:1.45}.critical p{margin:8px 0 0;color:var(--muted)}nav{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}nav a{background:white;border:1px solid var(--line);border-radius:8px;padding:7px 13px;text-decoration:none;font-size:14px}section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:28px;margin:22px 0;scroll-margin-top:20px}h2{font-size:25px;line-height:1.35;margin:0 0 18px}h3{font-size:18px;line-height:1.45;margin:12px 0}p{margin:10px 0 15px}.section-no{font-size:13px;letter-spacing:.08em;color:var(--brand);font-weight:700}.muted,.explain,small{color:var(--muted)}.explain{font-size:15px}.stage-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.stage{border:1px solid var(--line);border-radius:11px;padding:17px;background:#fafcfd}.stage p{font-size:14px;margin:0;color:var(--muted)}.stage-top{display:flex;justify-content:space-between;align-items:center;gap:8px}.stage-top b{font-size:24px;color:#537a88}.stage-top span{font-size:12px;border-radius:20px;padding:2px 8px;background:var(--warnbg);color:#7c510b}.stage.done .stage-top span{background:var(--soft);color:#206951}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:10px 0 18px}.metrics>div{background:#f0f7f7;padding:17px;border-radius:10px}.metrics strong{display:block;font-size:30px;color:#136766}.metrics span{font-size:14px;color:var(--muted)}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#edf3f5;text-align:left;color:#385664;font-weight:700}th,td{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top;line-height:1.55}td small{display:block;font-size:12px}.table-wrap table{min-width:680px}.source-families{display:inline-block;margin:0 0 3px 10px;color:var(--muted);font-size:12px}details{margin:18px 0 0;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:700;padding:6px 0}.empty{padding:25px;border:1px dashed #c8d5dc;background:#f7fafc;color:var(--muted);border-radius:10px}.note{border-left:4px solid #d1a447;background:var(--warnbg);padding:15px 18px;border-radius:6px;margin:16px 0}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}.two-col>div{min-width:0}ul{margin:8px 0;padding-left:23px}li+li{margin-top:9px}.viewer-controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;background:#f1f6f8;padding:15px;border-radius:10px}.viewer-controls select{max-width:100%}.timeline{display:flex;gap:13px;align-items:center;margin:16px 0}.timeline input{width:100%;accent-color:var(--brand)}.timeline output{white-space:nowrap;font-variant-numeric:tabular-nums}.images{display:grid;grid-template-columns:1.8fr 1fr;gap:15px}.image-box{background:#122932;border-radius:10px;overflow:hidden;color:#dce9ed}.image-box header{padding:9px 13px;font-size:13px;background:#284550}.image-box img{display:block;width:100%;height:370px;object-fit:contain}.image-error{padding:30px;display:none;color:#efdbc1}.frame-meta{padding:12px 0;color:var(--muted);font-size:14px}.feature-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px}.feature-card{border:1px solid var(--line);border-radius:10px;padding:15px}.feature-card h3{font-size:16px;margin:0 0 6px}.feature-state{font-size:20px;font-weight:700}.massbar{height:12px;background:#edf1f3;display:flex;overflow:hidden;border-radius:8px;margin:12px 0}.massbar i{display:block;height:100%}.massbar .yes{background:var(--brand)}.massbar .no{background:#7c99b6}.massbar .unknown{background:#d4ac67}.mass-text{font-size:12px;color:var(--muted)}.top3{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.top3 span{padding:4px 10px;background:#f0f4f7;border-radius:6px;font-size:13px}.links{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.links>a{display:block;border:1px solid var(--line);padding:12px;border-radius:8px;text-decoration:none}.links small{display:block;font-size:12px;margin-top:4px}.footer{font-size:12px;color:var(--muted);overflow-wrap:anywhere;padding:10px 4px}.statusline{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.footprint{font-size:13px;margin-top:15px;color:var(--muted)}.hidden{display:none}
@media(max-width:850px){.stage-grid{grid-template-columns:repeat(2,1fr)}.feature-grid,.two-col{grid-template-columns:1fr}.images{grid-template-columns:1fr}.image-box img{height:310px}.hero{padding:27px}.hero h1{font-size:29px}.wrap{padding:15px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:20px}.links{grid-template-columns:1fr}}
@media(max-width:480px){.stage-grid{grid-template-columns:1fr}.hero h1{font-size:25px}.critical strong{font-size:20px}.timeline{gap:7px}.viewer-controls{align-items:stretch}.viewer-controls select{width:100%}h2{font-size:22px}}
@media print{body{background:white}.wrap{max-width:none;padding:0}.hero{background:white;color:#18313e;border:1px solid #dce5e9}.hero p,.hero .eyebrow{color:#385664}nav,.viewer-controls,.timeline{display:none}section{break-inside:avoid}.stage-grid{grid-template-columns:repeat(3,1fr)}.image-box img{height:230px}details:not([open]){display:none}}
"""


JS = r"""
(() => {
  const data=JSON.parse(document.getElementById('viewer-data').textContent);
  const root=document.getElementById('viewer');
  if(!data.rows.length){root.innerHTML='<div class="empty">공개 KSL 표본의 저장 예측이 아직 없습니다. 최종 집계 뒤 생성하면 이곳에 연구 판독 화면이 표시됩니다.</div>';return;}
  const groups=new Map();
  for(const r of data.rows){const k=String(r.case_id||'미지정');if(!groups.has(k))groups.set(k,[]);groups.get(k).push(r);}
  for(const list of groups.values())list.sort((a,b)=>(a.frame_id??0)-(b.frame_id??0));
  const select=document.getElementById('case-select'),range=document.getElementById('frame-range');
  for(const [k,list]of groups){const r=list[0];const option=document.createElement('option');option.value=k;option.textContent=`${k} · ${r.label} · 기록 ID ${r.person??'미상'} · ${r.side||'손 미지정'} · ${list.length}표본`;select.appendChild(option);}
  const names={index_only:'검지만 폄',open_four:'네 손가락 폄',four_closed:'네 손가락 닫힘'};
  function mass(v){return typeof v==='number'&&Number.isFinite(v)?Math.max(0,Math.min(1,v)):null;}
  function percent(v){const n=mass(v);return n===null?'미확정':(n*100).toFixed(1)+'%';}
  function state(v){if(v===1||v===true||['present','positive','yes','supported','likely_present'].includes(v))return'해당 형태 추정';if(v===0||v===false||['absent','negative','no','likely_absent'].includes(v))return'비해당 형태 추정';if(v===null||v===undefined||['unknown','uncertain','hold','insufficient','abstain'].includes(v))return'판단 보류';return String(v);}
  function image(id,url){const img=document.getElementById(id),err=document.getElementById(id+'-error');img.style.display='none';err.style.display='none';img.onload=()=>{img.style.display='block';err.style.display='none';};img.onerror=()=>{img.style.display='none';err.style.display='block';};if(!url){img.removeAttribute('src');err.textContent=id==='hand-image'?'이 시점은 사용할 손 입력이 없습니다. 손모양 판단을 보류합니다.':'이 시점의 원본 이미지 경로가 없습니다.';err.style.display='block';}else {err.textContent='이미지를 불러오지 못했습니다. 파일 경로를 확인해야 합니다.';img.src=url;}}
  function draw(){const list=groups.get(select.value),i=Number(range.value),r=list[i];document.getElementById('frame-count').textContent=`${i+1} / ${list.length}`;document.getElementById('prev').disabled=i===0;document.getElementById('next').disabled=i===list.length-1;document.getElementById('frame-meta').textContent=`원본 프레임 ${r.frame_id??'미확정'} · ${typeof r.pts_us==='number'?(r.pts_us/1e6).toFixed(3)+'초':'시간 미확정'} · ${r.label} · 촬영자 기준 ${r.side==='R'?'오른손':r.side==='L'?'왼손':r.side||'손 미상'}`;image('full-image',r.full_image);image('hand-image',r.image);
    const cards=document.getElementById('feature-cards');cards.replaceChildren();
    for(const [key,title]of Object.entries(names)){const f=(r.features||{})[key]||{},card=document.createElement('article');card.className='feature-card';const h=document.createElement('h3');h.textContent=title;const s=document.createElement('div');s.className='feature-state';s.textContent=state(f.state);const bar=document.createElement('div');bar.className='massbar';bar.setAttribute('aria-label',`${title}: 해당 ${percent(f.p_pos)}, 비해당 ${percent(f.p_neg)}, 매핑 미정 ${percent(f.p_unknown)}`);for(const [cls,value]of [['yes',f.p_pos],['no',f.p_neg],['unknown',f.p_unknown]]){const part=document.createElement('i');part.className=cls;part.style.width=((mass(value)||0)*100)+'%';bar.appendChild(part);}const text=document.createElement('div');text.className='mass-text';text.textContent=`해당 ${percent(f.p_pos)} · 비해당 ${percent(f.p_neg)} · 매핑 미정 ${percent(f.p_unknown)}`;card.append(h,s,bar,text);cards.appendChild(card);}
    const top=document.getElementById('top3');top.replaceChildren();for(const [j,t]of (r.top3||[]).entries()){const span=document.createElement('span');span.textContent=`${j+1}위 ${t.class_name??'미상'} · ${percent(t.probability)}`;top.appendChild(span);}if(!(r.top3||[]).length)top.textContent='원형 수형 후보가 아직 기록되지 않았습니다.';
    document.getElementById('draft-note').textContent=r.draft_only?'이 표본의 기존 관찰 기록은 AI 초안입니다. 예측과 일치해도 독립적인 정답 확인이 아닙니다.':'이 화면은 저장한 모델 추정의 판독용입니다. 원본을 보면서 확인하며, 교육적 정오 판정으로 사용하지 않습니다.';
  }
  function changeCase(){range.min=0;range.max=groups.get(select.value).length-1;range.value=0;draw();}
  select.addEventListener('change',changeCase);range.addEventListener('input',draw);document.getElementById('prev').addEventListener('click',()=>{range.value=Math.max(0,Number(range.value)-1);draw();});document.getElementById('next').addEventListener('click',()=>{range.value=Math.min(Number(range.max),Number(range.value)+1);draw();});changeCase();
})();
"""


def build(summary, cards, predictions, source_hashes):
    models = summary.get("model_selection", [])
    conditions = summary.get("conditions", [])
    model_rows = [[m.get("name", "미지정"), m.get("kind", "미확정"), pct(m.get("val_accuracy")), pct(m.get("val_macro_f1")), pct(m.get("test_accuracy")), span(m.get("test_range")), f'{m["p95_ms"]:.2f} ms' if number(m.get("p95_ms")) else plain(m.get("p95_ms"))] for m in models]
    condition_rows = [[c.get("name", "미지정"), pct(c.get("test_mean")), span(c.get("test_range")), pct(c.get("macro_f1")), pct(c.get("missing_mean")), pct(c.get("change_mean"))] for c in conditions]
    facts = summary.get("facts", {})
    fact_rows = [[FACT_NAMES.get(k, k), plain(v)] for k, v in facts.items()]
    ksl = summary.get("ksl", {})
    ksl_rows = [[FACT_NAMES.get(k, k), plain(v)] for k, v in ksl.items()]
    labels = {p["word_id"]: p.get("display_label", p["word_id"]) for p in cards.get("profiles", [])}
    payload = viewer_payload(predictions, labels)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    citations = [
        ("Deep Hand · 수동 손모양 자료와 분류 도표", "https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/", "외부 손모양 주석 출처 · 해당 배포 페이지의 CC BY-NC-SA 3.0 조건 확인"),
        ("CC BY-NC-SA 3.0", "https://creativecommons.org/licenses/by-nc-sa/3.0/", "자료 이용 조건 · 운영 배포 자산과 분리"),
        ("MobileNetV3-Small · torchvision", "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html", "기존 비교 기준 · 코드와 학습 자료의 조건은 구분"),
        ("MobileNetV4 · 선택한 timm 가중치", "https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k", "2024 구조 · 선택한 가중치 카드 Apache-2.0"),
        ("TinyNeXt · 저자 공식 구현", "https://github.com/yuffeenn/TinyNeXt", "ICCV 2025 · 공식 저장소 MIT, 실제 가중치·호환성 별도 검산"),
        ("TinyNeXt · 원논문", "https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.html", "일반 이미지 벤치마크와 이번 손모양 실측을 구분"),
        ("한국수어사전", "https://sldict.korean.go.kr/", "표현 설명의 원출처 · 단어별 직접 링크는 위 표에 수록"),
        ("국립국어원 한국수어사전", "https://ksldict.korean.go.kr/kslserv/", "참고 항목과 표현 변이 연결 근거")]
    links = '<div class="links">' + "".join(f'<a href="{h(url)}" target="_blank" rel="noopener noreferrer">{h(title)}<small>{h(detail)}</small></a>' for title, url, detail in citations) + "</div>"
    hashes_html = "<br>".join(f"{h(name)} · {h(value)}" for name, value in source_hashes.items())
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손모양 관측 모델 연구 결과 · WordsInSilence</title><style>{CSS}</style></head><body><main class="wrap">
<header class="hero"><div class="eyebrow">WORDS IN SILENCE · RESEARCH EVIDENCE</div><h1>손모양을 더 잘 관측하기 위한<br>자료·모델·피드백 연결 결과</h1><p>출처 있는 설명과 공개 손모양 주석에서 시작해, 작은 모델들의 실제 성능을 비교하고 DTW 피드백에 연결할 수 있는 근거를 확인합니다.</p><div class="hero-meta"><span class="chip">기존 C·T3-F 보존</span><span class="chip">사용자·멘토 촬영 학습·튜닝 제외</span><span class="chip">공개 자료 연구 판독용</span></div></header>
<div class="critical"><strong>이번 점수는 ‘사진 속 손모양 45종’ 분류 결과입니다.</strong><p>기존 TCN의 ‘한국수어 단어 100종’ 인식률과 다른 과제입니다. 두 점수를 직접 비교해 수어 인식이 좋아지거나 나빠졌다고 해석할 수 없습니다.</p></div>
<p class="statusline">집계 상태: {h(status_name(summary.get('status')))} · 보고 생성: {h(now)}</p>
<section><h2>이번에 확인한 핵심 결과</h2>{list_html(summary.get('findings'))}</section>
<nav aria-label="보고서 목차"><a href="#stages">6단계 진행</a><a href="#sources">출처·정답</a><a href="#models">모델 비교</a><a href="#conditions">네 조건</a><a href="#ksl">KSL·DTW 적용</a><a href="#public-viewer">공개 표본 보기</a><a href="#limits">한계·다음 작업</a></nav>
<section id="stages"><div class="section-no">01 · 작업 범위</div><h2>계획의 어느 단계까지 확인했나</h2>{stage_cards(summary.get('stages', []))}<p class="footprint">연결 코드 구현, 외부 자료의 분류 성능, 한국수어의 교육적 피드백 검증은 각각 별도 결과로 기록합니다.</p></section>
<section id="sources"><div class="section-no">02 · 출처와 정답</div><h2>설명 자료와 학습 정답을 구분했습니다</h2>{source_profiles(cards)}<h3>이번 실험의 실제 조건</h3>{table(['항목','기록한 사실'], fact_rows)}<div class="note">외부 자료의 원래 사람 주석은 45개 수형입니다. ‘검지만 폄·네 손가락 폄·네 손가락 닫힘’은 공식 도표에서 근거가 분명한 부분만 연결한 파생 특징입니다. 새로 사람이 모든 사진을 세 특징으로 판정한 정답과는 구분합니다. ‘네 손가락 닫힘’은 엄지까지 포함한 전체 주먹을 뜻하지 않습니다.</div></section>
<section id="models"><div class="section-no">03 · 관측 모델</div><h2>최근 모델도 같은 공개 자료에서 비교했습니다</h2><p class="explain">출시 연도만으로 좋은 모델을 정하지 않았습니다. 표의 결과는 실제 비교한 후보와 고정 조건의 결론이며, 모든 모델·모든 수어에서 가장 좋다는 뜻은 아닙니다. 고정 특징만 학습한 조건과 모델을 추가 학습한 조건은 ‘조건’ 열에서 구분합니다.</p>{table(['모델','조건','개발 정확도','개발 macro-F1','최종 정확도','3회 범위','CPU p95'],model_rows)}<p class="explain">정확도는 맞힌 표본 비율입니다. macro-F1은 수형별 점수를 평균해 적은 수형도 살펴보는 지표입니다. 등장한 수형과 사람별 표본 수를 함께 확인해야 합니다. CPU 시간은 해당 측정 범위의 값이며 웹캠 전체 처리 속도를 뜻하지 않습니다.</p></section>
<section id="conditions"><div class="section-no">04 · 정보의 효과</div><h2>관절·영상·시간 정보를 나눠 확인했습니다</h2><p class="explain">J_T: 관절+앞뒤 표본 · R_T: RGB+앞뒤 표본 · RJ_F: 관절+RGB의 현재 표본 · RJ_T: 관절+RGB+앞뒤 표본. 앞뒤 자료는 실제 존재하는 원본 인접 프레임만 사용하며, 없는 사진을 복제해 시간 정보로 만들지 않습니다.</p>{table(['조건','최종 평균 정확도','3회 범위','macro-F1','관절 미검출 18장','라벨 변화 근처 19장'],condition_rows)}<div class="note">전체 분모는 매번 382장입니다. ‘인접 라벨 변화’는 5클립의 6개 변화 주변 19장으로, 수어의 필수 단계가 바뀌었다는 정답이 아닙니다. 각 표의 범위는 3 seed 범위이며 통계적 신뢰구간이 아닙니다.</div></section>
<section id="ksl"><div class="section-no">05 · 프로젝트 연결</div><h2>한국수어와 DTW에 무엇을 연결했나</h2>{table(['확인 항목','결과와 적용 범위'],ksl_rows,'한국수어 전이와 DTW 적용의 최종 집계는 아직 기록되지 않았습니다.')}<p class="explain">수형 관측이 개선돼도 좌우 손의 동일성, 가림, 정상 변이, 동작 시점 정렬이 함께 검증돼야 교정을 제시할 수 있습니다. 같은 모델끼리 비슷한 값을 냈다는 사실만으로 사용자의 행동이 옳다고 판정하지 않습니다.</p></section>
<section id="public-viewer"><div class="section-no">06 · 직접 확인</div><h2>공개 KSL 표본의 모델 추정을 살펴보세요</h2><p class="explain">원본 장면과 손 입력을 함께 보고 프레임별 추정이 어떤 모양에 반응하는지 확인하는 연구 화면입니다. 실제 사용자의 테스트 화면이나 운영 피드백 후보가 활성화된 상태는 아닙니다.</p><div id="viewer"><div class="viewer-controls"><label for="case-select">공개 구간</label><select id="case-select" aria-label="공개 구간 선택"></select><button id="prev" type="button">이전 표본</button><button id="next" type="button">다음 표본</button></div><div class="timeline"><input id="frame-range" type="range" min="0" max="0" value="0" step="1" aria-label="저장된 표본 이동"><output id="frame-count" for="frame-range"></output></div><div class="images"><div class="image-box"><header>원본 공개 장면</header><img id="full-image" alt="선택한 공개 구간의 원본 장면"><div class="image-error" id="full-image-error">이 원본 이미지를 불러오지 못했습니다. 파일 경로를 확인해야 합니다.</div></div><div class="image-box"><header>모델이 본 손 입력</header><img id="hand-image" alt="선택한 공개 장면의 손 입력"><div class="image-error" id="hand-image-error">이 손 입력 이미지를 불러오지 못했습니다. 파일 경로를 확인해야 합니다.</div></div></div><div class="frame-meta" id="frame-meta"></div><div class="feature-grid" id="feature-cards"></div><h3>원형 45수형 후보</h3><div class="top3" id="top3"></div><p class="explain" id="draft-note"></p></div><div class="note">표시한 비율은 모델의 확률 질량입니다. 사용자 동작의 정답률이나 검증된 교정 확률이 아닙니다. ‘매핑 미정’은 원래 수형과 해당 특징의 대응을 정하지 않았다는 뜻이며, 손이 가려졌다는 직접 판정이 아닙니다. 세 특징이 모두 해당하지 않아도 전체 손모양을 안다고 볼 수 없습니다.</div></section>
<section id="limits"><div class="section-no">07 · 남은 판단</div><h2>확인된 한계와 다음 작업</h2><div class="two-col"><div><h3>아직 해결하거나 검증해야 할 점</h3>{list_html(summary.get('limitations'))}</div><div><h3>근거를 갖추기 위한 다음 작업</h3>{list_html(summary.get('next_actions'))}</div></div><p class="explain">이 페이지는 공개 자료의 연구 결과와 저장 추정을 보여줍니다. 운영 적용 여부는 위 단계별 상태와 검증 결과로 확인합니다.</p></section>
<section id="references"><div class="section-no">08 · 재확인할 근거</div><h2>원출처·모델·이용 조건</h2>{links}<p class="explain">외부 연구 자료와 모델의 이용 조건을 각각 보존합니다. 자료의 비상업 조건을 코드 라이선스로 대신 판단하지 않습니다.</p></section><footer class="footer">입력 파일 해시<br>{hashes_html}<br>보고 생성기는 저장 결과를 표시하며 모델 추론·학습을 실행하지 않습니다.</footer>
</main><script type="application/json" id="viewer-data">{safe_json(payload)}</script><script>{JS}</script></body></html>'''
    profiles = cards.get("profiles", [])
    linked = sum(bool(p.get("source_variants")) for p in profiles)
    entries = sum(len(p.get("source_variants", [])) for p in profiles)
    claims = sum(len(v.get("feature_claims", [])) for p in profiles for v in p.get("source_variants", []))
    lines = ["WordsInSilence — 손모양 관측 모델 연구 결과", "", "이번 점수는 외부 손모양45종 분류 결과이며 기존 TCN의 한국수어100단어 정확도와 다른 과제다.", "기존 C·T3-F 보존, 사용자·멘토 촬영은 학습·튜닝에서 제외. 공개자료 연구 판독용.", "집계 상태: " + status_name(summary.get("status")), f"출처 프로필 {len(profiles)}개 / 연결 {linked}개 / 사전 항목 후보 {entries}개 / 출처 특징 설명 {claims}개. 영상 변이·허용경계는 자동 확정하지 않음.", "", "6단계 상태"]
    for s in summary.get("stages", []):
        lines.append(f'{s.get("n", "?")}. {s.get("title", "")} — {status_name(s.get("status"))}: {plain(s.get("detail"))}')
    lines += ["", "핵심 결과"] + ["- " + plain(x) for x in summary.get("findings", [])]
    lines += ["", "모델 비교: 개발 정확도 / 개발 macro-F1 / 최종 정확도 / 3회 범위 / CPU p95"]
    lines += [" | ".join(map(str, r)) for r in model_rows] or ["미확정"]
    lines += ["", "네 조건: 최종 평균 / 범위 / macro-F1 / 관절미검출 / 인접라벨변화"]
    lines += [" | ".join(map(str, r)) for r in condition_rows] or ["미확정"]
    lines += ["", "실제 조건"] + [f"{k}: {v}" for k, v in fact_rows]
    lines += ["", "한국수어·DTW 연결"] + [f"{k}: {v}" for k, v in ksl_rows]
    lines += ["", "남은 한계"] + ["- " + plain(x) for x in summary.get("limitations", [])]
    lines += ["", "다음 작업"] + ["- " + plain(x) for x in summary.get("next_actions", [])]
    lines += ["", "공개 표본 판독: " + str(len(payload["rows"])) + "개 저장 예측. 모델의 확률 질량이며 정답률·교정 확률이 아님. 매핑 미정 질량과 가림 판정을 혼동하지 않는다.", "", "출처"] + [f"{title}: {url} — {detail}" for title, url, detail in citations]
    lines += ["", "입력 해시"] + [f"{k}: {v}" for k, v in source_hashes.items()]
    return page, "\n".join(lines) + "\n"


def generate(summary_path, output_dir):
    if not summary_path.exists():
        raise FileNotFoundError("SUMMARY.json is required; no invented final results are generated")
    summary = read(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("SUMMARY must be an object")
    cards_path = U / "SOURCE_FEATURE_CARDS.json"
    predictions_path = U / "KSL_TRANSFER_PREDICTIONS.json"
    cards = read(cards_path, {})
    predictions = read(predictions_path, {})
    hashes = {summary_path.name: sha(summary_path)}
    for p in (cards_path, predictions_path):
        if p.exists():
            hashes[p.name] = sha(p)
    page, text = build(summary, cards, predictions, hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RESULT_KO.html").write_text(page, encoding="utf-8")
    (output_dir / "RESULT_KO.txt").write_text(text, encoding="utf-8")
    receipt = {"status": "GENERATED_FROM_SAVED_ROOT_SUMMARY", "input_hashes": hashes, "builder_sha256": sha(Path(__file__)), "report_html_sha256": sha(output_dir / "RESULT_KO.html"), "report_txt_sha256": sha(output_dir / "RESULT_KO.txt"), "viewer_rows": len(predictions.get("rows", [])), "new_inference": False, "training": False, "operating_activation": False}
    (output_dir / "REPORT_BUILD_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"html": str(output_dir / "RESULT_KO.html"), "txt": str(output_dir / "RESULT_KO.txt"), "viewer_rows": receipt["viewer_rows"]}, ensure_ascii=False))


def self_test():
    # Synthetic formatting fixture only, never placed at the final report location.
    sample = {"status": "SYNTHETIC_BUILDER_TEST_ONLY", "stages": [{"n": 1, "title": "<검사>", "status": "complete", "detail": "문자 & 이스케이프"}], "model_selection": [{"name": "fixture", "kind": "synthetic", "val_accuracy": .5, "val_macro_f1": .25, "test_accuracy": None, "test_range": [.4, .6], "p95_ms": 12.3}], "conditions": [], "facts": {"user_mentor_data_used": False}, "limitations": ["테스트용 가상값"], "next_actions": [], "ksl": {}}
    cards = {"profiles": [{"word_id": "X", "display_label": "가상검사", "source_variants": []}]}
    predictions = {"rows": [{"case_id": "SYNTHETIC", "word_id": "X", "side": "R", "person": "fixture", "frame_id": 1, "pts_us": 40000, "features": {"index_only": {"state": None, "p_pos": .2, "p_neg": .3, "p_unknown": .5}}, "top3": [{"class_name": "</script><script>alert(1)</script>", "probability": .3}], "draft_only": True}]}
    page, text = build(sample, cards, predictions, {"fixture": "synthetic"})
    assert "50.00%" in page and "25.00%" in page and "미확정" in page
    assert "&lt;검사&gt;" in page and "<검사>" not in page
    assert "</script><script>alert(1)</script>" not in page
    assert len(page.split('<article class="stage ')) - 1 == 6
    assert "getUserMedia" not in page and "fetch(" not in page
    assert "한국수어100단어" in text
    p2, _ = build({}, {}, {}, {})
    assert "미확정" in p2 and '"rows": []' in p2
    directory = U / "research" / "report_builder_selftest"
    directory.mkdir(exist_ok=True)
    (directory / "SYNTHETIC_ONLY.html").write_text(page, encoding="utf-8")
    (directory / "VIEWER_SCRIPT.js").write_text(JS, encoding="utf-8")
    result = {"status": "PASS", "checks": ["fraction_percent_display", "pending_values", "HTML_escape", "embedded_JSON_script_escape", "six_stages", "no_camera_or_network", "empty_prediction_placeholder"], "final_report_generated": False, "fixture_only": True}
    (directory / "SELF_TEST.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=U / "SUMMARY.json")
    parser.add_argument("--output", type=Path, default=U)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else generate(args.summary, args.output)
