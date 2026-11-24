"""Build a local viewer index and HTML only; never alter scientific results.

Run after EVALUATION.json and RESULT_MANIFEST.json exist. The local server must
serve /data/{id} as the corresponding decoded result JSON and /video/{alias}
with byte-range support. No full result records are embedded in the report.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json

U = Path(__file__).resolve().parent

COMPONENTS = {
    "L_wrist": {"label": "왼손목 · 몸 기준 위치", "axes": ["몸 기준 x", "몸 기준 y"]},
    "R_wrist": {"label": "오른손목 · 몸 기준 위치", "axes": ["몸 기준 x", "몸 기준 y"]},
    "hands_delta": {"label": "두 손목 사이의 상대 위치", "axes": ["손목 사이 Δx", "손목 사이 Δy"]},
    "L_to_R_forearm": {"label": "왼손 · 오른쪽 전완과의 관계", "axes": ["팔 방향 비율", "팔 옆 방향 비율"]},
    "R_to_L_forearm": {"label": "오른손 · 왼쪽 전완과의 관계", "axes": ["팔 방향 비율", "팔 옆 방향 비율"]},
}
CONTROLS = {
    "identity": "원본 그대로",
    "speed_075": "저장된 느린 속도 변형",
    "speed_125": "저장된 빠른 속도 변형 · 표본 삭제 포함",
    "hold_insert5": "같은 자세를 5프레임 더 유지",
    "small_common_coordinate_drift": "작은 공통 좌표 흔들림",
    "reverse_first_parent": "첫 상위 구간의 진행 순서 뒤집기",
    "omit_short_core_5f": "짧은 관측 구간 5프레임 삭제",
    "swap_adjacent_core_blocks": "인접 관측 구간 순서 교환",
    "right_hand_unobserved": "오른손 관측 가림",
    "right_middle_gap5": "오른손 중간 5프레임 공백",
}


def read(name):
    return json.loads((U / name).read_text(encoding="utf-8-sig"))


def build():
    evaluation, manifest, tasks = read("EVALUATION.json"), read("RESULT_MANIFEST.json"), read("TASKS.json")
    scores = {r["id"]: r for r in evaluation["rows"]}
    if set(scores) != {r["id"] for r in manifest["results"]}:
        raise ValueError("Evaluation and result manifest IDs differ")
    clips = {alias: {key: row[key] for key in ("alias", "label", "word_id", "person", "role")}
             for alias, row in tasks["rows"].items()}
    parents = {}
    for event in tasks["natural"]:
        parents.setdefault(event["reference"] + "::" + event["parent"]["segment_id"], event["parent"])
    for case in tasks["controls"] + tasks["missing_controls"]:
        for parent in case["parents"]:
            parents.setdefault(case["alias"] + "::" + parent["segment_id"], parent)
    index = []
    for rec in manifest["results"]:
        row = {key: rec.get(key) for key in ("id", "kind", "query", "reference", "role", "word_id", "component", "control", "parent_id")}
        row.update(available=True, complete=scores[rec["id"]]["complete"])
        index.append(row)
    for case in tasks["missing_controls"]:
        for parent in case["parents"]:
            for component in tasks["component_names"]:
                index.append(dict(id="missing::" + "::".join((case["alias"], case["control"], parent["segment_id"], component)),
                    kind="control", query=case["alias"], reference=case["alias"], role=case["role"],
                    word_id=clips[case["alias"]]["word_id"], component=component, control=case["control"],
                    parent_id=parent["segment_id"], available=False, complete=False, status="NO_SAVED_CONTROL"))
    summary = {key: evaluation[key] for key in ("natural", "controls", "timing", "decision", "technical_failure_count")}
    summary["control_cases"] = dict(possible=tasks["control_case_denominator"], computed=len(tasks["controls"]),
                                    missing=len(tasks["missing_controls"]),
                                    component_opportunities=tasks["control_component_opportunities"])
    data = dict(schema="wis_trajectory_view_index_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                summary=summary, clips=clips, components=COMPONENTS, controls=CONTROLS,
                parents={key: {name: value.get(name) for name in ("segment_id", "core", "description_ko")}
                         for key, value in parents.items()}, records=index,
                default=dict(query="P048", reference="P045", component="R_wrist"),
                endpoints=dict(data="/data/{id}", video="/video/{alias}"),
                source_sha256={name: hashlib.sha256((U / name).read_bytes()).hexdigest()
                               for name in ("EVALUATION.json", "RESULT_MANIFEST.json", "TASKS.json")})
    natural = evaluation["natural"]
    total_controls = summary["control_cases"]
    esc = html.escape
    page = TEMPLATE.replace("{{NATURAL_COMPLETE}}", str(natural["complete"]))
    page = page.replace("{{NATURAL_UNKNOWN}}", str(natural["unknown"]))
    page = page.replace("{{CONTROL_COMPONENTS}}", str(total_controls["component_opportunities"]))
    page = page.replace("{{TECHNICAL_STATUS}}", esc("고정 기술검사 통과" if evaluation["decision"]["technical_checks_pass"] else "고정 기술검사에서 미충족 항목 발견"))
    page = page.replace("{{RECORD_COUNT}}", str(manifest["geometry_record_count"]))
    natural_rows = [row for row in evaluation["rows"] if row["kind"] == "natural" and row["complete"]]
    page = page.replace("{{AMBIGUOUS_COUNT}}", str(sum(bool(row["optimal_path_ambiguous"]) for row in natural_rows)))
    controls = {row["control"]: row for row in evaluation["controls"]}
    page = page.replace("{{HOLD_ZERO}}", str(controls["hold_insert5"]["zero_distance"]))
    page = page.replace("{{HOLD_EDITED}}", str(sum(row["kind"] == "control" and row["control"] == "hold_insert5" and row["complete"] and row["intended_edit_in_parent"] for row in evaluation["rows"])))
    page = page.replace("{{SPEED_POSITIVE}}", str(controls["speed_125"]["positive_distance"]))
    page = page.replace("{{SWAP_OUTSIDE}}", str(controls["swap_adjacent_core_blocks"]["relevant_edit_outside_crop"]))
    (U / "VIEW_DATA.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (U / "RESULT_KO.html").write_text(page, encoding="utf-8")
    print(json.dumps(dict(status="VIEW_BUILT_ONLY", html="RESULT_KO.html", index="VIEW_DATA.json",
                          records=len(index), full_results_embedded=False), ensure_ascii=False))


TEMPLATE = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>멈춤과 움직임 · 이전 대응과 새 궤적 비교</title><link rel="stylesheet" href="viewer.css"><script defer src="viewer.js"></script></head>
<body><a class="skip" href="#comparison">비교 도구로 이동</a>
<header class="hero"><div class="eyebrow">WORDS IN SILENCE · 공개 자료 연구 비교</div>
<h1>멈춘 시간은 달라도,<br>움직인 경로는 같은가?</h1>
<p class="lead">기존 비교는 그대로 두고, <strong>주요 동작에서 손이 지나간 길과 순서</strong>를 따로 살펴보는 기능을 추가했습니다.</p>
<p class="hero-note"><strong>이번 결론:</strong> 새 비교 기능의 계산은 검증했습니다. 실제 수어 피드백이 이전보다 정확해졌는지는 아직 확인되지 않았습니다. 기존 C·T3-F와 운영 기능은 유지했습니다.</p>
<a class="jump" href="#comparison">영상과 경로 직접 비교하기 ↓</a></header>
<main>
<section aria-labelledby="change-title"><div class="section-heading"><span class="section-number">01</span><h2 id="change-title">무엇이 바뀌었나요?</h2></div>
<div class="flow">
<article><span class="pill">이전 흐름</span><h3>기존 경로에서 차이를 읽기</h3><p>이미 정해진 영상 간 대응에서 좌표 차이의 <strong>평균과 최대</strong>를 봅니다. 짧은 차이는 평균에서 작아질 수 있고, 관측이 있어도 같은 동작에 대응했다는 보장은 없습니다.</p></article>
<article><span class="pill">이번 추가</span><h3>경로 전체의 가장 큰 차이 보기</h3><p>같은 자세가 반복되는 정지는 허용하면서, 모든 점을 순서대로 지나는 대응 중 <strong>가장 큰 좌표 차이를 최소화</strong>합니다. 기존 경로와 별개인 연구용 궤적 경로입니다.</p></article>
<article><span class="pill">해결 범위와 한계</span><h3>공백은 잇지 않고 보류</h3><p>동일한 자세의 반복 길이에는 영향을 받지 않습니다. 관측 공백, 같은 공간 경로의 표본 삭제, 대칭적인 순서는 여전히 불명확할 수 있습니다. <strong>교육적 오류 여부는 판정하지 않습니다.</strong></p></article>
</div>
<div class="notice"><strong>숫자가 낮아졌다고 정확도가 좋아진 것은 아닙니다.</strong> 기존 평균·기존 최대·이번 최소 최대거리는 서로 다른 집계입니다. 새 값이 기존 최대보다 작거나 같은 것은 최소화 계산의 성질이며, 올바른 단계 대응이나 수어 수행의 증명이 아닙니다.</div>
</section>
<section aria-labelledby="scope-title"><div class="section-heading"><span class="section-number">02</span><h2 id="scope-title">무엇을 비교했나요?</h2></div>
<div class="scope-grid"><div><strong>16</strong><span>공개 정상 영상</span></div><div><strong>4</strong><span>표현</span></div><div><strong>36</strong><span>개별 영상 비교 쌍</span></div><div><strong>117</strong><span>참고 상위 구간 기회</span></div><div><strong>585</strong><span>구간 × 좌표 성분</span></div></div>
<p>자연 비교의 585성분 중 <strong>{{NATURAL_COMPLETE}}개는 양쪽 전체 관측</strong>, <strong>{{NATURAL_UNKNOWN}}개는 전체 거리 보류</strong>입니다. 보이는 부분과 원래 누락은 함께 남겼습니다.</p>
<p>기술 통제는 <strong>160사례 중 158계산·2저장 자료 없음</strong>입니다. 상위 구간과 5좌표 성분으로 나눈 전체 기회는 {{CONTROL_COMPONENTS}}개입니다. 저장 결과 {{RECORD_COUNT}}개는 선택할 때 한 건씩 불러옵니다.</p>
<p class="small">{{TECHNICAL_STATUS}} · 기하 입력은 seed와 무관해 한 번 계산했습니다. 기존 3seed의 손모양·양축 근거는 유지했습니다. 같은 개발 자료의 반복 확인이며 새로운 사람 대상의 독립 교육 성능 평가가 아닙니다.</p>
<div class="findings"><article><h3>정지 반복은 거리 0 유지</h3><p>정지 삽입의 완전 관측 {{HOLD_ZERO}}성분은 거리 0입니다. 실제 편집 구간에 해당하는 것은 {{HOLD_EDITED}}개이며 나머지는 미편집 구간입니다. 기존 대응도 거리 0이어서, 이전 대비 성능 향상으로 볼 수 없습니다.</p></article><article><h3>정상 속도에도 차이가 생김</h3><p>표본을 삭제한 빠른 속도 변형은 완전 관측 {{SPEED_POSITIVE}}성분 모두 양수 거리였습니다. 단순 시간 변화와 좌표 표본 차이를 아직 완전히 분리하지 못합니다.</p></article><article><h3>대표 경로는 정답이 아님</h3><p>자연 비교의 완전 관측 {{NATURAL_COMPLETE}}개 중 {{AMBIGUOUS_COUNT}}개는 같은 최소값의 경로가 여러 개입니다. 순서 교환 통제의 {{SWAP_OUTSIDE}}성분은 관련 변형이 기존 비교 구간 밖에도 남았습니다.</p></article></div>
</section>
<section id="comparison" aria-labelledby="compare-title"><div class="section-heading"><span class="section-number">03</span><h2 id="compare-title">영상과 경로를 직접 살펴보기</h2></div>
<p class="notice"><strong>이렇게 보세요.</strong> 처음에는 ‘잘하다’ 영상 두 개가 나옵니다. 아래의 <strong>‘기존 대응 경로’와 ‘이번 궤적 경로’</strong>를 번갈아 누르고 슬라이더를 움직여 보세요. 같은 손동작의 어느 순간끼리 연결했는지 영상과 그래프가 함께 바뀝니다. ‘이전/다음’ 버튼은 원본 영상 한 프레임이 아니라 선택한 비교 경로의 한 칸을 이동합니다.</p>
<div class="tabs" role="tablist" aria-label="비교 종류"><button id="tab-natural" role="tab" aria-selected="true" aria-controls="viewer-panel">실제 정상 영상끼리</button><button id="tab-control" role="tab" aria-selected="false" aria-controls="viewer-panel">저장 좌표 기술 통제</button></div>
<div id="viewer-panel" role="tabpanel" aria-labelledby="tab-natural">
<p id="kind-explanation">같은 표현의 다른 사람을 개별 참고와 비교합니다. 정상 영상 사이의 차이를 교육적 오답으로 해석하지 않습니다.</p>
<div class="selectors"><label id="pair-field">비교할 두 영상<select id="pair-select"></select></label><label id="alias-field" hidden>원본 영상<select id="alias-select"></select></label><label id="control-field" hidden>기술 변형<select id="control-select"></select></label><label>살펴볼 주요 구간<select id="parent-select"></select></label><label>살펴볼 위치 관계<select id="component-select"></select></label></div>
<div id="load-status" class="load-status" role="status" aria-live="polite">비교 목록을 불러오는 중입니다.</div>
<div id="record-panel" hidden>
<div class="scope-note"><h3 id="record-title"></h3><p id="parent-description"></p><p class="small" id="scope-detail"></p><p class="small">참고 구간은 표본 영상 관측에 따른 물리적 범위 초안입니다. 모든 프레임의 정답 단계나 필수 언어 동작을 뜻하지 않습니다.</p></div>
<div class="metric-grid"><article><h4>기존 대응 · 평균</h4><strong id="old-mean">—</strong><p>공통 관측이 있는 대응들의 평균</p></article><article><h4>기존 대응 · 최대</h4><strong id="old-max">—</strong><p>같은 기존 대응에서 가장 큰 차이</p></article><article><h4>이번 궤적 · 최소 최대거리</h4><strong id="new-distance">—</strong><p>모든 점을 순서대로 지나는 최소 병목</p></article></div>
<p id="support-detail" class="small"></p><p id="candidate-note" class="notice compact"></p>
<div class="videos"><figure id="query-video-wrap"><figcaption id="query-video-title">비교 영상</figcaption><video id="query-video" controls muted preload="metadata" playsinline></video><p id="query-video-time" class="small"></p></figure><figure id="reference-video-wrap"><figcaption id="reference-video-title">참고 영상</figcaption><video id="reference-video" controls muted preload="metadata" playsinline></video><p id="reference-video-time" class="small"></p></figure></div>
<p id="video-note" class="small"></p>
<div class="path-toolbar"><div class="mode-buttons" role="group" aria-label="표시할 대응 경로"><button id="mode-baseline" aria-pressed="true">기존 대응 경로</button><button id="mode-candidate" aria-pressed="false">이번 궤적 경로</button></div><button id="step-back" aria-label="이전 대응">← 이전</button><button id="step-next" aria-label="다음 대응">다음 →</button></div>
<label class="slider-label" for="path-slider">선택한 경로의 대응 위치 <output id="slider-position"></output></label><input id="path-slider" type="range" min="0" max="0" value="0" step="1">
<div id="frame-detail" class="frame-detail"></div>
<div class="plot-grid"><figure><figcaption>공간 경로 <span class="legend"><i class="query-line"></i>비교 좌표 <i class="reference-line"></i>참고 좌표</span></figcaption><svg id="xy-plot" viewBox="0 0 680 400" role="img" aria-label="두 좌표 궤적, 공백을 잇지 않음"></svg><p class="small" id="xy-caption"></p></figure><figure><figcaption>시간 대응 <span class="legend"><i class="baseline-line"></i>기존 경로 <i class="candidate-line"></i>이번 경로</span></figcaption><svg id="time-plot" viewBox="0 0 680 400" role="img" aria-label="기존 대응과 이번 궤적 경로의 시간 관계"></svg><p class="small">선이 지나간다는 것은 계산상 대응입니다. 같은 물리 단계나 정확한 수행의 확인이 아닙니다.</p></figure></div>
<div id="control-detail" hidden><h3>변형 좌표와 원본 프레임의 관계</h3><p class="notice compact">아래 자료는 저장된 특징 행의 순서·값·mask를 바꾼 기술 통제입니다. 변형 동작을 수행한 새 영상이나 RGB를 다시 추출한 결과가 아닙니다. 표시 영상은 원본 참고 한 개뿐입니다.</p><svg id="source-map" viewBox="0 0 1000 300" role="img" aria-label="변형 행과 원본 프레임 대응 지도"></svg><p id="edit-detail" class="small"></p></div>
<details><summary>관측 공백과 원자료 범위</summary><div id="missing-detail"></div><p class="small" id="record-id"></p></details>
</div></div></section>
<section aria-labelledby="controls-title"><div class="section-heading"><span class="section-number">04</span><h2 id="controls-title">기술 통제의 전체 분모</h2></div><p>0과 양수는 기하 거리의 결과입니다. 정상·오류 분류나 누락 검출률이 아닙니다. 표본 삭제와 기존 경로의 범위 선택 때문에 변형된 부분이 비교 구간 밖에 남을 수 있습니다.</p><div class="table-scroll"><table id="control-table"><caption>모든 기술 통제와 미지원 기회</caption><thead><tr><th>기술 통제</th><th>전체 성분</th><th>전체 관측</th><th>관측 보류</th><th>저장 없음</th><th>거리 0</th><th>양수 거리</th><th>관련 변형이 구간 밖</th></tr></thead><tbody></tbody></table></div></section>
<section class="limits" aria-labelledby="limits-title"><h2 id="limits-title">이 화면으로 알 수 있는 것과 남은 것</h2><ul><li>동일한 관측 자세의 연속 반복은 거리를 바꾸지 않습니다. 임의 표본 삭제나 모든 속도 변화까지 같은 값이라고 보장하지 않습니다.</li><li>관측된 짧은 공간 차이는 최대거리와 원시 궤적에 남습니다. 그 차이가 고쳐야 할 수어 오류인지는 별도 근거가 필요합니다.</li><li>중간 가림은 연결하지 않습니다. 보류와 작은 차이를 정상 판정으로 바꾸지 않습니다.</li><li>기존 global 경로, 원자료, C·T3F 및 운영 모델은 바뀌지 않았습니다. 추가 학습·모델 추론은 없습니다.</li></ul></section>
</main><footer>WordsInSilence · 정지 불변 궤적 연구 비교 · 로컬 공개 자료 뷰어 <span id="build-info"></span></footer>
<noscript><p class="notice">상단 설명은 읽을 수 있습니다. 영상·경로의 선택과 동기화에는 JavaScript가 필요합니다.</p></noscript>
</body></html>'''


if __name__ == "__main__":
    build()
