"""Independent explicit-span verification: only BC and CB are forwarded once.

No candidate helper/module import. BB/CC logits are copied from the closed prior
result only after their official inputs have independently matched exactly.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from independent_reference import (FIELDS, array_hash, exact, load_observation, need,
                                   pattern, prepare, rank, sha, state_digest)

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main_state_encoding(state):
    # The new runner records this encoding; old unit's digest uses a different
    # documented encoding. Compute both independently rather than comparing hashes
    # from different encodings or importing candidate code.
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode() + b"\0" + array_hash(state[name].detach().cpu().numpy()).encode())
    return digest.hexdigest()


def main():
    out_json, out_md = HERE / "INDEPENDENT_RESULT.json", HERE / "INDEPENDENT_RESULT.md"
    need(not out_json.exists() and not out_md.exists(), "Exclusive independent output already exists")
    cfg_path = HERE / "PREPARED_CONFIG.json"
    cfg = read(cfg_path)
    result_path = Path(cfg["output_dir"]) / "RESULT.json"
    result = read(result_path)
    need(result["status"] == "TECHNICAL_PASS_DIAGNOSTIC_ONLY", "Main result not completed")
    need(result["hashes_before"] == result["hashes_after"], "Main source mutation")
    need(cfg["full_frames"] == 78 and cfg["forwards"] == cfg["independent_forwards"] == 2, "Frozen scope")
    conditions = cfg["diagonals"] + cfg["cross_conditions"]
    need([(c["id"], c["observation"], c["span"]) for c in conditions] ==
         [("BB", "B", [12, 74]), ("CC", "C", [12, 69]), ("BC", "B", [12, 69]), ("CB", "C", [12, 74])],
         "Frozen conditions changed")
    protected = dict(cfg["pins"])
    for name, filename in (("config", "PREPARED_CONFIG.json"), ("runner", "span_factorial.py"),
                           ("test", "test_span_factorial.py"), ("frozen", "FROZEN_PLAN.md"),
                           ("receipt", "SYNTHETIC_CHECK.json")):
        protected[str(HERE / filename)] = result["hashes_before"][name]
    for p in (result_path, Path(__file__), HERE / "independent_reference.py"):
        protected[str(p)] = sha(p)
    for path, expected in protected.items():
        need(sha(path) == expected, "Source hash mismatch: " + path)
    prev, audit = read(cfg["previous_result"]), read(cfg["previous_independent"])
    need(audit["status"] == "PASS" and audit["max_abs_logit_difference"] == 0.0, "Previous independent status")
    need(prev["decision"]["B_correct"] == 18 and prev["decision"]["C_correct"] == 17 and prev["decision"]["denominator"] == 18,
         "Previous full18 result changed")
    prior_rows = [r for r in prev["rows"] if r["clip_id"] == cfg["clip_id"]]
    need(len(prior_rows) == 1, "Prior clip not unique")
    prior = prior_rows[0]
    need(prior["label"] == cfg["label"], "Previous truth identity")
    classes = prev["classes"]
    need(len(classes) == len(set(classes)) == 93 and result["classes"] == classes, "Full93 class order")
    model_record = result["model"]
    need(model_record["payload_state_sha"] == model_record["state_before"] == model_record["state_after"], "Main model mutated")
    obs = {arm: load_observation(cfg["observations"][arm]) for arm in ("B", "C")}
    obs_hash = {arm: {"obs_"+key: array_hash(a) for key, a in value.items()} for arm, value in obs.items()}
    need(obs_hash == result["obs_hashes"], "Saved observation identity")
    for arm in obs:
        need(all(len(obs[arm][key]) == 78 for key in FIELDS), "Must use full saved observations")
        need(np.array_equal(obs[arm]["frame_id"], np.arange(78)), "Original frame IDs")
    for key in ("timestamp_us", "frame_id"):
        need(exact(obs["B"][key], obs["C"][key]), "B/C source time mismatch")
    for key in ("xy_px", "usable", "observed", "conf"):
        need(exact(obs["B"][key][:, :33], obs["C"][key][:, :33]), "B/C pose mismatch")
    for values in obs.values():
        for a in values.values():
            a.setflags(write=False)

    # Official preprocessing only, before importing/constructing the model.
    sys.path.insert(0, cfg["pilot_src"])
    from pilot import data, web_input
    for module in (data, web_input):
        path = str(Path(module.__file__).resolve())
        match = [p for p in cfg["pins"] if Path(p).resolve() == Path(path)]
        need(len(match) == 1 and sha(path) == cfg["pins"][match[0]], "Official preprocessing import unpinned")
    prepared, inputs = {}, {}
    cells = result["cells"]
    need([c["id"] for c in cells] == [c["id"] for c in conditions], "Main cell order/count")
    saved = {c["id"]: c for c in cells}
    reused_logits = {}
    for condition in conditions:
        cid = condition["id"]
        prepared[cid], inputs[cid] = prepare(obs[condition["observation"]], condition["span"],
                                            web_input.to_native_like, data.build_channels, data.collate)
        need(inputs[cid] == saved[cid]["input"], "Main input/scale mismatch: " + cid)
        need(saved[cid]["observation"] == condition["observation"] and saved[cid]["span"] == condition["span"], "Cell identity")
        need(saved[cid]["pad_all_one"] is True and bool(np.all(prepared[cid][1] == 1)), "Real frames marked as padding")
        need(saved[cid]["reused"] is (cid in ("BB", "CC")), "Wrong reused/forwarded status")
        if "previous_arm" in condition:
            prior_arm = prior[condition["previous_arm"]]
            need(inputs[cid] == prior_arm["input"], "Diagonal previous input mismatch: " + cid)
            logits = np.asarray(prior_arm["logits"], dtype=np.float64)
            need(rank(logits, classes, cfg["label"]) == prior_arm["score"], "Diagonal saved ranking mismatch")
            need(np.array_equal(logits, np.asarray(saved[cid]["logits93"])), "Diagonal logits not reused exactly")
            reused_logits[cid] = logits
    tail = list(range(70, 75))
    expected_tail = {"rows": tail, "frame_id": obs["C"]["frame_id"][tail].tolist(),
                     "timestamp_us": obs["C"]["timestamp_us"][tail].tolist(),
                     "usable_hand_true_count": [int(obs["C"]["usable"][i, 33:].sum()) for i in tail],
                     "included_in_CB": set(tail) <= set(inputs["CB"]["frame_id"]), "raw_unchanged": True}
    need(expected_tail == result["c_longtail"], "C tail record mismatch")
    need(expected_tail["usable_hand_true_count"] == [0]*5, "Unexpected C tail hand information")

    # Exactly one load and two cross forwards; no calls for the two reused cells.
    import torch
    from pilot import model_masked_v1
    model_path = Path(model_masked_v1.__file__).resolve()
    matching = [p for p in cfg["pins"] if Path(p).resolve() == model_path]
    need(len(matching) == 1 and sha(model_path) == cfg["pins"][matching[0]], "Unpinned model import")
    torch.set_num_threads(cfg["cpu_threads"])
    need(sha(cfg["checkpoint"]) == cfg["checkpoint_sha256"], "Checkpoint identity")
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    need(list(payload["classes"]) == classes and payload["n_classes"] == 93, "Payload class order")
    model = model_masked_v1.MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                                         kernel_size=payload["kernel_size"], dropout=payload["dropout"], n_classes=93)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    need(state_digest(before) == prev["model_state"]["payload_state_sha256"], "Different model from prior unit")
    need(main_state_encoding(before) == model_record["payload_state_sha"] == main_state_encoding(payload["state_dict"]),
         "Different model from main execution")
    logits_by_id = dict(reused_logits)
    forwards = 0
    for cid in ("BC", "CB"):
        x, pad = prepared[cid]
        with torch.inference_mode():
            logits = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        forwards += 1
        need(logits.shape == (93,) and np.isfinite(logits).all(), "Non-finite scores")
        need(np.array_equal(logits, np.asarray(saved[cid]["logits93"], dtype=np.float64)), "Independent cross scores differ: " + cid)
        logits_by_id[cid] = logits
    need(forwards == result["calls"]["forward"] == 2 and result["calls"]["torch_load"] == 1, "Forward/load budget")
    need(all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "Independent model state mutated")

    # Truth and fixed-rival diagnostics are computed from completed forwards only.
    truth, rival = cfg["label"], prior["C"]["score"]["top1"]
    need(truth != rival and result["truth"] == truth and result["fixed_rival"] == rival, "Fixed comparator mismatch")
    scores, full_ranks = {}, {}
    for cid in ("BB", "CC", "BC", "CB"):
        logits = logits_by_id[cid]
        full_ranks[cid] = r = rank(logits, classes, truth)
        item = {"scope": "full93", "true_label": truth, "top1_index": r["top1_index"],
                "top1": r["top1"], "correct": r["correct"], "true_rank": r["true_rank"],
                "true_logit": float(logits[classes.index(truth)]), "best_other": r["best_other"],
                "true_vs_best_other_margin": r["true_vs_best_other_margin"], "fixed_rival": rival,
                "fixed_rival_gap": float(logits[classes.index(truth)]-logits[classes.index(rival)])}
        need(item == saved[cid]["score"], "Recomputed score mismatch: " + cid)
        scores[cid] = item
    need(scores["BB"]["correct"] and not scores["CC"]["correct"], "Diagonal truth pattern changed")
    effects = {}
    for metric in ("true_vs_best_other_margin", "fixed_rival_gap"):
        g = {cid: scores[cid][metric] for cid in scores}
        effects[metric] = {"span_on_B_obs_BC_minus_BB": g["BC"]-g["BB"],
                           "span_on_C_obs_CC_minus_CB": g["CC"]-g["CB"],
                           "obs_on_long_span_CB_minus_BB": g["CB"]-g["BB"],
                           "obs_on_short_span_CC_minus_BC": g["CC"]-g["BC"],
                           "interaction_CC_minus_BC_minus_CB_plus_BB": g["CC"]-g["BC"]-g["CB"]+g["BB"]}
    need(effects == result["contrasts"], "Contrast calculations differ")
    expected_patterns = {(True, False): "observation_bundle: B observation correct, C observation wrong at both spans",
                         (False, True): "span_renormalization_bundle: long span correct, short span wrong for both observations",
                         (True, True): "interaction: failure only in C-observation x C-short-span combination",
                         (False, False): "B_long_only: only B-observation x B-long-span is correct"}
    cross_pattern = (scores["BC"]["correct"], scores["CB"]["correct"])
    need(result["pattern"] == expected_patterns[cross_pattern], "Pattern classification differs")
    need({arm: {"obs_"+k: array_hash(v) for k, v in values.items()} for arm, values in obs.items()} == obs_hash,
         "Raw observation arrays changed")
    for path, expected in protected.items():
        need(sha(path) == expected, "Protected file changed during independent verification: " + path)
    need(not any(k == "av" or k == "mediapipe" or k.startswith("mediapipe.") for k in sys.modules), "Unexpected extractor/video import")
    report = {"status": "PASS", "source_result_sha256": sha(result_path), "source_config_sha256": sha(cfg_path),
              "verification_sources": {"verifier_sha256": sha(__file__), "reference_sha256": sha(HERE/"independent_reference.py")},
              "pins_unchanged": len(cfg["pins"]), "model_state_unchanged": True,
              "execution": {"model_loads": 1, "tcn_forwards": 2, "reused_diagonals": 2,
                            "video_decodes": 0, "mediapipe_calls": 0, "camera_calls": 0, "training_runs": 0},
              "all_inputs_exact": True, "all_cross_logits_exact": True, "max_abs_logit_difference": 0.0,
              "inputs": inputs, "scores": scores, "contrasts": effects, "pattern": pattern(*cross_pattern),
              "c_longtail": expected_tail,
              "limits": ["One post-regression diagnostic, not independent performance validation.",
                         "Explicit span changes normalization and temporal context together.",
                         "Prior IMAGE candidate remains rejected; no outcome-selected crop is promoted."]}
    with out_json.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# 손 관측 × 구간 독립 검산", "", "기술 검산 PASS. 모델 로드1회, 교차2조건만 독립2회 추론했습니다.",
             "대각선2조건은 입력·스케일·시각을 정확히 재현한 뒤 이전 점수를 재사용했습니다. 새 추출·카메라·학습0회.",
             "두 교차 조건의 전체93 점수186개가 본실행과 정확히 같습니다. 4조건의 순위·점수차와 대비값도 검산했습니다.",
             "", "| 조건 | 1위 | 정답 순위 | 정답−최고 경쟁 | 정답−고정 경쟁 |", "|---|---|---:|---:|---:|"]
    for cid, score in scores.items():
        lines.append(f"| {cid} | {score['top1']} | {score['true_rank']} | {score['true_vs_best_other_margin']:+.6f} | {score['fixed_rival_gap']:+.6f} |")
    lines += ["", "패턴: " + report["pattern"],
              "", "이것은 이미 실패가 알려진 한 원본의 조건부 진단입니다. 구간 변경은 시간 길이·말미 관측·정규화가 함께 바뀌며, 실제 손 좌표의 정답이나 다른 사람 성능을 입증하지 않습니다. IMAGE 회귀 실패와 운영 미적용 상태를 바꾸지 않습니다."]
    with out_md.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print(json.dumps({"status": "PASS", "forwards": forwards, "pattern": report["pattern"], "scores": scores}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] != ["--run"]:
        raise SystemExit("Explicit --run required; exactly two cross forwards after main result")
    main()
