#!/usr/bin/env python3
"""Read-only JSON evidence inventory for ONE fixed current model (stdlib only).

Aggregates stored JSON only. Computes no model outputs, loads no arrays/media/
checkpoints (the model file is hashed only), never selects another model, and
never infers sign-execution correctness or new-person generalization.
Usage: python inventory.py --config CONFIG.json --out-dir OUTDIR
"""
import argparse, hashlib, json, os, re, sys

ROLES = ("head_replay", "connection", "paired", "live", "public_evaluation", "ksl", "training_coverage")
USER_COHORTS = ("self_old18", "self_new18", "stored_session_replay", "live_current_model")
HEAD_KEEP, HEAD_SKIP = ("self_old18", "self_new18"), ("public_confirmed2",)
PUB_GROUPS = ("annotation_dev", "whole_dev", "public", "mp")
PUB_CONDS = ("intact", "reviewed_masked", "additional_train", "whole_clip", "provided_annotation")
UNCONFIRMED = (None, "UNCONFIRMED", "UNKNOWN", "PENDING", "NOT_CONFIRMED")
SELF_REPORT = "participant_self_report"
OUTPUTS = ("inventory.json", "INVENTORY.md")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NOTES = [
    "All user cohorts are trials from the SAME development-exposed participant; no new-person generalization claim.",
    "user_count unit is trials, not people. Ground truth = participant self-report / historical participant confirmation; expert_validation=false.",
    "Sign execution correctness is UNVALIDATED for every row; observation/feature statuses are not execution labels.",
    "self_old18/self_new18 are historical camera-input replays; old coordinates are not new current-extractor captures.",
    "stored_session_replay was captured while a different model was displayed; it is NOT live-current-model evidence.",
    "public intact/reviewed_masked = training-reference reuse (not independent); mp = known training signers; "
    "annotation_dev = provided coordinates, not webcam. KSL = source-label agreement, not expert truth; signer separation UNKNOWN.",
    "full93 is primary; restricted18 is summarized separately and is never fabricated when not recorded.",
    "No overall headline percentage mixing regimes. No plan to fix user-only rows; no training/tuning/selection based on user scores.",
]


class InventoryError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise InventoryError(msg)


def get(d, key, where):
    need(isinstance(d, dict), f"{where}: expected JSON object")
    need(key in d, f"{where}: missing field '{key}'")
    return d[key]


def pin(v, where):
    need(isinstance(v, str) and HEX64.match(v.lower()) is not None, f"{where}: invalid sha256 {v!r}")
    return v.lower()


def nonempty(v, where):
    need(isinstance(v, str) and v != "", f"{where}: expected non-empty string, got {v!r}")
    return v


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_config(path):
    cfg, base = load_json(path), os.path.dirname(os.path.abspath(path))
    res = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    m = get(cfg, "model", "config")
    targets = get(cfg, "target_words", "config")
    need(isinstance(targets, dict) and targets, "config.target_words must be a non-empty object")
    for k, v in targets.items():
        nonempty(k, "config.target_words id"), nonempty(v, f"config.target_words[{k}]")
    need(len(set(targets.values())) == len(targets), "config.target_words: duplicate label")
    roles = get(cfg, "source_roles", "config")
    need(isinstance(roles, dict) and set(roles) == set(ROLES), f"config.source_roles must have exactly {ROLES}")
    src = {r: {"path": res(nonempty(get(roles[r], "path", f"source_roles.{r}"), r)),
               "sha256": pin(get(roles[r], "sha256", f"source_roles.{r}"), f"source_roles.{r}.sha256")} for r in ROLES}
    model = {"path": res(nonempty(get(m, "path", "config.model"), "config.model.path")),
             "sha256": pin(get(m, "sha256", "config.model"), "config.model.sha256")}
    return {"model": model, "targets": dict(targets), "sources": src}


def verify_hashes(cfg, phase):
    for name, e in [("model", cfg["model"])] + list(cfg["sources"].items()):
        need(os.path.isfile(e["path"]), f"{phase}: {name} file missing: {e['path']}")
        got = sha256_file(e["path"])
        need(got == e["sha256"], f"{phase}: sha256 mismatch for {name} ({e['path']}): pinned {e['sha256']}, actual {got}")


def counter():
    return {"n": 0, "full93_matches": 0, "restricted18_n": 0, "restricted18_matches": 0, "errors": []}


def scored(block, truth, where, pk, ck, rk, mk, allowed=None):
    pred = nonempty(get(block, pk, where), f"{where}.{pk}")
    if allowed is not None:
        need(pred in allowed, f"{where}: restricted prediction {pred!r} is not a target label")
    rep = get(block, ck, where)
    need(isinstance(rep, bool), f"{where}.{ck} must be boolean")
    need(rep == (pred == truth), f"{where}: reported {ck}={rep} inconsistent with truth {truth!r} / prediction {pred!r}")
    return {"prediction": pred, "match": pred == truth, "rank": block.get(rk), "margin": block.get(mk)}


def items(doc, key, where):
    v = get(doc, key, where)
    need(isinstance(v, list), f"{where}.{key} must be a list")
    return v


class Inventory:
    def __init__(self, cfg, docs):
        self.cfg, self.docs, self.msha = cfg, docs, cfg["model"]["sha256"]
        self.t = cfg["targets"]
        self.l2i = {v: k for k, v in self.t.items()}
        self.rows, self.unknown, self.skipped, self.ids = [], [], {}, set()

    def ptr(self, role, index, rid):
        return {"role": role, "path": self.cfg["sources"][role]["path"],
                "sha256": self.cfg["sources"][role]["sha256"], "index": index, "id": rid}

    def add(self, tid, cohort, src, truth, selected, full, r18, extra):
        nonempty(tid, f"{cohort} trial id")
        need(tid not in self.ids, f"duplicate user trial id {tid!r} (ids must be unique across all user cohorts)")
        self.ids.add(tid)
        if truth is None:
            full["match"] = None
            if r18:
                r18["match"] = None
        self.rows.append(dict({"trial_id": tid, "cohort": cohort, "source": src,
            "actual": None if truth is None else {"word_id": self.l2i[truth], "label": truth},
            "selected": selected, "full93": full, "restricted18": r18, "expert_validation": False,
            "execution_correctness": "UNVALIDATED"}, **extra))

    def head(self):
        doc = self.docs["head_replay"]
        need(pin(get(get(doc, "candidate", "head_replay"), "sha256", "head_replay.candidate"), "head_replay.candidate")
             == self.msha, "head_replay: candidate model sha256 != pinned target model")
        self.skipped["head_replay_public_confirmed2"] = 0
        for i, r in enumerate(items(doc, "rows", "head_replay")):
            w, c = f"head_replay.rows[{i}]", get(r, "cohort", f"head_replay.rows[{i}]")
            if c in HEAD_SKIP:
                self.skipped["head_replay_public_confirmed2"] += 1
                continue
            need(c in HEAD_KEEP, f"{w}: unknown cohort {c!r}")
            truth = get(r, "truth", w)
            need(truth in self.l2i, f"{w}: truth {truth!r} is not a target label")
            m = get(get(r, "models", w), "candidate", w + ".models")
            f = scored(get(m, "full93", w), truth, w + ".full93", "prediction", "correct", "true_rank", "true_margin")
            p = scored(get(m, "project18", w), truth, w + ".project18", "prediction", "correct", "true_rank",
                       "true_margin", allowed=self.l2i)
            self.add(get(r, "id", w), c, [self.ptr("head_replay", i, r["id"])], truth, None, f, p,
                     {"input_sha256": r.get("input_sha256"), "truth_provenance": "historical participant confirmation (not expert)"})

    def session(self):
        def index(role):
            out = {}
            for i, t in enumerate(items(self.docs[role], "trials", role)):
                tid = nonempty(get(t, "trial_id", f"{role}.trials[{i}]"), f"{role}.trials[{i}].trial_id")
                need(tid not in out, f"{role}: duplicate trial_id {tid!r}")
                out[tid] = (i, t)
            return out
        cm, pm = index("connection"), index("paired")
        need(set(cm) == set(pm), f"connection/paired trial_id sets differ: only_connection={sorted(set(cm) - set(pm))}, "
             f"only_paired={sorted(set(pm) - set(cm))}")
        sid = get(self.docs["paired"], "session_id", "paired")
        for tid, (pi, p) in pm.items():
            ci, c, w = cm[tid][0], cm[tid][1], f"session trial {tid!r}"
            need(pin(get(get(c, "identity", w), "checkpoint_sha256", w), w) == self.msha, f"{w}: connection checkpoint != target model")
            need(get(c, "model_status", w) == "READY", f"{w}: model_status not READY")
            need(get(c, "full_candidate_count", w) == 93, f"{w}: full_candidate_count != 93")
            op = get(p, "operating", w)  # operating existing-hand-assignment only; 'candidate' is never read
            pred = nonempty(get(op, "pred", w), f"{w}.operating.pred")
            chain = {k: get(c, k, w) for k in ("service_top1_93", "post_hoc_top1_93", "stored_paired_top1")}
            need(all(v == pred for v in chain.values()), f"{w}: identity chain {chain} != operating.pred {pred!r}")
            r18id = get(c, "project18_top1", w)
            need(r18id in self.t, f"{w}: project18_top1 {r18id!r} is not a target word id")
            sel_id, sel_lab = get(p, "selected_word_id", w), get(p, "selected_label", w)
            need(sel_id in self.t and self.t[sel_id] == sel_lab, f"{w}: selected word/label invalid or inconsistent")
            conf = get(p, "confirmation", w)
            need(isinstance(conf, dict), f"{w}: confirmation must be an object")
            status, aid = conf.get("status"), conf.get("actual_word_id")
            need(conf.get("is_expert_verification") is not True, f"{w}: expert verification claimed; not supported")
            if status == "CONFIRMED":
                need(aid in self.t, f"{w}: CONFIRMED but actual_word_id {aid!r} is not a mapped target")
                need(conf.get("label") == self.t[aid], f"{w}: confirmation label {conf.get('label')!r} != mapping for {aid}")
                need(conf.get("source") == SELF_REPORT, f"{w}: confirmation source must be {SELF_REPORT}")
                need(conf.get("is_expert_verification") is False, f"{w}: is_expert_verification must be false")
                truth = self.t[aid]
                rep = get(op, "correct_vs_confirmed", w)
                need(rep is (pred == truth), f"{w}: correct_vs_confirmed={rep!r} inconsistent with recomputed")
            elif status in UNCONFIRMED:
                truth = None
                need(op.get("correct_vs_confirmed") is not True, f"{w}: unconfirmed trial claims correctness")
                self.unknown.append({"trial_id": tid, "cohort": "stored_session_replay", "status": status,
                                     "actual_word_id": aid, "selected_word_id": sel_id})
            else:
                raise InventoryError(f"{w}: unrecognized confirmation status {status!r}")
            f = {"prediction": pred, "match": pred == truth}
            r18 = {"prediction": self.t[r18id], "match": self.t[r18id] == truth}
            self.add(tid, "stored_session_replay", [self.ptr("connection", ci, tid), self.ptr("paired", pi, tid)],
                     truth, {"word_id": sel_id, "label": sel_lab}, f, r18,
                     {"session_id": sid, "confirmation_status": status, "not_live_current_model": True})

    def live(self):
        doc = self.docs["live"]
        sid = get(doc, "session_id", "live")
        for i, t in enumerate(items(doc, "trials", "live")):
            w = f"live.trials[{i}]"
            for k, v in (("origin", "camera"), ("model_status", "READY"), ("full_class_count", 93),
                         ("self_report_source", SELF_REPORT)):
                need(get(t, k, w) == v, f"{w}: {k} must be {v!r}, got {t.get(k)!r}")
            for k in ("checkpoint_sha256", "identity_fixed_sha256"):
                need(pin(get(t, k, w), f"{w}.{k}") == self.msha, f"{w}: {k} != target model")
            aid, req = get(t, "actual_word_id", w), get(t, "requested_word_id", w)
            need(aid in self.t, f"{w}: actual_word_id {aid!r} not a mapped target")
            need(get(t, "actual_label", w) == self.t[aid], f"{w}: actual_label inconsistent with mapping")
            need(req in self.t, f"{w}: requested_word_id {req!r} not a mapped target")
            truth, pred = self.t[aid], nonempty(get(t, "full_top1", w), f"{w}.full_top1")
            need(get(t, "top1_matches_self_report", w) is (pred == truth), f"{w}: top1_matches_self_report inconsistent")
            self.add(get(t, "trial_id", w), "live_current_model", [self.ptr("live", i, t.get("trial_id"))], truth,
                     {"word_id": req, "label": self.t[req]}, {"prediction": pred, "match": pred == truth}, None,
                     {"session_id": sid, "model_id": t.get("model_id"), "restricted18_status": "NOT_RECORDED",
                      "observation_status": t.get("observation_status"),
                      "execution_assessment": t.get("execution_assessment"), "accepted": t.get("accepted")})

    @staticmethod
    def role(g, c):
        r = []
        if c in ("intact", "reviewed_masked") and g == "public":
            r.append("TRAINING_REFERENCE_REUSE_NOT_INDEPENDENT")
        elif c == "reviewed_masked":
            r.append("REVIEWED_MASKED_REFERENCE_REUSE_NOT_INDEPENDENT")
        if c == "additional_train":
            r.append("ADDITIONAL_TRAINING_DATA_REUSE")
        if g == "mp":
            r.append("KNOWN_TRAINING_SIGNERS")
        if g == "annotation_dev":
            r.append("PROVIDED_COORDINATE_DEVELOPMENT_NOT_WEBCAM")
        if g == "whole_dev":
            r.append("WHOLE_CLIP_DEVELOPMENT_DIAGNOSTIC")
        return "+".join(r) or "PUBLIC_VIDEO_DIAGNOSTIC"

    def public(self):
        doc, self.pub, seen = self.docs["public_evaluation"], {}, set()
        need(pin(get(get(get(doc, "models", "public_evaluation"), "control", "public_evaluation.models"), "sha256",
                     "control"), "control") == self.msha, "public_evaluation: control model sha256 != target model")
        self.skipped["public_nontarget_truth"] = 0
        for i, r in enumerate(items(doc, "rows", "public_evaluation")):
            w = f"public_evaluation.rows[{i}]"
            g, c, rid, truth = get(r, "group", w), get(r, "condition", w), get(r, "id", w), get(r, "true", w)
            need(g in PUB_GROUPS and c in PUB_CONDS, f"{w}: unknown group/condition {g!r}/{c!r}")
            need((g, c, rid) not in seen, f"{w}: duplicate id {rid!r} in {g}|{c}")
            seen.add((g, c, rid))
            if truth not in self.l2i:
                self.skipped["public_nontarget_truth"] += 1
                continue
            ctl = get(get(r, "models", w), "control", w + ".models")
            f = scored(get(ctl, "full93", w), truth, w + ".full93", "pred", "correct", "rank", "margin")
            p = ctl.get("project18")
            p = None if p is None else scored(p, truth, w + ".project18", "pred", "correct", "rank", "margin", self.l2i)
            b = self.pub.setdefault(f"{g}|{c}", {"group": g, "condition": c, "evidence_role": self.role(g, c), "per_word": {}})
            k = b["per_word"].setdefault(truth, counter())
            k["n"] += 1
            k["full93_matches"] += f["match"]
            if p:
                k["restricted18_n"] += 1
                k["restricted18_matches"] += p["match"]
            if not f["match"] or (p and not p["match"]):
                k["errors"].append({"id": rid, "truth": truth, "full93": f, "restricted18": p,
                                    "signer_id": r.get("signer_id"), "source": self.ptr("public_evaluation", i, rid)})

    def ksl(self):
        doc, self.k, seen = self.docs["ksl"], {}, set()
        need(pin(get(doc, "model_sha256", "ksl"), "ksl.model_sha256") == self.msha, "ksl: model_sha256 != target model")
        self.skipped["ksl_nontarget_label"] = 0
        for i, r in enumerate(items(doc, "rows", "ksl")):
            w = f"ksl.rows[{i}]"
            rid = nonempty(get(r, "id", w), w + ".id")
            need(rid not in seen, f"ksl: duplicate source id {rid!r}")
            seen.add(rid)
            lab, exp = get(r, "candidate_label", w), get(r, "previous_diagnostic_exposure", w)
            need(isinstance(exp, bool), f"{w}: previous_diagnostic_exposure must be boolean")
            if lab not in self.l2i:
                self.skipped["ksl_nontarget_label"] += 1
                continue
            k = self.k.setdefault(lab, {}).setdefault("previous_exposure" if exp else "new", {"n": 0, "matches": 0, "unscored": 0, "errors": []})
            src = self.ptr("ksl", i, rid)
            if get(r, "status", w) != "OK":
                k["unscored"] += 1
                k["errors"].append({"id": rid, "status": r["status"], "unscored": True, "source": src})
                continue
            pred = nonempty(get(r, "prediction", w), w + ".prediction")
            need(get(r, "agreement", w) is (pred == lab), f"{w}: agreement inconsistent with candidate_label/prediction")
            k["n"] += 1
            k["matches"] += pred == lab
            if pred != lab:
                k["errors"].append({"id": rid, "candidate_label": lab, "prediction": pred, "target_rank": r.get("target_rank"),
                                    "meaning_evidence": r.get("meaning_evidence"), "source": src})

    def training(self):
        doc, self.tr = self.docs["training_coverage"], {}
        keys = ("registered_training_rows", "nikl_reference_rows", "other_training_rows", "provided_coordinate_development_rows")
        for i, e in enumerate(items(doc, "words", "training_coverage")):
            w = f"training_coverage.words[{i}]"
            lab = nonempty(get(e, "word", w), w + ".word")
            need(lab not in self.tr, f"training_coverage: duplicate word {lab!r}")
            vals = {k: get(e, k, w) for k in keys}
            need(all(type(v) is int and v >= 0 for v in vals.values()), f"{w}: counts must be non-negative integers")
            need(vals[keys[1]] + vals[keys[2]] == vals[keys[0]], f"{w}: nikl_reference+other != registered")
            self.tr[lab] = vals
        missing = [l for l in self.t.values() if l not in self.tr]
        need(not missing, f"training_coverage: target labels missing: {missing}")

    def build(self):
        self.head(), self.session(), self.live(), self.public(), self.ksl(), self.training()
        per_word, summ = [], {c: {"count": 0, "with_truth": 0, "full93_matches": 0, "restricted18_n": 0,
                                  "restricted18_matches": 0, "no_truth": 0} for c in USER_COHORTS}
        uc = {wid: {c: counter() for c in USER_COHORTS} for wid in self.t}
        for r in self.rows:
            s = summ[r["cohort"]]
            s["count"] += 1
            if r["actual"] is None:
                s["no_truth"] += 1
                continue
            k, f, p = uc[r["actual"]["word_id"]][r["cohort"]], r["full93"], r["restricted18"]
            s["with_truth"] += 1
            k["n"] += 1
            k["full93_matches"] += f["match"]
            s["full93_matches"] += f["match"]
            if p:
                for d in (k, s):
                    d["restricted18_n"] += 1
                    d["restricted18_matches"] += p["match"]
            if not f["match"] or (p and not p["match"]):
                k["errors"].append({"trial_id": r["trial_id"], "actual": r["actual"]["label"], "full93_prediction":
                                    f["prediction"], "restricted18_prediction": p["prediction"] if p else None})
        for wid, lab in self.t.items():
            per_word.append({"word_id": wid, "label": lab, "user": uc[wid],
                             "public": {key: b["per_word"][lab] for key, b in self.pub.items() if lab in b["per_word"]},
                             "ksl": self.k.get(lab, {}), "training": self.tr[lab]})
        pubc = {}
        for key, b in self.pub.items():
            tot = {x: sum(v[x] for v in b["per_word"].values()) for x in ("n", "full93_matches", "restricted18_n", "restricted18_matches")}
            pubc[key] = dict({"group": b["group"], "condition": b["condition"], "evidence_role": b["evidence_role"]}, **tot)
        ksum = {e: {x: sum(v.get(e, {}).get(x, 0) for v in self.k.values()) for x in ("n", "matches", "unscored")}
                for e in ("new", "previous_exposure")}
        return {"schema": "target18_evidence_inventory_v1", "notes": NOTES,
                "model": dict(self.cfg["model"], hash_verified_before_and_after=True, selection="exact pinned sha256 only"),
                "source_pins": self.cfg["sources"], "target_words": self.t, "user_count_unit": "trials_not_people",
                "execution_correctness": "UNVALIDATED", "user_rows": self.rows, "user_cohort_summary": summ,
                "unknown_confirmations": self.unknown, "per_word": per_word, "public_conditions": pubc,
                "ksl_summary": dict(ksum, truth_basis="source-label agreement, not expert truth", signer_separation="UNKNOWN",
                                    restricted18="NOT_RECORDED"),
                "training_coverage": {"per_label": self.tr, "nikl_only_words": self.docs["training_coverage"].get("nikl_only_words"),
                                      "scope": "current model registered training coverage; not all available data"},
                "skipped": self.skipped,
                "live_uncovered_target_words": [p["word_id"] for p in per_word if p["user"]["live_current_model"]["n"] == 0]}


def frac(k, key="full93_matches"):
    return "-" if not k or k["n"] == 0 else f"{k[key]}/{k['n']}"


def render_md(inv):
    L = ["# 목표 단어 증거 인벤토리 (Target-word evidence inventory)", "",
         f"- 현재 모델 exact sha256: `{inv['model']['sha256']}`", "- 저장된 JSON 집계만 수행; 모델 출력 새로 계산 없음.", ""]
    L += [f"- {n}" for n in NOTES] + ["", "## full93 (primary) — '-' = 시험 없음 (실패 아님)", "",
          "| 단어 | 과거 재생 old | 신규 재생 new | 세션 재생 | 실시간 현재모델 | NIKL 원본참조(학습) | 기타 공개KSL 신규 | 기타 공개KSL 이전노출 | 학습 행수 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in inv["per_word"]:
        u, ks = p["user"], p["ksl"]
        kf = lambda e: frac(ks.get(e), "matches") +(f" (+{ks[e]['unscored']} unscored)" if ks.get(e, {}).get("unscored") else "")
        L.append(f"| {p['label']} ({p['word_id']}) | {frac(u['self_old18'])} | {frac(u['self_new18'])} | "
                 f"{frac(u['stored_session_replay'])} | {frac(u['live_current_model'])} | {frac(p['public'].get('public|intact'))} | "
                 f"{kf('new')} | {kf('previous_exposure')} | {p['training']['registered_training_rows']} |")
    L += ["", "## restricted18 (separate summary)", ""]
    for c, s in inv["user_cohort_summary"].items():
        r = "NOT_RECORDED" if c == "live_current_model" else ("-" if s["restricted18_n"] == 0 else f"{s['restricted18_matches']}/{s['restricted18_n']}")
        L.append(f"- {c}: trials={s['count']}, full93={s['full93_matches']}/{s['with_truth']}, restricted18={r}, no_truth={s['no_truth']}")
    L += ["", "## 알려진 오류 (actual truth → prediction)", ""]
    for p in inv["per_word"]:
        for c, k in p["user"].items():
            L += [f"- [{c}] {e['trial_id']}: actual={e['actual']} full93={e['full93_prediction']} "
                  f"restricted18={e['restricted18_prediction'] or 'NOT_RECORDED/-'}" for e in k["errors"]]
    L += ["", "## 공개 조건별 (조건 간 합산 없음)", ""]
    L += [f"- {k}: {v['full93_matches']}/{v['n']} full93, role={v['evidence_role']}" for k, v in inv["public_conditions"].items()]
    L += ["", f"- 실시간 미커버 단어: {', '.join(inv['live_uncovered_target_words']) or '없음'}",
          f"- 미확인(unknown) 확인 상태: {len(inv['unknown_confirmations'])}건 (정답 없음, 정확도 주장 없음)", ""]
    return "\n".join(L)


def check_outdir(out_dir):
    for n in OUTPUTS:
        need(not os.path.exists(os.path.join(out_dir, n)), f"refusing: {n} already exists in {out_dir}")


def run(config_path, out_dir):
    check_outdir(out_dir)
    cfg = load_config(config_path)
    verify_hashes(cfg, "before")
    inv = Inventory(cfg, {r: load_json(cfg["sources"][r]["path"]) for r in ROLES}).build()
    verify_hashes(cfg, "after")
    js, md = json.dumps(inv, ensure_ascii=False, indent=2), render_md(inv)
    check_outdir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, text in zip(OUTPUTS, (js, md)):
        with open(os.path.join(out_dir, name), "x", encoding="utf-8") as f:
            f.write(text)
    return inv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    try:
        run(a.config, a.out_dir)
    except InventoryError as e:
        print(f"INVENTORY FAILED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
