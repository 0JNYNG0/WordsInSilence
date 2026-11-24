"""Bounded research runner: native source-data pair (control vs +9 native clips).

Metadata-only preflight by default; --execute runs the paired comparison using
existing project helpers only. No network, shell, deletion or registration.
Exploratory native source-domain comparison only: video PTS/action completeness
and new-webcam-person performance are NOT verified by this runner.
"""
import argparse, hashlib, json, math, os, sys

SCHEMA = "wis_native_data_pair_plan_v1"
SCOPE = "native_source_data_addition_only"
N_CLASSES = 93
EXP = {"source_train": 2095, "source_validation": 696, "nikl": 18}
TRAIN_SIGNERS = ["REAL%02d" % i for i in range(1, 10)]
DEV_SIGNERS = ["REAL11", "REAL12", "REAL13"]
INT_KEYS = ("seed", "steps", "batch_size", "cpu_threads")
FLOAT_KEYS = ("encoder_lr", "head_lr", "weight_decay")


class UniformSchedule(object):
    """Deterministic replay of a precomputed uniform stream (pure stdlib)."""

    def __init__(self, values):
        self._v = [float(x) for x in values]
        for x in self._v:
            if not (0.0 <= x < 1.0) or x != x:
                raise ValueError("uniform out of [0,1): %r" % (x,))
        self._i = 0

    def integers(self, high):
        high = int(high)
        if high <= 0:
            raise ValueError("high must be positive, got %d" % high)
        if self._i >= len(self._v):
            raise RuntimeError("uniform schedule exhausted at draw %d" % self._i)
        u = self._v[self._i]
        self._i += 1
        return min(int(math.floor(u * high)), high - 1)

    @property
    def consumed(self):
        return self._i

    def digest(self):
        blob = ",".join("%.17g" % x for x in self._v).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _under(root, path):
    r, p = os.path.abspath(root), os.path.abspath(path)
    return p == r or p.startswith(r + os.sep)


def _resolve(root, rel):
    p = os.path.abspath(os.path.normpath(os.path.join(root, rel)))
    if not _under(root, p):
        raise ValueError("path escapes root: %s" % rel)
    return p


def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _pos_float(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0


def preflight(plan, out):
    """Pure validation: returns (report, prepared_metadata). No side effects."""
    blocks = []
    add = blocks.append
    web, pilot = plan.get("web_root", ""), plan.get("pilot_root", "")
    if plan.get("schema") != SCHEMA:
        add("plan.schema must be %r" % SCHEMA)
    if plan.get("augmentation") is not False:
        add("plan.augmentation must be exactly false")
    if plan.get("scope") != SCOPE:
        add("plan.scope must be %r" % SCOPE)
    for k in ("web_root", "pilot_root", "base_inputs", "candidate_report", "checkpoint"):
        v = plan.get(k)
        if not isinstance(v, str) or not v or not os.path.isabs(v):
            add("plan.%s must be an absolute path string" % k)
    if not isinstance(plan.get("checkpoint_sha256"), str) or len(plan.get("checkpoint_sha256", "")) != 64:
        add("plan.checkpoint_sha256 must be a 64-hex string")
    st = plan.get("settings") or {}
    for k in INT_KEYS:
        if not _pos_int(st.get(k)):
            add("settings.%s must be a positive int" % k)
    for k in FLOAT_KEYS:
        if not _pos_float(st.get(k)):
            add("settings.%s must be a finite positive number" % k)
    rev = plan.get("semantic_review") or {}
    reviewer, evid = rev.get("reviewer"), rev.get("evidence")
    if rev.get("status") != "approved":
        add("semantic_review pending: status=%r (review not created by this tool)" % rev.get("status"))
    if not isinstance(reviewer, str) or not reviewer.strip():
        add("semantic_review.reviewer must be a non-empty string")
    evid_path = None
    if isinstance(evid, str) and evid.strip():
        evid_path = evid if os.path.isabs(evid) else os.path.join(web, evid)
        if not os.path.isfile(evid_path):
            add("semantic_review.evidence does not name an existing local file: %s" % evid)
    else:
        add("semantic_review.evidence must name an existing local file")
    mapping = plan.get("label_mapping")
    if not isinstance(mapping, dict) or not mapping:
        add("plan.label_mapping must be a non-empty object")
        mapping = {}
    if blocks:
        return _report(blocks, None), None

    src_meta, cand = _read_json(plan["base_inputs"]), _read_json(plan["candidate_report"])
    train = list(src_meta.get("source_train") or [])
    dev = list(src_meta.get("source_validation") or [])
    nikl = list(src_meta.get("nikl") or [])
    for name, rows in (("source_train", train), ("source_validation", dev), ("nikl", nikl)):
        if len(rows) != EXP[name]:
            add("%s count %d != expected %d" % (name, len(rows), EXP[name]))

    idx2lab, order = {}, []
    ids, paths, shas = set(), set(), set()

    def take(rows, root, kind):
        items = []
        for r in rows:
            rid, lab, ci = r.get("id"), r.get("label"), r.get("class_index")
            try:
                p = _resolve(root, r.get("npz", ""))
            except ValueError as exc:
                add("%s %s: %s" % (kind, rid, exc))
                continue
            if not os.path.isfile(p):
                add("%s %s: npz missing (%s)" % (kind, rid, r.get("npz")))
            for bucket, val, what in ((ids, rid, "id"), (paths, p, "path"), (shas, r.get("npz_sha256"), "sha256")):
                if val in bucket:
                    add("%s %s: duplicate %s across train/development" % (kind, rid, what))
                bucket.add(val)
            if not (isinstance(ci, int) and not isinstance(ci, bool) and 0 <= ci < N_CLASSES):
                add("%s %s: class_index out of 0..92" % (kind, rid))
            elif kind != "development":
                if ci in idx2lab and idx2lab[ci] != lab:
                    add("inconsistent index->label at %d: %r vs %r" % (ci, idx2lab[ci], lab))
                elif ci not in idx2lab:
                    idx2lab[ci] = lab
                    order.append(ci)
            items.append({"sample_id": rid, "label": lab, "class_index": ci, "npz": p,
                          "npz_sha256": r.get("npz_sha256"), "kind": kind})
        return items

    for r in train:
        if r.get("role") != "training" or not any(s in str(r.get("id", "")) for s in TRAIN_SIGNERS):
            add("source_train %s: role/signer restriction violated" % r.get("id"))
    for r in dev:
        if r.get("role") != "validation" or not any(s in str(r.get("id", "")) for s in DEV_SIGNERS):
            add("source_validation %s: role/signer restriction violated" % r.get("id"))
    for r in nikl:
        if r.get("role") != "training_original_and_every_augmentation" or not r.get("word_id"):
            add("nikl %s: historical role/word_id restriction violated" % r.get("id"))
    train_items = take(train, pilot, "source_train")
    dev_items = take(dev, pilot, "development")
    nikl_items = take(nikl, web, "nikl")
    if sorted(idx2lab) != list(range(N_CLASSES)):
        add("index->label mapping does not cover exactly indices 0..92 (got %d)" % len(idx2lab))
    labels = set(idx2lab.values())

    if cand.get("status") != "PASS_NATIVE_CANDIDATE_INPUT_ONLY" or cand.get("pins_unchanged") is not True \
            or cand.get("rows") != 12:
        add("candidate_report header invalid (status/pins_unchanged/rows)")
    crows = list(cand.get("clips") or cand.get("items") or [])
    if len(crows) != 12:
        add("candidate_report must list 12 clip rows, got %d" % len(crows))
    cand_train, cand_dev = [], []
    seen_train_sig, seen_dev_sig = set(), set()
    for r in crows:
        cid, role, raw = r.get("clip_id"), r.get("split_role", ""), r.get("raw_label")
        num = role.rsplit("_", 0)[0][-2:] if role else ""
        sig = "REAL" + num
        mapped = mapping.get(raw)
        if role.startswith("train_candidate") and sig in TRAIN_SIGNERS:
            cand_train.append(r); seen_train_sig.add(sig)
        elif role.startswith("development_only") and sig in DEV_SIGNERS:
            cand_dev.append(r); seen_dev_sig.add(sig)
        else:
            add("candidate %s: invalid split_role %r" % (cid, role))
            continue
        if sig not in str(cid):
            add("candidate %s: clip_id does not match REAL number of %s" % (cid, role))
        if not isinstance(r.get("signer_id"), int) or isinstance(r.get("signer_id"), bool):
            add("candidate %s: signer_id must be int" % cid)
        if mapped not in labels:
            add("candidate %s: raw_label %r does not map into the existing 93 labels" % (cid, raw))
        if r.get("numeric_contract_status") != "PASS" or not _pos_int(r.get("frames")):
            add("candidate %s: numeric_contract_status/frames invalid" % cid)
        p = r.get("npz", "")
        if not os.path.isabs(p) or not _under(os.path.join(plan["web_root"], "runs"), p):
            add("candidate %s: npz must be absolute inside WEB/runs" % cid)
        elif not os.path.isfile(p):
            add("candidate %s: npz missing" % cid)
        for bucket, val, what in ((ids, cid, "id"), (paths, os.path.abspath(p), "path"), (shas, r.get("npz_sha256"), "sha")):
            if val in bucket:
                add("candidate %s: duplicate %s versus existing rows" % (cid, what))
            bucket.add(val)
    if len(cand_train) != 9 or len(cand_dev) != 3:
        add("candidate split must be 9 train / 3 development, got %d/%d" % (len(cand_train), len(cand_dev)))
    if seen_train_sig != set(TRAIN_SIGNERS) or seen_dev_sig != set(DEV_SIGNERS):
        add("candidate signer sets do not match the expected REAL01..09 / REAL11..13 sets")
    if not os.path.isfile(plan["checkpoint"]):
        add("checkpoint file missing")

    control_n, treat_n = len(train_items) + len(nikl_items), len(train_items) + len(nikl_items) + 9
    eval_n = len(dev_items) + 3
    for got, want, what in ((control_n, 2113, "control"), (treat_n, 2122, "treatment"), (eval_n, 699, "eval")):
        if got != want and not blocks:
            add("%s sample count %d != expected %d" % (what, got, want))

    prepared = {"settings": dict(st), "label_mapping": dict(mapping), "index_to_label": idx2lab,
                "index_order": order, "train_items": train_items, "nikl_items": nikl_items,
                "dev_items": dev_items, "candidate_train": cand_train, "candidate_dev": cand_dev,
                "nikl_labels": sorted({i["label"] for i in nikl_items}),
                "counts": {"control": control_n, "treatment": treat_n, "eval": eval_n,
                           "source_eval": len(dev_items), "additional_eval": 3},
                "evidence": evid_path}
    return _report(blocks, prepared), (None if blocks else prepared)


def _report(blocks, prepared):
    rep = {"schema": SCHEMA, "scope": SCOPE, "augmentation": False,
           "status": "gated" if blocks else "ready", "blocks": blocks,
           "trained_models": 0, "arrays_loaded": 0,
           "note": "native source-domain exploratory comparison only; PTS/action completeness "
                   "and new-webcam-person performance unverified"}
    if prepared:
        rep["counts"] = prepared["counts"]
        rep["target18_available_gt"] = {"original_classes": 5, "original_rows": 30,
                                        "added_classes": 1, "added_rows": 3,
                                        "note": "not 18-class coverage"}
    return rep


def _load_items(u, build_channels, np, prepared, mapping):
    def finite(x, who, frames=None):
        a = np.asarray(x, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 280 or not np.isfinite(a).all():
            raise RuntimeError("%s: x must be finite Nx280, got %s" % (who, (a.shape,)))
        if frames is not None and a.shape[0] != frames:
            raise RuntimeError("%s: frames %d != reported %d" % (who, a.shape[0], frames))
        return a

    out = {"train": [], "nikl": [], "dev": [], "cand_train": [], "cand_dev": []}
    for it in prepared["train_items"]:
        x = u._channels_from_source(it["npz"], it["npz_sha256"])["x"]
        out["train"].append(dict(it, x=finite(x, it["sample_id"]), origin="source"))
    for it in prepared["nikl_items"]:
        nat, present, ts = u._native_from_nikl(it["npz"], it["npz_sha256"])
        x = build_channels(nat, present, ts)["x"]
        out["nikl"].append(dict(it, x=finite(x, it["sample_id"]), origin="nikl_original"))
    for it in prepared["dev_items"]:
        x = u._channels_from_source(it["npz"], it["npz_sha256"])["x"]
        out["dev"].append(dict(it, x=finite(x, it["sample_id"]), origin="source_dev"))
    idx = {v: k for k, v in prepared["index_to_label"].items()}
    for key, rows in (("cand_train", prepared["candidate_train"]), ("cand_dev", prepared["candidate_dev"])):
        for r in rows:
            if _sha256(r["npz"]) != r["npz_sha256"]:
                raise RuntimeError("candidate %s: sha256 mismatch" % r["clip_id"])
            with np.load(r["npz"], allow_pickle=False) as f:
                x = np.array(f["x"], dtype=np.float32, copy=True)
            lab = mapping[r["raw_label"]]
            out[key].append({"sample_id": r["clip_id"], "label": lab, "class_index": idx[lab],
                             "x": finite(x, r["clip_id"], int(r["frames"])), "origin": "native_candidate"})
    return out


def _summarize(ev, classes):
    gt = dict(ev.get("ground_truth_counts") or {})
    recall, conf = {}, {}
    for c in classes:
        n = int(gt.get(c, 0))
        recall[c] = None if n == 0 else sum(1 for r in ev["rows"] if r["true"] == c and r["pred"] == c) / float(n)
    for r in ev["rows"]:
        k = "%s>%s" % (r["true"], r["pred"])
        conf[k] = conf.get(k, 0) + 1
    return {"n": ev["n"], "correct": ev["correct"], "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
            "macro_f1_denominator": ev["macro_f1_denominator"], "ground_truth_counts": gt,
            "per_class_recall": recall, "sparse_confusion": conf, "rows": ev["rows"]}


def run(plan, prepared, out):
    web, pilot = plan["web_root"], plan["pilot_root"]
    for p in (web, os.path.join(web, "src"), os.path.join(pilot, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np, torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels

    st = prepared["settings"]
    torch.set_num_threads(int(st["cpu_threads"]))
    if _sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha256 mismatch")
    payload = torch.load(plan["checkpoint"], weights_only=True, map_location="cpu")
    probe, classes = u.rebuild_from_payload(payload)
    del probe
    if list(classes) != [prepared["index_to_label"][i] for i in range(N_CLASSES)]:
        raise RuntimeError("checkpoint classes do not match metadata index->label mapping")

    data = _load_items(u, build_channels, np, prepared, prepared["label_mapping"])
    control = data["train"] + data["nikl"]
    treatment = control + data["cand_train"]
    eval_all = data["dev"] + data["cand_dev"]
    if (len(control), len(treatment), len(eval_all)) != (2113, 2122, 699):
        raise RuntimeError("arm sizes %d/%d/%d unexpected" % (len(control), len(treatment), len(eval_all)))

    steps, bs = int(st["steps"]), int(st["batch_size"])
    uniforms = np.random.default_rng(int(st["seed"])).random(2 * steps * bs).tolist()
    setting = {"optimizer_steps": steps, "batch": {"total": bs, "source": bs, "nikl": 0},
               "lr": {"encoder": float(st["encoder_lr"]), "head": float(st["head_lr"])},
               "weight_decay": float(st["weight_decay"]), "seed": int(st["seed"]),
               "batch_size": bs, "cpu_threads": int(st["cpu_threads"]), "augmentation": False, "scope": SCOPE}

    def blocked(*a, **k):
        raise AssertionError("augment_nikl reached: augmentation is disabled for this run")

    results, original_aug = {}, u.augment_nikl
    u.augment_nikl = blocked
    try:
        for name, items in (("control", control), ("treatment", treatment)):
            u.set_seeds(int(st["seed"]))
            model, _ = u.rebuild_from_payload(payload)
            sched = UniformSchedule(uniforms)
            train_out = u.run_training(model, sorted(items, key=lambda d: d["sample_id"]), [],
                                       classes, setting, sched, log=print)
            evs = {"full": u.evaluate(model, eval_all, classes, restrict_to=None),
                   "source696": u.evaluate(model, data["dev"], classes, restrict_to=None),
                   "additional3": u.evaluate(model, data["cand_dev"], classes, restrict_to=None),
                   "target_nikl18": u.evaluate(model, eval_all, classes, restrict_to=prepared["nikl_labels"])}
            for ev in evs.values():
                if not math.isfinite(float(ev["accuracy"])) or not math.isfinite(float(ev["macro_f1"])):
                    raise RuntimeError("non-finite evaluation metric in arm %s" % name)
            results[name] = {"train": train_out, "uniforms_consumed": sched.consumed,
                             "evaluations": {k: _summarize(v, classes) for k, v in evs.items()}}
            ck = dict(payload)
            ck.update({"state_dict": model.state_dict(), "classes": list(classes), "step": steps,
                       "fixed_setting": setting, "settings": setting, "augmentation": False, "scope": SCOPE,
                       "parent_checkpoint_sha256": plan["checkpoint_sha256"], "arm": name})
            torch.save(ck, os.path.join(out, "checkpoint_%s_final.pt" % name))
    finally:
        u.augment_nikl = original_aug

    meta = {"schema": SCHEMA, "scope": SCOPE, "augmentation": False, "settings": setting,
            "counts": prepared["counts"], "schedule_digest": UniformSchedule(uniforms).digest(),
            "uniforms": len(uniforms), "parent_checkpoint_sha256": plan["checkpoint_sha256"],
            "semantic_review": plan["semantic_review"], "label_mapping": prepared["label_mapping"],
            "target18_available_gt": {"original_classes": 5, "original_rows": 30, "added_classes": 1,
                                      "added_rows": 3, "note": "not 18-class coverage"},
            "promotion": "none; exploratory native source-domain comparison only"}
    _write_json(os.path.join(out, "metadata.json"), meta)
    _write_json(os.path.join(out, "results.json"), {"metadata": meta, "arms": results})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="native source data-pair runner (preflight by default)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    plan = _read_json(args.plan)
    out = os.path.abspath(args.out)
    web = plan.get("web_root", "")
    allowed = [os.path.join(web, "runs"), os.path.join(web, "evidence")] if os.path.isabs(str(web)) else []
    if not any(_under(a, out) and os.path.abspath(out) != os.path.abspath(a) for a in allowed):
        sys.stderr.write("refusing: --out must be strictly below web_root/runs or web_root/evidence\n")
        return 2
    if os.path.exists(out):
        sys.stderr.write("refusing: --out already exists\n")
        return 2

    report, prepared = preflight(plan, out)
    os.makedirs(out)
    _write_json(os.path.join(out, "preflight.json"), report)
    if report["status"] != "ready":
        return 2
    if not args.execute:
        return 0
    try:
        return run(plan, prepared, out)
    except BaseException as exc:
        _write_json(os.path.join(out, "error.json"),
                    {"error": type(exc).__name__, "message": str(exc), "cleanup": "none"})
        sys.stderr.write("execution failed: %s: %s\n" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
