"""Independent saved CPU comparison arithmetic. No model forward or training.

Run only after TRAINING_COMPLETE.json exists. Missing completion is a harmless
PENDING result; source inference, extraction, and scoring are never repeated.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys

import numpy as np
import torch

U = Path(__file__).resolve().parents[1]
OLD = U.parent / "word100_feature_learning_pipeline_20261027"
sys.path.insert(0, str(U / "deps"))
sys.path.insert(0, str(OLD / "deps"))
from safetensors.torch import load_file


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    a = value.detach().cpu().contiguous()
    return hashlib.sha256(
        str(a.dtype).encode() + str(tuple(a.shape)).encode() + a.numpy().tobytes()
    ).hexdigest()


def independent_metrics(targets, logits, class_names):
    y = np.asarray(targets, dtype=np.int64)
    z = np.asarray(logits)
    assert z.shape == (len(y), len(class_names)) and np.isfinite(z).all()
    pred = np.argmax(z, axis=1)
    matrix = np.bincount(
        y * len(class_names) + pred, minlength=len(class_names) ** 2
    ).reshape(len(class_names), len(class_names))
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    detail = []
    for i, name in enumerate(class_names):
        support = int(true_counts[i])
        guessed = int(predicted_counts[i])
        hit = int(true_positive[i])
        precision = hit / guessed if guessed else 0.0
        recall = hit / support if support else 0.0
        f1 = 2 * hit / (support + guessed) if support + guessed else 0.0
        detail.append(dict(class_name=name, support=support, predicted=guessed,
                           tp=hit, precision=precision, recall=recall, f1=f1))
    return dict(n=len(y), correct=int(np.sum(pred == y)),
                accuracy=float(np.mean(pred == y)),
                macro_f1_supported=float(np.mean([
                    d["f1"] for d in detail if d["support"]
                ])), supported_classes=int(np.sum(true_counts > 0)),
                per_class=detail, confusion=matrix.tolist())


def assert_metrics(actual, saved):
    for key in ["n", "correct", "supported_classes", "confusion"]:
        assert actual[key] == saved[key], key
    for key in ["accuracy", "macro_f1_supported"]:
        assert abs(actual[key] - saved[key]) < 1e-12, key
    assert len(actual["per_class"]) == len(saved["per_class"])
    for a, b in zip(actual["per_class"], saved["per_class"]):
        assert a["class_name"] == b["class"]
        for key in ["support", "predicted", "tp"]:
            assert a[key] == b[key], key
        for key in ["precision", "recall", "f1"]:
            assert abs(a[key] - b[key]) < 1e-12, key


def main():
    if not (U / "TRAINING_COMPLETE.json").exists():
        print("PENDING: saved nine-run completion does not exist; no model operation performed.")
        return
    checks = []

    def ck(name, ok, detail=None):
        assert ok, name
        checks.append(dict(name=name, passed=True, detail=detail))

    protocol = read(U / "FROZEN_PROTOCOL.json")
    manifest = read(U / "TRAINING_MANIFEST.json")
    complete = read(U / "TRAINING_COMPLETE.json")
    ph = sha(U / "FROZEN_PROTOCOL.json")
    rows = manifest["rows"]
    names = protocol["models"]
    seeds = protocol["seeds"]
    ck("complete_pinned", complete["runs"] == 9 and complete["test_predictions"] == 0
       and complete["protocol_sha256"] == ph)
    ck("code_pinned", sha(U / "compare_cpu.py") == protocol["code_sha256"])
    ck("manifest_pinned", sha(U / "TRAINING_MANIFEST.json") == protocol["manifest_sha256"])
    ck("metrics_dependency_pinned", sha(OLD / "train_observer.py") == protocol["metrics_code_sha256"])
    ck("expected_models_and_seeds", names == ["v3", "repvit", "v4hybrid"]
       and seeds == [28101, 28102, 28103])
    ck("fit_dev_counts", len(rows) == 2977 and all(r["role"] == "fit" for r in rows[:2489])
       and all(r["role"] == "validation" for r in rows[2489:]))
    ck("fit_people", {r["signer"] for r in rows[:2489]} == {"signer01", "signer02", "signer04", "signer05"})
    ck("dev_people", {r["signer"] for r in rows[2489:]} == {"signer03", "signer06", "signer07"})
    ck("no_spent_final_people", all(r["signer"] not in {"signer08", "signer09"} for r in rows))
    ck("45_fit_classes", len({r["target"] for r in rows[:2489]}) == 45)
    ck("labels_match_author_order", all(manifest["classes"][r["target"]] == r["label"] for r in rows))
    for key in ["signer", "clip", "broadcast", "image_sha256", "pixel_sha256"]:
        ck(key + "_disjoint", not ({r[key] for r in rows[:2489]} & {r[key] for r in rows[2489:]}))
    targets = np.asarray([r["target"] for r in rows[2489:]], dtype=np.int64)
    people = np.asarray([r["signer"] for r in rows[2489:]])
    ids = [r["id"] for r in rows[2489:]]
    expected_stems = [f"{name}_{seed}" for name in names for seed in seeds]
    present = sorted(p.stem for p in (U / "models").glob("*.pt") if not p.stem.endswith("_resume"))
    ck("exact_nine_final_checkpoints", present == sorted(expected_stems), present)
    expected_orders = {}
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        expected_orders[seed] = [hashlib.sha256(torch.randperm(2489, generator=gen).numpy().tobytes()).hexdigest()
                                 for _ in range(20)]
    all_runs = []
    artifacts = {}
    for name in names:
        spec = protocol["specs"][name]
        source = Path(spec["source"])
        ck(name + "_official_source", sha(source) == spec["source_sha256"])
        if source.suffix == ".safetensors":
            initial = load_file(str(source), device="cpu")
        else:
            initial = torch.load(source, map_location="cpu", weights_only=True)
        for seed in seeds:
            stem = f"{name}_{seed}"
            cp = U / "models" / (stem + ".pt")
            result_path = cp.with_suffix(".json")
            npz_path = U / "models" / (stem + "_dev.npz")
            saved = read(result_path)
            ck(stem + "_saved_pins", saved["protocol_sha256"] == ph
               and saved["checkpoint_sha256"] == sha(cp) and saved["dev_sha256"] == sha(npz_path))
            ck(stem + "_metadata", saved["name"] == name and saved["seed"] == seed
               and saved["epoch"] == 20 and saved["test_predictions"] == 0 and saved["source_unchanged"] is True)
            with np.load(npz_path, allow_pickle=False) as data:
                logits = data["logits"].copy()
                ck(stem + "_saved_indices", data["ids"].tolist() == ids
                   and np.array_equal(data["targets"], targets) and np.array_equal(data["signer"], people))
            ck(stem + "_saved_shape", logits.shape == (488, 45) and np.isfinite(logits).all())
            overall = independent_metrics(targets, logits, manifest["classes"])
            assert_metrics(overall, saved["dev"])
            by_person = {}
            for signer in ["signer03", "signer06", "signer07"]:
                mask = people == signer
                by_person[signer] = independent_metrics(targets[mask], logits[mask], manifest["classes"])
                assert_metrics(by_person[signer], saved["per_signer"][signer])
            ck(stem + "_pooled_and_people_arithmetic", True)
            balanced = (by_person["signer03"]["macro_f1_supported"] + by_person["signer07"]["macro_f1_supported"]) / 2
            ck(stem + "_balanced_selection_arithmetic", abs(balanced - saved["balanced_selection_macro_f1"]) < 1e-12)
            trace = saved["trace"]
            ck(stem + "_exposure_trace", len(trace) == 20 and all(
                r["epoch"] == i + 1 and r["count"] == 2489 and r["unique"] == 2489
                and r["order_sha256"] == expected_orders[seed][i]
                and np.isfinite(r["fit_loss"]) and r["fit_loss"] >= 0
                for i, r in enumerate(trace)))
            checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
            ck(stem + "_checkpoint_identity", checkpoint["name"] == name and checkpoint["seed"] == seed
               and checkpoint["epoch"] == 20 and checkpoint["protocol_sha256"] == ph)
            state = checkpoint["model"]
            ck(stem + "_frozen_tensors", all(tensor_sha(state[k]) == h for k, h in spec["frozen_states"].items()))
            backbone = [k for k in spec["trainable_names"] if k not in spec["new_classifier_names"]]
            changed = [k for k in backbone if tensor_sha(state[k]) != tensor_sha(initial[k])]
            ck(stem + "_actual_backbone_learning", len(changed) > 0, len(changed))
            ck(stem + "_classifier45_and_finite", all(state[k].shape[0] == 45 and torch.isfinite(state[k]).all()
                 for k in spec["new_classifier_names"]))
            all_runs.append(dict(model=name, seed=seed, n=488, accuracy=overall["accuracy"],
                correct=overall["correct"], macro_f1=overall["macro_f1_supported"], balanced_macro_f1=balanced,
                per_person={s: {k: q[k] for k in ["n", "correct", "accuracy", "macro_f1_supported", "supported_classes"]}
                            for s, q in by_person.items()}))
            artifacts[stem] = dict(checkpoint=sha(cp), result=sha(result_path), predictions=sha(npz_path))
            del checkpoint, state, logits
        del initial
    summary = []
    for name in names:
        rr = [r for r in all_runs if r["model"] == name]
        acc = [r["accuracy"] for r in rr]
        score = [r["balanced_macro_f1"] for r in rr]
        summary.append(dict(model=name, n_per_seed=488, seeds=seeds,
                            accuracy_mean=float(np.mean(acc)), accuracy_range=[min(acc), max(acc)],
                            balanced_macro_f1_mean=float(np.mean(score)), balanced_macro_f1_range=[min(score), max(score)],
                            person_means={s: {k: float(np.mean([r["per_person"][s][k] for r in rr]))
                                               for k in ["accuracy", "macro_f1_supported"]}
                                          for s in ["signer03", "signer06", "signer07"]}))
    protected = read(U / "PROTECTED.json")
    ck("protected_files_unchanged", all(sha(path) == h for path, h in protected.items()), len(protected))
    output = dict(status="PASS", at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  check_count=len(checks), checks=checks, fixed_run_count=9,
                  inference_or_training_performed=False, producer_functions_imported=False,
                  final382_accessed=False, user_mentor_media_accessed=False,
                  scope="Independent saved-data arithmetic and tensor/contract checks, not independent new-person final evaluation.",
                  runs=all_runs, group_summaries=summary, artifacts=artifacts,
                  protocol_sha256=ph, script_sha256=sha(__file__),
                  limits=["Nine runs reuse the same488 development images; three seeds are not three new people.",
                          "Signer03 was used in historical fitting. Official pretrained restart is checked, but this is still a development split.",
                          "Exposure trace validates frozen code plus saved order/count contracts; it is not independent per-minibatch source-ID telemetry.",
                          "No latency winner is selected by this verifier; latency needs its own frozen-runtime review.",
                          "External45 handshape scores are not TCN word-recognition or Korean Sign Language teaching correctness."])
    target = U / "verification" / "SAVED9_INDEPENDENT.json"
    if target.exists():
        raise RuntimeError("Independent result already exists; preserve rather than overwrite")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", checks=len(checks), summaries=summary), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    main()
