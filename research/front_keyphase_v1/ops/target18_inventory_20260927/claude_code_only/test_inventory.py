"""Synthetic unittest suite for inventory.py. Fictitious labels only; no external reads/commands."""
import copy, hashlib, json, os, tempfile, unittest
from unittest import mock

import inventory as inv

TARGETS = {"WORD_A": "LabelA", "WORD_B": "LabelB", "WORD_C": "LabelC"}
MODEL_BYTES = b"synthetic-dummy-model-bytes-not-weights"
MSHA = hashlib.sha256(MODEL_BYTES).hexdigest()


def m2(pred, truth, rank=1, margin=1.0, pk="prediction", ck="correct", rk="true_rank", mk="true_margin"):
    return {pk: pred, ck: pred == truth, rk: rank, mk: margin}


def base_docs():
    conf = lambda wid, lab: {"source": "participant_self_report", "is_expert_verification": False,
                             "actual_word_id": wid, "label": lab, "status": "CONFIRMED"}
    ctrial = lambda tid, top, p18: {"trial_id": tid, "service_top1_93": top, "post_hoc_top1_93": top,
                                    "stored_paired_top1": top, "full_candidate_count": 93, "project18_top1": p18,
                                    "identity": {"checkpoint_sha256": MSHA}, "model_status": "READY"}
    pub = lambda g, c, rid, t, p, p18=None: {"group": g, "condition": c, "id": rid, "true": t, "signer_id": 11,
        "models": {"control": dict({"full93": m2(p, t, pk="pred", rk="rank", mk="margin")},
                                   **({"project18": m2(p18, t, pk="pred", rk="rank", mk="margin")} if p18 else {}))}}
    return {
        "head_replay": {"candidate": {"sha256": MSHA}, "rows": [
            {"cohort": "self_old18", "id": "old1", "truth": "LabelA", "input_sha256": "x",
             "models": {"candidate": {"full93": m2("LabelB", "LabelA", 2, -1.0), "project18": m2("LabelA", "LabelA")}}},
            {"cohort": "self_new18", "id": "new1", "truth": "LabelB", "input_sha256": "y",
             "models": {"candidate": {"full93": m2("LabelB", "LabelB"), "project18": m2("LabelB", "LabelB")}}},
            {"cohort": "public_confirmed2", "id": "pubc1", "truth": "LabelA", "input_sha256": "z",
             "models": {"candidate": {"full93": m2("LabelA", "LabelA"), "project18": m2("LabelA", "LabelA")}}}]},
        "connection": {"trials": [ctrial("s1", "LabelA", "WORD_A"), ctrial("s2", "LabelX", "WORD_C"),
                                  ctrial("s3", "LabelB", "WORD_B")]},
        "paired": {"session_id": "sess1", "trials": [
            {"trial_id": "s1", "selected_word_id": "WORD_B", "selected_label": "LabelB", "confirmation": conf("WORD_A", "LabelA"),
             "operating": {"pred": "LabelA", "correct_vs_confirmed": True}, "candidate": {"pred": "LabelB"}, "frames": 12},
            {"trial_id": "s2", "selected_word_id": "WORD_C", "selected_label": "LabelC", "confirmation": conf("WORD_C", "LabelC"),
             "operating": {"pred": "LabelX", "correct_vs_confirmed": False}, "candidate": {"pred": "LabelC"}, "frames": 12},
            {"trial_id": "s3", "selected_word_id": "WORD_B", "selected_label": "LabelB",
             "confirmation": {"source": None, "is_expert_verification": False, "actual_word_id": None, "label": None,
                              "status": "UNCONFIRMED"},
             "operating": {"pred": "LabelB", "correct_vs_confirmed": None}, "candidate": {"pred": "LabelB"}, "frames": 12}]},
        "live": {"session_id": "live1", "trials": [
            {"trial_id": "l1", "origin": "camera", "requested_word_id": "WORD_A", "actual_word_id": "WORD_A",
             "actual_label": "LabelA", "self_report_source": "participant_self_report", "model_id": "m",
             "checkpoint_sha256": MSHA, "identity_fixed_sha256": MSHA, "model_status": "READY", "full_class_count": 93,
             "full_top1": "LabelA", "top1_matches_self_report": True, "observation_status": "OK",
             "execution_assessment": "UNCONFIGURED", "accepted": None}]},
        "public_evaluation": {"models": {"control": {"sha256": MSHA}}, "rows": [
            pub("public", "intact", "p1", "LabelA", "LabelA"),
            pub("public", "reviewed_masked", "p1", "LabelA", "LabelB"),
            pub("mp", "whole_clip", "p2", "LabelB", "LabelA", "LabelB"),
            pub("public", "intact", "p3", "OtherLabel", "LabelA")]},
        "ksl": {"model_sha256": MSHA, "rows": [
            {"id": "k1", "candidate_label": "LabelA", "meaning_evidence": "USER_CONFIRMED", "previous_diagnostic_exposure": False,
             "status": "OK", "prediction": "LabelA", "agreement": True, "target_rank": 1, "support": {"observed_hand_fraction": 1.0}},
            {"id": "k2", "candidate_label": "LabelB", "meaning_evidence": "AI_VISUAL_REVIEW", "previous_diagnostic_exposure": True,
             "status": "OK", "prediction": "LabelC", "agreement": False, "target_rank": 5, "support": {}},
            {"id": "k3", "candidate_label": "LabelA", "meaning_evidence": "AI_VISUAL_REVIEW", "previous_diagnostic_exposure": False,
             "status": "NO_HANDS", "prediction": None, "agreement": None}]},
        "training_coverage": {"words": [
            {"word": lab, "registered_training_rows": 3, "nikl_reference_rows": 1, "other_training_rows": 2,
             "provided_coordinate_development_rows": 0} for lab in ["LabelA", "LabelB", "LabelC", "OtherLabel"]],
            "nikl_only_words": []},
    }


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.out = os.path.join(self.td.name, "out")

    def write(self, docs=None, cfg_mut=None):
        d = self.td.name
        with open(os.path.join(d, "model.bin"), "wb") as f:
            f.write(MODEL_BYTES)
        roles = {}
        for role, doc in (docs or base_docs()).items():
            p = os.path.join(d, role + ".json")
            with open(p, "w", encoding="utf-8-sig") as f:  # BOM allowed
                json.dump(doc, f, ensure_ascii=False)
            with open(p, "rb") as f:
                roles[role] = {"path": p, "sha256": hashlib.sha256(f.read()).hexdigest()}
        cfg = {"model": {"path": os.path.join(d, "model.bin"), "sha256": MSHA}, "target_words": TARGETS, "source_roles": roles}
        if cfg_mut:
            cfg_mut(cfg)
        cp = os.path.join(d, "config.json")
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return cp

    def run_docs(self, mutate=None, cfg_mut=None):
        docs = base_docs()
        if mutate:
            mutate(docs)
        return inv.run(self.write(docs, cfg_mut), self.out)

    def assertFails(self, mutate=None, cfg_mut=None, text=""):
        with self.assertRaises(inv.InventoryError) as cm:
            self.run_docs(mutate, cfg_mut)
        self.assertIn(text, str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.out, "inventory.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "INVENTORY.md")))

    def row(self, result, tid):
        return next(r for r in result["user_rows"] if r["trial_id"] == tid)


class TestInventory(Base):
    def test_baseline_aggregation(self):
        r = self.run_docs()
        self.assertEqual([p["word_id"] for p in r["per_word"]], list(TARGETS))
        s = r["user_cohort_summary"]
        self.assertEqual((s["self_old18"]["count"], s["self_old18"]["full93_matches"], s["self_old18"]["restricted18_matches"]), (1, 0, 1))
        self.assertEqual((s["stored_session_replay"]["count"], s["stored_session_replay"]["with_truth"],
                          s["stored_session_replay"]["no_truth"]), (3, 2, 1))
        pa = r["per_word"][0]
        self.assertEqual(pa["public"]["public|intact"]["n"], 1)
        self.assertEqual(pa["public"]["public|reviewed_masked"]["full93_matches"], 0)
        self.assertEqual((pa["ksl"]["new"]["n"], pa["ksl"]["new"]["matches"], pa["ksl"]["new"]["unscored"]), (1, 1, 1))
        self.assertEqual(r["per_word"][1]["ksl"]["previous_exposure"]["matches"], 0)
        self.assertEqual(r["skipped"]["public_nontarget_truth"], 1)
        self.assertIn("TRAINING_REFERENCE_REUSE", r["public_conditions"]["public|intact"]["evidence_role"])
        self.assertIn("KNOWN_TRAINING_SIGNERS", r["public_conditions"]["mp|whole_clip"]["evidence_role"])
        self.assertTrue(all(u["expert_validation"] is False and u["execution_correctness"] == "UNVALIDATED" for u in r["user_rows"]))
        self.assertTrue(all(u["source"] for u in r["user_rows"]))
        self.assertNotIn("overall_accuracy", json.dumps(r))
        with open(os.path.join(self.out, "INVENTORY.md"), encoding="utf-8") as f:
            self.assertIn(MSHA, f.read())

    def test_old_kept_public_confirmed_skipped(self):
        r = self.run_docs()
        ids = [u["trial_id"] for u in r["user_rows"]]
        self.assertNotIn("pubc1", ids)
        self.assertEqual(self.row(r, "old1")["cohort"], "self_old18")
        self.assertEqual(r["skipped"]["head_replay_public_confirmed2"], 1)

    def test_exact_model_mismatch_rejected(self):
        other = "0" * 64
        self.assertFails(lambda d: d["head_replay"]["candidate"].update(sha256=other), text="candidate model")
        self.assertFails(lambda d: d["live"]["trials"][0].update(identity_fixed_sha256=other), text="identity_fixed")
        self.assertFails(lambda d: d["connection"]["trials"][0]["identity"].update(checkpoint_sha256=other), text="checkpoint")
        self.assertFails(lambda d: d["public_evaluation"]["models"]["control"].update(sha256=other), text="control")
        self.assertFails(cfg_mut=lambda c: c["model"].update(sha256=other), text="model")

    def test_duplicate_user_trial_rejected(self):
        self.assertFails(lambda d: d["live"]["trials"][0].update(trial_id="old1"), text="duplicate user trial id")
        self.assertFails(lambda d: d["live"]["trials"].append(copy.deepcopy(d["live"]["trials"][0])), text="duplicate")
        self.assertFails(lambda d: d["ksl"]["rows"].append(copy.deepcopy(d["ksl"]["rows"][0])), text="duplicate source id")

    def test_selected_word_is_not_truth(self):
        u = self.row(self.run_docs(), "s1")
        self.assertEqual(u["actual"]["word_id"], "WORD_A")
        self.assertEqual(u["selected"]["word_id"], "WORD_B")
        self.assertTrue(u["full93"]["match"])

    def test_unconfirmed_not_counted_correct(self):
        r = self.run_docs()
        u = self.row(r, "s3")
        self.assertIsNone(u["actual"])
        self.assertIsNone(u["full93"]["match"])
        self.assertEqual(r["per_word"][1]["user"]["stored_session_replay"]["n"], 0)
        self.assertEqual([x["trial_id"] for x in r["unknown_confirmations"]], ["s3"])

    def test_confirmed_without_actual_or_bad_status_rejected(self):
        def null_actual(d):
            d["paired"]["trials"][0]["confirmation"]["actual_word_id"] = None
        self.assertFails(null_actual, text="CONFIRMED")
        self.assertFails(lambda d: d["paired"]["trials"][2]["confirmation"].update(status="MAYBE"), text="unrecognized")
        self.assertFails(lambda d: d["paired"]["trials"][0]["confirmation"].update(is_expert_verification=True), text="expert")

    def test_full93_and_restricted18_separate(self):
        r = self.run_docs()
        u = self.row(r, "s2")
        self.assertFalse(u["full93"]["match"])
        self.assertTrue(u["restricted18"]["match"])
        c = r["per_word"][2]["user"]["stored_session_replay"]
        self.assertEqual((c["n"], c["full93_matches"], c["restricted18_n"], c["restricted18_matches"]), (1, 0, 1, 1))

    def test_reported_correct_inconsistency_rejected(self):
        def lie(d):
            d["head_replay"]["rows"][0]["models"]["candidate"]["full93"]["correct"] = True
        self.assertFails(lie, text="inconsistent")

    def test_no_live_coverage_fabrication(self):
        r = self.run_docs()
        self.assertEqual(r["live_uncovered_target_words"], ["WORD_B", "WORD_C"])
        self.assertIsNone(self.row(r, "l1")["restricted18"])
        self.assertEqual(r["user_cohort_summary"]["live_current_model"]["restricted18_n"], 0)
        self.assertEqual(r["per_word"][1]["user"]["live_current_model"]["n"], 0)

    def test_source_hash_mismatch(self):
        self.assertFails(cfg_mut=lambda c: c["source_roles"]["ksl"].update(sha256="f" * 64), text="before: sha256 mismatch for ksl")

    def test_source_changed_during_run(self):
        cp, real = self.write(), inv.load_json

        def tamper(path):
            data = real(path)
            if path.endswith("live.json"):
                with open(path, "a", encoding="utf-8") as f:
                    f.write(" ")
            return data
        with mock.patch.object(inv, "load_json", side_effect=tamper):
            with self.assertRaises(inv.InventoryError) as cm:
                inv.run(cp, self.out)
        self.assertIn("after: sha256 mismatch for live", str(cm.exception))
        self.assertFalse(os.path.exists(self.out))

    def test_no_overwrite(self):
        os.makedirs(self.out)
        p = os.path.join(self.out, "inventory.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write("keep")
        with self.assertRaises(inv.InventoryError):
            inv.run(self.write(), self.out)
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep")
        self.assertFalse(os.path.exists(os.path.join(self.out, "INVENTORY.md")))


if __name__ == "__main__":
    unittest.main()
