"""Find target-labelled spans in already-owned AIHub103 sentence metadata.

Scan REAL01 front labels for discovery, then metadata only for REAL01..16.
No held-out media, model calls, source registration or downloads.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = Path("C:/Users/local-user/Downloads/수어 영상/1.Training")
LABEL = BASE / "[라벨]01_real_sen_morpheme.zip"
VIDEO = BASE / "[원천]01_real_sen_video.zip"
source = ROOT / "research/front_keyphase_v1/src/wis_lab/service/neural.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
targets = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "PROJECT_18" for t in n.targets))
assert len(targets) == 18
terms = {label: target for target, label in targets.items()}
terms.update({"감사": "GAMSAHAMNIDA", "죄송": "JOESONGHADA", "미안하다": "JOESONGHADA",
              "미안합니다": "JOESONGHADA", "잠자다": "JADA", "공부하다": "GONGBU",
              "사랑하다": "SARANG", "운동하다": "UNDONG", "인사하다": "INSA",
              "반갑다": "INSA", "안녕하세요": "INSA"})
gap_targets = {"GAM", "GONGBU", "INSA", "MAEKJU", "MEOKDA", "MUL", "PPANG", "SARANG", "UNDONG"}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def norm(value):
    return re.sub(r"\d+$", "", value.strip())


def match_annotations(document):
    matches = []
    for i, annotation in enumerate(document.get("data", [])):
        for attr in annotation.get("attributes", []):
            raw = attr.get("name", "")
            normalized = norm(raw)
            if normalized in terms:
                matches.append((i, raw, normalized, terms[normalized], float(annotation["start"]), float(annotation["end"])))
    return matches


with zipfile.ZipFile(VIDEO) as video_zip:
    local_media = {Path(m.filename).name: {"member": m.filename, "bytes": m.file_size,
                   "crc32": f"{m.CRC:08x}"} for m in video_zip.infolist() if m.filename.endswith("_F.mp4")}

all_rows, discovery, missing_labels = [], [], []
with zipfile.ZipFile(LABEL) as labels:
    names = set(labels.namelist())
    front01 = sorted(n for n in names if re.fullmatch(r"morpheme/01/NIA_SL_SEN\d{4}_REAL01_F_morpheme\.json", n))
    assert len(front01) == 2000
    selected_sentences = set()
    for name in front01:
        doc = json.loads(labels.read(name))
        matches = match_annotations(doc)
        if not matches:
            continue
        sentence = re.search(r"SEN(\d{4})", name).group(1)
        selected_sentences.add(sentence)
        discovery.append({"sentence_id": sentence, "label_member": name,
                          "target_candidates": sorted({m[3] for m in matches}),
                          "matched_raw_labels": sorted({m[1] for m in matches})})
    for sentence in sorted(selected_sentences):
        for signer in range(1, 17):
            clip = f"NIA_SL_SEN{sentence}_REAL{signer:02}_F"
            member = f"morpheme/{signer:02}/{clip}_morpheme.json"
            if member not in names:
                missing_labels.append(member)
                continue
            payload = labels.read(member)
            doc = json.loads(payload)
            duration = float(doc["metaData"]["duration"])
            for i, raw, normalized, target, start, end in match_annotations(doc):
                role = "train_candidate" if signer <= 9 else "unassigned" if signer == 10 else "development_only" if signer <= 13 else "reserved_test_metadata_only"
                all_rows.append({"clip_id": clip, "sentence_id": sentence, "signer_id": signer, "view": "F",
                                 "split_role": role, "annotation_index": i, "raw_label": raw,
                                 "normalized_search_label": normalized, "target_candidate_not_approved": target,
                                 "label_member": member, "label_member_sha256": hashlib.sha256(payload).hexdigest(),
                                 "annotation_start_s": start, "annotation_end_s": end, "clip_duration_s": duration,
                                 "numeric_bounds_valid": 0 <= start < end <= duration,
                                 "semantic_equivalence": "UNREVIEWED", "semantic_target_id": None,
                                 "training_approved": False, "full_action_boundary_verified": False,
                                 "source_video_basename": clip + ".mp4",
                                 "local_video_archive_match": local_media.get(clip + ".mp4")})
summary = []
for target, label in targets.items():
    rows = [r for r in all_rows if r["target_candidate_not_approved"] == target]
    clip_ids = {r["clip_id"] for r in rows}
    media = {r["clip_id"]: r["local_video_archive_match"] for r in rows if r["local_video_archive_match"]}
    summary.append({"target": target, "label": label, "previous_word_catalog_gap": target in gap_targets,
                    "candidate_annotation_rows": len(rows), "candidate_clips": len(clip_ids),
                    "sentence_ids": sorted({r["sentence_id"] for r in rows}),
                    "signer_ids_in_matched_metadata": sorted({r["signer_id"] for r in rows}),
                    "raw_labels": dict(Counter(r["raw_label"] for r in rows)),
                    "roles": dict(Counter(r["split_role"] for r in rows)),
                    "local_front_video_members": len(media), "local_front_video_bytes": sum(m["bytes"] for m in media.values()),
                    "approved_training_rows": 0})
report = {"status": "PASS_METADATA_DISCOVERY_NOT_SEMANTIC_APPROVAL", "label_archive": str(LABEL),
          "video_archive": str(VIDEO), "label_archive_bytes": LABEL.stat().st_size,
          "video_archive_bytes": VIDEO.stat().st_size,
          "discovery_signer": 1, "front_labels_scanned": len(front01),
          "selected_sentence_count": len(discovery), "candidate_annotation_rows": len(all_rows),
          "target_count_with_candidate": sum(s["candidate_annotation_rows"] > 0 for s in summary),
          "previous_gap_targets_with_candidate": [s["target"] for s in summary if s["previous_word_catalog_gap"] and s["candidate_annotation_rows"]],
          "invalid_numeric_bound_rows": sum(not r["numeric_bounds_valid"] for r in all_rows),
          "missing_label_members": missing_labels, "summary": summary,
          "normalization": "strip whitespace and trailing digits for search only; raw labels retained",
          "limitations": ["Discovery limited to labels present for REAL01; no absence claim for the whole corpus.",
                         "Equivalent meaning and complete action boundaries are not established by a label hit.",
                         "Training ZIP signer14-16 labels are metadata only; no reserved pixels or arrays read.",
                         "View variants must stay in the same signer/recording split; annotations within one clip are correlated."],
          "new_downloads": 0, "new_extractions": 0, "training_runs": 0, "classifier_inference_calls": 0}
save("catalog.json", report)
save("candidate_rows.json", {"rows": all_rows, "training_approved": False})
save("discovery_real01.json", discovery)
print(json.dumps({k: v for k, v in report.items() if k not in ("summary", "limitations", "missing_label_members")}, ensure_ascii=False))
print(json.dumps([{k: s[k] for k in ("target", "candidate_annotation_rows", "local_front_video_members", "local_front_video_bytes", "raw_labels")} for s in summary], ensure_ascii=False))
