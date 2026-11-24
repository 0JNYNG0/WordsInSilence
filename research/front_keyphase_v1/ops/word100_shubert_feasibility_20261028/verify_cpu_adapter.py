"""Independent CPU numerical check against AST-extracted official SHuBERT.

No network, installation, training, or project/user input search. The official
source is executed, not the adapter's computations copied into a reference.
Only model registration and distributed wrappers are replaced for local eval.
Optional --public-features points to one explicitly selected public NPZ with
face/left_hand/right_hand/body_posture arrays, not raw videos or private data.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
import types
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter, functional as F

UNIT = Path(__file__).resolve().parent
OFFICIAL = UNIT / "vendor/shubert/fairseq"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference_classes():
    ns = dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Parameter=Parameter,
              np=np, math=math, uuid=uuid, logger=logging.getLogger("official"),
              BaseFairseqModel=nn.Module, has_fused_layernorm=False,
              fsdp_wrap=lambda module: module)
    loaded = []

    def load(relative, names, strip_registration=False):
        path = OFFICIAL / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        chosen = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                  and n.name in names]
        assert {n.name for n in chosen} == set(names), (path, names)
        for node in chosen:
            if strip_registration:
                node.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + chosen, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        loaded.append({"path": str(path), "sha256": sha256(path), "objects": names,
                       "registration_decorators_removed": strip_registration})

    load("fairseq/modules/gelu.py", ["gelu", "gelu_accurate"])
    # Preserve the official activation factory's local import without importing
    # all of Fairseq (which needs unavailable training/distributed dependencies).
    fairseq_stub = types.ModuleType("fairseq")
    modules_stub = types.ModuleType("fairseq.modules")
    modules_stub.gelu, modules_stub.gelu_accurate = ns["gelu"], ns["gelu_accurate"]
    fairseq_stub.modules = modules_stub
    sys.modules["fairseq"] = fairseq_stub
    sys.modules["fairseq.modules"] = modules_stub
    load("fairseq/utils.py", ["get_activation_fn", "is_xla_tensor", "index_put"])
    ns["utils"] = types.SimpleNamespace(get_activation_fn=ns["get_activation_fn"])
    load("fairseq/modules/layer_norm.py", ["LayerNorm"])
    load("fairseq/modules/quant_noise.py", ["quant_noise"])
    load("fairseq/incremental_decoding_utils.py", ["FairseqIncrementalState", "with_incremental_state"])
    load("fairseq/modules/fairseq_dropout.py", ["FairseqDropout"])
    load("fairseq/modules/multihead_attention.py", ["MultiheadAttention"])
    load("fairseq/modules/transformer_sentence_encoder.py", ["init_bert_params"])
    load("fairseq/modules/same_pad.py", ["SamePad"])
    load("fairseq/models/wav2vec/utils.py", ["pad_to_multiple"])
    load("fairseq/models/wav2vec/wav2vec2.py", ["make_conv_pos", "TransformerEncoder", "TransformerSentenceEncoderLayer"])
    load("examples/shubert/models/shubert.py", ["SHubertModel"], strip_registration=True)
    return ns["SHubertModel"], loaded


def comparison(expected, actual):
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    delta = (expected - actual).abs()
    return {"shape": list(expected.shape), "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()), "bit_exact": bool(torch.equal(expected, actual)),
            "allclose_atol_1e-5_rtol_1e-5": bool(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))}


def source_with_labels(arrays):
    result = []
    for sample in arrays:
        row = dict(sample)
        length = row["face"].shape[0]
        for name in ("label_face", "label_left_hand", "label_right_hand", "label_body_posture"):
            row[name] = torch.zeros(length, dtype=torch.long)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-features", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(281028)
    from cpu_shubert_adapter import CPUShubertAdapter, CPUShubertConfig, stack_official_features

    checkpoint_path = UNIT / "weights/checkpoint_836_400000.pt"
    assert sha256(checkpoint_path) == "9087f47602401c07c44a9c70164ca1387373938ac4642149126a43e61c20cd7d"
    assert torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path) == []
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["cfg"]["model"]
    inspection = json.loads((UNIT / "WEIGHT_INSPECTION.json").read_text(encoding="utf-8"))
    assert config == inspection["rows"][0]["cfg"]["model"]
    adapter_config = CPUShubertConfig.from_mapping(config)
    cfg_values = dataclasses.asdict(adapter_config)
    mismatches = {key: {"checkpoint": config.get(key), "adapter": value}
                  for key, value in cfg_values.items() if config.get(key) != value}
    assert not mismatches, mismatches
    assert config["checkpoint_activations"] is False
    assert config["layer_type"] == "transformer"
    assert config["pos_conv_depth"] == 1
    assert config["conv_pos"] == 32 and config["conv_pos_groups"] == 16
    assert config["required_seq_len_multiple"] == 2
    assert config["layer_norm_first"] is False
    Reference, ast_manifest = reference_classes()
    reference = Reference(types.SimpleNamespace(**config)).eval()
    adapter = CPUShubertAdapter(adapter_config).eval()
    state = checkpoint["model"]
    assert set(reference.state_dict()) == set(adapter.state_dict()) == set(state)
    assert all(torch.isfinite(t).all() for t in state.values())
    ref_load = reference.load_state_dict(state, strict=True)
    load_report = adapter.load_exact_state_dict(state)
    assert not ref_load.missing_keys and not ref_load.unexpected_keys
    loaded_equal = {key: torch.equal(reference.state_dict()[key], adapter.state_dict()[key])
                    and torch.equal(adapter.state_dict()[key], value.float()) for key, value in state.items()}
    assert all(loaded_equal.values())
    assert list(adapter.encoder.pos_conv[0].weight_g.shape) == [1, 1, 32]
    tests = []

    @torch.inference_mode()
    def run_case(name, samples, padding_mask=None, layer=None):
        then = time.perf_counter()
        source = source_with_labels(samples)
        expected = reference.extract_features(source, padding_mask=padding_mask, kmeans_labels=None, mask=False, layer=layer)
        actual = adapter.extract_features(source, padding_mask=padding_mask, kmeans_labels=None, mask=False, layer=layer)
        row = {"name": name, "batch": len(source), "frames": source[0]["face"].shape[0],
               "padding_frames": int(padding_mask.sum()) if padding_mask is not None else 0,
               "stop_layer": layer, "final_x": comparison(expected["x"], actual["x"]), "layers": []}
        assert len(expected["layer_results"]) == len(actual["layer_results"])
        for idx, (left, right) in enumerate(zip(expected["layer_results"], actual["layer_results"])):
            row["layers"].append({"index": idx, "context": comparison(left[0], right[0]), "ffn": comparison(left[-1], right[-1])})
        if len(source) == 1:
            official_saved = torch.stack([item[-1].squeeze(1) for item in expected["layer_results"]])
            row["saved_ffn_feature"] = comparison(official_saved, stack_official_features(actual))
            row["last_ffn_equals_final_context"] = torch.equal(actual["layer_results"][-1][-1].transpose(0, 1), actual["x"])
            assert not row["last_ffn_equals_final_context"], "Test fails to distinguish official FFN feature and final context"
        row["seconds_both_models"] = time.perf_counter() - then
        checks = [row["final_x"]] + [x[k] for x in row["layers"] for k in ("context", "ffn")]
        assert all(c["allclose_atol_1e-5_rtol_1e-5"] for c in checks), row
        tests.append(row)
        print(json.dumps({"case": name, "max_abs": max(c["max_abs"] for c in checks), "seconds": row["seconds_both_models"]}), flush=True)

    def random_source(length, batch=1):
        return [{name: torch.randn(length, width) for name, width in
                 (("face", 384), ("left_hand", 384), ("right_hand", 384), ("body_posture", 14))}
                for _ in range(batch)]

    for length in (1, 2, 7, 8, 31, 32, 33):
        run_case(f"random_T{length}", random_source(length))
    samples = random_source(9, 2)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[1, -3:] = True
    run_case("batch2_odd_explicit_padding", samples, mask)
    run_case("stop_first_layer", random_source(5), layer=0)
    run_case("stop_middle_layer", random_source(6), layer=5)
    if args.public_features:
        selected = args.public_features.resolve()
        # Caller must explicitly select a public feature file inside this unit.
        assert selected.is_relative_to(UNIT.resolve())
        with np.load(selected, allow_pickle=False) as values:
            sample = {key: torch.tensor(values[key], dtype=torch.float32) for key in
                      ("face", "left_hand", "right_hand", "body_posture")}
        run_case("explicit_public_feature_sequence", [sample])
    else:
        selected = None
    rejected = {}
    for name, mutated in (("missing_key", {k: v for k, v in state.items() if k != "mask_emb"}),
                          ("extra_key", {**state, "unexpected_test_key": torch.zeros(1)}),
                          ("wrong_shape", {**state, "mask_emb": torch.zeros(1)})):
        try:
            adapter.load_exact_state_dict(mutated)
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError(f"Strict loader accepted {name}")
    result = {"status": "PASS", "scope": "CPU FP32 adapter equivalence to official pinned inference math; not sign-language semantic accuracy",
              "torch_version": torch.__version__, "threads": torch.get_num_threads(),
              "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_keys": len(state),
              "checkpoint_elements": sum(v.numel() for v in state.values()),
              "adapter_sha256": sha256(UNIT / "cpu_shubert_adapter.py"),
              "verifier_sha256": sha256(__file__), "model_config_checked": cfg_values,
              "config_mismatches": mismatches, "all_loaded_weight_values_exact": all(loaded_equal.values()),
              "strict_load_report": load_report, "invalid_state_rejected": rejected,
              "official_source_ast": ast_manifest,
              "reference_changes": ["SHubertModel registration decorator removed; BaseFairseqModel replaced by nn.Module (no feature math in base)",
                                    "fsdp_wrap identity for single-process CPU; checkpoint_activations=False verified",
                                    "Only official GELU definitions provided through fairseq.modules local import; all active math uses official source"],
              "tests": tests, "public_feature_path": str(selected) if selected else None,
              "public_feature_sha256": sha256(selected) if selected else None,
              "limitations": ["Random synthetic features do not establish detection, Korean sign understanding, feedback correctness, or end-to-end speed.",
                              "Public feature numerical match, if present, validates adapter calculation only; input preparation is a separate audit.",
                              "No training, candidate selection, private footage, or production replacement."],
              "elapsed_seconds": time.perf_counter() - started}
    (UNIT / "ADAPTER_INDEPENDENT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["SHuBERT CPU adapter 독립 수치 검산: PASS",
             f"공식 원본 AST와 실제 체크포인트 {len(state)}개 키/모든 가중치가 일치합니다.",
             f"{len(tests)}개 시험에서 12층의 FFN 출력과 문맥 출력, 홀짝 길이/패딩/중간층 종료를 비교했습니다.",
             "FFN residual 이전의 layer[-1]을 공식 저장 특징으로 확인했습니다.",
             "conv_pos=32, weight_norm dim=2, padding multiple=2, 채널별 LayerNorm을 검산했습니다.",
             "코드 구현의 동등성 검산이며 수어 의미, 정확한 손 위치 또는 교육 피드백의 검증이 아닙니다.",
             "공개 실제 특징: " + (str(selected) if selected else "아직 없음; 임의 수치 입력만 검산"),
             "최대 절대 오차: " + str(max(c["ffn"]["max_abs"] for t in tests for c in t["layers"])),
             "독립 검산 담당: Codex 별도 agent shubert_adapter_verifier. 독립 사람 정답이 아닙니다."]
    (UNIT / "ADAPTER_INDEPENDENT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "tests": len(tests), "elapsed": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
