"""Summarize already evaluated checkpoints. Never train, infer, or select on test."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from statistics import mean

U = Path(__file__).resolve().parent


def read(name):
    return json.loads((U / name).read_text(encoding="utf-8-sig"))


def write(name, value):
    (U / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    verified = read("verification/ALL42_SCORES_INDEPENDENT.json")
    assert verified["status"] == "PASS" and verified["fixed_run_count"] == 42
    groups = {r["group"]: r for r in verified["group_summaries"]}
    frozen = read("FROZEN_PREPARATION.json")
    protected = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
                 for p, expected in frozen["protected"].items()}
    assert len(protected) == 32 and all(protected.values())
    write("verification/FINAL_PROTECTION.json", {"status": "PASS", "count": len(protected), "files": protected})
    names = {"mobilenet_v3_small": "MobileNetV3-Small", "mobilenetv4_conv_small": "MobileNetV4-Conv-Small (2024)", "tinynext_t": "TinyNeXt-T (2025)"}
    lat = read("CPU_LATENCY.json")["models"]
    adapted = read("ADAPTED_ENCODER_SELECTION.json")
    ftlat = {r["name"]: r["p95_ms"] for r in adapted["candidates"]}
    models = []
    for kind in ("probe", "ft"):
        for name, display in names.items():
            g = groups[f"{kind}_{name}"]
            vals = [read(f"models/{kind}_{name}_{s}.json")["validation"] for s in g["seeds"]]
            models.append({"name": display, "kind": "영상부 고정·분류부 학습" if kind == "probe" else "영상 마지막 단계 미세조정",
                           "val_accuracy": mean(v["accuracy"] for v in vals),
                           "val_macro_f1": mean(v["macro_f1_supported"] for v in vals),
                           "test_accuracy": g["accuracy_mean"], "test_range": g["accuracy_range"],
                           "p95_ms": lat[name]["p95_ms"] if kind == "probe" else ftlat[name]})
    labels = {
        "observer_J_T": "관절 + 앞뒤 (J_T)",
        "observer_R_T": "RGB 고정 + 앞뒤 (R_T)",
        "observer_RJ_F": "RGB 고정 + 관절·현재 (RJ_F)",
        "observer_RJ_T": "RGB 고정 + 관절 + 앞뒤 (RJ_T)",
        "adapted_R_T": "RGB 미세조정 + 앞뒤 (R_T)",
        "adapted_RJ_F": "RGB 미세조정 + 관절·현재 (RJ_F)",
        "adapted_RJ_T": "RGB 미세조정 + 관절 + 앞뒤 (RJ_T)",
        "observer_MASK_T": "자료 존재 표시만 (진단 대조군)",
    }
    conditions = []
    for key, label in labels.items():
        g = groups[key]
        conditions.append({"name": label, "test_mean": g["accuracy_mean"], "test_range": g["accuracy_range"],
                           "macro_f1": g["macro_f1_supported_mean"], "correct_per_seed": g["correct_per_seed"],
                           "missing_mean": g["subgroups"]["detector_missing"]["accuracy_mean"],
                           "change_mean": g["subgroups"]["near_label_change"]["accuracy_mean"]})
    cards = read("SOURCE_FEATURE_CARDS.json")["profiles"]
    ks = read("KSL_TRANSFER_PREDICTIONS.json")["summary"]
    assert ks["frames"] == 468 and ks["reference_pairs"] == 24 and not ks["real_correctness_validation"]
    selection = read("OBSERVER_SELECTION.json")["selected"]
    assert selection["kind"] == "observer" and selection["condition"] == "RJ_F"
    improved = (groups["adapted_R_T"]["accuracy_mean"] - groups["observer_R_T"]["accuracy_mean"]) * 100
    dropped = (groups["adapted_RJ_T"]["accuracy_mean"] - groups["observer_RJ_T"]["accuracy_mean"]) * 100
    summary = {
        "at": dt.datetime.now().astimezone().isoformat(),
        "status": "학습·고정 비교 완료 / 통합 후보 미채택 / 한국수어 교정 검증은 미완료",
        "findings": [
            f"영상+앞뒤 정보로 손모양을 분류하는 후보는 미세조정 후 41.45% → 68.24%로 개선됐습니다 (+{improved:.2f}%p).",
            f"그러나 관절과 모두 합친 후보는 72.25% → 67.89%로 낮아졌습니다 ({dropped:.2f}%p). 정보가 많다고 자동으로 좋아지지 않았습니다.",
            "개발 검증으로 미리 고른 통합 후보도 다른 사람의 최종 자료에서 관절 기준을 넘지 못했습니다. 기존 C·T3-F와 운영 피드백은 유지합니다.",
            "최근 후보까지 실제로 비교한 결과, 이번 작은 자료·CPU·학습 조건에서는 MobileNetV3-Small이 선택됐습니다. 모든 모델 중 최적이라는 뜻은 아닙니다.",
        ],
        "stages": [
            {"n": 1, "title": "출처 있는 특징 프로필", "status": "partial", "detail": "100개 프로필을 정리하고 32개 이름에 사전 항목 후보 49개·특징 설명 168개를 연결했습니다. 동음이의어·영상 변이와 교육적 필수 규칙은 자동 확정하지 않았습니다."},
            {"n": 2, "title": "믿을 수 있는 물리 특징 자료", "status": "complete", "detail": "사람이 수형을 주석한 외부 공개 손 사진 3,359장·45종을 확보했습니다. 공식 촬영자 9명을 학습 5명·개발 2명·최종 2명으로 분리했습니다. 한국수어 교육 정답 자료와는 다릅니다."},
            {"n": 3, "title": "모델 선택과 실제 학습", "status": "complete", "detail": "10개 후보 계열을 조사하고 V3·V4·TinyNeXt를 실측했습니다. 3 seed 반복으로 분류부·영상 미세조정·정보 결합을 합쳐 42개 학습 결과를 완성했습니다."},
            {"n": 4, "title": "네 조건과 최종 비교", "status": "complete", "detail": "고정한 다른 사람의 382장으로 한 번씩 평가하고 저장 결과를 별도 코드로 검산했습니다. RGB 단독 보완은 효과가 있었지만 통합 후보의 일반화가 부족해 채택하지 않았습니다."},
            {"n": 5, "title": "다중 참고 DTW 연결", "status": "partial", "detail": "세 물리 특징을 손·시간·참고별로 연결하고 공개 한국수어 36구간에서 기술 재생을 했습니다. 누락 시 분모 보존은 확인했지만 정상 교차·오류 교정의 정확도를 증명할 독립 정답은 부족합니다."},
            {"n": 6, "title": "별도 웹 후보", "status": "not_activated", "detail": "원본과 추정을 보는 연구 보고 화면은 완성했습니다. 오프라인 통과 조건을 만족하지 못해 사용자 촬영을 받는 운영 후보는 활성화하지 않았습니다."},
        ],
        "model_selection": models, "conditions": conditions,
        "facts": {
            "source_profiles": len(cards), "source_supported": sum(bool(p.get("source_variants")) for p in cards),
            "source_claims": sum(len(v.get("feature_claims", [])) for p in cards for v in p.get("source_variants", [])),
            "manual_images": 3359, "classes": 45, "fit": "2,693장 / 공식 ID 01–05", "validation": "284장 / ID 06–07", "test": "382장 / ID 08–09",
            "seeds": "27101 / 27102 / 27103", "training_runs": 42,
            "보조 진단 자료 수": "최종 관절 미검출 18장 / 라벨 변화 근처 19장(6개 변화·5클립)",
            "최종 자료의 클래스 범위": "45종 출력 중 정답이 등장하는 수형 28종; macro-F1은 이 28종 평균",
            "protected_files_unchanged": "32개 해시 일치", "user_mentor_data_used": False, "operating_changed": False, "educational_verdict_enabled": False,
        },
        "ksl": {
            "재생 범위": "공개 한국수어 12표현 × 3기록 ID = 36구간·468프레임 (손 입력 456개)",
            "사용 후보": "최종 점수 전에 선택한 RJ_F 3seed 동일 비율 평균; 성능 합격 후보가 아닌 기술 판독용",
            "비교 연결": "ID 03의 12구간을 ID 01·02의 24개 참고와 각각 정렬. 기존 위치·팔·손 관절 기반 DTW 경로 고정.",
            "구분한 정보": "검지만 폄 / 네 손가락 폄 / 네 손가락 닫힘. 확률·판단 보류·참고별 비교 가능 범위 보존.",
            "중요한 설계": "관측이 사라져도 참고의 필요한 시간 분모를 유지. 비교 가능한 수치만 표시하고 근거 부족은 별도 기록.",
            "실제 비교 가능 범위": "참고별 13표본 × 3특징 × 24참고 = 936칸. 완전 비교 600·부분 5·비교 불가 331. 손 정보를 모두 지운 대조에서는 분모 936 그대로 전부 비교 불가.",
            "분모의 의미": "이번 기술시험에서 고정한 비교 범위입니다. 13표본 모두 언어적으로 필수라는 뜻이 아니며 정밀 각도·위치 오차의 정확도 검증도 아닙니다.",
            "기존 AI 초안과의 관계": "기존 745개 확인값 중 모델이 답한 605개에서 547개 일치. AI끼리의 일치이며 정확도가 아님; 학습·선택에 미사용.",
            "미검증": "정상 교차 보존 정확도, 필수 동작 누락 탐지, 교육적으로 틀린 행동의 교정 정확도, 새 사용자 일반화",
            "운영 상태": "C·T3-F·기존 웹·DTW 유지 / 사용자 촬영 접근 0 / 교정 판정 비활성",
        },
        "limitations": [
            "손모양 분류를 학습한 것이며 단어의 의미·허용 범위·교정 규칙을 학습한 것은 아닙니다. TCN 인식률 개선도 이번 시험에서 측정하지 않았습니다.",
            "개발 자료는 284장 중 273장이 한 ID에 치우쳐 있습니다. 최종 시험도 2명뿐이며 3 seed는 새 사람이 3배 늘었다는 뜻이 아닙니다.",
            "새 사람이 최종 평가에 포함됐지만 최종 자료에 없는 17종, 다른 촬영 환경, 한국수어 전체로의 일반화는 미검증입니다.",
            "앞뒤 정보는 실제 존재하는 인접 사진에 한정됩니다. 긴 연속 동영상 학습이 아니며 빠른 A→B→A 변화의 최종 정답 표본은 없습니다.",
            "관절 미검출 18장에서는 미세조정 RGB+시간이 각 seed 15장씩 맞혔습니다. 손 사진은 보이지만 검출기가 놓친 경우이며 완전 가림 복구나 정밀 좌표 복구 성공을 뜻하지 않습니다.",
            "한국수어 36구간의 기존 확인값은 AI 초안입니다. 독립적인 사람 정답 없이 정상 교차가 보존됐다거나 올바른 교정을 한다고 결론낼 수 없습니다.",
            "세 모델의 학습 가능한 마지막 단계 크기는 다릅니다. 작은 고정 예산의 적합성 비교이며 각 모델을 최대 성능까지 최적화한 순위가 아닙니다.",
            "CPU p95는 손 입력 한 개의 영상 모델 처리 범위입니다. 손 탐지·양손·영상 읽기·DTW·웹캠까지 포함한 실사용 속도가 아닙니다.",
            "외부 DeepHand 자료는 CC BY-NC-SA 3.0 연구 자산으로 분리했습니다. 이번 산출물을 운영 배포 허가로 해석하지 않습니다.",
        ],
        "next_actions": [
            "기존 관절 경로와 C·T3-F를 유지합니다. 새 검출기·거대 모델을 더 얹는 작업은 우선하지 않습니다.",
            "다음 후보는 관절이 잘 보일 때 그대로 사용하고, 손은 보이는데 관절만 놓쳤을 때 RGB로 손모양을 보조하는 방식으로 좁힙니다. 이번 18장 결과는 가설 근거이며 같은 최종 시험으로 조건을 다시 맞추지 않습니다.",
            "기존 독립 물리 주석을 우선 찾고, 부족한 한국수어 장면만 손모양 유지·변화·회전·교차·미관측으로 나눈 작은 확인 묶음을 준비합니다. AI 초안을 정답으로 바꾸지 않고 필요한 사람 확인을 집중합니다.",
            "참고의 정상 차이와 명확한 오류를 분리한 새 공개 평가 자료를 먼저 고정합니다. 비교 가능한 비율, 잘못된 교정, 놓친 오류를 함께 비교해야 합니다.",
            "이 근거로 선택적 RGB 보조가 통과한 뒤에만 DTW의 실제 교정 정확도를 검증하고 웹의 별도 후보를 활성화합니다. 현재 사용자에게 재촬영을 요구할 단계는 아닙니다.",
        ],
        "selection_unchanged_after_test": True,
        "new_training_or_inference_by_summary": False,
    }
    write("SUMMARY.json", summary)
    write("FINAL_GATE.json", {
        "status": "COMPARISON_COMPLETE_FUSION_NOT_ADOPTED_KSL_FEEDBACK_UNVALIDATED",
        "observer_selected_before_test": selection,
        "selection_changed_after_test": False,
        "learned_candidate_generalization_gate": "FAIL: preselected fusion does not improve held-person baseline",
        "ksl_educational_feedback_gate": "NOT_ESTABLISHED: independent KSL correctness/normal-variation truth unavailable",
        "technical_bridge": "implemented, saved public replay; separate verification recorded in verification/",
        "production_activation": False, "user_mentor_data_used": False,
        "all_six_operating_stages_complete": False,
        "no_more_test_tuning": True,
    })
    print(json.dumps({"summary": "written", "models": len(models), "conditions": len(conditions), "training_runs": 42, "protected": len(protected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
