#!/usr/bin/env python3
"""
Tools/train_svm.py
------------------
extract_features.py 가 출력한 CSV에서 LinearSVC 를 학습하고
C# SVMClassifier 가 읽을 수 있는 JSON 모델을 저장한다.

사용법:
    python train_svm.py --features features.csv
    python train_svm.py --features features.csv --output Assets/StreamingAssets/svm_model.json
    python train_svm.py --features features.csv --c 0.1 --eval

필요 패키지:
    pip install scikit-learn numpy
"""

import argparse
import csv
import json
import math
import os
import sys

try:
    import numpy as np
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    print("[오류] 필요 패키지 없음: pip install scikit-learn numpy")
    sys.exit(1)

FEATURE_NAMES = [
    "ApertureTrend",
    "WristTrajDx", "WristTrajDy",
    "OscCountX", "OscCountY",
    "PathClosure", "WristZAmp",
    "WristCenterY", "WristYOscillation",
    "FaceDistNose", "FaceDistEar", "FaceDistMouth",
    "IndexDirX", "IndexDirY", "IndexDirZ",
    "WristCenterX", "TwoHandDist", "TwoHandCorr",
    "PalmNormalX", "PalmNormalY", "PalmNormalZ",
    "FingerExtThumb", "FingerExtIndex", "FingerExtMiddle",
    "FingerExtRing", "FingerExtPinky",
]
N_FEATURES = len(FEATURE_NAMES)

# ─── CSV 로드 ─────────────────────────────────────────────────────────────────

def load_csv(path):
    labels = []
    feats  = []
    with open(path, encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = row.get("label", "").strip()
            if not label:
                continue
            vals = []
            for name in FEATURE_NAMES:
                v = row.get(name, "")
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(math.nan)
            labels.append(label)
            feats.append(vals)
    return labels, feats

# ─── 전처리 ──────────────────────────────────────────────────────────────────

def impute_with_means(X_raw, means=None):
    """NaN → 열 평균 대체. means가 없으면 X_raw에서 계산."""
    arr = np.array(X_raw, dtype=float)
    if means is None:
        means = np.nanmean(arr, axis=0)
        # 전체가 NaN인 열은 0으로 대체
        means = np.where(np.isnan(means), 0.0, means)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = means[j]
    return arr, means

# ─── 학습 ────────────────────────────────────────────────────────────────────

def train(features_path, output_path, C=1.0, do_eval=False):
    print(f"[로드] {features_path}")
    labels, feats_raw = load_csv(features_path)

    if len(labels) == 0:
        print("[오류] 샘플 없음.")
        sys.exit(1)

    classes = sorted(set(labels))
    print(f"[정보] 샘플 {len(labels)}개 / 클래스 {len(classes)}개: {classes}")

    # 단어별 샘플 수 출력
    from collections import Counter
    counts = Counter(labels)
    for word in sorted(counts):
        cnt = counts[word]
        warn = " ⚠️ 부족" if cnt < 10 else ""
        print(f"  {word:12s}: {cnt}개{warn}")

    if len(classes) < 2:
        print("[오류] 클래스가 2개 이상이어야 합니다.")
        sys.exit(1)

    # NaN 대체
    X_raw, col_means = impute_with_means(feats_raw)

    # StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    y = np.array(labels)

    # 교차 검증 + 혼동행렬 (선택)
    if do_eval:
        svm_eval = LinearSVC(C=C, max_iter=5000, random_state=42)
        n_splits = min(5, min(counts.values()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # 정확도
            scores = cross_val_score(svm_eval, X, y, cv=cv, scoring="accuracy")
            print(f"\n[CV {n_splits}-fold] 정확도: {scores.mean():.1%} ± {scores.std():.1%}")
            print(f"  각 폴드: {[f'{s:.1%}' for s in scores]}")

            # 클래스별 정밀도/재현율/F1
            y_pred_cv = cross_val_predict(svm_eval, X, y, cv=cv)
            print("\n[클래스별 성능]")
            report = classification_report(y, y_pred_cv, zero_division=0)
            print(report)

            # 혼동행렬 — 오인식이 10% 이상인 쌍만 출력
            cm = confusion_matrix(y, y_pred_cv, labels=classes)
            print("[혼동행렬 — 자주 혼동되는 쌍 (오인식 10% 이상)]")
            found_any = False
            for i, true_cls in enumerate(classes):
                row_total = cm[i].sum()
                if row_total == 0:
                    continue
                for j, pred_cls in enumerate(classes):
                    if i == j:
                        continue
                    pct = cm[i][j] / row_total
                    if pct >= 0.10:
                        print(f"  {true_cls:12s} → {pred_cls:12s}  {cm[i][j]}/{row_total}  ({pct:.0%})")
                        found_any = True
            if not found_any:
                print("  (10% 이상 혼동되는 쌍 없음 — 양호)")

            # 전체 혼동행렬 (단어 수 <= 12일 때만 출력, 그 이상은 너무 넓음)
            if len(classes) <= 12:
                print("\n[전체 혼동행렬] (행=실제, 열=예측)")
                header = f"{'':12s}" + "".join(f"{c[:8]:>9s}" for c in classes)
                print(header)
                for i, true_cls in enumerate(classes):
                    row = f"{true_cls[:12]:12s}" + "".join(
                        f"{'→' + str(cm[i][j]) if i != j and cm[i][j] > 0 else str(cm[i][j]):>9s}"
                        for j in range(len(classes))
                    )
                    print(row)
        else:
            print("[경고] 샘플이 너무 적어 교차 검증 생략 (클래스당 최소 2개 필요)")

    # 최종 학습
    svm = LinearSVC(C=C, max_iter=5000, random_state=42)
    svm.fit(X, y)

    train_acc = (svm.predict(X) == y).mean()
    print(f"\n[학습 완료] 훈련 정확도: {train_acc:.1%}")

    # 다중 클래스: coef_ shape = (n_classes, n_features)
    # 이진 클래스: coef_ shape = (1, n_features) — 강제로 (2, n_features) 로 확장
    coef = svm.coef_
    intercept = svm.intercept_
    svm_classes = list(svm.classes_)

    if len(svm_classes) == 2:
        # 이진 분류: OVR 확장 (class0=+1, class1=-1 방향으로 2행)
        coef      = np.vstack([coef, -coef])
        intercept = np.array([intercept[0], -intercept[0]])
        print("[정보] 이진 분류 → OVR 2행으로 확장")

    # 피처 중요도 (|가중치| 합계 기준 상위 10개)
    importance = np.abs(coef).mean(axis=0)
    top_idx    = importance.argsort()[::-1][:10]
    print("\n[피처 중요도 Top10]")
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {FEATURE_NAMES[idx]:25s} {importance[idx]:.4f}")

    # 모델 저장
    model_data = {
        "classes":        svm_classes,
        "feature_names":  FEATURE_NAMES,
        "n_classes":      len(svm_classes),
        "n_features":     N_FEATURES,
        "feature_means":  col_means.tolist(),
        "feature_scales": scaler.scale_.tolist(),
        "weights_flat":   coef.flatten().tolist(),
        "biases":         intercept.tolist(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(model_data, fp, ensure_ascii=False, indent=2)

    print(f"\n[저장] {os.path.abspath(output_path)}")
    print("       Unity StreamingAssets/ 에 svm_model.json 을 복사하면 자동 로드됩니다.")
    print("       (Editor 빌드 시: Assets/StreamingAssets/svm_model.json)")

# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="피처 CSV → LinearSVC 학습 → JSON 모델 저장"
    )
    parser.add_argument("--features", required=True,
                        help="extract_features.py 출력 CSV 경로")
    parser.add_argument("--output", default="Assets/StreamingAssets/svm_model.json",
                        help="모델 JSON 저장 경로 (기본: Assets/StreamingAssets/svm_model.json)")
    parser.add_argument("--c", type=float, default=1.0,
                        help="LinearSVC 정규화 파라미터 C (기본: 1.0, 작을수록 더 강한 정규화)")
    parser.add_argument("--eval", action="store_true",
                        help="교차 검증으로 정확도 평가 후 학습")
    args = parser.parse_args()

    train(args.features, args.output, C=args.c, do_eval=args.eval)


if __name__ == "__main__":
    main()
