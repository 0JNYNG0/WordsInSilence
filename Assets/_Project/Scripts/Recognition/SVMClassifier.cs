using System;
using System.IO;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// train_svm.py 로 학습된 Linear SVM 모델을 로드하여 수어 동작을 분류한다.
    ///
    /// 사용법:
    ///   SVMClassifier.TryLoad();          // 앱 시작 시 한 번
    ///   var result = SVMClassifier.Classify(recording);
    ///   if (result.HasValue) Debug.Log(result.Value.ClassName);
    ///
    /// 모델 파일: StreamingAssets/svm_model.json (train_svm.py 출력)
    /// </summary>
    public static class SVMClassifier
    {
        // ─── 직렬화 가능 모델 데이터 ────────────────────────────────────────────

        [Serializable]
        class ModelData
        {
            public string[] classes;
            public string[] feature_names;
            public int      n_classes;
            public int      n_features;
            public float[]  feature_means;   // StandardScaler μ
            public float[]  feature_scales;  // StandardScaler σ
            public float[]  weights_flat;    // [n_classes × n_features] row-major
            public float[]  biases;          // [n_classes]
        }

        // ─── 상태 ──────────────────────────────────────────────────────────────

        static ModelData _model;

        /// <summary>모델이 로드된 상태인지 여부.</summary>
        public static bool IsLoaded => _model != null && _model.classes?.Length > 0;

        // ─── 결과 ──────────────────────────────────────────────────────────────

        public struct SVMResult
        {
            /// <summary>예측 단어 (DisplayName).</summary>
            public string ClassName;
            /// <summary>1위 점수 − 2위 점수. 클수록 확실한 예측.</summary>
            public float  ConfidenceMargin;
        }

        // ─── 로드 ──────────────────────────────────────────────────────────────

        /// <summary>
        /// StreamingAssets/svm_model.json 을 로드한다.
        /// 선택적으로 절대 경로를 직접 지정할 수 있다.
        /// 성공 시 true, 파일 없음 또는 오류 시 false.
        /// </summary>
        public static bool TryLoad(string jsonPath = null)
        {
            try
            {
                string path = jsonPath
                    ?? Path.Combine(Application.streamingAssetsPath, "svm_model.json");

                if (!File.Exists(path))
                {
                    Debug.LogWarning($"[SVMClassifier] 모델 파일 없음: {path}");
                    return false;
                }

                string json = File.ReadAllText(path);
                _model = JsonUtility.FromJson<ModelData>(json);

                if (_model == null || _model.classes == null || _model.classes.Length == 0)
                {
                    Debug.LogError("[SVMClassifier] 모델 파싱 실패 (빈 데이터).");
                    _model = null;
                    return false;
                }

                Debug.Log($"[SVMClassifier] 모델 로드 완료: {_model.classes.Length}개 클래스, {_model.n_features}개 피처");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[SVMClassifier] 로드 실패: {e.Message}");
                _model = null;
                return false;
            }
        }

        // ─── 분류 ──────────────────────────────────────────────────────────────

        /// <summary>
        /// 녹화 데이터를 분류하여 예측 단어와 신뢰도 마진을 반환한다.
        /// 모델이 로드되지 않았으면 null.
        /// </summary>
        public static SVMResult? Classify(HandPoseRecording recording)
        {
            if (!IsLoaded || recording?.Frames == null) return null;

            var frames = HandPoseEvaluator.TrimInactivePublic(recording.Frames);
            if (frames == null || frames.Length == 0) return null;

            float[] rawFeats = HandPoseEvaluator.ExtractFeatureVector(frames);
            float[] feats    = Normalize(rawFeats);

            int nC = _model.n_classes;
            int nF = _model.n_features;

            float best1 = float.MinValue, best2 = float.MinValue;
            int   bestIdx = 0;

            for (int c = 0; c < nC; c++)
            {
                float score = _model.biases[c];
                int   offset = c * nF;
                for (int k = 0; k < nF && k < feats.Length; k++)
                    score += _model.weights_flat[offset + k] * feats[k];

                if (score > best1)
                {
                    best2   = best1;
                    best1   = score;
                    bestIdx = c;
                }
                else if (score > best2)
                {
                    best2 = score;
                }
            }

            return new SVMResult
            {
                ClassName        = _model.classes[bestIdx],
                ConfidenceMargin = best1 - best2,
            };
        }

        // ─── 내부 ──────────────────────────────────────────────────────────────

        /// <summary>StandardScaler 정규화. NaN 값은 학습 데이터 평균으로 대체.</summary>
        static float[] Normalize(float[] raw)
        {
            int n = _model.n_features;
            var result = new float[n];
            for (int i = 0; i < n && i < raw.Length; i++)
            {
                float v = (float.IsNaN(raw[i]) || float.IsInfinity(raw[i]))
                          ? _model.feature_means[i]
                          : raw[i];
                float s = _model.feature_scales[i] > 1e-8f ? _model.feature_scales[i] : 1f;
                result[i] = (v - _model.feature_means[i]) / s;
            }
            return result;
        }
    }
}
