using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Canonical;
using WordsInSilence.Recognition.Quality;

namespace WordsInSilence.Recognition.ExpectedSign
{
    /// <summary>
    /// ISignVerifier 구현체.
    ///
    /// 세 가지 모드를 단일 진입점에서 처리한다:
    ///   - LegacyDtw:           HandPoseEvaluator.Compare (기존 DTW 경로)
    ///   - LegacySvm:           SVMClassifier.Classify    (기존 SVM 경로)
    ///   - ExpectedSignPrototype: 좌표 표준화 → 구간 검출 → (템플릿 비교 예정)
    ///
    /// 레거시 경로(DTW/SVM)는 수정 없이 그대로 호출한다.
    /// </summary>
    public sealed class ExpectedSignVerifier : ISignVerifier
    {
        // ─── 의존성 ──────────────────────────────────────────────────────────

        /// <summary>LegacyDtw 모드에서 비교할 레퍼런스 녹화 목록.</summary>
        readonly IEnumerable<HandPoseRecording> _references;

        /// <summary>구간 검출 설정. null 이면 SegmentationConfig.Default 사용.</summary>
        readonly SegmentationConfig _segConfig;

        // ─── 생성자 ──────────────────────────────────────────────────────────

        /// <summary>
        /// </summary>
        /// <param name="references">LegacyDtw 모드에서 사용할 레퍼런스 녹화 목록. null 허용 (그 경우 DTW 모드는 실패 반환).</param>
        /// <param name="segConfig">구간 검출 설정. null 이면 기본값 사용.</param>
        public ExpectedSignVerifier(
            IEnumerable<HandPoseRecording> references = null,
            SegmentationConfig segConfig = null)
        {
            _references = references;
            _segConfig  = segConfig;
        }

        // ─── ISignVerifier 구현 ──────────────────────────────────────────────

        /// <inheritdoc/>
        public SignVerificationResult Verify(
            HandPoseRecording recording,
            string expectedSignId,
            RecognitionMode mode = RecognitionMode.LegacyDtw)
        {
            if (recording == null)
                return SignVerificationResult.QualityFailure(
                    expectedSignId,
                    FrameQualityStatus.RecordingTooShort,
                    "recording is null",
                    mode);

            try
            {
                return mode switch
                {
                    RecognitionMode.LegacyDtw            => RunLegacyDtw(recording, expectedSignId),
                    RecognitionMode.LegacySvm            => RunLegacySvm(recording, expectedSignId),
                    RecognitionMode.ExpectedSignPrototype => RunPrototype(recording, expectedSignId),
                    _                                     => throw new ArgumentOutOfRangeException(nameof(mode), mode, null),
                };
            }
            catch (Exception ex)
            {
                Debug.LogError($"[ExpectedSignVerifier] Verify exception (mode={mode}): {ex}");
                return SignVerificationResult.QualityFailure(
                    expectedSignId,
                    FrameQualityStatus.InternalError,
                    ex.Message,
                    mode);
            }
        }

        // ─── LegacyDtw ───────────────────────────────────────────────────────

        SignVerificationResult RunLegacyDtw(HandPoseRecording recording, string expectedSignId)
        {
            var result = new SignVerificationResult
            {
                ExpectedSignId = expectedSignId,
                Mode           = RecognitionMode.LegacyDtw,
                QualityStatus  = FrameQualityStatus.Analyzable,
                Debug          = new DebugMetrics { TotalFrames = recording.Frames?.Length ?? 0 },
            };

            if (_references == null)
            {
                result.Accepted      = false;
                result.FailureReason = "No reference recordings provided for LegacyDtw.";
                return result;
            }

            var bestMatch = HandPoseEvaluator.FindBestMatch(recording, _references);
            result.PredictedSignId = bestMatch.MotionId;
            result.Debug.DtwScore  = bestMatch.Score;
            result.OverallScore    = bestMatch.Score;

            // expectedSignId 와 비교 (DisplayName 또는 MotionId 중 하나라도 일치)
            bool nameMatch = !string.IsNullOrEmpty(bestMatch.DisplayName)
                             && (bestMatch.DisplayName == expectedSignId
                                 || bestMatch.MotionId  == expectedSignId);

            result.Accepted = nameMatch && bestMatch.Score >= HandPoseEvaluator.MinConfidence;
            if (!result.Accepted && string.IsNullOrEmpty(result.FailureReason))
                result.FailureReason = nameMatch
                    ? $"Score {bestMatch.Score:F3} < MinConfidence {HandPoseEvaluator.MinConfidence}"
                    : $"Best match '{bestMatch.DisplayName}' != expected '{expectedSignId}'";

            return result;
        }

        // ─── LegacySvm ───────────────────────────────────────────────────────

        SignVerificationResult RunLegacySvm(HandPoseRecording recording, string expectedSignId)
        {
            var result = new SignVerificationResult
            {
                ExpectedSignId = expectedSignId,
                Mode           = RecognitionMode.LegacySvm,
                QualityStatus  = FrameQualityStatus.Analyzable,
                Debug          = new DebugMetrics { TotalFrames = recording.Frames?.Length ?? 0 },
            };

            if (!SVMClassifier.IsLoaded)
            {
                result.Accepted      = false;
                result.FailureReason = "SVM model not loaded. Call SVMClassifier.TryLoad() first.";
                return result;
            }

            var svmResult = SVMClassifier.Classify(recording);
            if (!svmResult.HasValue)
            {
                result.Accepted      = false;
                result.FailureReason = "SVMClassifier returned no result.";
                return result;
            }

            result.PredictedSignId            = svmResult.Value.ClassName;
            result.Debug.SvmConfidenceMargin  = svmResult.Value.ConfidenceMargin;
            result.OverallScore               = Mathf.Clamp01(svmResult.Value.ConfidenceMargin);

            result.Accepted = svmResult.Value.ClassName == expectedSignId;
            if (!result.Accepted)
                result.FailureReason = $"SVM predicted '{svmResult.Value.ClassName}' != expected '{expectedSignId}'";

            return result;
        }

        // ─── ExpectedSignPrototype ────────────────────────────────────────────

        SignVerificationResult RunPrototype(HandPoseRecording recording, string expectedSignId)
        {
            var debug = new DebugMetrics { TotalFrames = recording.Frames?.Length ?? 0 };

            // 1. 좌표 표준화 (Unity 런타임 = 변환 불필요)
            var canonical = FrameCanonicalizer.Canonicalize(
                recording, CoordinateConvention.UnityRuntime);
            debug.CoordinateTransform = "UnityRuntime (no transform)";

            if (canonical == null || canonical.Length == 0)
                return SignVerificationResult.QualityFailure(
                    expectedSignId, FrameQualityStatus.RecordingTooShort,
                    "Canonicalization produced 0 frames.", RecognitionMode.ExpectedSignPrototype);

            // 2. 구간 검출
            var segResult = MotionSegmentDetector.Detect(canonical, _segConfig);
            debug.HandFrameRatio = segResult.HandFrameRatio;
            debug.PoseFrameRatio = segResult.PoseFrameRatio;
            debug.SegmentFrames  = segResult.Segment?.Length ?? 0;
            debug.Notes          = segResult.DebugNote;

            if (segResult.Status != FrameQualityStatus.Analyzable)
                return new SignVerificationResult
                {
                    ExpectedSignId = expectedSignId,
                    Mode           = RecognitionMode.ExpectedSignPrototype,
                    QualityStatus  = segResult.Status,
                    Accepted       = false,
                    FailureReason  = segResult.DebugNote,
                    Debug          = debug,
                };

            // 3. 템플릿 비교 (미구현 — 향후 ExpectedSignTemplate 에셋 시스템 추가 예정)
            //    현재는 구간 검출 성공 여부만 반환한다.
            debug.Notes += " | Template comparison: not implemented (prototype stub).";

            return new SignVerificationResult
            {
                ExpectedSignId = expectedSignId,
                Mode           = RecognitionMode.ExpectedSignPrototype,
                QualityStatus  = FrameQualityStatus.Analyzable,
                Accepted       = false,                    // 템플릿 없이 승인 불가
                OverallScore   = float.NaN,                // 미평가
                MotionScore    = float.NaN,
                HandShapeScore = float.NaN,
                LocationScore  = float.NaN,
                DirectionScore = float.NaN,
                FailureReason  = "Prototype: template comparison not yet implemented.",
                Debug          = debug,
            };
        }
    }
}
