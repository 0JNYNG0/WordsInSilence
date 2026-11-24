using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Segmentation;
using WordsInSilence.Settings;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// 공용 레퍼런스 기반 수어 인식기.
    /// 요소별 점수(손모양, 위치, 궤적, 양손)를 분해하고 피드백 코드를 생성한다.
    /// 기존 HandPoseEvaluator(DTW 기반)를 내부적으로 활용한다.
    /// </summary>
    public sealed class SignRecognizer
    {
        readonly RecognitionSettings _settings;

        public SignRecognizer(RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
        }

        /// <summary>
        /// 캡처된 프레임을 레퍼런스들과 비교하여 판정한다.
        /// </summary>
        public SignRecognitionResult Recognize(
            List<HandPoseFrame> capturedFrames,
            IList<SignReference> references,
            CaptureQualityInfo quality = null)
        {
            var result = new SignRecognitionResult();
            quality = quality ?? new CaptureQualityInfo();

            if (capturedFrames == null || capturedFrames.Count < 3)
            {
                result.Status = AttemptResultStatus.Invalid;
                result.InvalidReason = "insufficient_frames";
                return result;
            }

            // 품질 게이트: 화면 이탈
            if (quality.OutOfFrame)
            {
                result.Status = AttemptResultStatus.Uncertain;
                result.AddFeedback("CAMERA_OUT_OF_FRAME");
                return result;
            }

            // 사용자 특징 추출
            var userFeatures = SignFeatureExtractor.Extract(capturedFrames);
            result.UserFeatures = userFeatures;

            if (!userFeatures.Valid)
            {
                result.Status = AttemptResultStatus.Invalid;
                result.InvalidReason = "feature_extraction_failed";
                return result;
            }

            // 결측 마스크 검사: 필수 랜드마크 관찰 가능성
            if (userFeatures.HandFrameRatio < 0.3f)
            {
                result.Status = AttemptResultStatus.Uncertain;
                result.AddFeedback("CAMERA_HAND_NOT_VISIBLE");
                return result;
            }

            // 각 레퍼런스와 비교
            float bestScore = 0f;
            float secondBestScore = 0f;
            string bestSignId = null;
            SignComponentScores bestScores = null;

            foreach (var reference in references)
            {
                var scores = CompareWithReference(capturedFrames, userFeatures, reference);
                float combined = scores.CombinedScore;

                if (combined > bestScore)
                {
                    secondBestScore = bestScore;
                    bestScore = combined;
                    bestSignId = reference.SignId;
                    bestScores = scores;
                }
                else if (combined > secondBestScore)
                {
                    secondBestScore = combined;
                }
            }

            result.BestSignId = bestSignId;
            result.GlobalScore = bestScore;
            result.ComponentScores = bestScores ?? new SignComponentScores();
            result.AmbiguityGap = bestScore - secondBestScore;

            // 판정
            if (bestScore >= _settings.PassThreshold &&
                result.AmbiguityGap >= _settings.AmbiguityGap)
            {
                result.Status = AttemptResultStatus.Passed;
            }
            else if (result.AmbiguityGap < _settings.AmbiguityGap && bestScore > _settings.GlobalScoreFloor)
            {
                result.Status = AttemptResultStatus.Uncertain;
                result.AddFeedback("AMBIGUOUS_MATCH");
            }
            else if (bestScore < _settings.GlobalScoreFloor)
            {
                result.Status = AttemptResultStatus.Incorrect;
            }
            else
            {
                result.Status = AttemptResultStatus.Incorrect;
            }

            // 피드백 코드: 가장 낮은 요소 1~2개
            if (result.Status != AttemptResultStatus.Passed && bestScores != null)
            {
                var feedbacks = GenerateFeedbackCodes(bestScores);
                foreach (var fb in feedbacks)
                    result.AddFeedback(fb);
            }

            return result;
        }

        /// <summary>
        /// 단일 레퍼런스와의 요소별 점수를 계산한다.
        /// </summary>
        SignComponentScores CompareWithReference(
            List<HandPoseFrame> userFrames,
            SignFeatureSet userFeatures,
            SignReference reference)
        {
            var scores = new SignComponentScores();

            // 기존 DTW 기반 전체 점수 활용
            if (reference.Recording != null)
            {
                // HandPoseFrame → HandPoseRecording 변환 후 기존 evaluator 사용
                var userRecording = FramesToRecording(userFrames);
                float dtwScore = HandPoseEvaluator.Compare(userRecording, reference.Recording);
                scores.DtwScore = dtwScore;
            }

            // 레퍼런스 특징이 있으면 요소별 비교
            if (reference.Features != null && reference.Features.Valid)
            {
                scores.PositionScore = ComparePosition(userFeatures.Position, reference.Features.Position);
                scores.TrajectoryScore = CompareTrajectory(userFeatures.Trajectory, reference.Features.Trajectory);
                scores.TwoHandScore = CompareTwoHand(userFeatures.TwoHandRelation, reference.Features.TwoHandRelation);
                scores.HandshapeScore = scores.DtwScore; // DTW가 손모양 포함
            }
            else
            {
                // 특징 없으면 DTW 점수만 사용
                scores.PositionScore = scores.DtwScore;
                scores.TrajectoryScore = scores.DtwScore;
                scores.TwoHandScore = scores.DtwScore;
                scores.HandshapeScore = scores.DtwScore;
            }

            // 가중 합산
            scores.CombinedScore =
                scores.HandshapeScore * 0.30f +
                scores.PositionScore * 0.25f +
                scores.TrajectoryScore * 0.25f +
                scores.TwoHandScore * 0.20f;

            return scores;
        }

        float ComparePosition(PositionFeatures user, PositionFeatures reference)
        {
            float score = 0f;
            float weight = 0f;

            if (user.HasRight && reference.HasRight)
            {
                float dist = Vector2.Distance(user.RightMeanPosition, reference.RightMeanPosition);
                score += Mathf.Clamp01(1f - dist / 1.5f);
                weight += 1f;
            }

            if (user.HasLeft && reference.HasLeft)
            {
                float dist = Vector2.Distance(user.LeftMeanPosition, reference.LeftMeanPosition);
                score += Mathf.Clamp01(1f - dist / 1.5f);
                weight += 1f;
            }

            return weight > 0f ? score / weight : 0.5f;
        }

        float CompareTrajectory(TrajectoryFeatures user, TrajectoryFeatures reference)
        {
            if (!user.Valid || !reference.Valid) return 0.5f;

            float score = 0f;

            // 이동량 비교
            float dispRatio = reference.TotalDisplacement > 0.01f
                ? user.TotalDisplacement / reference.TotalDisplacement
                : 1f;
            score += Mathf.Clamp01(1f - Mathf.Abs(1f - dispRatio));

            // 방향 전환 횟수
            float oscDiffX = Mathf.Abs(user.DirectionChangesX - reference.DirectionChangesX);
            score += Mathf.Clamp01(1f - oscDiffX * 0.3f);

            float oscDiffY = Mathf.Abs(user.DirectionChangesY - reference.DirectionChangesY);
            score += Mathf.Clamp01(1f - oscDiffY * 0.3f);

            // 경로 폐합도
            float closureDiff = Mathf.Abs(user.PathClosure - reference.PathClosure);
            score += Mathf.Clamp01(1f - closureDiff * 2f);

            return score / 4f;
        }

        float CompareTwoHand(TwoHandFeatures user, TwoHandFeatures reference)
        {
            if (!user.Valid || !reference.Valid) return 0.5f;

            float distDiff = Mathf.Abs(user.MeanDistance - reference.MeanDistance);
            float heightDiff = Mathf.Abs(user.MeanHeightDiff - reference.MeanHeightDiff);

            float s1 = Mathf.Clamp01(1f - distDiff / 1.0f);
            float s2 = Mathf.Clamp01(1f - heightDiff / 0.5f);

            return (s1 + s2) * 0.5f;
        }

        List<string> GenerateFeedbackCodes(SignComponentScores scores)
        {
            var feedbacks = new List<(string code, float score)>
            {
                ("HANDSHAPE_MISMATCH", scores.HandshapeScore),
                ("POSITION_MISMATCH", scores.PositionScore),
                ("TRAJECTORY_MISMATCH", scores.TrajectoryScore),
                ("TWO_HAND_MISMATCH", scores.TwoHandScore),
            };

            feedbacks.Sort((a, b) => a.score.CompareTo(b.score));

            var result = new List<string>();
            // 가장 낮은 1~2개
            int count = Mathf.Min(2, feedbacks.Count);
            for (int i = 0; i < count; i++)
            {
                if (feedbacks[i].score < 0.7f) // 충분히 낮은 것만
                    result.Add(feedbacks[i].code);
            }

            return result;
        }

        static HandPoseRecording FramesToRecording(List<HandPoseFrame> frames)
        {
            var recording = new HandPoseRecording
            {
                SchemaVersion = 2,
                MotionId = "user_capture",
                TotalFrames = frames.Count,
                Frames = new HandPoseRecordingFrame[frames.Count],
            };

            for (int i = 0; i < frames.Count; i++)
            {
                var f = frames[i];
                recording.Frames[i] = new HandPoseRecordingFrame
                {
                    TimestampMs = f.TimestampMs,
                    HasLeftHand = f.HasLeftHand,
                    HasRightHand = f.HasRightHand,
                    HasPose = f.HasPose,
                    HasValidShoulders = f.HasValidShoulders,
                    ShoulderCenter = f.ShoulderCenter,
                    ShoulderWidth = f.ShoulderWidth,
                    LeftWristBodyRelative = f.LeftWristBodyRelative,
                    RightWristBodyRelative = f.RightWristBodyRelative,
                    Quality = f.Quality,
                    PosePoints = f.PoseFrame?.AllPoints,
                };
            }

            if (frames.Count > 0)
            {
                recording.DurationMs = frames[frames.Count - 1].TimestampMs - frames[0].TimestampMs;
            }

            return recording;
        }
    }

    /// <summary>
    /// 수어 레퍼런스. 기존 HandPoseRecording + 사전 추출된 특징.
    /// </summary>
    public sealed class SignReference
    {
        public string SignId;
        public string DisplayName;
        public HandPoseRecording Recording;
        public SignFeatureSet Features;
        public CaptureViewPolicy ViewPolicy = CaptureViewPolicy.Front;

        /// <summary>
        /// 필수 관찰 랜드마크 (예: "right_hand", "left_hand", "both_hands").
        /// </summary>
        public string[] RequiredLandmarks = Array.Empty<string>();
    }

    /// <summary>
    /// 캡처 품질 정보.
    /// </summary>
    public sealed class CaptureQualityInfo
    {
        public float ValidFrameRatio = 1f;
        public bool OutOfFrame;
        public bool Occluded;
        public long TrackingGapMs;
    }

    /// <summary>
    /// 요소별 점수.
    /// </summary>
    public sealed class SignComponentScores
    {
        public float HandshapeScore;
        public float PositionScore;
        public float TrajectoryScore;
        public float TwoHandScore;
        public float DtwScore;
        public float CombinedScore;
    }

    /// <summary>
    /// 인식 결과.
    /// </summary>
    public sealed class SignRecognitionResult
    {
        public AttemptResultStatus Status = AttemptResultStatus.Invalid;
        public string BestSignId;
        public float GlobalScore;
        public float PersonalScore = float.NaN;
        public float CombinedScore = float.NaN;
        public float AmbiguityGap;
        public string InvalidReason;

        public SignComponentScores ComponentScores = new SignComponentScores();
        public SignFeatureSet UserFeatures;

        public List<string> FeedbackCodes = new List<string>();

        public void AddFeedback(string code)
        {
            if (!string.IsNullOrEmpty(code) && !FeedbackCodes.Contains(code))
                FeedbackCodes.Add(code);
        }

        public bool IsPassed => Status == AttemptResultStatus.Passed;
    }
}
