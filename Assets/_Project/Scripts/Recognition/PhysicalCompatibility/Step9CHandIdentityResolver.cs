using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Pose;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    /// <summary>
    /// Step 9C-only policy. It resolves a duplicate provider category only when the same-frame
    /// pose wrists make the two possible assignments objectively separable.
    /// </summary>
    public sealed class Step9CHandIdentityResolver
    {
        public const float MaximumMatchedWristDistance = 0.30f;
        public const float MinimumPoseWristSeparation = 0.10f;
        public const float MinimumAssignmentMargin = 0.05f;
        public const float MinimumAssignmentMarginRatio = 0.50f;

        public HandPoseFrame Resolve(HandPoseFrame frame, int providerFrameIndex, out Step9CHandIdentityDiagnostic diagnostic)
        {
            if (frame == null) throw new ArgumentNullException(nameof(frame));
            if (providerFrameIndex < 0) throw new ArgumentOutOfRangeException(nameof(providerFrameIndex));
            diagnostic = null;

            List<HandLandmarks> hands = NonNullHands(frame.HandFrame);
            if (!ContainsDuplicateCategory(hands)) return frame;

            string category0 = hands.Count > 0 ? Category(hands[0]) : string.Empty;
            string category1 = hands.Count > 1 ? Category(hands[1]) : string.Empty;
            double score0 = hands.Count > 0 ? hands[0].Confidence : double.NaN;
            double score1 = hands.Count > 1 ? hands[1].Confidence : double.NaN;
            double handTimestamp = frame.HandFrame == null ? double.NaN : frame.HandFrame.Timestamp;

            if (hands.Count != 2)
                throw Ambiguous(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                    "duplicate_category_requires_exactly_two_non_null_hands");
            if (!IsScore(score0) || !IsScore(score1))
                throw Ambiguous(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                    "provider_handedness_score_invalid");
            if (!TryWrist(hands[0], out Vector2 hand0) || !TryWrist(hands[1], out Vector2 hand1))
                throw Ambiguous(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                    "hand_wrist_landmark_unavailable");
            if (!TryPoseWrists(frame.PoseFrame, out Vector2 poseLeft, out Vector2 poseRight, out float leftVisibility, out float rightVisibility))
                throw Ambiguous(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                    "same_frame_pose_wrists_unavailable_or_low_visibility", hand0, hand1);

            float separation = Vector2.Distance(poseLeft, poseRight);
            float hand0Left = Vector2.Distance(hand0, poseLeft);
            float hand0Right = Vector2.Distance(hand0, poseRight);
            float hand1Left = Vector2.Distance(hand1, poseLeft);
            float hand1Right = Vector2.Distance(hand1, poseRight);
            float directCost = hand0Left + hand1Right;
            float swappedCost = hand0Right + hand1Left;
            bool direct = directCost <= swappedCost;
            float bestMaxDistance = direct ? Mathf.Max(hand0Left, hand1Right) : Mathf.Max(hand0Right, hand1Left);
            float margin = Mathf.Abs(directCost - swappedCost);
            float requiredMargin = Mathf.Max(MinimumAssignmentMargin, MinimumAssignmentMarginRatio * separation);

            string ambiguity = null;
            if (!IsFinite(separation) || separation < MinimumPoseWristSeparation) ambiguity = "pose_wrists_not_sufficiently_separated";
            else if (!IsFinite(bestMaxDistance) || bestMaxDistance > MaximumMatchedWristDistance) ambiguity = "best_pose_wrist_assignment_too_far";
            else if (!IsFinite(margin) || margin < requiredMargin) ambiguity = "pose_wrist_assignment_margin_ambiguous";

            if (ambiguity != null)
            {
                diagnostic = BuildDiagnostic(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                    hand0, hand1, poseLeft, poseRight, leftVisibility, rightVisibility, directCost, swappedCost,
                    bestMaxDistance, margin, separation, string.Empty, Step9CHandIdentityOutcome.Ambiguous, ambiguity);
                throw new Step9CAmbiguousHandIdentityException(diagnostic);
            }

            bool hand0IsLeft = direct;
            HandLandmarks[] resolvedHands =
            {
                CloneHand(hands[0], hand0IsLeft),
                CloneHand(hands[1], !hand0IsLeft),
            };
            diagnostic = BuildDiagnostic(frame, providerFrameIndex, hands.Count, category0, score0, category1, score1,
                hand0, hand1, poseLeft, poseRight, leftVisibility, rightVisibility, directCost, swappedCost,
                bestMaxDistance, margin, separation, hand0IsLeft ? "Left" : "Right",
                Step9CHandIdentityOutcome.ResolvedByPoseWrist, "pose_wrist_assignment_unambiguous");
            return CloneFrame(frame, resolvedHands);
        }

        static Step9CAmbiguousHandIdentityException Ambiguous(
            HandPoseFrame frame, int frameIndex, int handCount, string category0, double score0, string category1, double score1,
            string reason, Vector2? hand0 = null, Vector2? hand1 = null)
        {
            Vector2 missing = new Vector2(float.NaN, float.NaN);
            Step9CHandIdentityDiagnostic diagnostic = BuildDiagnostic(
                frame, frameIndex, handCount, category0, score0, category1, score1,
                hand0 ?? missing, hand1 ?? missing, missing, missing, double.NaN, double.NaN,
                double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, string.Empty,
                Step9CHandIdentityOutcome.Ambiguous, reason);
            return new Step9CAmbiguousHandIdentityException(diagnostic);
        }

        static Step9CHandIdentityDiagnostic BuildDiagnostic(
            HandPoseFrame frame, int frameIndex, int handCount, string category0, double score0, string category1, double score1,
            Vector2 hand0, Vector2 hand1, Vector2 poseLeft, Vector2 poseRight, double leftVisibility, double rightVisibility,
            double directCost, double swappedCost, double bestMaxDistance, double margin, double separation,
            string selectedHand0, Step9CHandIdentityOutcome outcome, string reason)
            => new Step9CHandIdentityDiagnostic(
                frameIndex,
                frame.Timestamp,
                frame.HandFrame == null ? double.NaN : frame.HandFrame.Timestamp,
                handCount,
                category0,
                score0,
                category1,
                score1,
                hand0.x,
                hand0.y,
                hand1.x,
                hand1.y,
                poseLeft.x,
                poseLeft.y,
                poseRight.x,
                poseRight.y,
                leftVisibility,
                rightVisibility,
                directCost,
                swappedCost,
                bestMaxDistance,
                margin,
                separation,
                selectedHand0,
                outcome,
                reason);

        static List<HandLandmarks> NonNullHands(LandmarkFrame frame)
        {
            var result = new List<HandLandmarks>();
            if (frame?.Hands == null) return result;
            foreach (HandLandmarks hand in frame.Hands) if (hand != null) result.Add(hand);
            return result;
        }

        static bool ContainsDuplicateCategory(IReadOnlyList<HandLandmarks> hands)
        {
            bool left = false, right = false;
            foreach (HandLandmarks hand in hands)
            {
                if (hand.IsLeftHand)
                {
                    if (left) return true;
                    left = true;
                }
                else
                {
                    if (right) return true;
                    right = true;
                }
            }
            return false;
        }

        static bool TryWrist(HandLandmarks hand, out Vector2 wrist)
        {
            wrist = new Vector2(float.NaN, float.NaN);
            if (hand?.Points == null || hand.Points.Length == 0 || hand.Points[0] == null) return false;
            LandmarkPoint point = hand.Points[0];
            if (!IsFinite(point.X) || !IsFinite(point.Y)) return false;
            wrist = new Vector2(point.X, point.Y);
            return true;
        }

        static bool TryPoseWrists(UpperBodyPoseFrame pose, out Vector2 left, out Vector2 right, out float leftVisibility, out float rightVisibility)
        {
            left = right = new Vector2(float.NaN, float.NaN);
            leftVisibility = rightVisibility = float.NaN;
            if (pose?.AllPoints == null || pose.AllPoints.Length <= 16 || pose.Visibility == null || pose.Visibility.Length <= 16) return false;
            Vector3 leftPoint = pose.AllPoints[15];
            Vector3 rightPoint = pose.AllPoints[16];
            leftVisibility = pose.Visibility[15];
            rightVisibility = pose.Visibility[16];
            if (!IsFinite(leftPoint.x) || !IsFinite(leftPoint.y) || !IsFinite(rightPoint.x) || !IsFinite(rightPoint.y)) return false;
            if (!IsFinite(leftVisibility) || !IsFinite(rightVisibility) ||
                leftVisibility < UpperBodyPoseFrame.MinVisibilityThreshold || rightVisibility < UpperBodyPoseFrame.MinVisibilityThreshold) return false;
            left = new Vector2(leftPoint.x, leftPoint.y);
            right = new Vector2(rightPoint.x, rightPoint.y);
            return true;
        }

        static HandLandmarks CloneHand(HandLandmarks source, bool isLeft) => new HandLandmarks
        {
            IsLeftHand = isLeft,
            Points = source.Points,
            Confidence = source.Confidence,
        };

        static HandPoseFrame CloneFrame(HandPoseFrame source, HandLandmarks[] hands) => new HandPoseFrame
        {
            Timestamp = source.Timestamp,
            TimestampMs = source.TimestampMs,
            HandPoseTimestampDiffMs = source.HandPoseTimestampDiffMs,
            HandFrame = new LandmarkFrame
            {
                Timestamp = source.HandFrame.Timestamp,
                Hands = hands,
                InferenceGeometryResult = source.HandFrame.InferenceGeometryResult,
            },
            HasLeftHand = true,
            HasRightHand = true,
            PoseFrame = source.PoseFrame,
            HasPose = source.HasPose,
            HasValidShoulders = source.HasValidShoulders,
            ShoulderCenter = source.ShoulderCenter,
            ShoulderWidth = source.ShoulderWidth,
            IsValidForBodyRelativeEvaluation = source.IsValidForBodyRelativeEvaluation,
            LeftWristBodyRelative = source.LeftWristBodyRelative,
            RightWristBodyRelative = source.RightWristBodyRelative,
            LeftArmAngleDegrees = source.LeftArmAngleDegrees,
            RightArmAngleDegrees = source.RightArmAngleDegrees,
            LeftHandPoseWristDist = source.LeftHandPoseWristDist,
            RightHandPoseWristDist = source.RightHandPoseWristDist,
            Quality = source.Quality,
            MergedGeometry = source.MergedGeometry,
        };

        static string Category(HandLandmarks hand) => hand.IsLeftHand ? "Left" : "Right";
        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        static bool IsScore(double value) => !double.IsNaN(value) && !double.IsInfinity(value) && value >= 0.0 && value <= 1.0;
    }

    public sealed class Step9CAmbiguousHandIdentityException : InvalidOperationException
    {
        public Step9CHandIdentityDiagnostic Diagnostic { get; }

        public Step9CAmbiguousHandIdentityException(Step9CHandIdentityDiagnostic diagnostic)
            : base(FormatMessage(diagnostic))
        {
            Diagnostic = diagnostic ?? throw new ArgumentNullException(nameof(diagnostic));
        }

        static string FormatMessage(Step9CHandIdentityDiagnostic diagnostic)
        {
            if (diagnostic == null) return "duplicate handedness input is objectively ambiguous";
            return $"duplicate handedness is objectively ambiguous at provider frame {diagnostic.ProviderFrameIndex}, " +
                   $"timestamp={diagnostic.ProviderTimestampSeconds:R}, categories=" +
                   $"{diagnostic.SourceCategory0}({diagnostic.SourceScore0:R})," +
                   $"{diagnostic.SourceCategory1}({diagnostic.SourceScore1:R}), reason={diagnostic.Reason}";
        }
    }

    public static class Step9CHandIdentityCapturePolicy
    {
        public const int MaximumTransientDuplicateFrames = 2;
        public const double MaximumTransientDuplicateRatio = 0.02;
        public const int MaximumTransientConsecutiveFrames = 2;

        public static IReadOnlyList<string> InvalidReasons(int providerFrameCount, IReadOnlyList<Step9CHandIdentityDiagnostic> diagnostics)
        {
            if (diagnostics == null) throw new ArgumentNullException(nameof(diagnostics));
            var reasons = new List<string>();
            int resolved = 0;
            int ambiguous = 0;
            int longestRun = 0;
            int currentRun = 0;
            int previous = int.MinValue;
            foreach (Step9CHandIdentityDiagnostic diagnostic in diagnostics)
            {
                if (diagnostic == null) throw new ArgumentException("hand identity diagnostic is null", nameof(diagnostics));
                if (diagnostic.Outcome == Step9CHandIdentityOutcome.ResolvedByPoseWrist) resolved++;
                else ambiguous++;
                currentRun = diagnostic.ProviderFrameIndex == previous + 1 ? currentRun + 1 : 1;
                if (currentRun > longestRun) longestRun = currentRun;
                previous = diagnostic.ProviderFrameIndex;
            }
            if (ambiguous > 0) reasons.Add("ambiguous_hand_identity_observed");
            double ratio = providerFrameCount <= 0 ? (diagnostics.Count == 0 ? 0.0 : double.PositiveInfinity) : (double)diagnostics.Count / providerFrameCount;
            if (resolved > MaximumTransientDuplicateFrames || ratio > MaximumTransientDuplicateRatio || longestRun > MaximumTransientConsecutiveFrames)
                reasons.Add("duplicate_handedness_not_transient");
            return reasons.AsReadOnly();
        }
    }
}
