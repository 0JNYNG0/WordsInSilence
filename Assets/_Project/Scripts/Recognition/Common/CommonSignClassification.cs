using System;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// Why a frozen common-2D classification could not be produced. Any value other
    /// than <see cref="None"/> means the attempt is a technical failure for this
    /// auxiliary signal and must never fall through to success.
    /// </summary>
    public enum CommonSignFailureReason
    {
        None = 0,
        ModelUnavailable,
        ModelIntegrityMismatch,
        SequenceMissing,
        InvalidShape,
        NonFiniteInput,
        AllLandmarksMissing,
        UnsupportedTarget,
    }

    /// <summary>
    /// Immutable result of the frozen Step 10O v1.6 common-2D classifier.
    ///
    /// This is an AUXILIARY signal only. It carries no acceptance decision: it may not
    /// create StrictSuccess, activate a personal profile, approve a candidate or
    /// complete a quest, and it introduces no new confidence threshold.
    /// <see cref="ExpectedTargetSupport"/> is a plain argmax comparison against the
    /// target word the game already knows; probabilities and margin are supplied as
    /// evidence for the existing conservative fusion, not as a gate.
    /// </summary>
    public sealed class CommonSignClassification
    {
        public string ArtifactVersion { get; }
        public string ArtifactSha256 { get; }
        public bool Eligible { get; }
        public CommonSignFailureReason FailureReason { get; }
        public string FailureDetail { get; }
        public string RequestedTarget { get; }
        public double GreetingProbability { get; }
        public double ThankYouProbability { get; }
        public string TopClass { get; }
        public double TopProbability { get; }
        public double ProbabilityMargin { get; }
        public bool ExpectedTargetSupport { get; }
        public double DecisionValue { get; }
        public int SourceFrameCount { get; }

        CommonSignClassification(
            string artifactVersion,
            string artifactSha256,
            bool eligible,
            CommonSignFailureReason failureReason,
            string failureDetail,
            string requestedTarget,
            double greetingProbability,
            double thankYouProbability,
            string topClass,
            double topProbability,
            double probabilityMargin,
            bool expectedTargetSupport,
            double decisionValue,
            int sourceFrameCount)
        {
            ArtifactVersion = artifactVersion ?? string.Empty;
            ArtifactSha256 = artifactSha256 ?? string.Empty;
            Eligible = eligible;
            FailureReason = failureReason;
            FailureDetail = failureDetail ?? string.Empty;
            RequestedTarget = requestedTarget ?? string.Empty;
            GreetingProbability = greetingProbability;
            ThankYouProbability = thankYouProbability;
            TopClass = topClass ?? string.Empty;
            TopProbability = topProbability;
            ProbabilityMargin = probabilityMargin;
            ExpectedTargetSupport = expectedTargetSupport;
            DecisionValue = decisionValue;
            SourceFrameCount = sourceFrameCount;
        }

        /// <summary>A produced classification. Never asserts acceptance by itself.</summary>
        public static CommonSignClassification Success(
            string artifactVersion,
            string artifactSha256,
            string requestedTarget,
            double greetingProbability,
            double thankYouProbability,
            double decisionValue,
            int sourceFrameCount)
        {
            // Tie-break matches the frozen v1.6 rule: the first entry of the frozen
            // class order wins on exact equality.
            bool greetingTop = greetingProbability >= thankYouProbability;
            string top = greetingTop
                ? FrozenCommonSignModel.ClassGreeting
                : FrozenCommonSignModel.ClassThankYou;
            double topProbability = greetingTop ? greetingProbability : thankYouProbability;
            double margin = Math.Abs(greetingProbability - thankYouProbability);
            bool support = string.Equals(top, requestedTarget, StringComparison.Ordinal);
            return new CommonSignClassification(
                artifactVersion, artifactSha256, true, CommonSignFailureReason.None,
                string.Empty, requestedTarget, greetingProbability, thankYouProbability,
                top, topProbability, margin, support, decisionValue, sourceFrameCount);
        }

        /// <summary>An ineligible attempt. Support is always false and never a success.</summary>
        public static CommonSignClassification Failed(
            string artifactVersion,
            string artifactSha256,
            string requestedTarget,
            CommonSignFailureReason reason,
            string detail,
            int sourceFrameCount = 0)
        {
            if (reason == CommonSignFailureReason.None)
                throw new ArgumentOutOfRangeException(nameof(reason),
                    "a failed classification requires a non-None reason");
            return new CommonSignClassification(
                artifactVersion, artifactSha256, false, reason, detail, requestedTarget,
                double.NaN, double.NaN, string.Empty, double.NaN, double.NaN,
                false, double.NaN, sourceFrameCount);
        }
    }
}
