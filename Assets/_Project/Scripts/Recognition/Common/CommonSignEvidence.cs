using System;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// Fused state of the frozen common-2D evidence for one attempt.
    ///
    /// This is deliberately an EVIDENCE state, not an acceptance state. No value here
    /// promotes, demotes or vetoes the strict outcome, and no value performs profile
    /// activation, candidate writing or quest completion.
    /// </summary>
    public enum CommonEvidenceState
    {
        /// <summary>No common input was supplied, or it could not be attempted at all.</summary>
        Unavailable = 0,

        /// <summary>
        /// The requested target is outside the frozen two-class contract, for example the
        /// legacy LOVE route. This is NOT a common-model failure and must leave the legacy
        /// outcome completely unchanged.
        /// </summary>
        NotApplicable,

        /// <summary>Eligible classification whose top-1 equals the requested target.</summary>
        Support,

        /// <summary>Eligible classification whose top-1 opposes the requested target.</summary>
        NonSupport,

        /// <summary>
        /// A GREETING/THANK_YOU attempt with a model, input, geometry, shape, mask or
        /// non-finite failure. Preserved as technical evidence with a typed reason; it
        /// never marks the learner wrong and never creates a strict technical failure.
        /// </summary>
        TechnicalEvidence,
    }

    /// <summary>
    /// Immutable fused common evidence attached to the outcome that the game actually
    /// consumes. Produced by the outcome/fusion seam from the common classification, so
    /// removing or ignoring the common input changes this value and is detectable.
    /// </summary>
    public sealed class CommonEvidence
    {
        public CommonEvidenceState State { get; }
        public string ArtifactVersion { get; }
        public string ArtifactSha256 { get; }
        public string RequestedTarget { get; }
        public string TopClass { get; }
        public double GreetingProbability { get; }
        public double ThankYouProbability { get; }
        public double ProbabilityMargin { get; }
        public CommonSignFailureReason FailureReason { get; }
        public string FailureDetail { get; }

        /// <summary>Always false. The common evidence never carries acceptance authority.</summary>
        public bool GrantsSuccess { get { return false; } }

        /// <summary>Always false. The common evidence never marks the learner wrong.</summary>
        public bool GrantsLearnerError { get { return false; } }

        /// <summary>Always false. The common evidence never vetoes an existing outcome.</summary>
        public bool GrantsVeto { get { return false; } }

        CommonEvidence(
            CommonEvidenceState state, string artifactVersion, string artifactSha256,
            string requestedTarget, string topClass, double greeting, double thankYou,
            double margin, CommonSignFailureReason failureReason, string failureDetail)
        {
            State = state;
            ArtifactVersion = artifactVersion ?? string.Empty;
            ArtifactSha256 = artifactSha256 ?? string.Empty;
            RequestedTarget = requestedTarget ?? string.Empty;
            TopClass = topClass ?? string.Empty;
            GreetingProbability = greeting;
            ThankYouProbability = thankYou;
            ProbabilityMargin = margin;
            FailureReason = failureReason;
            FailureDetail = failureDetail ?? string.Empty;
        }

        public static readonly CommonEvidence Unavailable = new CommonEvidence(
            CommonEvidenceState.Unavailable, string.Empty, string.Empty, string.Empty,
            string.Empty, double.NaN, double.NaN, double.NaN,
            CommonSignFailureReason.None, "no common classification was supplied");

        /// <summary>
        /// Fuses one classification into the evidence state. The requested target decides
        /// NotApplicable before any model outcome is considered, so the legacy LOVE route
        /// is never reported as a common-model technical failure.
        /// </summary>
        public static CommonEvidence From(
            CommonSignClassification classification, string requestedTarget)
        {
            if (!FrozenCommonSignModel.IsSupportedTarget(requestedTarget))
            {
                return new CommonEvidence(
                    CommonEvidenceState.NotApplicable,
                    classification == null ? string.Empty : classification.ArtifactVersion,
                    classification == null ? string.Empty : classification.ArtifactSha256,
                    requestedTarget, string.Empty, double.NaN, double.NaN, double.NaN,
                    CommonSignFailureReason.None,
                    "target is outside the frozen two-class contract; the legacy route is unaffected");
            }
            if (classification == null)
            {
                return new CommonEvidence(
                    CommonEvidenceState.Unavailable, string.Empty, string.Empty,
                    requestedTarget, string.Empty, double.NaN, double.NaN, double.NaN,
                    CommonSignFailureReason.None, "no common classification was supplied");
            }
            if (!classification.Eligible)
            {
                return new CommonEvidence(
                    CommonEvidenceState.TechnicalEvidence,
                    classification.ArtifactVersion, classification.ArtifactSha256,
                    requestedTarget, string.Empty, double.NaN, double.NaN, double.NaN,
                    classification.FailureReason, classification.FailureDetail);
            }
            return new CommonEvidence(
                classification.ExpectedTargetSupport
                    ? CommonEvidenceState.Support
                    : CommonEvidenceState.NonSupport,
                classification.ArtifactVersion, classification.ArtifactSha256,
                requestedTarget, classification.TopClass,
                classification.GreetingProbability, classification.ThankYouProbability,
                classification.ProbabilityMargin,
                CommonSignFailureReason.None, string.Empty);
        }
    }
}
