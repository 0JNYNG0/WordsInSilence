using System;
using System.Collections.Generic;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum PetRobotPracticeState
    {
        Ready,
        Capturing,
        Feedback,
        Completed,
        Exited,
    }

    public enum PetRobotWordLearningStatus
    {
        Pending,
        Mastered,
        NeedsPractice,
    }

    public enum PersonalizationCandidateEligibility
    {
        Eligible,
        ReviewOnly,
    }

    public enum CandidateStorageStatus
    {
        NotRequested,
        Saved,
        Duplicate,
        Failed,
    }

    public enum TrustedPersonalizationTrustBasis
    {
        StrictEligible,
        GuidedLowConfidenceSeed,
    }

    public enum GuidedEnrollmentSeedCommitStatus
    {
        NotRequested,
        Committed,
        AlreadyCommitted,
        NotRequired,
        InsufficientCandidates,
        DuplicateIdentity,
        Failed,
    }

    public enum PetRobotPersonalizationEnrollmentState
    {
        Pending,
        Personalized,
        PersonalizationIncomplete,
        TechnicalStop,
    }

    public sealed class CandidateCountSnapshot
    {
        readonly IReadOnlyDictionary<string, int> eligible;
        readonly IReadOnlyDictionary<string, int> reviewOnly;

        public IReadOnlyDictionary<string, int> Eligible => eligible;
        public IReadOnlyDictionary<string, int> ReviewOnly => reviewOnly;
        public int TotalCandidateCount =>
            eligible["GREETING"] + eligible["LOVE"] + eligible["THANK_YOU"] +
            reviewOnly["GREETING"] + reviewOnly["LOVE"] + reviewOnly["THANK_YOU"];
        public bool ProfileRefreshReady =>
            eligible["GREETING"] >= 3 && eligible["LOVE"] >= 3 && eligible["THANK_YOU"] >= 3;

        public CandidateCountSnapshot(
            IDictionary<string, int> eligibleCounts,
            IDictionary<string, int> reviewOnlyCounts)
        {
            eligible = CopyCounts(eligibleCounts);
            reviewOnly = CopyCounts(reviewOnlyCounts);
        }

        public static CandidateCountSnapshot Empty => new CandidateCountSnapshot(
            new Dictionary<string, int>(StringComparer.Ordinal),
            new Dictionary<string, int>(StringComparer.Ordinal));

        static IReadOnlyDictionary<string, int> CopyCounts(IDictionary<string, int> source)
        {
            var copy = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in PersonalLiveRecognitionController.SupportedWords)
            {
                int count;
                copy[word] = source != null && source.TryGetValue(word, out count) ? count : 0;
            }
            return copy;
        }
    }

    public sealed class CandidateSaveResult
    {
        public CandidateStorageStatus Status { get; }
        public string CandidateDirectory { get; }
        public string Message { get; }
        public CandidateCountSnapshot Counts { get; }
        public bool IsSaved => Status == CandidateStorageStatus.Saved;

        public CandidateSaveResult(
            CandidateStorageStatus status,
            string candidateDirectory,
            string message,
            CandidateCountSnapshot counts)
        {
            Status = status;
            CandidateDirectory = candidateDirectory ?? string.Empty;
            Message = message ?? string.Empty;
            Counts = counts ?? CandidateCountSnapshot.Empty;
        }
    }

    /// <summary>
    /// A strictly verified, identity-unique practice capture. Its trust basis is either the
    /// original strict Eligible transaction or an explicit guided-enrollment seed manifest.
    /// The underlying candidate transaction is never promoted or rewritten.
    /// </summary>
    public sealed class TrustedPersonalizationCandidate
    {
        public string CaptureSessionId { get; }
        public string Word { get; }
        public string CandidateDirectory { get; }
        public string SequenceFileSha256 { get; }
        public string SourcePayloadSha256 { get; }
        public string LogicalPayloadSha256 { get; }
        public RecognitionSequence Sequence { get; }
        public TrustedPersonalizationTrustBasis TrustBasis { get; }

        public TrustedPersonalizationCandidate(
            string captureSessionId,
            string word,
            string candidateDirectory,
            string sequenceFileSha256,
            string sourcePayloadSha256,
            string logicalPayloadSha256,
            RecognitionSequence sequence)
            : this(captureSessionId, word, candidateDirectory, sequenceFileSha256,
                sourcePayloadSha256, logicalPayloadSha256, sequence,
                TrustedPersonalizationTrustBasis.StrictEligible)
        {
        }

        public TrustedPersonalizationCandidate(
            string captureSessionId,
            string word,
            string candidateDirectory,
            string sequenceFileSha256,
            string sourcePayloadSha256,
            string logicalPayloadSha256,
            RecognitionSequence sequence,
            TrustedPersonalizationTrustBasis trustBasis)
        {
            CaptureSessionId = captureSessionId ?? string.Empty;
            Word = word ?? string.Empty;
            CandidateDirectory = candidateDirectory ?? string.Empty;
            SequenceFileSha256 = sequenceFileSha256 ?? string.Empty;
            SourcePayloadSha256 = sourcePayloadSha256 ?? string.Empty;
            LogicalPayloadSha256 = logicalPayloadSha256 ?? string.Empty;
            Sequence = sequence ?? throw new ArgumentNullException(nameof(sequence));
            TrustBasis = trustBasis;
        }
    }

    public sealed class GuidedEnrollmentSeedCommitResult
    {
        public GuidedEnrollmentSeedCommitStatus Status { get; }
        public string Word { get; }
        public string ManifestPath { get; }
        public string Message { get; }
        public IReadOnlyList<string> SelectedCaptureSessionIds { get; }
        public bool IsSuccess =>
            Status == GuidedEnrollmentSeedCommitStatus.Committed ||
            Status == GuidedEnrollmentSeedCommitStatus.AlreadyCommitted ||
            Status == GuidedEnrollmentSeedCommitStatus.NotRequired;

        public GuidedEnrollmentSeedCommitResult(
            GuidedEnrollmentSeedCommitStatus status,
            string word,
            string manifestPath,
            string message,
            IEnumerable<string> selectedCaptureSessionIds = null)
        {
            Status = status;
            Word = word ?? string.Empty;
            ManifestPath = manifestPath ?? string.Empty;
            Message = message ?? string.Empty;
            var selected = new List<string>();
            if (selectedCaptureSessionIds != null) selected.AddRange(selectedCaptureSessionIds);
            SelectedCaptureSessionIds = selected.AsReadOnly();
        }

        public static GuidedEnrollmentSeedCommitResult NotRequested =>
            new GuidedEnrollmentSeedCommitResult(
                GuidedEnrollmentSeedCommitStatus.NotRequested,
                string.Empty,
                string.Empty,
                "not requested");
    }

    public sealed class TrustedCandidateInventory
    {
        readonly IReadOnlyDictionary<string, int> trustedCounts;
        readonly IReadOnlyList<TrustedPersonalizationCandidate> candidates;

        public CandidateCountSnapshot RawCounts { get; }
        public IReadOnlyDictionary<string, int> TrustedCounts => trustedCounts;
        public IReadOnlyList<TrustedPersonalizationCandidate> Candidates => candidates;
        public int ExcludedDuplicateOrUntrustedCount { get; }
        public bool OverallEnrollmentGateReady =>
            trustedCounts["GREETING"] >= 3 &&
            trustedCounts["LOVE"] >= 3 &&
            trustedCounts["THANK_YOU"] >= 3;

        public TrustedCandidateInventory(
            CandidateCountSnapshot rawCounts,
            IDictionary<string, int> trustedCounts,
            IEnumerable<TrustedPersonalizationCandidate> candidates,
            int excludedDuplicateOrUntrustedCount)
        {
            RawCounts = rawCounts ?? CandidateCountSnapshot.Empty;
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in PersonalLiveRecognitionController.SupportedWords)
            {
                int count;
                counts[word] = trustedCounts != null && trustedCounts.TryGetValue(word, out count) ? count : 0;
            }
            this.trustedCounts = counts;
            var copy = new List<TrustedPersonalizationCandidate>();
            if (candidates != null) copy.AddRange(candidates);
            this.candidates = copy.AsReadOnly();
            ExcludedDuplicateOrUntrustedCount = Math.Max(0, excludedDuplicateOrUntrustedCount);
        }

        public int CountForWord(string word)
        {
            int count;
            if (!trustedCounts.TryGetValue(word, out count))
                throw new ArgumentException("unsupported practice word", nameof(word));
            return count;
        }

        public static TrustedCandidateInventory Empty => new TrustedCandidateInventory(
            CandidateCountSnapshot.Empty,
            new Dictionary<string, int>(StringComparer.Ordinal),
            Array.Empty<TrustedPersonalizationCandidate>(),
            0);
    }

    public sealed class PracticeRecognitionAttempt
    {
        public string CaptureSessionId { get; }
        public string ExpectedWord { get; }
        public string PredictedWord { get; }
        public PersonalLiveCaptureState CaptureState { get; }
        public GuidedQuestOutcome Outcome { get; }
        public PersonalLiveRecognitionSnapshot Recognition { get; }
        public RecognitionSequence NormalizedSequence { get; }
        public string RouteId { get; }
        public string ProfileId { get; }
        public double ExpectedWordThreshold { get; }
        public PersonalLiveFailure Failure { get; }
        public PersonalLivePrimaryFailureSnapshot PrimaryFailure { get; }
        public bool StrictAccepted { get; }
        public bool StrictExpectedWordClassAccept { get; }
        public bool StrictQuestSuccess { get; }
        public double GreetingDistance { get; }
        public double LoveDistance { get; }
        public double ThankYouDistance { get; }
        public double BestDistance { get; }
        public string WinningPrototypeId { get; }
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }

        /// <summary>
        /// Auxiliary frozen Step 10O v1.6 common-2D classification for this attempt, or
        /// null when no classifier was supplied. Advisory evidence only: it never
        /// changes StrictAccepted / StrictQuestSuccess / Outcome, never approves a
        /// candidate, never activates a profile and never completes a quest.
        /// </summary>
        public CommonSignClassification CommonSignal { get; }

        /// <summary>
        /// Optional typed F21 full-hand pipeline evidence for this attempt, or null when the
        /// seam was not configured. Research evidence only: it never changes StrictAccepted /
        /// StrictExpectedWordClassAccept / StrictQuestSuccess / Outcome, never approves a
        /// candidate, never activates a profile and never completes a quest.
        /// </summary>
        public WordsInSilence.Recognition.FullHandRuntime.FullHandGameCaptureResult
            FullHandEvidence { get; }

        /// <summary>
        /// F27. Non-null ONLY when the explicit full-hand gameplay mode decided this
        /// attempt. When it is non-null it is the authority: Outcome, PredictedWord and the
        /// Strict* flags above describe THIS decision, not the legacy 16-point snapshot,
        /// and it carries its own method/bank/model/gate/source provenance.
        ///
        /// A full-hand attempt is never a 16-point personalization candidate: 46 source
        /// points cannot be stored as a 16-point normalized sequence, so NormalizedSequence
        /// is null on this route and the repository refuses such a request outright.
        /// </summary>
        public WordsInSilence.Recognition.FullHandRuntime.FullHandGameplayDecision
            FullHandGameplay { get; }

        /// <summary>True when the full-hand decision, not the legacy route, is authority.</summary>
        public bool FullHandGameplayAuthority { get { return FullHandGameplay != null; } }

        public PracticeRecognitionAttempt(
            string captureSessionId,
            string expectedWord,
            string predictedWord,
            PersonalLiveCaptureState captureState,
            GuidedQuestOutcome outcome,
            PersonalLiveRecognitionSnapshot recognition,
            RecognitionSequence normalizedSequence,
            string routeId,
            string profileId,
            double expectedWordThreshold,
            PersonalLiveFailure failure,
            PersonalLivePrimaryFailureSnapshot primaryFailure,
            PracticeRecognitionMetrics metrics = null,
            CommonSignClassification commonSignal = null,
            WordsInSilence.Recognition.FullHandRuntime.FullHandGameCaptureResult fullHandEvidence = null,
            WordsInSilence.Recognition.FullHandRuntime.FullHandGameplayDecision fullHandGameplay = null)
        {
            FullHandGameplay = fullHandGameplay;
            FullHandEvidence = fullHandEvidence;
            CommonSignal = commonSignal;
            CaptureSessionId = captureSessionId ?? string.Empty;
            ExpectedWord = expectedWord ?? string.Empty;
            PredictedWord = predictedWord ?? string.Empty;
            CaptureState = captureState;
            Outcome = outcome;
            Recognition = recognition;
            NormalizedSequence = normalizedSequence;
            RouteId = routeId ?? string.Empty;
            ProfileId = profileId ?? string.Empty;
            ExpectedWordThreshold = expectedWordThreshold;
            Failure = failure;
            PrimaryFailure = primaryFailure;
            PracticeRecognitionMetrics value = metrics ?? PracticeRecognitionMetrics.From(recognition);
            StrictAccepted = value.StrictAccepted;
            StrictExpectedWordClassAccept = value.StrictExpectedWordClassAccept;
            StrictQuestSuccess = value.StrictQuestSuccess;
            GreetingDistance = value.GreetingDistance;
            LoveDistance = value.LoveDistance;
            ThankYouDistance = value.ThankYouDistance;
            BestDistance = value.BestDistance;
            WinningPrototypeId = value.WinningPrototypeId;
            RawFrameCount = value.RawFrameCount;
            NormalizedFrameCount = value.NormalizedFrameCount;
            ValidLandmarkCount = value.ValidLandmarkCount;
            InterpolatedLandmarkCount = value.InterpolatedLandmarkCount;
        }
    }

    public sealed class PracticeRecognitionMetrics
    {
        public bool StrictAccepted;
        public bool StrictExpectedWordClassAccept;
        public bool StrictQuestSuccess;
        public double GreetingDistance;
        public double LoveDistance;
        public double ThankYouDistance;
        public double BestDistance;
        public string WinningPrototypeId;
        public int RawFrameCount;
        public int NormalizedFrameCount;
        public int ValidLandmarkCount;
        public int InterpolatedLandmarkCount;

        public static PracticeRecognitionMetrics From(PersonalLiveRecognitionSnapshot value)
        {
            if (value == null) return new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty };
            return new PracticeRecognitionMetrics
            {
                StrictAccepted = value.Accepted,
                StrictExpectedWordClassAccept = value.ExpectedWordClassAccept,
                StrictQuestSuccess = value.QuestSuccess,
                GreetingDistance = value.GreetingDistance,
                LoveDistance = value.LoveDistance,
                ThankYouDistance = value.ThankYouDistance,
                BestDistance = value.BestDistance,
                WinningPrototypeId = value.WinningPrototypeId,
                RawFrameCount = value.RawFrameCount,
                NormalizedFrameCount = value.NormalizedFrameCount,
                ValidLandmarkCount = value.ValidLandmarkCount,
                InterpolatedLandmarkCount = value.InterpolatedLandmarkCount,
            };
        }
    }

    public sealed class PersonalizationCandidateSaveRequest
    {
        public PracticeRecognitionAttempt Attempt { get; }
        public PersonalizationCandidateEligibility Eligibility { get; }
        public DateTime CreatedUtc { get; }

        public PersonalizationCandidateSaveRequest(
            PracticeRecognitionAttempt attempt,
            PersonalizationCandidateEligibility eligibility,
            DateTime createdUtc)
        {
            Attempt = attempt ?? throw new ArgumentNullException(nameof(attempt));
            Eligibility = eligibility;
            CreatedUtc = createdUtc.Kind == DateTimeKind.Utc ? createdUtc : createdUtc.ToUniversalTime();
        }
    }

    public interface IPersonalLivePracticeRecognitionAdapter
    {
        PersonalLiveCaptureState CaptureState { get; }
        string CaptureSessionId { get; }
        string ActiveProfileId { get; }
        bool CandidateRouteActive { get; }
        bool SetExpectedWord(string word);
        bool BeginCapture();
        PracticeRecognitionAttempt FinalizeCapture();
        bool ResetCapture();
    }

    /// <summary>
    /// Optional current-native segmentation seam. The existing manual no-argument interface
    /// remains unchanged for all current callers and test doubles.
    /// </summary>
    public interface ISegmentedPersonalLivePracticeRecognitionAdapter :
        IPersonalLivePracticeRecognitionAdapter
    {
        bool BeginCaptureWithPreRoll(IReadOnlyList<RawMediaPipeObservation> preRoll);
        PracticeRecognitionAttempt FinalizeCaptureWithTrailingTrim(double trailingTrimSeconds);
        PracticeRecognitionAttempt FailActiveCaptureFromSource(Exception exception);
    }

    public interface IPersonalizationCandidateRepository
    {
        string RootDirectory { get; }
        string LearnerId { get; }
        CandidateCountSnapshot Counts { get; }
        CandidateCountSnapshot ReloadAndVerify();
        CandidateSaveResult Save(PersonalizationCandidateSaveRequest request);
    }

    /// <summary>
    /// Optional strict enrollment view. Legacy repository test doubles keep their original
    /// single-result behavior by implementing only IPersonalizationCandidateRepository.
    /// </summary>
    public interface ITrustedPersonalizationCandidateRepository : IPersonalizationCandidateRepository
    {
        TrustedCandidateInventory TrustedInventory { get; }
        TrustedCandidateInventory ReloadTrustedAndVerify();
    }

    /// <summary>
    /// Explicit write capability used only by the guided practice session after a successful
    /// ReviewOnly save. Runtime/NPC consumers receive only read-only trusted snapshots.
    /// </summary>
    public interface IGuidedEnrollmentSeedPromotionRepository
    {
        GuidedEnrollmentSeedCommitResult TryCommitGuidedEnrollmentSeedQuorum(string word);
    }

    public interface ISignDemonstrationSlot
    {
        void ShowDemonstration(string word);
        void ClearDemonstration();
    }
}
