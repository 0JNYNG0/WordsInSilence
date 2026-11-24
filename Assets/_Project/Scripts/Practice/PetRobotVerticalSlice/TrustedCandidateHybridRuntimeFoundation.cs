using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum TrustedHybridOutcomeKind
    {
        StrictSuccess,
        LowConfidenceCorrect,
        WrongWord,
        TechnicalFailure,

        /// <summary>F27 full-hand route only: a valid abstention. Never grants the item.</summary>
        NotDistinguishedRetry,

        /// <summary>F27 full-hand route only: the prompted word is outside the model
        /// vocabulary. Never grants the item and never falls back to another route.</summary>
        UnsupportedWord,
    }

    public static class TrustedHybridOutcomePolicy
    {
        public static TrustedHybridOutcomeKind Evaluate(
            bool globalExpected,
            bool globalAcceptedConflict,
            bool personalExpected,
            bool personalAcceptedConflict,
            string globalPredictedWord,
            string personalPredictedWord)
        {
            if (globalExpected && personalExpected &&
                string.Equals(globalPredictedWord, personalPredictedWord, StringComparison.Ordinal))
                return TrustedHybridOutcomeKind.StrictSuccess;
            if (globalAcceptedConflict || personalAcceptedConflict)
                return TrustedHybridOutcomeKind.WrongWord;
            if (globalExpected ^ personalExpected)
                return TrustedHybridOutcomeKind.LowConfidenceCorrect;
            return TrustedHybridOutcomeKind.WrongWord;
        }
    }

    public sealed class TrustedHybridRecognitionResult
    {
        public TrustedHybridOutcomeKind Kind { get; }
        public string ExpectedWord { get; }
        public RuntimeRecognitionResult GlobalResult { get; }
        public PersonalRuntimeRecognitionResult PersonalResult { get; }
        public string GlobalPredictedWord => GlobalResult == null ? string.Empty : GlobalResult.PredictedWord;
        public string PersonalPredictedWord => PersonalResult == null ? string.Empty : PersonalResult.PredictedWord;
        public bool GlobalAccepted => GlobalResult != null && GlobalResult.Accepted;
        public bool PersonalAccepted => PersonalResult != null && PersonalResult.Accepted;
        public bool GlobalExpectedWordAccepted => GlobalAccepted && GlobalPredictedWord == ExpectedWord;
        public bool PersonalExpectedWordAccepted => PersonalResult != null && PersonalResult.QuestSuccess;
        public string GlobalWinningPrototypeId => GlobalResult == null ? string.Empty : GlobalResult.WinningPrototypeId;
        public string PersonalWinningPrototypeId => PersonalResult == null ? string.Empty : PersonalResult.WinningPrototypeId;
        public string FailureType { get; }
        public string FailureMessage { get; }

        internal TrustedHybridRecognitionResult(
            TrustedHybridOutcomeKind kind,
            string expectedWord,
            RuntimeRecognitionResult globalResult,
            PersonalRuntimeRecognitionResult personalResult,
            Exception failure)
        {
            Kind = kind;
            ExpectedWord = expectedWord ?? string.Empty;
            GlobalResult = globalResult;
            PersonalResult = personalResult;
            FailureType = failure == null ? string.Empty : failure.GetType().Name;
            FailureMessage = failure == null ? string.Empty : failure.Message;
        }
    }

    /// <summary>
    /// Immutable base-plus-qualified-personal view. Candidate repository changes cannot mutate a
    /// view already acquired by a quest/session.
    /// </summary>
    public sealed class TrustedPersonalizationActiveView
    {
        readonly IReadOnlyDictionary<string, IReadOnlyList<string>> candidateCaptureIds;
        internal IPersonalRuntimeEvaluator Evaluator { get; }

        public long Generation { get; }
        public string ProfileId { get; }
        public PersonalMultiPrototypeRuntimeProfile Profile { get; }
        public IReadOnlyDictionary<string, IReadOnlyList<string>> CandidateCaptureIds => candidateCaptureIds;

        internal TrustedPersonalizationActiveView(
            long generation,
            PersonalMultiPrototypeRuntimeProfile profile,
            IPersonalRuntimeEvaluator evaluator,
            IDictionary<string, List<string>> captureIds)
        {
            Generation = generation;
            Profile = profile ?? throw new ArgumentNullException(nameof(profile));
            ProfileId = profile.ProfileId;
            Evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
            candidateCaptureIds = new ReadOnlyDictionary<string, IReadOnlyList<string>>(
                captureIds.ToDictionary(
                    item => item.Key,
                    item => (IReadOnlyList<string>)Array.AsReadOnly(item.Value.ToArray()),
                    StringComparer.Ordinal));
        }
    }

    public sealed class TrustedActiveViewBuildResult
    {
        public bool Succeeded { get; }
        public bool Replaced { get; }
        public TrustedPersonalizationActiveView ActiveView { get; }
        public string FailureType { get; }
        public string FailureMessage { get; }

        internal TrustedActiveViewBuildResult(
            bool succeeded,
            bool replaced,
            TrustedPersonalizationActiveView activeView,
            Exception failure)
        {
            Succeeded = succeeded;
            Replaced = replaced;
            ActiveView = activeView;
            FailureType = failure == null ? string.Empty : failure.GetType().Name;
            FailureMessage = failure == null ? string.Empty : failure.Message;
        }
    }

    /// <summary>
    /// Read-only active-view owner for future NPC consumers. It never participates in practice
    /// enrollment recognition and never saves runtime observations as candidates.
    /// </summary>
    public sealed class TrustedPersonalizationActiveViewService
    {
        readonly RuntimeRecognitionService globalService;
        readonly string fixedPersonalProfileDirectory;
        readonly ITrustedPersonalizationCandidateRepository repository;
        TrustedPersonalizationActiveView activeView;

        public TrustedPersonalizationActiveView Current => activeView;
        public bool RuntimeObservationCanMutateEnrollment => false;

        public TrustedPersonalizationActiveViewService(
            RuntimeRecognitionService globalService,
            string fixedPersonalProfileDirectory,
            ITrustedPersonalizationCandidateRepository repository)
        {
            this.globalService = globalService ?? throw new ArgumentNullException(nameof(globalService));
            if (string.IsNullOrWhiteSpace(fixedPersonalProfileDirectory))
                throw new ArgumentException("fixed Step 9L profile directory is required", nameof(fixedPersonalProfileDirectory));
            this.fixedPersonalProfileDirectory = fixedPersonalProfileDirectory;
            this.repository = repository ?? throw new ArgumentNullException(nameof(repository));
            PersonalMultiPrototypeRuntimeProfile fixedProfile =
                PersonalMultiPrototypeRuntimeProfileLoader.Load(fixedPersonalProfileDirectory);
            activeView = CreateView(0, fixedProfile, Array.Empty<TrustedPersonalizationCandidate>());
        }

        public TrustedActiveViewBuildResult TryRefresh()
        {
            TrustedPersonalizationActiveView previous = activeView;
            try
            {
                TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
                PersonalMultiPrototypeRuntimeProfile fixedProfile =
                    PersonalMultiPrototypeRuntimeProfileLoader.Load(fixedPersonalProfileDirectory);
                TrustedPersonalizationCandidate[] qualified = inventory.Candidates
                    .Where(item => inventory.CountForWord(item.Word) >= 3)
                    .OrderBy(item => item.Word, StringComparer.Ordinal)
                    .ThenBy(item => item.CaptureSessionId, StringComparer.Ordinal)
                    .ToArray();
                TrustedPersonalizationActiveView candidate =
                    CreateView(previous.Generation + 1, fixedProfile, qualified);
                activeView = candidate;
                return new TrustedActiveViewBuildResult(true, true, candidate, null);
            }
            catch (Exception exception)
            {
                return new TrustedActiveViewBuildResult(false, false, previous, exception);
            }
        }

        public TrustedHybridRuntimeSession AcquireSession()
        {
            return new TrustedHybridRuntimeSession(globalService, activeView);
        }

        static TrustedPersonalizationActiveView CreateView(
            long generation,
            PersonalMultiPrototypeRuntimeProfile fixedProfile,
            IEnumerable<TrustedPersonalizationCandidate> candidates)
        {
            var prototypes = fixedProfile.Prototypes.ToDictionary(
                item => item.Key,
                item => new List<RecognitionSequence>(item.Value),
                StringComparer.Ordinal);
            var provenance = fixedProfile.PrototypeProvenance.ToDictionary(
                item => item.Key,
                item => new List<PersonalTemplateProvenance>(item.Value),
                StringComparer.Ordinal);
            var captureIds = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                word => word, word => new List<string>(), StringComparer.Ordinal);
            foreach (TrustedPersonalizationCandidate candidate in candidates)
            {
                if (!prototypes.ContainsKey(candidate.Word))
                    throw new InvalidDataException("trusted candidate has unsupported word: " + candidate.Word);
                string sequencePath = Path.Combine(candidate.CandidateDirectory, "normalized_sequence.wisrq");
                var file = new FileInfo(sequencePath);
                if (!file.Exists || file.Length <= 0)
                    throw new InvalidDataException("trusted candidate sequence disappeared after verification");
                prototypes[candidate.Word].Add(candidate.Sequence);
                provenance[candidate.Word].Add(new PersonalTemplateProvenance(
                    candidate.Word,
                    "practice_candidate/" + candidate.Word + "/" + candidate.CaptureSessionId + "/normalized_sequence.wisrq",
                    file.Length,
                    candidate.SequenceFileSha256,
                    candidate.Sequence.BinaryPayloadSha256,
                    candidate.CaptureSessionId,
                    "practice_enrollment::" + candidate.CaptureSessionId,
                    string.Empty,
                    candidate.SourcePayloadSha256,
                    string.Empty,
                    string.Empty,
                    candidate.LogicalPayloadSha256));
                captureIds[candidate.Word].Add(candidate.CaptureSessionId);
            }

            var thresholds = fixedProfile.Thresholds.ToDictionary(
                item => item.Key, item => item.Value, StringComparer.Ordinal);
            var warnings = fixedProfile.Warnings.Concat(new[]
            {
                "STEP10K_V2_1_TRUSTED_PRACTICE_CANDIDATES_READ_ONLY_ACTIVE_VIEW",
                "RUNTIME_OBSERVATIONS_NEVER_AUTO_PROMOTE",
            });
            var profile = new PersonalMultiPrototypeRuntimeProfile(
                fixedProfile.ProfileId + "+trusted-practice-g" + generation,
                prototypes,
                provenance,
                thresholds,
                warnings);
            return new TrustedPersonalizationActiveView(
                generation, profile, new PersonalMultiPrototypeRuntimeEvaluator(profile), captureIds);
        }
    }

    /// <summary>Session-local immutable evaluator pair; it has no candidate repository reference.</summary>
    public sealed class TrustedHybridRuntimeSession
    {
        readonly RuntimeRecognitionService globalService;
        readonly IPersonalRuntimeEvaluator personalEvaluator;

        public TrustedPersonalizationActiveView ActiveView { get; }
        public long ActiveViewGeneration => ActiveView.Generation;
        public bool RuntimeObservationCanMutateEnrollment => false;

        internal TrustedHybridRuntimeSession(
            RuntimeRecognitionService globalService,
            TrustedPersonalizationActiveView activeView)
        {
            this.globalService = globalService ?? throw new ArgumentNullException(nameof(globalService));
            ActiveView = activeView ?? throw new ArgumentNullException(nameof(activeView));
            personalEvaluator = activeView.Evaluator;
        }

        public TrustedHybridRecognitionResult Recognize(
            string querySessionId,
            RecognitionSequence query,
            string expectedWord)
        {
            try
            {
                RuntimeRecognitionResult global = globalService.Recognize(querySessionId, query);
                PersonalRuntimeRecognitionResult personal =
                    personalEvaluator.Recognize(querySessionId, query, expectedWord);
                bool globalExpected = global.Accepted && global.PredictedWord == expectedWord;
                bool personalExpected = personal.QuestSuccess;
                bool globalConflict = global.Accepted && global.PredictedWord != expectedWord;
                bool personalConflict = personal.Accepted && personal.PredictedWord != expectedWord;

                TrustedHybridOutcomeKind kind = TrustedHybridOutcomePolicy.Evaluate(
                    globalExpected,
                    globalConflict,
                    personalExpected,
                    personalConflict,
                    global.PredictedWord,
                    personal.PredictedWord);
                return new TrustedHybridRecognitionResult(kind, expectedWord, global, personal, null);
            }
            catch (Exception exception)
            {
                return new TrustedHybridRecognitionResult(
                    TrustedHybridOutcomeKind.TechnicalFailure,
                    expectedWord,
                    null,
                    null,
                    exception);
            }
        }
    }
}
