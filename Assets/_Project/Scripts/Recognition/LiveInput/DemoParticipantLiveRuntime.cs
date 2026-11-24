using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Explicit marker for the F18 fresh-participant route. A runtime implementing this
    /// interface is allowed to be *not* personally active: while only the approved public
    /// references are used, <see cref="IPersonalLiveRecognitionRuntime.IsPersonalActive"/>
    /// stays false and <see cref="IPersonalLiveRecognitionRuntime.ActiveProfileId"/> stays
    /// empty, so nothing in the pipeline can report common mode as a personal model.
    /// </summary>
    public interface IDemoParticipantLiveRuntime : IPersonalLiveRecognitionRuntime
    {
        string ParticipantId { get; }

        /// <summary>Honest provenance: the public-reference identity until this participant's
        /// own enrollment is complete, then this participant's own profile id.</summary>
        string RuntimeIdentity { get; }

        IReadOnlyDictionary<string, double> Thresholds { get; }

        bool IsReady { get; }
    }

    /// <summary>
    /// Fresh-participant runtime factory. It never reads the fixed Step 9H/9L personal
    /// profiles: the global load is the approved public one, and personalization can only
    /// come from candidates this participant enrolled under their own candidate root.
    /// </summary>
    public sealed class DemoParticipantLiveRuntimeFactory : IPersonalLiveRuntimeFactory
    {
        public const string PublicReferenceRuntimeIdentity = "public_reference_only_v1";
        public const string ParticipantProfileIdPrefix = "demo_participant_only_v1::";
        const string SequenceFileName = "normalized_sequence.wisrq";
        const int RequiredPerWord = 3;

        readonly DemoParticipantIdentity identity;
        readonly ApprovedPersonalLiveRuntimeFactory globalLoader = new ApprovedPersonalLiveRuntimeFactory();

        public DemoParticipantLiveRuntimeFactory(DemoParticipantIdentity identity)
        {
            this.identity = identity ?? throw new ArgumentNullException(nameof(identity));
        }

        /// <summary>Delegates to the approved global loader: same public templates, same frozen threshold.</summary>
        public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths) =>
            globalLoader.LoadGlobal(paths);

        public IPersonalLiveRecognitionRuntime CreateAndActivate(
            IPersonalLiveGlobalRuntime globalRuntime,
            PersonalLiveRuntimePaths paths)
        {
            if (!(globalRuntime is ApprovedGlobalRuntimeHandle approved))
                throw new ArgumentException("approved global runtime handle is required", nameof(globalRuntime));

            var repository = new PersonalizationCandidateRepository(identity.LearnerId, identity.CandidateRoot)
                as ITrustedPersonalizationCandidateRepository;
            if (repository == null)
                throw new InvalidOperationException("the demo participant route requires a trusted candidate repository");

            // Malformed candidate storage must surface, never be treated as "no candidates yet".
            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();

            if (!inventory.OverallEnrollmentGateReady)
                return new PublicReferenceOnlyLiveRuntime(identity.LearnerId, approved.Service);

            return CreateParticipantRuntimeWithCommonThreshold(
                approved.Service.Threshold, repository, identity);
        }

        // ---------------------------------------------------------------- public route

        /// <summary>
        /// Maps an approved global result onto the personal routing shape without changing any
        /// score, ranking or acceptance. The expected word is only consulted after the class
        /// scores and the ranking are already fixed, so it can never re-rank anything.
        /// </summary>
        public static PersonalLiveRoutingResult MapGlobalResult(
            RuntimeRecognitionResult result,
            string expectedWord)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));

            var scores = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                { "GREETING", result.GreetingDistance },
                { "LOVE", result.LoveDistance },
                { "THANK_YOU", result.ThankYouDistance },
            };
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, double> score in scores)
                accepts[score.Key] = score.Value <= result.Threshold;

            string predicted = result.PredictedWord;
            bool accepted = result.Accepted;

            bool expectedAccept = !string.IsNullOrEmpty(expectedWord) &&
                accepts.TryGetValue(expectedWord, out bool value) && value;
            bool questSuccess = accepted &&
                string.Equals(predicted, expectedWord, StringComparison.Ordinal) &&
                expectedAccept;

            return new PersonalLiveRoutingResult(
                predicted,
                accepted,
                expectedAccept,
                questSuccess,
                scores,
                accepts,
                result.BestDistance,
                result.WinningPrototypeId,
                new[] { "F18_PUBLIC_REFERENCE_ONLY_NO_PERSONAL_MODEL" });
        }

        // ------------------------------------------------------------ participant route

        public static TrustedPersonalizationActiveView CreateParticipantView(
            RuntimeRecognitionService global,
            ITrustedPersonalizationCandidateRepository repository,
            DemoParticipantIdentity identity)
        {
            if (global == null) throw new ArgumentNullException(nameof(global));
            return CreateParticipantViewWithCommonThreshold(global.Threshold, repository, identity);
        }

        public static TrustedHybridRuntimeSession CreateQuestSession(
            RuntimeRecognitionService global,
            ITrustedPersonalizationCandidateRepository repository,
            DemoParticipantIdentity identity)
        {
            if (global == null) throw new ArgumentNullException(nameof(global));
            return new TrustedHybridRuntimeSession(
                global, CreateParticipantView(global, repository, identity));
        }

        /// <summary>
        /// Shared core of the participant route. The threshold argument is the unchanged common
        /// threshold; no calibration, no per-class cutoff and no old personal threshold exists here.
        /// </summary>
        public static TrustedPersonalizationActiveView CreateParticipantViewWithCommonThreshold(
            double commonThreshold,
            ITrustedPersonalizationCandidateRepository repository,
            DemoParticipantIdentity identity)
        {
            if (repository == null) throw new ArgumentNullException(nameof(repository));
            if (identity == null) throw new ArgumentNullException(nameof(identity));
            if (double.IsNaN(commonThreshold) || double.IsInfinity(commonThreshold) || commonThreshold < 0.0)
                throw new InvalidDataException("common threshold is invalid");

            if (!string.Equals(repository.LearnerId, identity.LearnerId, StringComparison.Ordinal))
                throw new InvalidDataException("candidate repository belongs to another participant");
            if (!SamePath(repository.RootDirectory, identity.CandidateRoot))
                throw new InvalidDataException("candidate repository root is not this participant's candidate root");

            string learnerArea = Path.Combine(
                Canonical(identity.CandidateRoot), identity.LearnerId, "candidates");

            TrustedCandidateInventory inventory = repository.ReloadTrustedAndVerify();
            if (!inventory.OverallEnrollmentGateReady)
                throw new InvalidDataException("participant enrollment is incomplete");

            var prototypes = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                word => word, word => new List<RecognitionSequence>(), StringComparer.Ordinal);
            var provenance = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                word => word, word => new List<PersonalTemplateProvenance>(), StringComparer.Ordinal);
            var captureIds = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                word => word, word => new List<string>(), StringComparer.Ordinal);

            foreach (TrustedPersonalizationCandidate candidate in inventory.Candidates)
            {
                if (!prototypes.ContainsKey(candidate.Word))
                    throw new InvalidDataException("trusted candidate has unsupported word: " + candidate.Word);
                if (string.IsNullOrWhiteSpace(candidate.CaptureSessionId))
                    throw new InvalidDataException("trusted candidate has no capture session id");
                RequireHash(candidate.SequenceFileSha256, "sequence file");
                RequireHash(candidate.SourcePayloadSha256, "source payload");
                RequireHash(candidate.LogicalPayloadSha256, "logical payload");
                RequireHash(candidate.Sequence.SourceNpzSha256, "sequence source");
                RequireHash(candidate.Sequence.LogicalPayloadSha256, "sequence logical");

                string directory = Canonical(candidate.CandidateDirectory);
                if (!IsInside(learnerArea, directory))
                    throw new InvalidDataException("trusted candidate is outside this participant's candidate area");
                var directoryInfo = new DirectoryInfo(directory);
                if (directoryInfo.Parent == null ||
                    !string.Equals(directoryInfo.Parent.Name, candidate.Word, StringComparison.Ordinal))
                    throw new InvalidDataException("trusted candidate is not stored under its own word group");

                string sequencePath = Path.Combine(candidate.CandidateDirectory, SequenceFileName);
                var file = new FileInfo(sequencePath);
                if (!file.Exists || file.Length <= 0)
                    throw new InvalidDataException("trusted candidate sequence disappeared after verification");

                prototypes[candidate.Word].Add(candidate.Sequence);
                provenance[candidate.Word].Add(new PersonalTemplateProvenance(
                    candidate.Word,
                    "practice_candidate/" + candidate.Word + "/" + candidate.CaptureSessionId + "/" + SequenceFileName,
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

            foreach (string word in PersonalRuntimeProfileLoader.RequiredWords)
            {
                if (prototypes[word].Count < RequiredPerWord)
                    throw new InvalidDataException("participant enrollment is incomplete for " + word);
                if (prototypes[word].Count != inventory.CountForWord(word))
                    throw new InvalidDataException("trusted candidate membership does not match the counts for " + word);
            }

            var thresholds = PersonalRuntimeProfileLoader.RequiredWords.ToDictionary(
                word => word, word => commonThreshold, StringComparer.Ordinal);

            var profile = new PersonalMultiPrototypeRuntimeProfile(
                ParticipantProfileIdPrefix + identity.LearnerId,
                prototypes,
                provenance,
                thresholds,
                new[]
                {
                    "F18_PARTICIPANT_ONLY_PROTOTYPES_NO_FIXED_PERSONAL_BASELINE",
                    "F18_COMMON_THRESHOLD_UNCHANGED_FOR_EVERY_CLASS",
                    "RUNTIME_OBSERVATIONS_NEVER_AUTO_PROMOTE",
                });

            var guard = new DemoParticipantSelfReferenceGuardEvaluator(
                new PersonalMultiPrototypeRuntimeEvaluator(profile),
                identity.LearnerId,
                inventory.Candidates);

            return new TrustedPersonalizationActiveView(1L, profile, guard, captureIds);
        }

        public static IDemoParticipantLiveRuntime CreateParticipantRuntimeWithCommonThreshold(
            double commonThreshold,
            ITrustedPersonalizationCandidateRepository repository,
            DemoParticipantIdentity identity)
        {
            TrustedPersonalizationActiveView view =
                CreateParticipantViewWithCommonThreshold(commonThreshold, repository, identity);
            return new ParticipantOnlyLiveRuntime(identity.LearnerId, view);
        }

        // ------------------------------------------------------------------- helpers

        internal static string Canonical(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new InvalidDataException("a candidate path is required");
            return Path.GetFullPath(path)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }

        static bool SamePath(string left, string right) =>
            string.Equals(Canonical(left), Canonical(right), StringComparison.OrdinalIgnoreCase);

        static bool IsInside(string area, string candidate) =>
            candidate.Length > area.Length &&
            candidate.StartsWith(area + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);

        static void RequireHash(string value, string label)
        {
            if (value == null || value.Length != 64 || !value.All(IsLowerHex))
                throw new InvalidDataException("trusted candidate " + label + " hash is invalid");
        }

        static bool IsLowerHex(char value) =>
            (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
    }

    /// <summary>
    /// Binds a runtime to the participant that was active when it was created. Once the active
    /// participant changes, or the session becomes unavailable, the runtime is stale: it stops
    /// reporting ready and refuses to recognize. A runtime constructed while no session was
    /// active at all (synthetic/edit-mode construction) is never considered stale.
    /// </summary>
    internal sealed class DemoParticipantSessionBinding
    {
        readonly bool boundToLiveSession;

        internal string ParticipantId { get; }

        internal DemoParticipantSessionBinding(string participantId)
        {
            ParticipantId = participantId ?? string.Empty;
            boundToLiveSession = DemoParticipantSession.IsActive;
        }

        internal bool IsCurrent
        {
            get
            {
                if (!boundToLiveSession) return true;
                try
                {
                    return DemoParticipantSession.IsActive &&
                        string.Equals(
                            DemoParticipantSession.Current.LearnerId, ParticipantId, StringComparison.Ordinal);
                }
                catch (InvalidOperationException)
                {
                    return false;
                }
            }
        }

        internal void ThrowIfStale()
        {
            if (!IsCurrent)
                throw new InvalidOperationException(
                    "the active demo participant changed or is unavailable; this runtime is stale");
        }
    }

    /// <summary>
    /// Public-reference-only runtime. It is explicitly NOT a personal model: it forwards to the
    /// approved global service and reports the global scores, ranking and acceptance unchanged.
    /// </summary>
    public sealed class PublicReferenceOnlyLiveRuntime : IDemoParticipantLiveRuntime
    {
        readonly RuntimeRecognitionService service;
        readonly IReadOnlyDictionary<string, double> thresholds;
        readonly DemoParticipantSessionBinding binding;

        public string ParticipantId { get; }
        public string RuntimeIdentity => DemoParticipantLiveRuntimeFactory.PublicReferenceRuntimeIdentity;
        public IReadOnlyDictionary<string, double> Thresholds => thresholds;
        public bool IsReady => binding.IsCurrent;

        public bool IsPersonalActive => false;
        public string ActiveProfileId => string.Empty;

        public PublicReferenceOnlyLiveRuntime(string participantId, RuntimeRecognitionService service)
        {
            ParticipantId = participantId ?? string.Empty;
            this.service = service ?? throw new ArgumentNullException(nameof(service));
            binding = new DemoParticipantSessionBinding(ParticipantId);
            thresholds = new ReadOnlyDictionary<string, double>(
                PersonalLiveRecognitionController.SupportedWords.ToDictionary(
                    word => word, word => service.Threshold, StringComparer.Ordinal));
        }

        public PersonalLiveRoutingResult Recognize(
            string sessionId,
            RecognitionSequence sequence,
            string expectedWord)
        {
            binding.ThrowIfStale();
            return DemoParticipantLiveRuntimeFactory.MapGlobalResult(
                service.Recognize(sessionId, sequence), expectedWord);
        }
    }

    /// <summary>
    /// Participant-only personal runtime: every prototype comes from this participant's own
    /// verified candidates, and the class thresholds are the unchanged common threshold.
    /// </summary>
    public sealed class ParticipantOnlyLiveRuntime : IDemoParticipantLiveRuntime
    {
        readonly TrustedPersonalizationActiveView view;
        readonly IReadOnlyDictionary<string, double> thresholds;
        readonly DemoParticipantSessionBinding binding;

        public string ParticipantId { get; }
        public string RuntimeIdentity => view.ProfileId;
        public IReadOnlyDictionary<string, double> Thresholds => thresholds;
        public bool IsReady => binding.IsCurrent;

        public bool IsPersonalActive => true;
        public string ActiveProfileId => view.ProfileId;
        public TrustedPersonalizationActiveView ActiveView => view;

        internal ParticipantOnlyLiveRuntime(string participantId, TrustedPersonalizationActiveView view)
        {
            ParticipantId = participantId ?? string.Empty;
            this.view = view ?? throw new ArgumentNullException(nameof(view));
            binding = new DemoParticipantSessionBinding(ParticipantId);
            thresholds = view.Profile.Thresholds;
        }

        public PersonalLiveRoutingResult Recognize(
            string sessionId,
            RecognitionSequence sequence,
            string expectedWord)
        {
            binding.ThrowIfStale();
            PersonalRuntimeRecognitionResult result =
                view.Evaluator.Recognize(sessionId, sequence, expectedWord);
            return new PersonalLiveRoutingResult(
                result.PredictedWord,
                result.Accepted,
                result.ExpectedWordClassAccept,
                result.QuestSuccess,
                result.Scores,
                result.PerClassAccept,
                result.BestDistance,
                result.WinningPrototypeId,
                result.Warnings);
        }
    }

    /// <summary>
    /// Delegating wrapper around the approved multi-prototype evaluator. It changes no maths;
    /// it only refuses to score an enrolled sample against itself and refuses to serve a
    /// participant that is no longer the active one.
    /// </summary>
    internal sealed class DemoParticipantSelfReferenceGuardEvaluator : IPersonalRuntimeEvaluator
    {
        readonly IPersonalRuntimeEvaluator inner;
        readonly DemoParticipantSessionBinding binding;
        readonly HashSet<string> enrolledIds = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> enrolledSources = new HashSet<string>(StringComparer.Ordinal);
        readonly HashSet<string> enrolledLogical = new HashSet<string>(StringComparer.Ordinal);

        internal DemoParticipantSelfReferenceGuardEvaluator(
            IPersonalRuntimeEvaluator inner,
            string participantId,
            IEnumerable<TrustedPersonalizationCandidate> candidates)
        {
            this.inner = inner ?? throw new ArgumentNullException(nameof(inner));
            binding = new DemoParticipantSessionBinding(participantId);
            foreach (TrustedPersonalizationCandidate candidate in candidates)
            {
                Add(enrolledIds, candidate.CaptureSessionId);
                Add(enrolledIds, candidate.Sequence.SampleId);
                Add(enrolledSources, candidate.SourcePayloadSha256);
                Add(enrolledSources, candidate.Sequence.SourceNpzSha256);
                Add(enrolledLogical, candidate.LogicalPayloadSha256);
                Add(enrolledLogical, candidate.Sequence.LogicalPayloadSha256);
            }
        }

        static void Add(HashSet<string> target, string value)
        {
            if (!string.IsNullOrEmpty(value)) target.Add(value);
        }

        public PersonalRuntimeRecognitionResult Recognize(
            string querySessionId,
            RecognitionSequence query,
            string expectedWord)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            binding.ThrowIfStale();
            if (enrolledIds.Contains(query.SampleId) ||
                enrolledIds.Contains(querySessionId ?? string.Empty) ||
                enrolledSources.Contains(query.SourceNpzSha256) ||
                enrolledLogical.Contains(query.LogicalPayloadSha256))
                throw new InvalidOperationException("an enrolled candidate cannot be scored against itself");
            return inner.Recognize(querySessionId, query, expectedWord);
        }
    }
}
