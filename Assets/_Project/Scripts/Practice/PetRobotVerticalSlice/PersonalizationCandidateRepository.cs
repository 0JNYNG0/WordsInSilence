using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum CandidateRepositoryFailurePoint
    {
        AfterSequenceWrite,
        AfterMetadataWrite,
        AfterFinalDirectoryMove,
        AfterIndexCommitBeforeComplete,
        BeforeGuidedEnrollmentSeedManifestMove,
    }

    /// <summary>
    /// Local, opt-in candidate accumulation only. It writes the approved WISRCB01 normalized
    /// sequence payload and provenance; it never reads or changes an active runtime profile.
    /// </summary>
    public sealed class PersonalizationCandidateRepository :
        ITrustedPersonalizationCandidateRepository,
        IGuidedEnrollmentSeedPromotionRepository
    {
        const string SchemaVersion = "step10a_personalization_candidate_v1_0";
        const string GuidedSeedSchemaVersion = "step10k_guided_enrollment_seed_manifest_v1_1";
        const string GuidedSeedPolicyVersion = "guided_low_confidence_seed_exact_three_frozen_input_v1_1";
        const string GuidedSeedTrustBasis = "GuidedLowConfidenceSeed";
        const string GuidedSeedSelectionPolicy =
            "closed_set_margin_desc,threshold_margin_desc,capture_session_id_ordinal_asc";
        const string IndexFileName = "candidate_index.json";
        const string SequenceFileName = "normalized_sequence.wisrq";
        const string MetadataFileName = "candidate_metadata.json";
        const string ChecksumsFileName = "checksum_manifest.json";
        const string GuidedSeedDirectoryName = ".guided_enrollment_seeds";
        static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(false, true);
        static readonly string[] ForbiddenMediaExtensions =
            { ".mp4", ".mov", ".avi", ".webm", ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".ogg" };

        readonly Action<CandidateRepositoryFailurePoint> failureInjector;
        CandidateIndexDocument index;

        public string RootDirectory { get; }
        public string LearnerId { get; }
        public CandidateCountSnapshot Counts { get; private set; } = CandidateCountSnapshot.Empty;
        public TrustedCandidateInventory TrustedInventory { get; private set; } = TrustedCandidateInventory.Empty;

        public static string DefaultRootDirectory => Path.Combine(
            Application.persistentDataPath,
            "WordsInSilence",
            "PersonalizationCandidates",
            "v1");

        public PersonalizationCandidateRepository(
            string learnerId,
            string rootDirectory = null,
            Action<CandidateRepositoryFailurePoint> failureInjector = null)
        {
            LearnerId = RequireSafeSegment(learnerId, "learner/profile ID");
            RootDirectory = Path.GetFullPath(string.IsNullOrWhiteSpace(rootDirectory)
                ? DefaultRootDirectory
                : rootDirectory);
            this.failureInjector = failureInjector;
            ReloadAndVerify();
        }

        public CandidateCountSnapshot ReloadAndVerify()
        {
            string profileRoot = ProfileRoot;
            string indexPath = Path.Combine(profileRoot, IndexFileName);
            if (!File.Exists(indexPath))
            {
                RejectSeedFilesWithoutCandidateIndex();
                index = NewIndex();
                Counts = CandidateCountSnapshot.Empty;
                TrustedInventory = new TrustedCandidateInventory(
                    Counts, new Dictionary<string, int>(StringComparer.Ordinal),
                    Array.Empty<TrustedPersonalizationCandidate>(), 0);
                return Counts;
            }

            index = ParseJson<CandidateIndexDocument>(File.ReadAllBytes(indexPath), "candidate index");
            if (index.schema_version != SchemaVersion || index.learner_id != LearnerId || index.entries == null)
                throw new InvalidDataException("candidate index schema or learner ID mismatch");

            var keys = new HashSet<string>(StringComparer.Ordinal);
            var verified = new List<VerifiedCandidatePayload>();
            foreach (CandidateIndexEntry entry in index.entries)
            {
                ValidateIndexEntry(entry);
                if (!keys.Add(entry.capture_session_id))
                    throw new InvalidDataException("duplicate capture/session ID in candidate index");
                verified.Add(VerifyCandidateDirectory(entry));
            }

            string candidatesRoot = Path.Combine(profileRoot, "candidates");
            if (Directory.Exists(candidatesRoot))
            {
                int actualDirectories = Directory.GetDirectories(candidatesRoot, "*", SearchOption.AllDirectories)
                    .Count(path => File.Exists(Path.Combine(path, MetadataFileName)));
                if (actualDirectories != index.entries.Count)
                    throw new InvalidDataException("candidate index membership mismatch");
                if (Directory.GetFiles(candidatesRoot, "*", SearchOption.AllDirectories)
                    .Any(path => ForbiddenMediaExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase)))
                    throw new InvalidDataException("candidate storage contains prohibited media");
            }

            Counts = BuildCounts(index.entries);
            List<GuidedEnrollmentSeedManifest> manifests = LoadGuidedSeedManifests();
            ValidateGuidedSeedManifestSet(manifests, verified);
            TrustedInventory = BuildTrustedInventory(Counts, verified, manifests);
            return Counts;
        }

        public TrustedCandidateInventory ReloadTrustedAndVerify()
        {
            ReloadAndVerify();
            return TrustedInventory;
        }

        public GuidedEnrollmentSeedCommitResult TryCommitGuidedEnrollmentSeedQuorum(string word)
        {
            if (!PersonalLiveRecognitionController.IsSupportedWord(word))
                throw new ArgumentException("unsupported practice word", nameof(word));

            string targetPath = GuidedSeedManifestPath(word);
            try
            {
                ReloadAndVerify();
            }
            catch (Exception exception)
            {
                return SeedResult(GuidedEnrollmentSeedCommitStatus.Failed, word, targetPath,
                    "repository or existing guided seed manifest failed strict verification: " + exception.Message);
            }
            var verified = VerifyAllIndexedCandidates();
            if (File.Exists(targetPath))
            {
                try
                {
                    GuidedEnrollmentSeedManifest existing = ParseJson<GuidedEnrollmentSeedManifest>(
                        File.ReadAllBytes(targetPath), "guided enrollment seed manifest");
                    ValidateGuidedSeedManifest(existing, verified);
                    return SeedResult(GuidedEnrollmentSeedCommitStatus.AlreadyCommitted, word, targetPath,
                        "the existing frozen-input guided seed manifest is already committed",
                        existing.selected_candidates);
                }
                catch (Exception exception)
                {
                    return SeedResult(GuidedEnrollmentSeedCommitStatus.Failed, word, targetPath,
                        "existing guided seed manifest failed strict verification: " + exception.Message);
                }
            }

            if (TrustedInventory.Candidates.Count(candidate =>
                    candidate.Word == word &&
                    candidate.TrustBasis == TrustedPersonalizationTrustBasis.StrictEligible) >= 3)
                return SeedResult(GuidedEnrollmentSeedCommitStatus.NotRequired, word, targetPath,
                    "three strict Eligible candidates already satisfy the word quorum");

            GuidedEnrollmentSeedManifest manifest;
            GuidedEnrollmentSeedCommitStatus selectionFailure;
            string selectionMessage;
            if (!TryBuildGuidedSeedManifest(word, verified, out manifest, out selectionFailure, out selectionMessage))
                return SeedResult(selectionFailure, word, targetPath, selectionMessage);

            byte[] bytes = JsonBytes(manifest);

            try
            {
                var proposed = LoadGuidedSeedManifests();
                proposed.Add(manifest);
                ValidateGuidedSeedManifestSet(proposed, verified);
            }
            catch (Exception exception)
            {
                return SeedResult(GuidedEnrollmentSeedCommitStatus.DuplicateIdentity, word, targetPath,
                    "guided seed manifest conflicts with the existing trusted inventory: " + exception.Message);
            }

            string directory = Path.GetDirectoryName(targetPath);
            string stagePath = Path.Combine(ProfileRoot, ".guided_seed_stage_" + word + "." + Guid.NewGuid().ToString("N") + ".tmp");
            try
            {
                Directory.CreateDirectory(directory);
                File.WriteAllBytes(stagePath, bytes);
                GuidedEnrollmentSeedManifest staged = ParseJson<GuidedEnrollmentSeedManifest>(
                    File.ReadAllBytes(stagePath), "staged guided enrollment seed manifest");
                var stagedSet = LoadGuidedSeedManifests();
                stagedSet.Add(staged);
                ValidateGuidedSeedManifestSet(stagedSet, verified);
                Inject(CandidateRepositoryFailurePoint.BeforeGuidedEnrollmentSeedManifestMove);
                if (File.Exists(targetPath))
                    throw new IOException("guided seed manifest target appeared before atomic publish");
                File.Move(stagePath, targetPath);
                ReloadAndVerify();
                return SeedResult(GuidedEnrollmentSeedCommitStatus.Committed, word, targetPath,
                    "guided low-confidence seed quorum committed without changing candidate bytes",
                    manifest.selected_candidates);
            }
            catch (Exception exception)
            {
                DeleteFileIfExists(stagePath);
                DeleteDirectoryIfEmpty(directory);
                return SeedResult(GuidedEnrollmentSeedCommitStatus.Failed, word, targetPath,
                    exception.GetType().Name + ": " + exception.Message);
            }
        }

        public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            ValidateRequest(request);
            ReloadAndVerify();

            string word = request.Attempt.ExpectedWord;
            string sessionId = RequireSafeSegment(request.Attempt.CaptureSessionId, "capture/session ID");
            string finalDirectory = Path.Combine(ProfileRoot, "candidates", word, sessionId);
            byte[] sequenceBytes = RecognitionSequenceCandidateBinary.Write(request.Attempt.NormalizedSequence);
            string sequenceFileSha256 = Sha256(sequenceBytes);
            bool duplicatePayload = index.entries.Any(item => item.sequence_sha256 == sequenceFileSha256) ||
                TrustedInventory.Candidates.Any(item =>
                    item.SourcePayloadSha256 == request.Attempt.NormalizedSequence.SourceNpzSha256 ||
                    item.LogicalPayloadSha256 == request.Attempt.NormalizedSequence.LogicalPayloadSha256);
            if (Directory.Exists(finalDirectory) || index.entries.Any(item => item.capture_session_id == sessionId) || duplicatePayload)
                return new CandidateSaveResult(CandidateStorageStatus.Duplicate, finalDirectory,
                    duplicatePayload
                        ? "candidate payload identity was already committed"
                        : "capture/session ID was already committed",
                    Counts);

            string transactionId = Guid.NewGuid().ToString("N");
            string transactionRoot = Path.Combine(ProfileRoot, ".transactions");
            string stageDirectory = Path.Combine(transactionRoot, transactionId);
            string indexTempPath = Path.Combine(ProfileRoot, ".candidate_index." + transactionId + ".tmp");
            string indexBackupPath = Path.Combine(ProfileRoot, ".candidate_index." + transactionId + ".bak");
            string indexPath = Path.Combine(ProfileRoot, IndexFileName);
            bool finalCreated = false;
            bool indexCommitted = false;
            bool indexPreviouslyExisted = File.Exists(indexPath);

            try
            {
                Directory.CreateDirectory(stageDirectory);
                File.WriteAllBytes(Path.Combine(stageDirectory, SequenceFileName), sequenceBytes);
                Inject(CandidateRepositoryFailurePoint.AfterSequenceWrite);

                CandidateMetadata metadata = BuildMetadata(request, sequenceBytes);
                metadata.learner_id = LearnerId;
                byte[] metadataBytes = JsonBytes(metadata);
                File.WriteAllBytes(Path.Combine(stageDirectory, MetadataFileName), metadataBytes);
                Inject(CandidateRepositoryFailurePoint.AfterMetadataWrite);

                var checksums = new CandidateChecksumManifest
                {
                    schema_version = SchemaVersion,
                    files = new[]
                    {
                        new CandidateChecksumEntry { path = SequenceFileName, size_bytes = sequenceBytes.LongLength, sha256 = Sha256(sequenceBytes) },
                        new CandidateChecksumEntry { path = MetadataFileName, size_bytes = metadataBytes.LongLength, sha256 = Sha256(metadataBytes) },
                    },
                };
                File.WriteAllBytes(Path.Combine(stageDirectory, ChecksumsFileName), JsonBytes(checksums));
                VerifyStagedCandidate(stageDirectory, metadata.capture_session_id);

                Directory.CreateDirectory(Path.GetDirectoryName(finalDirectory));
                Directory.Move(stageDirectory, finalDirectory);
                finalCreated = true;
                Inject(CandidateRepositoryFailurePoint.AfterFinalDirectoryMove);

                CandidateIndexDocument nextIndex = CloneIndex(index);
                nextIndex.entries.Add(new CandidateIndexEntry
                {
                    capture_session_id = metadata.capture_session_id,
                    word = metadata.word,
                    outcome = metadata.outcome,
                    eligibility = metadata.eligibility,
                    relative_directory = RelativePath(finalDirectory),
                    sequence_sha256 = metadata.sequence_binary_sha256,
                    metadata_sha256 = Sha256(metadataBytes),
                });
                nextIndex.entries = nextIndex.entries
                    .OrderBy(item => item.capture_session_id, StringComparer.Ordinal).ToList();

                File.WriteAllBytes(indexTempPath, JsonBytes(nextIndex));
                ParseJson<CandidateIndexDocument>(File.ReadAllBytes(indexTempPath), "candidate index temp");
                if (indexPreviouslyExisted)
                    File.Replace(indexTempPath, indexPath, indexBackupPath, true);
                else
                    File.Move(indexTempPath, indexPath);
                indexCommitted = true;
                Inject(CandidateRepositoryFailurePoint.AfterIndexCommitBeforeComplete);

                index = nextIndex;
                Counts = ReloadAndVerify();
                DeleteFileIfExists(indexBackupPath);
                DeleteDirectoryIfEmpty(transactionRoot);
                return new CandidateSaveResult(CandidateStorageStatus.Saved, finalDirectory,
                    "candidate committed; active personal profile remains unchanged", Counts);
            }
            catch (Exception exception)
            {
                DeleteDirectoryIfExists(stageDirectory);
                if (finalCreated) DeleteDirectoryIfExists(finalDirectory);
                DeleteFileIfExists(indexTempPath);

                if (indexCommitted)
                {
                    DeleteFileIfExists(indexPath);
                    if (File.Exists(indexBackupPath)) File.Move(indexBackupPath, indexPath);
                }
                else
                {
                    DeleteFileIfExists(indexBackupPath);
                }
                DeleteDirectoryIfEmpty(transactionRoot);

                try { ReloadAndVerify(); }
                catch
                {
                    index = NewIndex();
                    Counts = CandidateCountSnapshot.Empty;
                    TrustedInventory = TrustedCandidateInventory.Empty;
                }
                return new CandidateSaveResult(CandidateStorageStatus.Failed, string.Empty,
                    exception.GetType().Name + ": " + exception.Message, Counts);
            }
        }

        void RejectSeedFilesWithoutCandidateIndex()
        {
            string directory = GuidedSeedDirectory;
            if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
                throw new InvalidDataException("guided seed manifests require a committed candidate index");
        }

        List<VerifiedCandidatePayload> VerifyAllIndexedCandidates()
        {
            var verified = new List<VerifiedCandidatePayload>();
            foreach (CandidateIndexEntry entry in index.entries)
                verified.Add(VerifyCandidateDirectory(entry));
            return verified;
        }

        List<GuidedEnrollmentSeedManifest> LoadGuidedSeedManifests()
        {
            string directory = GuidedSeedDirectory;
            var manifests = new List<GuidedEnrollmentSeedManifest>();
            if (!Directory.Exists(directory)) return manifests;
            if ((File.GetAttributes(directory) & FileAttributes.ReparsePoint) != 0 ||
                Directory.GetDirectories(directory, "*", SearchOption.TopDirectoryOnly).Length != 0)
                throw new InvalidDataException("guided seed manifest directory is not a regular flat directory");

            var expectedNames = new HashSet<string>(
                PersonalLiveRecognitionController.SupportedWords.Select(word => word + ".json"),
                StringComparer.Ordinal);
            foreach (string path in Directory.GetFiles(directory, "*", SearchOption.TopDirectoryOnly))
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0 ||
                    !expectedNames.Contains(Path.GetFileName(path)))
                    throw new InvalidDataException("unexpected guided seed manifest directory member");
                GuidedEnrollmentSeedManifest manifest = ParseJson<GuidedEnrollmentSeedManifest>(
                    File.ReadAllBytes(path), "guided enrollment seed manifest");
                if (Path.GetFileNameWithoutExtension(path) != manifest.word)
                    throw new InvalidDataException("guided seed manifest filename/word mismatch");
                manifests.Add(manifest);
            }
            return manifests;
        }

        bool TryBuildGuidedSeedManifest(
            string word,
            IReadOnlyList<VerifiedCandidatePayload> verified,
            out GuidedEnrollmentSeedManifest manifest,
            out GuidedEnrollmentSeedCommitStatus failureStatus,
            out string failureMessage)
        {
            manifest = null;
            failureStatus = GuidedEnrollmentSeedCommitStatus.Failed;
            failureMessage = string.Empty;

            var captureIds = new HashSet<string>(StringComparer.Ordinal);
            var sequenceHashes = new HashSet<string>(StringComparer.Ordinal);
            var sourceHashes = new HashSet<string>(StringComparer.Ordinal);
            var logicalHashes = new HashSet<string>(StringComparer.Ordinal);
            foreach (VerifiedCandidatePayload payload in verified.Where(IsStrictTrustedContract))
            {
                AddIdentityOrThrow(payload, captureIds, sequenceHashes, sourceHashes, logicalHashes,
                    "strict trusted candidate identity is duplicated");
            }

            var candidates = verified.Where(payload =>
                    payload.Metadata.word == word && IsGuidedSeedCandidateContract(payload))
                .ToList();
            if (candidates.Count < 3)
            {
                failureStatus = GuidedEnrollmentSeedCommitStatus.InsufficientCandidates;
                failureMessage = "fewer than three strictly verified LowConfidenceCorrect ReviewOnly candidates exist";
                return false;
            }

            try
            {
                foreach (VerifiedCandidatePayload payload in candidates)
                    AddIdentityOrThrow(payload, captureIds, sequenceHashes, sourceHashes, logicalHashes,
                        "guided seed candidate identity is duplicated");
            }
            catch (InvalidDataException exception)
            {
                failureStatus = GuidedEnrollmentSeedCommitStatus.DuplicateIdentity;
                failureMessage = exception.Message;
                return false;
            }

            List<VerifiedCandidatePayload> ordered = candidates
                .OrderByDescending(payload => payload.Metadata.closed_set_margin)
                .ThenByDescending(payload => payload.Metadata.threshold_margin)
                .ThenBy(payload => payload.Metadata.capture_session_id, StringComparer.Ordinal)
                .ToList();
            manifest = new GuidedEnrollmentSeedManifest
            {
                schema_version = GuidedSeedSchemaVersion,
                policy_version = GuidedSeedPolicyVersion,
                learner_id = LearnerId,
                word = word,
                trust_basis = GuidedSeedTrustBasis,
                selection_policy = GuidedSeedSelectionPolicy,
                frozen_selection_input = ordered.Select(payload =>
                    GuidedSeedFrozenInputEntry.From(payload, RelativePath(payload.Directory))).ToArray(),
                selected_candidates = ordered.Take(3).Select((payload, position) =>
                    GuidedSeedCandidateEntry.From(payload, position + 1, RelativePath(payload.Directory))).ToArray(),
            };
            manifest.selection_input_count = manifest.frozen_selection_input.Length;
            manifest.selection_input_sha256 = SelectionInputSha256(manifest.frozen_selection_input);
            return true;
        }

        void ValidateGuidedSeedManifest(
            GuidedEnrollmentSeedManifest manifest,
            IReadOnlyList<VerifiedCandidatePayload> verified)
        {
            if (manifest == null || manifest.schema_version != GuidedSeedSchemaVersion ||
                manifest.policy_version != GuidedSeedPolicyVersion || manifest.learner_id != LearnerId ||
                !PersonalLiveRecognitionController.IsSupportedWord(manifest.word) ||
                manifest.trust_basis != GuidedSeedTrustBasis ||
                manifest.selection_policy != GuidedSeedSelectionPolicy ||
                manifest.frozen_selection_input == null || manifest.frozen_selection_input.Length < 3 ||
                manifest.selected_candidates == null || manifest.selected_candidates.Length != 3)
                throw new InvalidDataException("guided seed manifest schema or identity mismatch");

            // Candidate metadata bytes and hashes are verified below, and ordering is rebuilt from
            // that verified metadata. Do not make duplicated JSON double text identity-bearing:
            // JsonUtility can parse the same value to different low bits across Unity/runtime paths.
            if (manifest.selection_input_count != manifest.frozen_selection_input.Length ||
                !RecognitionSequence.IsLowerHexSha256(manifest.selection_input_sha256))
                throw new InvalidDataException("guided seed manifest selection inventory mismatch");

            var byCapture = verified.ToDictionary(
                payload => payload.Metadata.capture_session_id,
                payload => payload,
                StringComparer.Ordinal);
            var captureIds = new HashSet<string>(StringComparer.Ordinal);
            var sequenceHashes = new HashSet<string>(StringComparer.Ordinal);
            var sourceHashes = new HashSet<string>(StringComparer.Ordinal);
            var logicalHashes = new HashSet<string>(StringComparer.Ordinal);
            var frozenPayloads = new List<VerifiedCandidatePayload>();
            foreach (GuidedSeedFrozenInputEntry frozen in manifest.frozen_selection_input)
            {
                VerifiedCandidatePayload payload;
                if (frozen == null || !byCapture.TryGetValue(frozen.capture_session_id, out payload) ||
                    payload.Metadata.word != manifest.word || !IsGuidedSeedCandidateContract(payload) ||
                    !frozen.EqualsIdentity(payload, RelativePath(payload.Directory)))
                    throw new InvalidDataException("guided seed frozen input is absent, changed, or untrusted");
                AddIdentityOrThrow(payload, captureIds, sequenceHashes, sourceHashes, logicalHashes,
                    "guided seed frozen input identity is duplicated");
                frozenPayloads.Add(payload);
            }

            List<VerifiedCandidatePayload> ordered = frozenPayloads
                .OrderByDescending(payload => payload.Metadata.closed_set_margin)
                .ThenByDescending(payload => payload.Metadata.threshold_margin)
                .ThenBy(payload => payload.Metadata.capture_session_id, StringComparer.Ordinal)
                .ToList();
            for (int index = 0; index < ordered.Count; index++)
                if (!manifest.frozen_selection_input[index].EqualsIdentity(
                        ordered[index], RelativePath(ordered[index].Directory)))
                    throw new InvalidDataException("guided seed frozen input ordering mismatch");
            for (int index = 0; index < 3; index++)
                if (!manifest.selected_candidates[index].EqualsIdentity(
                        GuidedSeedCandidateEntry.From(ordered[index], index + 1, RelativePath(ordered[index].Directory))))
                    throw new InvalidDataException("guided seed manifest selection or candidate identity mismatch");
        }

        void ValidateGuidedSeedManifestSet(
            IReadOnlyList<GuidedEnrollmentSeedManifest> manifests,
            IReadOnlyList<VerifiedCandidatePayload> verified)
        {
            var words = new HashSet<string>(StringComparer.Ordinal);
            var captureIds = new HashSet<string>(StringComparer.Ordinal);
            var sequenceHashes = new HashSet<string>(StringComparer.Ordinal);
            var sourceHashes = new HashSet<string>(StringComparer.Ordinal);
            var logicalHashes = new HashSet<string>(StringComparer.Ordinal);
            foreach (VerifiedCandidatePayload payload in verified.Where(IsStrictTrustedContract))
                AddIdentityOrThrow(payload, captureIds, sequenceHashes, sourceHashes, logicalHashes,
                    "strict trusted candidate identity is duplicated");
            var byCapture = verified.ToDictionary(
                payload => payload.Metadata.capture_session_id,
                payload => payload,
                StringComparer.Ordinal);
            foreach (GuidedEnrollmentSeedManifest manifest in manifests)
            {
                ValidateGuidedSeedManifest(manifest, verified);
                if (!words.Add(manifest.word))
                    throw new InvalidDataException("multiple guided seed manifests exist for one word");
                foreach (GuidedSeedFrozenInputEntry seed in manifest.frozen_selection_input)
                {
                    VerifiedCandidatePayload payload;
                    if (!byCapture.TryGetValue(seed.capture_session_id, out payload))
                        throw new InvalidDataException("guided seed candidate is absent from the verified index");
                    AddIdentityOrThrow(payload, captureIds, sequenceHashes, sourceHashes, logicalHashes,
                        "guided seed identity duplicates another trusted candidate");
                }
            }
        }

        static bool IsStrictTrustedContractForWord(string word, VerifiedCandidatePayload payload) =>
            payload.Metadata.word == word && IsStrictTrustedContract(payload);

        static bool IsStrictTrustedContract(VerifiedCandidatePayload payload)
        {
            CandidateMetadata metadata = payload.Metadata;
            return payload.Entry.eligibility == PersonalizationCandidateEligibility.Eligible.ToString() &&
                metadata.eligibility == PersonalizationCandidateEligibility.Eligible.ToString() &&
                metadata.eligible_for_profile &&
                metadata.outcome == GuidedQuestOutcomeKind.StrictSuccess.ToString() &&
                metadata.controller_state == PersonalLiveCaptureState.Completed.ToString() &&
                metadata.word == metadata.expected_word && metadata.expected_word == metadata.predicted_word &&
                metadata.strict_accepted && metadata.strict_expected_class_accept && metadata.strict_quest_success &&
                string.IsNullOrEmpty(metadata.failure_code) && string.IsNullOrEmpty(metadata.primary_failure_code) &&
                metadata.sequence_format == "WISRCB01" &&
                metadata.sequence_binary_sha256 == payload.Entry.sequence_sha256 &&
                metadata.sequence_source_payload_sha256 == payload.Sequence.SourceNpzSha256 &&
                metadata.sequence_normalized_payload_sha256 == payload.Sequence.LogicalPayloadSha256;
        }

        static bool IsGuidedSeedCandidateContract(VerifiedCandidatePayload payload)
        {
            CandidateMetadata metadata = payload.Metadata;
            return payload.Entry.eligibility == PersonalizationCandidateEligibility.ReviewOnly.ToString() &&
                metadata.eligibility == PersonalizationCandidateEligibility.ReviewOnly.ToString() &&
                !metadata.eligible_for_profile &&
                metadata.outcome == GuidedQuestOutcomeKind.LowConfidenceCorrect.ToString() &&
                metadata.controller_state == PersonalLiveCaptureState.Completed.ToString() &&
                metadata.word == metadata.expected_word && metadata.expected_word == metadata.predicted_word &&
                !metadata.strict_accepted && !metadata.strict_expected_class_accept && !metadata.strict_quest_success &&
                string.IsNullOrEmpty(metadata.failure_code) && string.IsNullOrEmpty(metadata.primary_failure_code) &&
                metadata.route_id == PersonalLiveProfileMode.MultiPrototypeCandidate.ToString() &&
                !string.IsNullOrWhiteSpace(metadata.profile_id) &&
                IsFinite(metadata.expected_word_distance) && IsFinite(metadata.best_distance) &&
                IsFinite(metadata.approved_threshold) && IsFinite(metadata.threshold_margin) &&
                IsFinite(metadata.closed_set_margin) && metadata.closed_set_margin > 0d &&
                metadata.raw_frame_count > 0 && metadata.normalized_frame_count > 0 &&
                metadata.valid_landmark_count >= 0 && metadata.interpolated_landmark_count >= 0 &&
                metadata.sequence_format == "WISRCB01" &&
                metadata.sequence_binary_sha256 == payload.Entry.sequence_sha256 &&
                metadata.sequence_source_payload_sha256 == payload.Sequence.SourceNpzSha256 &&
                metadata.sequence_normalized_payload_sha256 == payload.Sequence.LogicalPayloadSha256;
        }

        static void AddIdentityOrThrow(
            VerifiedCandidatePayload payload,
            HashSet<string> captureIds,
            HashSet<string> sequenceHashes,
            HashSet<string> sourceHashes,
            HashSet<string> logicalHashes,
            string message)
        {
            CandidateMetadata metadata = payload.Metadata;
            if (!captureIds.Add(metadata.capture_session_id) ||
                !sequenceHashes.Add(metadata.sequence_binary_sha256) ||
                !sourceHashes.Add(metadata.sequence_source_payload_sha256) ||
                !logicalHashes.Add(metadata.sequence_normalized_payload_sha256))
                throw new InvalidDataException(message);
        }

        static string SelectionInputSha256(IEnumerable<GuidedSeedFrozenInputEntry> candidates)
        {
            string text = string.Join("\n", candidates.Select(payload =>
                payload.capture_session_id + "\t" + payload.candidate_relative_directory + "\t" +
                payload.candidate_metadata_sha256 + "\t" + payload.sequence_file_sha256 + "\t" +
                payload.source_payload_sha256 + "\t" + payload.logical_payload_sha256 + "\t" +
                BitConverter.DoubleToInt64Bits(payload.closed_set_margin).ToString("x16", CultureInfo.InvariantCulture) + "\t" +
                BitConverter.DoubleToInt64Bits(payload.threshold_margin).ToString("x16", CultureInfo.InvariantCulture)));
            return Sha256(Utf8NoBom.GetBytes(text));
        }

        static GuidedEnrollmentSeedCommitResult SeedResult(
            GuidedEnrollmentSeedCommitStatus status,
            string word,
            string path,
            string message,
            IEnumerable<GuidedSeedCandidateEntry> selected = null) =>
            new GuidedEnrollmentSeedCommitResult(status, word, path, message,
                selected == null ? null : selected.Select(item => item.capture_session_id));

        string ProfileRoot => Path.Combine(RootDirectory, LearnerId);
        string GuidedSeedDirectory => Path.Combine(ProfileRoot, GuidedSeedDirectoryName);
        string GuidedSeedManifestPath(string word) => Path.Combine(GuidedSeedDirectory, word + ".json");

        static void ValidateRequest(PersonalizationCandidateSaveRequest request)
        {
            PracticeRecognitionAttempt attempt = request.Attempt;
            // F27: the full-hand route has no 16-point normalized sequence and no full-hand
            // enrollment exists yet. Refusing here keeps "enrolled" meaning what it meant.
            if (attempt.FullHandGameplay != null)
                throw new InvalidOperationException(
                    "a full-hand (46-point) attempt cannot be stored as a 16-point " +
                    "personalization candidate; full-hand enrollment is not implemented");
            if (attempt.CaptureState != PersonalLiveCaptureState.Completed ||
                attempt.NormalizedSequence == null || attempt.Outcome == null || attempt.Failure != null ||
                attempt.PrimaryFailure != null)
                throw new InvalidOperationException("only failure-free Completed results can become candidates");
            if (!PersonalLiveRecognitionController.IsSupportedWord(attempt.ExpectedWord) ||
                attempt.ExpectedWord != attempt.PredictedWord)
                throw new InvalidOperationException("candidate expected/predicted word contract failed");
            if (attempt.RouteId != PersonalLiveProfileMode.MultiPrototypeCandidate.ToString() ||
                string.IsNullOrWhiteSpace(attempt.ProfileId))
                throw new InvalidOperationException("candidate route/profile provenance is invalid");
            if (!IsFiniteNonnegative(attempt.ExpectedWordThreshold))
                throw new InvalidOperationException("candidate threshold is invalid");
            if (!IsFiniteNonnegative(attempt.GreetingDistance) || !IsFiniteNonnegative(attempt.LoveDistance) ||
                !IsFiniteNonnegative(attempt.ThankYouDistance) || !IsFiniteNonnegative(attempt.BestDistance) ||
                string.IsNullOrWhiteSpace(attempt.WinningPrototypeId) || attempt.RawFrameCount <= 0 ||
                attempt.NormalizedFrameCount <= 0)
                throw new InvalidOperationException("candidate recognition diagnostics are invalid");

            if (request.Eligibility == PersonalizationCandidateEligibility.Eligible)
            {
                if (attempt.Outcome.Kind != GuidedQuestOutcomeKind.StrictSuccess ||
                    !attempt.Outcome.StrictAccepted || !attempt.Outcome.StrictExpectedWordClassAccept ||
                    !attempt.Outcome.StrictQuestSuccess)
                    throw new InvalidOperationException("Eligible requires the unchanged strict success gate");
            }
            else if (attempt.Outcome.Kind != GuidedQuestOutcomeKind.LowConfidenceCorrect ||
                     attempt.Outcome.StrictQuestSuccess)
            {
                throw new InvalidOperationException("ReviewOnly requires low-confidence correct with strict failure preserved");
            }
        }

        static CandidateMetadata BuildMetadata(PersonalizationCandidateSaveRequest request, byte[] sequenceBytes)
        {
            PracticeRecognitionAttempt attempt = request.Attempt;
            double expectedDistance = DistanceForWord(attempt, attempt.ExpectedWord);
            double[] ordered = { attempt.GreetingDistance, attempt.LoveDistance, attempt.ThankYouDistance };
            Array.Sort(ordered);
            return new CandidateMetadata
            {
                schema_version = SchemaVersion,
                learner_id = string.Empty,
                capture_session_id = attempt.CaptureSessionId,
                word = attempt.ExpectedWord,
                expected_word = attempt.ExpectedWord,
                predicted_word = attempt.PredictedWord,
                outcome = attempt.Outcome.Kind.ToString(),
                route_id = attempt.RouteId,
                profile_id = attempt.ProfileId,
                controller_state = attempt.CaptureState.ToString(),
                strict_accepted = attempt.StrictAccepted,
                strict_expected_class_accept = attempt.StrictExpectedWordClassAccept,
                strict_quest_success = attempt.StrictQuestSuccess,
                raw_frame_count = attempt.RawFrameCount,
                normalized_frame_count = attempt.NormalizedFrameCount,
                valid_landmark_count = attempt.ValidLandmarkCount,
                interpolated_landmark_count = attempt.InterpolatedLandmarkCount,
                expected_word_distance = expectedDistance,
                best_distance = attempt.BestDistance,
                approved_threshold = attempt.ExpectedWordThreshold,
                threshold_margin = attempt.ExpectedWordThreshold - expectedDistance,
                closed_set_margin = ordered[1] - ordered[0],
                winning_prototype_id = attempt.WinningPrototypeId,
                failure_code = string.Empty,
                primary_failure_code = string.Empty,
                eligibility = request.Eligibility.ToString(),
                eligible_for_profile = request.Eligibility == PersonalizationCandidateEligibility.Eligible,
                created_utc = request.CreatedUtc.ToString("O", CultureInfo.InvariantCulture),
                sequence_format = "WISRCB01",
                sequence_binary_sha256 = Sha256(sequenceBytes),
                sequence_source_payload_sha256 = attempt.NormalizedSequence.SourceNpzSha256,
                sequence_normalized_payload_sha256 = attempt.NormalizedSequence.LogicalPayloadSha256,
            };
        }

        VerifiedCandidatePayload VerifyCandidateDirectory(CandidateIndexEntry entry)
        {
            string directory = FullRelativePath(entry.relative_directory);
            RecognitionSequence sequence = VerifyStagedCandidate(directory, entry.capture_session_id);
            CandidateMetadata metadata = ParseJson<CandidateMetadata>(
                File.ReadAllBytes(Path.Combine(directory, MetadataFileName)), "candidate metadata");
            byte[] metadataBytes = File.ReadAllBytes(Path.Combine(directory, MetadataFileName));
            if (entry.word != metadata.word || entry.outcome != metadata.outcome ||
                entry.eligibility != metadata.eligibility || entry.sequence_sha256 != metadata.sequence_binary_sha256 ||
                entry.metadata_sha256 != Sha256(metadataBytes))
                throw new InvalidDataException("candidate index entry does not match candidate payload");
            return new VerifiedCandidatePayload
            {
                Entry = entry,
                Metadata = metadata,
                Directory = directory,
                Sequence = sequence,
            };
        }

        static RecognitionSequence VerifyStagedCandidate(string directory, string expectedSessionId)
        {
            string sequencePath = Path.Combine(directory, SequenceFileName);
            string metadataPath = Path.Combine(directory, MetadataFileName);
            string checksumsPath = Path.Combine(directory, ChecksumsFileName);
            if (!File.Exists(sequencePath) || !File.Exists(metadataPath) || !File.Exists(checksumsPath))
                throw new InvalidDataException("candidate transaction is incomplete");
            CandidateMetadata metadata = ParseJson<CandidateMetadata>(File.ReadAllBytes(metadataPath), "candidate metadata");
            if (metadata.schema_version != SchemaVersion || metadata.capture_session_id != expectedSessionId)
                throw new InvalidDataException("candidate metadata identity mismatch");
            byte[] sequenceBytes = File.ReadAllBytes(sequencePath);
            RecognitionSequence loaded = RecognitionSequenceBinary.Load(sequenceBytes);
            if (loaded.SampleId != expectedSessionId || metadata.sequence_binary_sha256 != Sha256(sequenceBytes) ||
                loaded.LogicalPayloadSha256 != metadata.sequence_normalized_payload_sha256)
                throw new InvalidDataException("candidate sequence round-trip mismatch");
            CandidateChecksumManifest manifest = ParseJson<CandidateChecksumManifest>(
                File.ReadAllBytes(checksumsPath), "candidate checksum manifest");
            if (manifest.schema_version != SchemaVersion || manifest.files == null || manifest.files.Length != 2)
                throw new InvalidDataException("candidate checksum manifest schema mismatch");
            var paths = new HashSet<string>(StringComparer.Ordinal);
            foreach (CandidateChecksumEntry file in manifest.files)
            {
                if (!paths.Add(file.path) || (file.path != SequenceFileName && file.path != MetadataFileName))
                    throw new InvalidDataException("candidate checksum manifest path mismatch");
                byte[] bytes = File.ReadAllBytes(Path.Combine(directory, file.path));
                if (file.size_bytes != bytes.LongLength || file.sha256 != Sha256(bytes))
                    throw new InvalidDataException("candidate checksum mismatch");
            }
            return loaded;
        }

        static void ValidateIndexEntry(CandidateIndexEntry entry)
        {
            if (entry == null || !PersonalLiveRecognitionController.IsSupportedWord(entry.word) ||
                string.IsNullOrWhiteSpace(entry.capture_session_id) ||
                (entry.eligibility != PersonalizationCandidateEligibility.Eligible.ToString() &&
                 entry.eligibility != PersonalizationCandidateEligibility.ReviewOnly.ToString()) ||
                !RecognitionSequence.IsLowerHexSha256(entry.sequence_sha256) ||
                !RecognitionSequence.IsLowerHexSha256(entry.metadata_sha256))
                throw new InvalidDataException("candidate index entry is invalid");
        }

        CandidateIndexDocument NewIndex() => new CandidateIndexDocument
        {
            schema_version = SchemaVersion,
            learner_id = LearnerId,
            entries = new List<CandidateIndexEntry>(),
        };

        static CandidateIndexDocument CloneIndex(CandidateIndexDocument value) => new CandidateIndexDocument
        {
            schema_version = value.schema_version,
            learner_id = value.learner_id,
            entries = value.entries.Select(item => new CandidateIndexEntry
            {
                capture_session_id = item.capture_session_id,
                word = item.word,
                outcome = item.outcome,
                eligibility = item.eligibility,
                relative_directory = item.relative_directory,
                sequence_sha256 = item.sequence_sha256,
                metadata_sha256 = item.metadata_sha256,
            }).ToList(),
        };

        static CandidateCountSnapshot BuildCounts(IEnumerable<CandidateIndexEntry> entries)
        {
            var eligible = new Dictionary<string, int>(StringComparer.Ordinal);
            var review = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in PersonalLiveRecognitionController.SupportedWords) { eligible[word] = 0; review[word] = 0; }
            foreach (CandidateIndexEntry entry in entries)
            {
                if (entry.eligibility == PersonalizationCandidateEligibility.Eligible.ToString()) eligible[entry.word]++;
                else review[entry.word]++;
            }
            return new CandidateCountSnapshot(eligible, review);
        }

        static TrustedCandidateInventory BuildTrustedInventory(
            CandidateCountSnapshot rawCounts,
            IEnumerable<VerifiedCandidatePayload> verified,
            IEnumerable<GuidedEnrollmentSeedManifest> manifests)
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in PersonalLiveRecognitionController.SupportedWords) counts[word] = 0;
            var trusted = new List<TrustedPersonalizationCandidate>();
            var captureIds = new HashSet<string>(StringComparer.Ordinal);
            var sequenceHashes = new HashSet<string>(StringComparer.Ordinal);
            var sourceHashes = new HashSet<string>(StringComparer.Ordinal);
            var logicalHashes = new HashSet<string>(StringComparer.Ordinal);
            int excluded = 0;

            foreach (VerifiedCandidatePayload payload in verified.OrderBy(
                item => item.Entry.capture_session_id, StringComparer.Ordinal))
            {
                CandidateMetadata metadata = payload.Metadata;
                bool strictContract = IsStrictTrustedContract(payload);
                bool unique = strictContract &&
                    !captureIds.Contains(metadata.capture_session_id) &&
                    !sequenceHashes.Contains(metadata.sequence_binary_sha256) &&
                    !sourceHashes.Contains(metadata.sequence_source_payload_sha256) &&
                    !logicalHashes.Contains(metadata.sequence_normalized_payload_sha256);
                if (!unique)
                {
                    excluded++;
                    continue;
                }

                captureIds.Add(metadata.capture_session_id);
                sequenceHashes.Add(metadata.sequence_binary_sha256);
                sourceHashes.Add(metadata.sequence_source_payload_sha256);
                logicalHashes.Add(metadata.sequence_normalized_payload_sha256);

                trusted.Add(new TrustedPersonalizationCandidate(
                    metadata.capture_session_id,
                    metadata.word,
                    payload.Directory,
                    metadata.sequence_binary_sha256,
                    metadata.sequence_source_payload_sha256,
                    metadata.sequence_normalized_payload_sha256,
                    payload.Sequence,
                    TrustedPersonalizationTrustBasis.StrictEligible));
                counts[metadata.word]++;
            }

            var byCapture = verified.ToDictionary(
                payload => payload.Metadata.capture_session_id,
                payload => payload,
                StringComparer.Ordinal);
            foreach (GuidedEnrollmentSeedManifest manifest in (manifests ??
                Enumerable.Empty<GuidedEnrollmentSeedManifest>()).OrderBy(value => value.word, StringComparer.Ordinal))
            {
                int guidedFillers = Math.Max(0, 3 - counts[manifest.word]);
                foreach (GuidedSeedCandidateEntry seed in manifest.selected_candidates
                    .OrderBy(item => item.selection_order).Take(guidedFillers))
                {
                    VerifiedCandidatePayload payload = byCapture[seed.capture_session_id];
                    CandidateMetadata metadata = payload.Metadata;
                    if (!captureIds.Add(metadata.capture_session_id) ||
                        !sequenceHashes.Add(metadata.sequence_binary_sha256) ||
                        !sourceHashes.Add(metadata.sequence_source_payload_sha256) ||
                        !logicalHashes.Add(metadata.sequence_normalized_payload_sha256))
                        throw new InvalidDataException("guided seed identity duplicates another trusted candidate");
                    trusted.Add(new TrustedPersonalizationCandidate(
                        metadata.capture_session_id,
                        metadata.word,
                        payload.Directory,
                        metadata.sequence_binary_sha256,
                        metadata.sequence_source_payload_sha256,
                        metadata.sequence_normalized_payload_sha256,
                        payload.Sequence,
                        TrustedPersonalizationTrustBasis.GuidedLowConfidenceSeed));
                    counts[metadata.word]++;
                    excluded = Math.Max(0, excluded - 1);
                }
            }
            return new TrustedCandidateInventory(rawCounts, counts, trusted, excluded);
        }

        string RelativePath(string fullPath)
        {
            string root = ProfileRoot.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("candidate path escaped learner root");
            return fullPath.Substring(root.Length).Replace(Path.DirectorySeparatorChar, '/');
        }

        string FullRelativePath(string relativePath)
        {
            if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath) ||
                relativePath.Split('/').Any(part => part == ".." || part == "." || part.Length == 0))
                throw new InvalidDataException("unsafe candidate relative path");
            string full = Path.GetFullPath(Path.Combine(ProfileRoot, relativePath.Replace('/', Path.DirectorySeparatorChar)));
            string root = Path.GetFullPath(ProfileRoot).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("candidate relative path escaped learner root");
            return full;
        }

        static string RequireSafeSegment(string value, string field)
        {
            if (string.IsNullOrWhiteSpace(value) || value.Length > 80 ||
                value.Any(c => !(char.IsLetterOrDigit(c) || c == '-' || c == '_')))
                throw new ArgumentException(field + " must contain only letters, digits, '-' or '_'", field);
            return value;
        }

        static double DistanceForWord(PracticeRecognitionAttempt result, string word)
        {
            if (word == "GREETING") return result.GreetingDistance;
            if (word == "LOVE") return result.LoveDistance;
            if (word == "THANK_YOU") return result.ThankYouDistance;
            throw new ArgumentException("unsupported word", nameof(word));
        }

        static bool IsFiniteNonnegative(double value) =>
            !double.IsNaN(value) && !double.IsInfinity(value) && value >= 0.0;

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

        void Inject(CandidateRepositoryFailurePoint point) => failureInjector?.Invoke(point);

        static byte[] JsonBytes(object value) => Utf8NoBom.GetBytes(JsonUtility.ToJson(value, true) + "\n");

        static T ParseJson<T>(byte[] bytes, string label) where T : class
        {
            string text = Utf8NoBom.GetString(bytes);
            if (text.Contains("NaN") || text.Contains("Infinity"))
                throw new InvalidDataException(label + " contains a non-standard number");
            T value = JsonUtility.FromJson<T>(text);
            if (value == null) throw new InvalidDataException(label + " is invalid JSON");
            return value;
        }

        static string Sha256(byte[] bytes)
        {
            using (var sha = SHA256.Create()) return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(bytes));
        }

        static void DeleteFileIfExists(string path) { if (File.Exists(path)) File.Delete(path); }
        static void DeleteDirectoryIfExists(string path) { if (Directory.Exists(path)) Directory.Delete(path, true); }
        static void DeleteDirectoryIfEmpty(string path)
        {
            if (Directory.Exists(path) && !Directory.EnumerateFileSystemEntries(path).Any()) Directory.Delete(path);
        }

        [Serializable]
        sealed class GuidedEnrollmentSeedManifest
        {
            public string schema_version;
            public string policy_version;
            public string learner_id;
            public string word;
            public string trust_basis;
            public string selection_policy;
            public int selection_input_count;
            public string selection_input_sha256;
            public GuidedSeedFrozenInputEntry[] frozen_selection_input;
            public GuidedSeedCandidateEntry[] selected_candidates;
        }

        [Serializable]
        sealed class GuidedSeedFrozenInputEntry
        {
            public string capture_session_id;
            public string candidate_relative_directory;
            public string candidate_metadata_sha256;
            public string sequence_file_sha256;
            public string source_payload_sha256;
            public string logical_payload_sha256;
            public double closed_set_margin;
            public double threshold_margin;

            public static GuidedSeedFrozenInputEntry From(
                VerifiedCandidatePayload payload,
                string relativeDirectory)
            {
                return new GuidedSeedFrozenInputEntry
                {
                    capture_session_id = payload.Metadata.capture_session_id,
                    candidate_relative_directory = relativeDirectory,
                    candidate_metadata_sha256 = payload.Entry.metadata_sha256,
                    sequence_file_sha256 = payload.Metadata.sequence_binary_sha256,
                    source_payload_sha256 = payload.Metadata.sequence_source_payload_sha256,
                    logical_payload_sha256 = payload.Metadata.sequence_normalized_payload_sha256,
                    closed_set_margin = payload.Metadata.closed_set_margin,
                    threshold_margin = payload.Metadata.threshold_margin,
                };
            }

            public bool EqualsIdentity(VerifiedCandidatePayload payload, string relativeDirectory)
            {
                return payload != null && capture_session_id == payload.Metadata.capture_session_id &&
                    candidate_relative_directory == relativeDirectory &&
                    candidate_metadata_sha256 == payload.Entry.metadata_sha256 &&
                    sequence_file_sha256 == payload.Metadata.sequence_binary_sha256 &&
                    source_payload_sha256 == payload.Metadata.sequence_source_payload_sha256 &&
                    logical_payload_sha256 == payload.Metadata.sequence_normalized_payload_sha256;
            }
        }

        [Serializable]
        sealed class GuidedSeedCandidateEntry
        {
            public int selection_order;
            public string capture_session_id;
            public string candidate_relative_directory;
            public string candidate_metadata_sha256;
            public string sequence_file_sha256;
            public string source_payload_sha256;
            public string logical_payload_sha256;
            public double closed_set_margin;
            public double threshold_margin;

            public static GuidedSeedCandidateEntry From(
                VerifiedCandidatePayload payload,
                int order,
                string relativeDirectory)
            {
                return new GuidedSeedCandidateEntry
                {
                    selection_order = order,
                    capture_session_id = payload.Metadata.capture_session_id,
                    candidate_relative_directory = relativeDirectory,
                    candidate_metadata_sha256 = payload.Entry.metadata_sha256,
                    sequence_file_sha256 = payload.Metadata.sequence_binary_sha256,
                    source_payload_sha256 = payload.Metadata.sequence_source_payload_sha256,
                    logical_payload_sha256 = payload.Metadata.sequence_normalized_payload_sha256,
                    closed_set_margin = payload.Metadata.closed_set_margin,
                    threshold_margin = payload.Metadata.threshold_margin,
                };
            }

            public bool EqualsIdentity(GuidedSeedCandidateEntry other)
            {
                return other != null && selection_order == other.selection_order &&
                    capture_session_id == other.capture_session_id &&
                    candidate_relative_directory == other.candidate_relative_directory &&
                    candidate_metadata_sha256 == other.candidate_metadata_sha256 &&
                    sequence_file_sha256 == other.sequence_file_sha256 &&
                    source_payload_sha256 == other.source_payload_sha256 &&
                    logical_payload_sha256 == other.logical_payload_sha256;
            }
        }

        [Serializable]
        sealed class CandidateMetadata
        {
            public string schema_version;
            public string learner_id;
            public string capture_session_id;
            public string word;
            public string expected_word;
            public string predicted_word;
            public string outcome;
            public string route_id;
            public string profile_id;
            public string controller_state;
            public bool strict_accepted;
            public bool strict_expected_class_accept;
            public bool strict_quest_success;
            public int raw_frame_count;
            public int normalized_frame_count;
            public int valid_landmark_count;
            public int interpolated_landmark_count;
            public double expected_word_distance;
            public double best_distance;
            public double approved_threshold;
            public double threshold_margin;
            public double closed_set_margin;
            public string winning_prototype_id;
            public string failure_code;
            public string primary_failure_code;
            public string eligibility;
            public bool eligible_for_profile;
            public string created_utc;
            public string sequence_format;
            public string sequence_binary_sha256;
            public string sequence_source_payload_sha256;
            public string sequence_normalized_payload_sha256;
        }

        [Serializable]
        sealed class CandidateChecksumManifest
        {
            public string schema_version;
            public CandidateChecksumEntry[] files;
        }

        [Serializable]
        sealed class CandidateChecksumEntry
        {
            public string path;
            public long size_bytes;
            public string sha256;
        }

        [Serializable]
        sealed class CandidateIndexDocument
        {
            public string schema_version;
            public string learner_id;
            public List<CandidateIndexEntry> entries;
        }

        [Serializable]
        sealed class CandidateIndexEntry
        {
            public string capture_session_id;
            public string word;
            public string outcome;
            public string eligibility;
            public string relative_directory;
            public string sequence_sha256;
            public string metadata_sha256;
        }

        sealed class VerifiedCandidatePayload
        {
            public CandidateIndexEntry Entry;
            public CandidateMetadata Metadata;
            public string Directory;
            public RecognitionSequence Sequence;
        }
    }

    public static class RecognitionSequenceCandidateBinary
    {
        static readonly byte[] Magic = Encoding.ASCII.GetBytes("WISRCB01");

        public static byte[] Write(RecognitionSequence sequence)
        {
            if (sequence == null) throw new ArgumentNullException(nameof(sequence));
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(Magic);
                writer.Write(RecognitionSequenceBinary.FormatVersion);
                writer.Write(RecognitionSequenceBinary.EndianMarker);
                writer.Write((uint)sequence.LandmarkCount);
                writer.Write((uint)sequence.Length);
                WriteString(writer, sequence.SampleId);
                WriteString(writer, sequence.Word);
                WriteString(writer, sequence.Speed);
                WriteString(writer, sequence.Split);
                WriteString(writer, sequence.SourceNpzSha256);
                WriteString(writer, sequence.LogicalPayloadSha256);

                double[] coordinates = sequence.CopyCoordinates();
                writer.Write((ulong)coordinates.Length);
                foreach (double value in coordinates) writer.Write(BitConverter.DoubleToInt64Bits(value));

                bool[] mask = sequence.CopyValidMask();
                writer.Write((ulong)mask.Length);
                var packed = new byte[(mask.Length + 7) / 8];
                for (int index = 0; index < mask.Length; index++)
                    if (mask[index]) packed[index >> 3] |= (byte)(1 << (index & 7));
                writer.Write((uint)packed.Length);
                writer.Write(packed);
                writer.Flush();
                byte[] payload = stream.ToArray();
                byte[] digest;
                using (var sha = SHA256.Create()) digest = sha.ComputeHash(payload);
                var result = new byte[payload.Length + digest.Length];
                Buffer.BlockCopy(payload, 0, result, 0, payload.Length);
                Buffer.BlockCopy(digest, 0, result, payload.Length, digest.Length);
                return result;
            }
        }

        static void WriteString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((uint)bytes.Length);
            writer.Write(bytes);
        }
    }
}
