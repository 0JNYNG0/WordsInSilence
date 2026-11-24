// F28 - the ONE pinned binding that the full-hand practice route, the full-hand record store
// and the world/quest gate all read, so they can never disagree about which participant,
// which model, which gate configuration and which vocabulary a stored record belongs to.
//
// This file adds NO geometry, NO classifier, NO threshold, NO fitted number and NO model of
// its own. It reads the SAME operator-supplied bundle manifest and gate document the F21
// configuration already reads, and holds the resulting identities. It never scores anything,
// never enrolls anything, never activates a personalized reference and never relaxes a gate.
//
// R1: there is NO cached pinning. A binding is built when an owning component opens (a
// practice bootstrap, an in-world runtime, one gate decision), so a changed gate or model at
// the same path can never be reused from a stale cache. It is not built per capture and not
// per UI frame.
//
// A record collected through this binding is a MODEL/GATE ACCEPTED research candidate for the
// current participant. It is not expert-verified, not a correctness certificate and it is NOT
// loaded back as a personalized reference: the gameplay authority stays the common-only
// "Before" bank and the "After" bank stays comparison-only, exactly as F27 froze them.
using System;
using System.IO;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.FullHandEnrollment
{
    /// <summary>Safe, non-secret refusal codes. No paths, no identities, no capture data.</summary>
    public static class FullHandEnrollmentCodes
    {
        public const string None = "";
        public const string ModeDisabled = "ENROLLMENT_MODE_DISABLED";
        /// <summary>Gameplay is on while the full-hand pipeline itself is not enabled.</summary>
        public const string PipelineNotEnabled = "ENROLLMENT_PIPELINE_NOT_ENABLED";
        public const string ParticipantMissing = "ENROLLMENT_PARTICIPANT_MISSING";
        public const string ParticipantChanged = "ENROLLMENT_PARTICIPANT_CHANGED";
        public const string ConfigurationMissing = "ENROLLMENT_CONFIGURATION_MISSING";
        public const string ModelUnreadable = "ENROLLMENT_MODEL_UNREADABLE";
        public const string GateUnreadable = "ENROLLMENT_GATE_UNREADABLE";
        public const string GateInvalid = "ENROLLMENT_GATE_INVALID";
        public const string VocabularyInvalid = "ENROLLMENT_VOCABULARY_INVALID";
        /// <summary>A model word this build cannot introduce or demonstrate to a learner.</summary>
        public const string VocabularyNotPresentable = "ENROLLMENT_VOCABULARY_NOT_PRESENTABLE";
        public const string RootRejected = "ENROLLMENT_ROOT_REJECTED";
        public const string StoreUnavailable = "ENROLLMENT_STORE_UNAVAILABLE";
        /// <summary>The directory could not be listed completely within the accepted bound.</summary>
        public const string InventoryIncomplete = "ENROLLMENT_INVENTORY_INCOMPLETE";
        public const string OwnerMismatch = "ENROLLMENT_OWNER_MISMATCH";
        public const string ModelMismatch = "ENROLLMENT_MODEL_MISMATCH";
        public const string GateMismatch = "ENROLLMENT_GATE_MISMATCH";
        public const string EvidenceUnusable = "ENROLLMENT_EVIDENCE_UNUSABLE";
        public const string NotAccepted = "ENROLLMENT_NOT_ACCEPTED";
        public const string RecordCorrupt = "ENROLLMENT_RECORD_CORRUPT";
        public const string BudgetExceeded = "ENROLLMENT_BUDGET_EXCEEDED";
        public const string WriteFailed = "ENROLLMENT_WRITE_FAILED";
    }

    /// <summary>
    /// Immutable pinned identities for ONE full-hand enrollment session. Everything here is
    /// copied from material that already existed; nothing is recomputed or invented.
    /// </summary>
    public sealed class FullHandEnrollmentBinding
    {
        /// <summary>
        /// The existing enrollment quorum, reused mechanically and unchanged. It is a RECORD
        /// COUNT, not a quality bar and not a recognition threshold.
        /// </summary>
        public const int RequiredAcceptedRecordings = 3;

        readonly string[] vocabulary;

        public string ParticipantId { get; private set; }
        /// <summary>The loaded bundle's own payload hash. Never a name, never a guess.</summary>
        public string ModelId { get; private set; }
        public string GateConfigId { get; private set; }
        /// <summary>
        /// The gate identity of the AUTHORITY method for this configuration. The production
        /// gate loader composes it as configId + "_" + method, and that is the default here;
        /// an explicitly supplied value is used verbatim so a differently composed gate
        /// identity is pinned rather than guessed.
        /// </summary>
        public string AuthorityGateId { get; private set; }
        public int RequiredRecordings { get { return RequiredAcceptedRecordings; } }
        public int VocabularyCount { get { return vocabulary.Length; } }

        public FullHandEnrollmentBinding(string participantId, string modelId,
                                         string gateConfigId, string[] modelVocabulary,
                                         string authorityGateId = null)
        {
            if (string.IsNullOrWhiteSpace(participantId))
                throw new ArgumentException("participant identity is required", "participantId");
            if (string.IsNullOrWhiteSpace(modelId))
                throw new ArgumentException("model identity is required", "modelId");
            if (string.IsNullOrWhiteSpace(gateConfigId))
                throw new ArgumentException("gate configuration identity is required", "gateConfigId");
            if (modelVocabulary == null || modelVocabulary.Length == 0)
                throw new ArgumentException("a non-empty model vocabulary is required",
                                            "modelVocabulary");
            var copy = (string[])modelVocabulary.Clone();
            for (int index = 0; index < copy.Length; index++)
            {
                if (!IsPlainWord(copy[index]))
                    throw new ArgumentException("a model word is not a plain token",
                                                "modelVocabulary");
                for (int other = 0; other < index; other++)
                    if (string.Equals(copy[other], copy[index], StringComparison.Ordinal))
                        throw new ArgumentException("the model vocabulary repeats a word",
                                                    "modelVocabulary");
            }
            ParticipantId = participantId;
            ModelId = modelId;
            GateConfigId = gateConfigId;
            AuthorityGateId = string.IsNullOrWhiteSpace(authorityGateId)
                ? gateConfigId + "_" + FullHandMethods.Hybrid
                : authorityGateId;
            vocabulary = copy;
        }

        public string WordAt(int index) { return vocabulary[index]; }
        public string[] CopyVocabulary() { return (string[])vocabulary.Clone(); }

        public bool Supports(string word)
        {
            if (string.IsNullOrEmpty(word)) return false;
            for (int index = 0; index < vocabulary.Length; index++)
                if (string.Equals(vocabulary[index], word, StringComparison.Ordinal)) return true;
            return false;
        }

        public bool IsLastWord(string word)
        {
            return vocabulary.Length > 0 &&
                   string.Equals(vocabulary[vocabulary.Length - 1], word, StringComparison.Ordinal);
        }

        /// <summary>One line for a log or a technical panel. No paths and no capture data.</summary>
        public string ProvenanceSummary()
        {
            return "participant=" + Short(ParticipantId) + " model=" + Short(ModelId) +
                   " gate_config=" + Short(GateConfigId) + " words=" + vocabulary.Length +
                   " required=" + RequiredAcceptedRecordings;
        }

        static string Short(string value)
        {
            if (string.IsNullOrEmpty(value)) return "<none>";
            return value.Length <= 12 ? value : value.Substring(0, 12);
        }

        /// <summary>Plain uppercase word tags only, so a word can never shape a file name.</summary>
        public static bool IsPlainWord(string word)
        {
            if (string.IsNullOrEmpty(word) || word.Length > 48) return false;
            for (int index = 0; index < word.Length; index++)
            {
                char c = word[index];
                bool allowed = (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
                if (!allowed) return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Builds the pinned binding for the CURRENTLY ACTIVE demo participant.
    ///
    /// The ROUTE is owned by the gameplay opt-in alone: with WIS_FULL_HAND_GAMEPLAY=1 the
    /// full-hand route is selected, and a missing or invalid pipeline configuration is a
    /// visible REFUSAL of that route - never a silent fall back to the legacy world/quest
    /// code. Building a binding additionally requires the pipeline opt-in and its files.
    /// </summary>
    public static class FullHandEnrollmentBindingFactory
    {
        /// <summary>The route selector: the gameplay opt-in, and nothing else.</summary>
        public static bool IsGameplayRouteRequested()
        {
            return FullHandGameplayConfiguration.IsEnabled();
        }

        /// <summary>True only when the route is requested AND the pipeline is enabled.</summary>
        public static bool IsFullyConfigured()
        {
            return IsGameplayRouteRequested() && FullHandGameCaptureConfiguration.IsEnabled();
        }

        /// <summary>
        /// Reads the operator configuration NOW - no cache - and pins it. Every failure is
        /// explicit and fails closed: this never invents an identity, never invents a
        /// vocabulary and never returns a partially valid binding.
        /// </summary>
        public static bool TryCreateForCurrentParticipant(out FullHandEnrollmentBinding binding,
                                                          out string failureCode,
                                                          out string failureDetail)
        {
            binding = null;
            failureCode = FullHandEnrollmentCodes.None;
            failureDetail = string.Empty;

            if (!IsGameplayRouteRequested())
            {
                failureCode = FullHandEnrollmentCodes.ModeDisabled;
                failureDetail = "the full-hand gameplay route is not enabled";
                return false;
            }
            if (!FullHandGameCaptureConfiguration.IsEnabled())
            {
                failureCode = FullHandEnrollmentCodes.PipelineNotEnabled;
                failureDetail = "the full-hand gameplay route is enabled while the full-hand " +
                                "pipeline is not";
                return false;
            }

            string participantId = FullHandGameCaptureConfiguration.CurrentParticipantIdOrNull();
            if (string.IsNullOrEmpty(participantId))
            {
                failureCode = FullHandEnrollmentCodes.ParticipantMissing;
                failureDetail = "an active participant session is required";
                return false;
            }

            string manifestPath = ReadVariable(FullHandGameCaptureConfiguration.ManifestVariable);
            string gatesPath = ReadVariable(FullHandGameCaptureConfiguration.GatesVariable);
            if (!IsUsableFile(manifestPath) || !IsUsableFile(gatesPath))
            {
                failureCode = FullHandEnrollmentCodes.ConfigurationMissing;
                failureDetail = "the bundle manifest and the gate document must both be " +
                                "fully qualified existing files";
                return false;
            }

            return TryBuild(participantId, manifestPath, gatesPath, out binding,
                            out failureCode, out failureDetail);
        }

        static bool TryBuild(string participantId, string manifestPath, string gatesPath,
                             out FullHandEnrollmentBinding binding, out string failureCode,
                             out string failureDetail)
        {
            binding = null;
            failureCode = FullHandEnrollmentCodes.None;
            failureDetail = string.Empty;

            byte[] gateBytes;
            try
            {
                var info = new FileInfo(gatesPath);
                if (info.Length <= 0 || info.Length > FullHandGateDocument.MaxBytes)
                {
                    failureCode = FullHandEnrollmentCodes.GateUnreadable;
                    failureDetail = "the gate document is empty or larger than the accepted limit";
                    return false;
                }
                gateBytes = File.ReadAllBytes(gatesPath);
            }
            catch (Exception)
            {
                failureCode = FullHandEnrollmentCodes.GateUnreadable;
                failureDetail = "the gate document could not be read";
                return false;
            }

            FullHandRuntimeBundle bundle;
            try { bundle = FullHandRuntimeBundle.Load(manifestPath); }
            catch (Exception) { bundle = null; }
            if (bundle == null)
            {
                failureCode = FullHandEnrollmentCodes.ModelUnreadable;
                failureDetail = "the full-hand bundle could not be loaded";
                return false;
            }

            FullHandGateConfig gates;
            string reviewHash, parseCode, parseDetail;
            if (!FullHandGateDocument.TryParse(gateBytes, bundle.PayloadSha256, out gates,
                                               out reviewHash, out parseCode, out parseDetail))
            {
                failureCode = FullHandEnrollmentCodes.GateInvalid;
                failureDetail = parseDetail ?? parseCode ?? "the gate document was refused";
                return false;
            }

            string[] vocabulary = bundle.CopyVocabulary();
            // an unknown future model word this build cannot introduce or demonstrate must
            // REFUSE the whole route; it may never be silently marked collected
            for (int index = 0; index < vocabulary.Length; index++)
            {
                if (!MoaStoryGuidedPracticeFlow.HasPresentationFor(vocabulary[index]))
                {
                    failureCode = FullHandEnrollmentCodes.VocabularyNotPresentable;
                    failureDetail = "the model vocabulary contains a word this build cannot " +
                                    "introduce or demonstrate to a learner";
                    return false;
                }
            }

            try
            {
                binding = new FullHandEnrollmentBinding(participantId, bundle.PayloadSha256,
                                                        gates.ConfigId, vocabulary);
            }
            catch (Exception exception)
            {
                failureCode = FullHandEnrollmentCodes.VocabularyInvalid;
                failureDetail = exception.GetType().Name;
                return false;
            }
            return true;
        }

        static bool IsUsableFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try
            {
                if (!Path.IsPathFullyQualified(path)) return false;
                foreach (string segment in path.Split('/', '\\'))
                    if (segment == "." || segment == "..") return false;
                return File.Exists(path);
            }
            catch (Exception) { return false; }
        }

        static string ReadVariable(string name)
        {
            try { return Environment.GetEnvironmentVariable(name); }
            catch (Exception) { return null; }
        }
    }

    /// <summary>
    /// The ONE authoritative record-collection view of the current word. The session produces
    /// it from the store; the story flow and the UGUI only render it, so no layer can invent a
    /// count or claim a save that did not happen. The default value is INACTIVE, so every
    /// caller that does not supply one keeps the previous behaviour.
    /// </summary>
    public readonly struct FullHandRecordCollectionView
    {
        public bool Active { get; }
        /// <summary>Records actually stored for the current word.</summary>
        public int Collected { get; }
        public int Required { get; }
        /// <summary>The current word has its full quorum of stored records.</summary>
        public bool WordCollected { get; }
        /// <summary>What the store actually did with the last accepted attempt.</summary>
        public FullHandEnrollmentSaveStatus SaveStatus { get; }
        /// <summary>Store-derived: every word of the round is collected.</summary>
        public bool AllCollected { get; }
        public int CollectedWordCount { get; }
        public int WordCount { get; }
        /// <summary>An unfinished round may be ended honestly, collecting nothing.</summary>
        public bool CanLeaveIncomplete { get; }
        /// <summary>F29: finalized attempts spent on the current word in this session.</summary>
        public int Attempts { get; }
        /// <summary>F29: the attempt budget for one word in one round.</summary>
        public int MaxAttempts { get; }
        /// <summary>F29: the budget is spent and the records are still incomplete.</summary>
        public bool AttemptCapReached { get; }
        /// <summary>
        /// F32: ending the current word early CONTINUES this visit with another pending,
        /// non-deferred word instead of finishing it. The label and the copy must say so.
        /// </summary>
        public bool MovesToNextWord { get; }

        public FullHandRecordCollectionView(bool active, int collected, int required,
                                            bool wordCollected,
                                            FullHandEnrollmentSaveStatus saveStatus,
                                            bool allCollected, int collectedWordCount,
                                            int wordCount, bool canLeaveIncomplete,
                                            int attempts = 0, int maxAttempts = 0,
                                            bool attemptCapReached = false,
                                            bool movesToNextWord = false)
        {
            Active = active;
            Collected = collected < 0 ? 0 : collected;
            Required = required <= 0
                ? FullHandEnrollmentBinding.RequiredAcceptedRecordings : required;
            WordCollected = wordCollected;
            SaveStatus = saveStatus;
            AllCollected = allCollected;
            CollectedWordCount = collectedWordCount < 0 ? 0 : collectedWordCount;
            WordCount = wordCount < 0 ? 0 : wordCount;
            CanLeaveIncomplete = canLeaveIncomplete;
            Attempts = attempts < 0 ? 0 : attempts;
            MaxAttempts = maxAttempts <= 0
                ? FullHandEnrollmentCopy.DefaultMaxAttempts : maxAttempts;
            AttemptCapReached = attemptCapReached;
            MovesToNextWord = movesToNextWord;
        }
    }

    /// <summary>
    /// The learner-facing Korean copy for the record-collection route. It states exactly what
    /// was collected and never claims a completed personalized model, an expert-verified
    /// sample or a probability of correctness. It carries no hash, no identity, no file name
    /// and no enum name.
    /// </summary>
    public static class FullHandEnrollmentCopy
    {
        /// <summary>e.g. "연습 기록 1/3".</summary>
        public static string Progress(int collected, int required)
        {
            int done = collected < 0 ? 0 : collected;
            int total = required <= 0
                ? FullHandEnrollmentBinding.RequiredAcceptedRecordings : required;
            if (done > total) done = total;
            return "연습 기록 " + done + "/" + total;
        }

        public const string Collected = "연습 기록 완료";

        public const string SuccessPrefix =
            "잘했어! 이번 동작을 지금 배우는 표현으로 알아봤어.";

        /// <summary>One accepted recording was collected; the count moved.</summary>
        public static string Saved(int collected, int required)
        {
            return SuccessPrefix + " " + Progress(collected, required);
        }

        /// <summary>The quorum of accepted recordings for this word is now collected.</summary>
        public static string CollectedForWord(int required)
        {
            return SuccessPrefix + " " + Collected + " (" + Progress(required, required) + ")";
        }

        /// <summary>The same recording again: nothing was added and nothing was lost.</summary>
        public static string Duplicate(int collected, int required)
        {
            return "이미 같은 기록이 있어서 새로 저장하지 않았어. " +
                   Progress(collected, required) + " 한 번 더 해볼까?";
        }

        /// <summary>
        /// The save did NOT happen. This must never be shown as progress: the count did not
        /// move and the learner is asked to try again.
        /// </summary>
        public static string NotSaved(int collected, int required)
        {
            return "이번 동작은 알아봤지만 기록으로 저장하지는 못했어. " +
                   Progress(collected, required) + " 한 번 더 해보자.";
        }

        /// <summary>Shown wherever the older route promised personalization.</summary>
        public const string CollectionNote =
            "모아둔 기록은 네 기기에만 저장돼. 아직 개인 맞춤 인식에 쓰이지는 않아.";

        /// <summary>Every word of this round has its records. A count, not a model.</summary>
        public const string RoundCollected =
            "이번에 배운 표현들의 연습 기록을 모두 모았어. 이제 밖에서 직접 써볼 준비가 됐어.";

        /// <summary>The round ended without collecting every record. Never a false completion.</summary>
        public static string RoundIncomplete(int collectedWords, int totalWords)
        {
            return "이번 연습은 여기까지 할게. 아직 연습 기록을 다 모으지 못한 표현이 있어 (" +
                   collectedWords + "/" + totalWords + "). 나중에 이어서 연습할 수 있어.";
        }

        /// <summary>F29: the attempt budget for one word in one practice round.</summary>
        public const int DefaultMaxAttempts = 10;

        /// <summary>e.g. "연습 기록 1/3 · 시도 4/10". A small counter, not a diagnostic.</summary>
        public static string AttemptProgress(int collected, int required, int attempts,
                                             int maxAttempts)
        {
            int limit = maxAttempts <= 0 ? DefaultMaxAttempts : maxAttempts;
            int used = attempts < 0 ? 0 : attempts > limit ? limit : attempts;
            return Progress(collected, required) + " · 시도 " + used + "/" + limit;
        }

        /// <summary>
        /// F29. The attempt budget for this word is spent and the records are NOT complete.
        /// This is not a camera fault and not a claim that the learner failed: it states the
        /// real collected count and offers to continue later.
        /// </summary>
        public static string AttemptCapReached(int collected, int required, int attempts,
                                               int maxAttempts)
        {
            int limit = maxAttempts <= 0 ? DefaultMaxAttempts : maxAttempts;
            return "이번 표현은 " + limit + "번까지 해봤어. 지금까지 모은 건 " +
                   Progress(collected, required) + "야. 여기서 연습을 마치고 " +
                   "나중에 이어서 남은 기록을 모을 수 있어.";
        }

        /// <summary>
        /// F32. The budget for THIS word is spent while another word of the visit is still
        /// waiting. It states the real collected count, claims no completion and says exactly
        /// what the button does: move on to the next expression.
        /// </summary>
        public static string AttemptCapReachedMovingOn(int collected, int required,
                                                       int attempts, int maxAttempts)
        {
            int limit = maxAttempts <= 0 ? DefaultMaxAttempts : maxAttempts;
            return "이번 표현은 " + limit + "번까지 해봤어. 지금까지 모은 건 " +
                   Progress(collected, required) + "야. 이 표현은 나중에 이어서 하고, " +
                   "지금은 다음 표현을 연습해보자.";
        }

        /// <summary>A technical stop: not learner performance, and nothing was recorded.</summary>
        public const string TechnicalStop =
            "기술 문제가 계속돼서 이번 표현은 여기서 멈출게. 네 동작이 틀린 게 아니야. " +
            "카메라와 자세를 확인한 뒤 나중에 이어서 연습하자.";

        /// <summary>
        /// F32. A technical stop on THIS word while another word is still waiting. It claims
        /// no completion and does not pretend the practice is finished.
        /// </summary>
        public const string TechnicalStopMovingOn =
            "기술 문제가 계속돼서 이번 표현은 여기까지만 할게. 네 동작이 틀린 게 아니야. " +
            "이 표현은 나중에 이어서 하고, 지금은 다음 표현을 연습해보자.";

        /// <summary>An attempt that did not arrive as a full-hand result at all.</summary>
        public const string UnexpectedResult =
            "이번 결과를 확인하지 못했어. 기록으로 저장하지 않았어. 다시 시도해 줄래?";
    }
}
