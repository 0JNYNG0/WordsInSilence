using System;
using System.Collections.Generic;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>Deterministic, manual-start/manual-finalize learning progression.</summary>
    public sealed class PetRobotSignPracticeSession : IDisposable
    {
        static readonly string[] legacyWords = { "GREETING", "LOVE", "THANK_YOU" };

        /// <summary>
        /// F28. The words THIS session practises. Off-mode it is exactly the legacy three; in
        /// full-hand record-collection mode it is the CURRENT model vocabulary.
        /// </summary>
        readonly string[] words;
        readonly FullHandEnrollmentBinding fullHandBinding;
        readonly FullHandParticipantCandidateStore fullHandStore;
        readonly IPersonalLivePracticeRecognitionAdapter recognition;
        readonly IPersonalizationCandidateRepository repository;
        readonly ITrustedPersonalizationCandidateRepository trustedRepository;
        readonly IGuidedEnrollmentSeedPromotionRepository guidedSeedRepository;
        readonly IPetRobotCandidateCameraReadiness cameraReadiness;
        readonly Func<DateTime> utcNow;
        readonly HashSet<string> processedSessions = new HashSet<string>(StringComparer.Ordinal);
        PetRobotWordLearningStatus[] wordStatuses;
        PracticeRecognitionAttempt[] finalAttempts;
        int[] wrongWordAttemptCounts;
        int[] scoreableAttemptCounts;
        int[] consecutiveTechnicalFailureCounts;
        PetRobotPersonalizationEnrollmentState[] enrollmentStates;
        /// <summary>F28: the stored record count per word, read back from the store.</summary>
        int[] fullHandRecordCounts;
        /// <summary>
        /// F29: the per-word ATTEMPT BUDGET of THIS session. It counts finalized full-hand
        /// practice attempts for the current word whatever their outcome, and it is never a
        /// stored-record count. It is not persisted: a later visit starts a fresh budget.
        /// </summary>
        int[] fullHandAttemptCounts;
        /// <summary>
        /// F32: the words DEFERRED for THIS visit. A word is deferred when the learner ends
        /// it early at the attempt cap or at the technical stop. Its stored records are kept
        /// untouched, it is never marked mastered, it is not practised again during this
        /// visit, and the flag is not persisted: a fresh session clears it.
        /// </summary>
        bool[] fullHandDeferredWords;
        int currentIndex;

        readonly bool questMode;
        readonly IQuestHybridRecognitionSession frozenHybridSession;
        readonly PetRobotGreetingQuestToken questToken;

        public event Action Changed;
        public event Action ExitRequested;

        public IReadOnlyList<string> Words => Array.AsReadOnly((string[])words.Clone());
        public PetRobotPracticeState State { get; private set; } = PetRobotPracticeState.Ready;
        public string CurrentWord => words[currentIndex];
        public int CurrentStep => currentIndex + 1;
        public bool IsCurrentWordComplete => wordStatuses[currentIndex] != PetRobotWordLearningStatus.Pending;
        public bool AllWordsComplete
        {
            get
            {
                for (int index = 0; index < wordStatuses.Length; index++)
                    if (wordStatuses[index] == PetRobotWordLearningStatus.Pending) return false;
                return wordStatuses.Length > 0;
            }
        }

        /// <summary>
        /// F28. True while this session collects full-hand practice RECORDS for the current
        /// model vocabulary. Collecting records is not personalization: nothing stored here is
        /// loaded back as a reference and no personalized model is produced or claimed.
        /// </summary>
        public bool FullHandRecordCollectionActive => fullHandStore != null && !questMode;
        public FullHandEnrollmentBinding FullHandBinding => fullHandBinding;
        public int FullHandRequiredRecordCount =>
            fullHandBinding == null ? 0 : fullHandBinding.RequiredRecordings;
        public int CurrentFullHandRecordCount =>
            fullHandStore == null ? 0 : fullHandRecordCounts[currentIndex];
        public bool IsCurrentFullHandWordCollected =>
            fullHandStore != null && FullHandRequiredRecordCount > 0 &&
            CurrentFullHandRecordCount >= FullHandRequiredRecordCount;
        /// <summary>F29: the maximum finalized attempts per word in ONE practice round.</summary>
        public const int FullHandMaxAttemptsPerWord = 10;
        /// <summary>F29: attempts already spent on the current word in THIS session.</summary>
        public int CurrentFullHandAttemptCount =>
            fullHandStore == null ? 0 : fullHandAttemptCounts[currentIndex];
        /// <summary>F29: the spent budget of any word of this round.</summary>
        public int GetFullHandAttemptCount(string word)
        {
            int index = IndexOfWordOrMinusOne(word);
            return index < 0 || fullHandStore == null ? 0 : fullHandAttemptCounts[index];
        }
        /// <summary>
        /// F29: the budget is spent and the records for this word are still incomplete. No new
        /// capture and no retry is accepted; the round may be ended honestly and resumed later
        /// with a fresh budget. Nothing is marked mastered and no record is invented.
        /// </summary>
        public bool IsCurrentFullHandAttemptCapReached =>
            fullHandStore != null && !questMode && !IsCurrentFullHandWordCollected &&
            fullHandAttemptCounts[currentIndex] >= FullHandMaxAttemptsPerWord;
        /// <summary>The real total of collected recordings for this round.</summary>
        public int FullHandRecordTotal
        {
            get
            {
                if (fullHandStore == null) return 0;
                int total = 0;
                for (int index = 0; index < fullHandRecordCounts.Length; index++)
                    total += fullHandRecordCounts[index];
                return total;
            }
        }
        /// <summary>How many words of this round have their full quorum of recordings.</summary>
        public int CollectedFullHandWordCount
        {
            get
            {
                if (fullHandStore == null) return 0;
                int collected = 0;
                for (int index = 0; index < fullHandRecordCounts.Length; index++)
                    if (fullHandRecordCounts[index] >= FullHandRequiredRecordCount) collected++;
                return collected;
            }
        }
        /// <summary>Store-derived. Never inferred from a word status or from a UI state.</summary>
        public bool AllFullHandRecordsCollected =>
            fullHandStore != null && fullHandStore.AllWordsCollected;
        /// <summary>The last store verdict. Diagnostics; it carries no learner copy.</summary>
        public FullHandEnrollmentSaveResult LastFullHandSaveResult { get; private set; } =
            FullHandEnrollmentSaveResult.NotRequested;
        /// <summary>Learner-safe collection text for the last attempt, or empty.</summary>
        public string LastFullHandCollectionMessage { get; private set; } = string.Empty;
        /// <summary>
        /// A record round that hit the technical stop may be ended honestly. It collects
        /// nothing, marks nothing mastered and leaves the world/NPC gate locked.
        /// </summary>
        public bool CanLeaveRoundIncomplete =>
            fullHandStore != null && !questMode &&
            State == PetRobotPracticeState.Feedback &&
            (enrollmentStates[currentIndex] == PetRobotPersonalizationEnrollmentState.TechnicalStop ||
             IsCurrentFullHandAttemptCapReached);

        /// <summary>The one authoritative view the story flow and the UGUI render.</summary>
        public FullHandRecordCollectionView FullHandRecordView =>
            new FullHandRecordCollectionView(
                FullHandRecordCollectionActive,
                CurrentFullHandRecordCount,
                FullHandRequiredRecordCount,
                IsCurrentFullHandWordCollected,
                LastFullHandSaveResult.Status,
                AllFullHandRecordsCollected,
                CollectedFullHandWordCount,
                words.Length,
                CanLeaveRoundIncomplete,
                CurrentFullHandAttemptCount,
                FullHandMaxAttemptsPerWord,
                IsCurrentFullHandAttemptCapReached,
                CanLeaveRoundIncomplete && HasNextEligibleFullHandWord);

        /// <summary>
        /// F32: another word of this round is still pending and not deferred, so ending the
        /// current word early continues the visit instead of finishing it.
        /// </summary>
        public bool HasNextEligibleFullHandWord =>
            fullHandStore != null && !questMode && NextEligibleFullHandWord() >= 0;

        /// <summary>F32: this word was ended early during THIS visit.</summary>
        public bool IsFullHandWordDeferred(string word)
        {
            int index = IndexOfWordOrMinusOne(word);
            return index >= 0 && fullHandStore != null && fullHandDeferredWords[index];
        }

        public bool SupportsWord(string word) => IndexOfWordOrMinusOne(word) >= 0;

        public int GetFullHandRecordCount(string word)
        {
            int index = IndexOfWordOrMinusOne(word);
            return index < 0 || fullHandStore == null ? 0 : fullHandRecordCounts[index];
        }
        public int MasteredCount => CountStatus(PetRobotWordLearningStatus.Mastered);
        public int NeedsPracticeCount => CountStatus(PetRobotWordLearningStatus.NeedsPractice);
        public int CurrentWrongWordAttemptCount => wrongWordAttemptCounts[currentIndex];
        public bool TrustedEnrollmentEnabled => trustedRepository != null && fullHandStore == null;
        public int CurrentTrustedSampleCount => GetTrustedSampleCount(CurrentWord);
        public int CurrentScoreableAttemptCount => scoreableAttemptCounts[currentIndex];
        public int CurrentConsecutiveTechnicalFailureCount => consecutiveTechnicalFailureCounts[currentIndex];
        public PetRobotPersonalizationEnrollmentState CurrentEnrollmentState => enrollmentStates[currentIndex];
        public bool OverallEnrollmentGateReady => fullHandStore != null
            ? fullHandStore.AllWordsCollected
            : trustedRepository != null &&
              trustedRepository.TrustedInventory.OverallEnrollmentGateReady;
        public bool CanRetry => questMode
            ? State == PetRobotPracticeState.Feedback && !QuestCompleted
            : fullHandStore != null
            ? State == PetRobotPracticeState.Feedback && !IsCurrentWordComplete &&
              !IsCurrentFullHandAttemptCapReached &&
              enrollmentStates[currentIndex] != PetRobotPersonalizationEnrollmentState.TechnicalStop
            : trustedRepository != null
            ? State == PetRobotPracticeState.Feedback &&
              enrollmentStates[currentIndex] == PetRobotPersonalizationEnrollmentState.Pending &&
              scoreableAttemptCounts[currentIndex] < 10 &&
              consecutiveTechnicalFailureCounts[currentIndex] < 3
            : State == PetRobotPracticeState.Feedback && !IsCurrentWordComplete &&
              (LastStorageResult.Status == CandidateStorageStatus.Duplicate ||
              LastAttempt == null || LastAttempt.Outcome == null ||
              LastAttempt.Outcome.Kind == GuidedQuestOutcomeKind.TechnicalFailure ||
              LastAttempt.Outcome.Kind == GuidedQuestOutcomeKind.NotDistinguishedRetry ||
              (LastAttempt.Outcome.Kind == GuidedQuestOutcomeKind.WrongWord &&
              wrongWordAttemptCounts[currentIndex] == 1));
        public bool CanContinueWithNeedsPractice => questMode ? false :
            fullHandStore != null ? false :
            trustedRepository != null
                ? State == PetRobotPracticeState.Feedback &&
                  (enrollmentStates[currentIndex] == PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete ||
                   enrollmentStates[currentIndex] == PetRobotPersonalizationEnrollmentState.TechnicalStop)
                : State == PetRobotPracticeState.Feedback && !IsCurrentWordComplete &&
            LastAttempt != null && LastAttempt.Outcome != null &&
            ((LastAttempt.Outcome.Kind == GuidedQuestOutcomeKind.WrongWord &&
              wrongWordAttemptCounts[currentIndex] == 2) ||
             LastAttempt.Outcome.Kind == GuidedQuestOutcomeKind.UnsupportedWord);
        public PracticeRecognitionAttempt LastAttempt { get; private set; }
        public GuidedQuestOutcome LastLearningOutcome => LastAttempt == null ? null : LastAttempt.Outcome;

        /// <summary>
        /// F27. Non-null only when the 46-point full-hand decision was the authority for the
        /// last attempt. It carries method/bank/model/gate/source provenance and never
        /// changes any value above.
        /// </summary>
        public FullHandGameplayDecision LastFullHandGameplayDecision =>
            LastAttempt == null ? null : LastAttempt.FullHandGameplay;
        public CandidateSaveResult LastStorageResult { get; private set; } =
            new CandidateSaveResult(CandidateStorageStatus.NotRequested, string.Empty, "not requested", CandidateCountSnapshot.Empty);
        public GuidedEnrollmentSeedCommitResult LastGuidedSeedCommitResult { get; private set; } =
            GuidedEnrollmentSeedCommitResult.NotRequested;
        public CandidateCountSnapshot CandidateCounts => repository.Counts;
        public PersonalLiveCaptureState CaptureState => recognition.CaptureState;
        public bool CandidateRouteActive => recognition.CandidateRouteActive;
        public CandidateCameraStartupState CameraState => cameraReadiness.CameraState;
        public bool CameraReady => cameraReadiness.IsReady;
        public bool CameraStarted => cameraReadiness.CameraStarted;
        public bool CameraFrameReady => cameraReadiness.HasReceivedFrame;
        public bool ProviderObservationReady => cameraReadiness.ProviderObservationReady;
        public int CameraFrameCount => cameraReadiness.CameraFrameCount;
        public int ProviderObservationCount => cameraReadiness.ProviderObservationCount;
        public string CameraFailureCode => cameraReadiness.LastErrorCode;
        public string CameraFailureMessage => cameraReadiness.LastErrorMessage;

        public bool QuestMode => questMode;
        public bool QuestCompleted { get; private set; }
        public TrustedHybridOutcomeKind? QuestAuthoritativeOutcome { get; private set; }
        public TrustedHybridRecognitionResult QuestHybridResult { get; private set; }

        /// <summary>
        /// Step 10W v1.0. The legacy frozen-hybrid reading of the same sequence, kept for
        /// comparison and for the diagnostic record only. It has no authority over
        /// <see cref="QuestAuthoritativeOutcome"/> and no consumer may branch on it.
        /// Null means the legacy pass was not run, returned nothing, or threw.
        /// </summary>
        public TrustedHybridOutcomeKind? LegacyHybridDiagnosticOutcome => legacyHybridOutcome;

        /// <summary>
        /// Fixed literal naming the policy that decides the World branch, so a reader of
        /// one record never has to infer which layer won.
        /// </summary>
        public const string WorldOutcomeAuthority =
            "canonical_guided_outcome_mapped_v1_0w";

        TrustedHybridOutcomeKind? legacyHybridOutcome;
        public PetRobotGreetingQuestToken QuestToken => questToken;
        public long QuestSessionGeneration => frozenHybridSession != null ? frozenHybridSession.Generation : 0;
        public string QuestInitializationFailure { get; internal set; }

        public PetRobotSignPracticeSession(
            IPersonalLivePracticeRecognitionAdapter recognition,
            IPersonalizationCandidateRepository repository,
            Func<DateTime> utcNow = null,
            IPetRobotCandidateCameraReadiness cameraReadiness = null)
        {
            words = legacyWords;
            AllocateWordState();
            this.recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
            this.repository = repository ?? throw new ArgumentNullException(nameof(repository));
            trustedRepository = repository as ITrustedPersonalizationCandidateRepository;
            guidedSeedRepository = repository as IGuidedEnrollmentSeedPromotionRepository;
            this.utcNow = utcNow ?? (() => DateTime.UtcNow);
            this.cameraReadiness = cameraReadiness ?? AlwaysReadyCameraReadiness.Instance;
            if (!recognition.CandidateRouteActive)
                throw new InvalidOperationException("candidate route must be explicitly active for the practice session");
            if (trustedRepository != null)
                RestorePersistedEnrollment();
            if (State != PetRobotPracticeState.Completed && !recognition.SetExpectedWord(CurrentWord))
                throw new InvalidOperationException("failed to select the initial practice prompt");
            this.cameraReadiness.Changed += OnCameraReadinessChanged;
            OnCameraReadinessChanged();
        }

        public PetRobotSignPracticeSession(
            IPersonalLivePracticeRecognitionAdapter recognition,
            IPersonalizationCandidateRepository repository,
            IQuestHybridRecognitionSession frozenHybridSession,
            PetRobotGreetingQuestToken questToken,
            IPetRobotCandidateCameraReadiness cameraReadiness = null)
        {
            this.recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
            this.repository = repository ?? throw new ArgumentNullException(nameof(repository));
            this.frozenHybridSession = frozenHybridSession ?? throw new ArgumentNullException(nameof(frozenHybridSession));
            this.questToken = questToken ?? throw new ArgumentNullException(nameof(questToken));
            words = legacyWords;
            AllocateWordState();
            trustedRepository = repository as ITrustedPersonalizationCandidateRepository;
            guidedSeedRepository = null;
            this.utcNow = () => DateTime.UtcNow;
            this.cameraReadiness = cameraReadiness ?? AlwaysReadyCameraReadiness.Instance;
            questMode = true;
            if (!recognition.CandidateRouteActive)
                throw new InvalidOperationException("candidate route must be explicitly active for the practice session");
            if (!recognition.SetExpectedWord("GREETING"))
                throw new InvalidOperationException("failed to select the GREETING quest prompt");
            this.cameraReadiness.Changed += OnCameraReadinessChanged;
            OnCameraReadinessChanged();
        }

        /// <summary>
        /// F28 practice route. The prompts are the CURRENT model vocabulary and every accepted
        /// capture is offered to the local full-hand record store BEFORE any progress moves.
        /// No legacy 16-point candidate is written and no personalized model is produced.
        /// </summary>
        public PetRobotSignPracticeSession(
            IPersonalLivePracticeRecognitionAdapter recognition,
            IPersonalizationCandidateRepository repository,
            FullHandEnrollmentBinding fullHandBinding,
            FullHandParticipantCandidateStore fullHandStore,
            Func<DateTime> utcNow = null,
            IPetRobotCandidateCameraReadiness cameraReadiness = null)
        {
            this.recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
            this.repository = repository ?? throw new ArgumentNullException(nameof(repository));
            this.fullHandBinding = fullHandBinding ??
                throw new ArgumentNullException(nameof(fullHandBinding));
            this.fullHandStore = fullHandStore ??
                throw new ArgumentNullException(nameof(fullHandStore));
            if (!ReferenceEquals(fullHandStore.Binding, fullHandBinding))
                throw new InvalidOperationException(
                    "the record store is bound to a different model/gate/participant pinning");
            words = fullHandBinding.CopyVocabulary();
            AllocateWordState();
            trustedRepository = repository as ITrustedPersonalizationCandidateRepository;
            guidedSeedRepository = null;
            this.utcNow = utcNow ?? (() => DateTime.UtcNow);
            this.cameraReadiness = cameraReadiness ?? AlwaysReadyCameraReadiness.Instance;
            if (!recognition.CandidateRouteActive)
                throw new InvalidOperationException("candidate route must be explicitly active for the practice session");
            RestoreFullHandCollection();
            if (State != PetRobotPracticeState.Completed && !recognition.SetExpectedWord(CurrentWord))
                throw new InvalidOperationException("failed to select the initial practice prompt");
            this.cameraReadiness.Changed += OnCameraReadinessChanged;
            OnCameraReadinessChanged();
        }

        /// <summary>
        /// F28 in-world quest route, explicit full-hand mode. There is no frozen 16-point
        /// hybrid session and no legacy profile: the quest consumes exactly the same accepted
        /// full-hand decision the practice route consumes, verifies it against this binding,
        /// and stores nothing.
        /// </summary>
        public PetRobotSignPracticeSession(
            IPersonalLivePracticeRecognitionAdapter recognition,
            IPersonalizationCandidateRepository repository,
            PetRobotGreetingQuestToken questToken,
            FullHandEnrollmentBinding fullHandBinding,
            FullHandParticipantCandidateStore fullHandStore,
            IPetRobotCandidateCameraReadiness cameraReadiness = null,
            string questWord = "GREETING")
        {
            this.recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
            this.repository = repository ?? throw new ArgumentNullException(nameof(repository));
            this.questToken = questToken ?? throw new ArgumentNullException(nameof(questToken));
            this.fullHandBinding = fullHandBinding ??
                throw new ArgumentNullException(nameof(fullHandBinding));
            this.fullHandStore = fullHandStore ??
                throw new ArgumentNullException(nameof(fullHandStore));
            if (!ReferenceEquals(fullHandStore.Binding, fullHandBinding))
                throw new InvalidOperationException(
                    "the record store is bound to a different model/gate/participant pinning");
            if (!fullHandBinding.Supports(questWord))
                throw new InvalidOperationException(
                    "the quest word is outside the current model vocabulary");
            words = new[] { questWord };
            AllocateWordState();
            trustedRepository = repository as ITrustedPersonalizationCandidateRepository;
            guidedSeedRepository = null;
            this.utcNow = () => DateTime.UtcNow;
            this.cameraReadiness = cameraReadiness ?? AlwaysReadyCameraReadiness.Instance;
            questMode = true;
            if (!recognition.CandidateRouteActive)
                throw new InvalidOperationException("candidate route must be explicitly active for the practice session");
            if (!recognition.SetExpectedWord(questWord))
                throw new InvalidOperationException("failed to select the quest prompt");
            this.cameraReadiness.Changed += OnCameraReadinessChanged;
            OnCameraReadinessChanged();
        }

        void AllocateWordState()
        {
            wordStatuses = new PetRobotWordLearningStatus[words.Length];
            finalAttempts = new PracticeRecognitionAttempt[words.Length];
            wrongWordAttemptCounts = new int[words.Length];
            scoreableAttemptCounts = new int[words.Length];
            consecutiveTechnicalFailureCounts = new int[words.Length];
            enrollmentStates = new PetRobotPersonalizationEnrollmentState[words.Length];
            fullHandRecordCounts = new int[words.Length];
            fullHandAttemptCounts = new int[words.Length];
            fullHandDeferredWords = new bool[words.Length];
        }

        /// <summary>
        /// F28/F32. Ends the CURRENT word early after its attempt cap or its technical stop.
        ///
        /// F32: the word is DEFERRED for this visit instead of ending the whole visit. Its
        /// records are preserved exactly as they are, it is never marked mastered, and it is
        /// not offered again during this visit even after another word succeeds. If another
        /// pending, non-deferred word is left, the visit continues there with its own fresh
        /// attempt budget; if none is left, the visit ends with the honest incomplete summary
        /// and the NPC gate stays locked. Leaving the whole session remains available
        /// elsewhere (the completed-state exit and the world route are unchanged).
        /// </summary>
        public bool LeaveRoundIncomplete()
        {
            if (!CanLeaveRoundIncomplete) return false;
            fullHandDeferredWords[currentIndex] = true;
            int next = NextEligibleFullHandWord();
            if (next >= 0)
            {
                if (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[next]))
                {
                    fullHandDeferredWords[currentIndex] = false;
                    return false;
                }
                currentIndex = next;
                LastAttempt = null;
                LastFullHandSaveResult = FullHandEnrollmentSaveResult.NotRequested;
                LastFullHandCollectionMessage = string.Empty;
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "the previous word was left for later in this visit; nothing was collected " +
                    "for it and nothing was marked complete; ready for the next word",
                    repository.Counts);
                State = PetRobotPracticeState.Ready;
                Notify();
                return true;
            }

            LastAttempt = null;
            LastFullHandCollectionMessage = string.Empty;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "the record round ended early; no word was marked collected", repository.Counts);
            State = PetRobotPracticeState.Completed;
            Notify();
            return true;
        }

        /// <summary>
        /// F32: the next OTHER word of this round whose records are still incomplete AND
        /// which was not deferred during this visit.
        ///
        /// The current word is deliberately excluded (the loop stops one short of a full
        /// turn). <see cref="HasNextEligibleFullHandWord"/> asks this BEFORE the current word
        /// is deferred, so including it would report a next word that does not exist and the
        /// view would promise "the next expression" on the last exhausted word. Advancing a
        /// collected word is unaffected: that word is Mastered, not Pending.
        /// </summary>
        int NextEligibleFullHandWord()
        {
            if (fullHandStore == null) return -1;
            for (int offset = 1; offset < words.Length; offset++)
            {
                int index = (currentIndex + offset) % words.Length;
                if (wordStatuses[index] == PetRobotWordLearningStatus.Pending &&
                    !fullHandDeferredWords[index]) return index;
            }
            return -1;
        }

        /// <summary>
        /// F29. Spends ONE unit of the current word's attempt budget. It is called only from
        /// the full-hand practice finalization paths, i.e. AFTER the blank-or-duplicate
        /// capture-session gate has accepted the attempt, so a duplicate callback and a
        /// rejected BeginCapture spend nothing. Every finalized outcome spends exactly one:
        /// abstention, wrong word, technical failure and an accepted capture that the store
        /// duplicated, refused or failed to write all count the same.
        /// </summary>
        void CountFullHandAttempt()
        {
            if (fullHandStore == null || questMode) return;
            if (fullHandAttemptCounts[currentIndex] < FullHandMaxAttemptsPerWord)
                fullHandAttemptCounts[currentIndex]++;
        }

        /// <summary>
        /// F29. Replaces the outcome copy with the cap copy when the budget is spent and the
        /// records are still incomplete. It states the real collected count and never claims
        /// a completed word.
        /// </summary>
        bool ApplyFullHandAttemptCapMessage()
        {
            if (!IsCurrentFullHandAttemptCapReached) return false;
            LastFullHandCollectionMessage = FullHandEnrollmentCopy.AttemptCapReached(
                CurrentFullHandRecordCount, FullHandRequiredRecordCount,
                CurrentFullHandAttemptCount, FullHandMaxAttemptsPerWord);
            return true;
        }

        public bool StartExpression()
        {
            return StartExpressionCore(() => recognition.BeginCapture());
        }

        public bool StartAutoExpression(IReadOnlyList<RawMediaPipeObservation> preRoll)
        {
            var segmented = recognition as ISegmentedPersonalLivePracticeRecognitionAdapter;
            return segmented != null && preRoll != null &&
                StartExpressionCore(() => segmented.BeginCaptureWithPreRoll(preRoll));
        }

        public bool PrepareAutoAttempt()
        {
            if (State != PetRobotPracticeState.Ready || IsCurrentWordComplete ||
                IsCurrentFullHandAttemptCapReached) return false;
            if (recognition.CaptureState == PersonalLiveCaptureState.Failed &&
                !recognition.ResetCapture()) return false;
            if (recognition.CaptureState != PersonalLiveCaptureState.Ready) return false;
            return recognition.SetExpectedWord(CurrentWord);
        }

        bool StartExpressionCore(Func<bool> beginCapture)
        {
            if (State != PetRobotPracticeState.Ready || IsCurrentWordComplete ||
                IsCurrentFullHandAttemptCapReached) return false;
            if (!cameraReadiness.IsReady)
            {
                if (cameraReadiness.CameraState == CandidateCameraStartupState.Failed)
                    RecordCameraStartupFailure();
                return false;
            }
            if (!recognition.SetExpectedWord(CurrentWord) || !beginCapture()) return false;
            LastAttempt = null;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty, "capture in progress", repository.Counts);
            State = PetRobotPracticeState.Capturing;
            Notify();
            return true;
        }

        public bool FinishExpression()
        {
            return FinishExpressionCore(() => recognition.FinalizeCapture());
        }

        public bool FinishAutoExpression(double trailingTrimSeconds)
        {
            var segmented = recognition as ISegmentedPersonalLivePracticeRecognitionAdapter;
            return segmented != null &&
                FinishExpressionCore(() =>
                    segmented.FinalizeCaptureWithTrailingTrim(trailingTrimSeconds));
        }

        public bool FinishAutoTechnicalFailure(Exception exception)
        {
            if (State != PetRobotPracticeState.Capturing) return false;
            var segmented = recognition as ISegmentedPersonalLivePracticeRecognitionAdapter;
            if (segmented == null) return false;
            PracticeRecognitionAttempt attempt = segmented.FailActiveCaptureFromSource(exception);
            return attempt != null && FinishExpressionCore(() => attempt);
        }

        bool FinishExpressionCore(Func<PracticeRecognitionAttempt> finalizeCapture)
        {
            if (State != PetRobotPracticeState.Capturing) return false;
            PracticeRecognitionAttempt attempt = finalizeCapture();
            if (attempt == null || attempt.Outcome == null)
            {
                // F29: a finalized attempt with no usable result still spends the budget.
                CountFullHandAttempt();
                ApplyFullHandAttemptCapMessage();
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty, "technical result unavailable; no candidate stored", repository.Counts);
                if (questMode)
                {
                    QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                    QuestHybridResult = null;
                    legacyHybridOutcome = null;
                }
                State = PetRobotPracticeState.Feedback;
                Notify();
                return true;
            }

            LastAttempt = attempt;
            // F29. Malformed-identity full-hand practice result: NOT a duplicate callback.
            if (fullHandStore != null && !questMode &&
                string.IsNullOrWhiteSpace(attempt.CaptureSessionId))
            {
                HandleUnexpectedLegacyAttempt();
                State = PetRobotPracticeState.Feedback;
                Notify();
                return true;
            }
            if (string.IsNullOrWhiteSpace(attempt.CaptureSessionId) || !processedSessions.Add(attempt.CaptureSessionId))
            {
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.Duplicate, string.Empty, "duplicate callback ignored", repository.Counts);
                if (questMode)
                {
                    QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                    QuestHybridResult = null;
                    legacyHybridOutcome = null;
                }
                State = PetRobotPracticeState.Feedback;
                Notify();
                return true;
            }

            // Step 10P v1.0c1 observability: emitted only AFTER the blank-or-duplicate
            // capture-session gate above has accepted this attempt, so a repeated
            // callback adds no second record. The diagnostic reads values the approved
            // seams already produced and changes no state, so removing this line
            // restores the previous behaviour exactly.
            finalAttempts[currentIndex] = attempt;

            if (questMode)
            {
                HandleQuestOutcome(attempt);
            }
            else if (attempt.FullHandGameplay != null)
            {
                HandleFullHandPracticeOutcome(attempt);
            }
            else if (fullHandStore != null)
            {
                HandleUnexpectedLegacyAttempt();
            }
            else if (trustedRepository != null)
            {
                HandleTrustedEnrollmentOutcome(attempt);
            }
            else switch (attempt.Outcome.Kind)
            {
                case GuidedQuestOutcomeKind.StrictSuccess:
                    wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                    SaveCandidate(attempt, PersonalizationCandidateEligibility.Eligible);
                    break;
                case GuidedQuestOutcomeKind.LowConfidenceCorrect:
                    wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                    SaveCandidate(attempt, PersonalizationCandidateEligibility.ReviewOnly);
                    break;
                case GuidedQuestOutcomeKind.WrongWord:
                    wrongWordAttemptCounts[currentIndex]++;
                    LastStorageResult = new CandidateSaveResult(
                        CandidateStorageStatus.NotRequested, string.Empty,
                        "wrong-word result is not stored as a personalization sequence", repository.Counts);
                    break;
                case GuidedQuestOutcomeKind.TechnicalFailure:
                    LastStorageResult = new CandidateSaveResult(
                        CandidateStorageStatus.NotRequested, string.Empty,
                        "technical failure is not learner performance and is not stored", repository.Counts);
                    break;
                case GuidedQuestOutcomeKind.NotDistinguishedRetry:
                case GuidedQuestOutcomeKind.UnsupportedWord:
                    // full-hand-only verdicts; they never approve a candidate
                    LastStorageResult = new CandidateSaveResult(
                        CandidateStorageStatus.NotRequested, string.Empty,
                        FullHandGameplayFeedback.EnrollmentPending, repository.Counts);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            // Step 10W v1.0 observability: emitted AFTER the outcome branch above, so the
            // canonical World outcome is already decided and the record can carry it. The
            // blank-or-duplicate capture-session gate still runs earlier and returns
            // before this point, so an accepted attempt is still recorded exactly once
            // and a repeated callback still adds no second record.
            CommonSignalAttemptDiagnostics.Emit(
                attempt, questMode ? QuestAuthoritativeOutcome : null,
                questMode ? legacyHybridOutcome : null, questMode);

            State = PetRobotPracticeState.Feedback;
            Notify();
            return true;
        }

        public bool Retry()
        {
            if (!CanRetry) return false;
            if (!recognition.ResetCapture() || !recognition.SetExpectedWord(CurrentWord)) return false;
            LastAttempt = null;
            if (questMode)
            {
                QuestAuthoritativeOutcome = null;
                QuestHybridResult = null;
                legacyHybridOutcome = null;
            }
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty, "ready for a manual retry", repository.Counts);
            State = PetRobotPracticeState.Ready;
            Notify();
            return true;
        }

        public bool ContinueToNextWord()
        {
            if (State != PetRobotPracticeState.Feedback ||
                wordStatuses[currentIndex] != PetRobotWordLearningStatus.Mastered) return false;
            return AdvanceResolvedWord();
        }

        public bool ContinueWithNeedsPractice()
        {
            if (!CanContinueWithNeedsPractice) return false;
            if (trustedRepository != null) return AdvanceStoppedEnrollmentRound();
            if (currentIndex < words.Length - 1 &&
                (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[currentIndex + 1])))
                return false;

            wordStatuses[currentIndex] = PetRobotWordLearningStatus.NeedsPractice;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested,
                string.Empty,
                "wrong-word retry exhausted; marked NeedsPractice without storing a candidate",
                repository.Counts);
            if (currentIndex == words.Length - 1)
            {
                State = PetRobotPracticeState.Completed;
                Notify();
                return true;
            }
            currentIndex++;
            LastAttempt = null;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "previous word marked NeedsPractice; ready for the next word", repository.Counts);
            State = PetRobotPracticeState.Ready;
            Notify();
            return true;
        }

        public bool ExitPractice()
        {
            if (State != PetRobotPracticeState.Completed) return false;
            State = PetRobotPracticeState.Exited;
            Notify();
            ExitRequested?.Invoke();
            return true;
        }

        public bool IsWordComplete(string word)
        {
            for (int index = 0; index < words.Length; index++)
                if (words[index] == word) return wordStatuses[index] != PetRobotWordLearningStatus.Pending;
            return false;
        }

        public PetRobotWordLearningStatus GetWordLearningStatus(string word)
        {
            for (int index = 0; index < words.Length; index++)
                if (words[index] == word) return wordStatuses[index];
            throw new ArgumentException("unsupported practice word", nameof(word));
        }

        public int GetTrustedSampleCount(string word)
        {
            if (trustedRepository == null) return 0;
            return trustedRepository.TrustedInventory.CountForWord(word);
        }

        public int GetScoreableAttemptCount(string word) => scoreableAttemptCounts[IndexOfWord(word)];

        public int GetConsecutiveTechnicalFailureCount(string word) =>
            consecutiveTechnicalFailureCounts[IndexOfWord(word)];

        public PetRobotPersonalizationEnrollmentState GetEnrollmentState(string word) =>
            enrollmentStates[IndexOfWord(word)];

        public PracticeRecognitionAttempt GetWordFinalAttempt(string word)
        {
            for (int index = 0; index < words.Length; index++)
                if (words[index] == word) return finalAttempts[index];
            throw new ArgumentException("unsupported practice word", nameof(word));
        }

        public void Dispose()
        {
            cameraReadiness.Changed -= OnCameraReadinessChanged;
        }

        void OnCameraReadinessChanged()
        {
            if (cameraReadiness.CameraState == CandidateCameraStartupState.Failed &&
                State == PetRobotPracticeState.Ready && LastAttempt == null)
                RecordCameraStartupFailure();
            else
                Notify();
        }

        void RecordCameraStartupFailure()
        {
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Failed,
                false,
                CurrentWord,
                string.Empty,
                false,
                false,
                false,
                true,
                true);
            LastAttempt = new PracticeRecognitionAttempt(
                string.Empty,
                CurrentWord,
                string.Empty,
                PersonalLiveCaptureState.Failed,
                outcome,
                null,
                null,
                PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                recognition.ActiveProfileId,
                0d,
                null,
                null,
                new PracticeRecognitionMetrics { WinningPrototypeId = string.Empty });
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested,
                string.Empty,
                "camera startup failure is technical and no personalization candidate was stored",
                repository.Counts);
            Notify();
        }

        void SaveCandidate(PracticeRecognitionAttempt attempt, PersonalizationCandidateEligibility eligibility)
        {
            try
            {
                LastStorageResult = repository.Save(new PersonalizationCandidateSaveRequest(attempt, eligibility, utcNow()));
            }
            catch (Exception exception)
            {
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.Failed, string.Empty,
                    exception.GetType().Name + ": " + exception.Message, repository.Counts);
            }
        }

        void RestorePersistedEnrollment()
        {
            TrustedCandidateInventory inventory = trustedRepository.ReloadTrustedAndVerify();
            int firstPending = -1;
            for (int index = 0; index < words.Length; index++)
            {
                if (inventory.CountForWord(words[index]) >= 3)
                {
                    enrollmentStates[index] = PetRobotPersonalizationEnrollmentState.Personalized;
                    wordStatuses[index] = PetRobotWordLearningStatus.Mastered;
                }
                else if (firstPending < 0)
                {
                    firstPending = index;
                }
            }
            if (firstPending < 0)
            {
                currentIndex = words.Length - 1;
                State = PetRobotPracticeState.Completed;
            }
            else
            {
                currentIndex = firstPending;
            }
        }

        /// <summary>
        /// Step 10W v1.0. The one-to-one map from the finalized guided outcome to the
        /// canonical World branch. This is the ONLY producer of a World outcome value.
        ///
        /// Step 10V observed the defect this replaces: the same capture produced guided
        /// LowConfidenceCorrect and World WrongWord, because the World branch was taken
        /// from a second, independent decision made by the legacy frozen hybrid over the
        /// same sequence. The guided outcome is already finalized when this runs, so
        /// re-deciding it could only ever disagree with itself.
        ///
        /// The map is total and label-preserving. LowConfidenceCorrect stays
        /// LowConfidenceCorrect: nothing here promotes it to StrictSuccess and nothing
        /// here touches strict acceptance, thresholds or the model.
        /// </summary>
        public static TrustedHybridOutcomeKind MapGuidedToCanonicalWorldOutcome(
            GuidedQuestOutcomeKind guided)
        {
            switch (guided)
            {
                case GuidedQuestOutcomeKind.StrictSuccess:
                    return TrustedHybridOutcomeKind.StrictSuccess;
                case GuidedQuestOutcomeKind.LowConfidenceCorrect:
                    return TrustedHybridOutcomeKind.LowConfidenceCorrect;
                case GuidedQuestOutcomeKind.WrongWord:
                    return TrustedHybridOutcomeKind.WrongWord;
                case GuidedQuestOutcomeKind.TechnicalFailure:
                    return TrustedHybridOutcomeKind.TechnicalFailure;
                case GuidedQuestOutcomeKind.NotDistinguishedRetry:
                    return TrustedHybridOutcomeKind.NotDistinguishedRetry;
                case GuidedQuestOutcomeKind.UnsupportedWord:
                    return TrustedHybridOutcomeKind.UnsupportedWord;
                default:
                    throw new ArgumentOutOfRangeException(nameof(guided));
            }
        }

        /// <summary>
        /// F27 quest branch. The authority is the typed full-hand decision that the shared
        /// adapter already produced for this capture. No 16-point sequence is required and
        /// the legacy frozen-hybrid diagnostic is deliberately NOT run: it reads a sequence
        /// this route does not have, and a second opinion could only disagree with an
        /// already finalized decision.
        /// </summary>
        void HandleFullHandQuestOutcome(PracticeRecognitionAttempt attempt)
        {
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "quest mode does not store personalization candidates; " +
                FullHandGameplayFeedback.EnrollmentPending, repository.Counts);
            legacyHybridOutcome = null;
            QuestHybridResult = null;

            if (string.IsNullOrWhiteSpace(attempt.CaptureSessionId))
            {
                QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                return;
            }

            TrustedHybridOutcomeKind canonical =
                MapGuidedToCanonicalWorldOutcome(attempt.Outcome.Kind);
            if (canonical == TrustedHybridOutcomeKind.StrictSuccess && fullHandBinding != null &&
                !FullHandEvidenceMatchesBinding(attempt, CurrentWord))
            {
                QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                QuestCompleted = false;
                return;
            }
            QuestAuthoritativeOutcome = canonical;
            // ONLY an accepted capture whose Top1 matched the supported prompted word can
            // complete the quest; every other verdict leaves it incomplete.
            QuestCompleted = canonical == TrustedHybridOutcomeKind.StrictSuccess;
        }

        /// <summary>
        /// F27 practice branch. It reports the full-hand verdict and deliberately stores
        /// nothing: full-hand personalization enrollment is not implemented, so no candidate
        /// is written and no word is reported as personalized.
        /// </summary>
        void HandleFullHandPracticeOutcome(PracticeRecognitionAttempt attempt)
        {
            if (fullHandStore != null)
            {
                HandleFullHandCollectionOutcome(attempt);
                return;
            }
            LastGuidedSeedCommitResult = GuidedEnrollmentSeedCommitResult.NotRequested;
            switch (attempt.Outcome.Kind)
            {
                case GuidedQuestOutcomeKind.StrictSuccess:
                    // The learning step advances so the learner is never stuck, but nothing
                    // is saved and the enrollment state is set to the honest
                    // PersonalizationIncomplete: full-hand enrollment does not exist yet, so
                    // Personalized would be a false green.
                    wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                    enrollmentStates[currentIndex] =
                        PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete;
                    scoreableAttemptCounts[currentIndex]++;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    break;
                case GuidedQuestOutcomeKind.WrongWord:
                    wrongWordAttemptCounts[currentIndex]++;
                    scoreableAttemptCounts[currentIndex]++;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    break;
                case GuidedQuestOutcomeKind.NotDistinguishedRetry:
                    scoreableAttemptCounts[currentIndex]++;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    break;
                case GuidedQuestOutcomeKind.UnsupportedWord:
                    enrollmentStates[currentIndex] =
                        PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete;
                    break;
                default:
                    consecutiveTechnicalFailureCounts[currentIndex]++;
                    if (consecutiveTechnicalFailureCounts[currentIndex] >= 3)
                        enrollmentStates[currentIndex] =
                            PetRobotPersonalizationEnrollmentState.TechnicalStop;
                    break;
            }
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                FullHandGameplayFeedback.EnrollmentPending, repository.Counts);
        }

        /// <summary>
        /// F28. The store-backed practice branch.
        ///
        /// An ACCEPTED capture is offered to the local record store FIRST; the count is then
        /// read back from the store, so a duplicate, a refused record or a failed write can
        /// never move progress. A storage failure does not consume a sample: the learner tries
        /// the same word again.
        ///
        /// Reaching the quorum means the practice RECORDS for this word are collected. It is
        /// not expert correctness, not certified sign quality and not a personalized model.
        /// </summary>
        void HandleFullHandCollectionOutcome(PracticeRecognitionAttempt attempt)
        {
            LastGuidedSeedCommitResult = GuidedEnrollmentSeedCommitResult.NotRequested;
            LastFullHandSaveResult = FullHandEnrollmentSaveResult.NotRequested;
            LastFullHandCollectionMessage = string.Empty;
            // F29: this attempt is finalized and unique, so it spends one unit of the budget
            // whatever the store decides below.
            CountFullHandAttempt();
            switch (attempt.Outcome.Kind)
            {
                case GuidedQuestOutcomeKind.StrictSuccess:
                {
                    FullHandEnrollmentSaveResult save = fullHandStore.TrySave(
                        attempt.FullHandGameplay, attempt.FullHandEvidence, CurrentWord, utcNow());
                    LastFullHandSaveResult = save;
                    fullHandRecordCounts[currentIndex] = save.CountForWord;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    if (save.Committed)
                    {
                        scoreableAttemptCounts[currentIndex]++;
                        if (save.CountForWord >= fullHandBinding.RequiredRecordings)
                        {
                            wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                            LastFullHandCollectionMessage =
                                FullHandEnrollmentCopy.CollectedForWord(
                                    fullHandBinding.RequiredRecordings);
                        }
                        else
                        {
                            LastFullHandCollectionMessage = FullHandEnrollmentCopy.Saved(
                                save.CountForWord, fullHandBinding.RequiredRecordings);
                        }
                    }
                    else if (save.Status == FullHandEnrollmentSaveStatus.Duplicate)
                    {
                        LastFullHandCollectionMessage = FullHandEnrollmentCopy.Duplicate(
                            save.CountForWord, fullHandBinding.RequiredRecordings);
                    }
                    else
                    {
                        // NOT saved. The count did not move and this is never shown as progress.
                        LastFullHandCollectionMessage = FullHandEnrollmentCopy.NotSaved(
                            save.CountForWord, fullHandBinding.RequiredRecordings);
                    }
                    break;
                }
                case GuidedQuestOutcomeKind.WrongWord:
                    wrongWordAttemptCounts[currentIndex]++;
                    scoreableAttemptCounts[currentIndex]++;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    break;
                case GuidedQuestOutcomeKind.NotDistinguishedRetry:
                    scoreableAttemptCounts[currentIndex]++;
                    consecutiveTechnicalFailureCounts[currentIndex] = 0;
                    break;
                case GuidedQuestOutcomeKind.UnsupportedWord:
                    // outside the CURRENT model vocabulary: never stored, never collected
                    enrollmentStates[currentIndex] =
                        PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete;
                    break;
                default:
                    consecutiveTechnicalFailureCounts[currentIndex]++;
                    if (consecutiveTechnicalFailureCounts[currentIndex] >= 3)
                        enrollmentStates[currentIndex] =
                            PetRobotPersonalizationEnrollmentState.TechnicalStop;
                    break;
            }
            // F29: the save above already ran. If it supplied the last missing record the word
            // is collected and the cap does not apply; otherwise the budget is spent and the
            // learner is told what was actually collected.
            ApplyFullHandAttemptCapMessage();
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "full-hand practice records are kept in the local full-hand record store; " +
                "no 16-point personalization candidate is written on this route",
                repository.Counts);
        }

        /// <summary>
        /// F28. A bound record session received an attempt that is not a full-hand result. It
        /// is a technical refusal: nothing is stored anywhere, no word advances, and the legacy
        /// 16-point branch is not reached.
        /// </summary>
        void HandleUnexpectedLegacyAttempt()
        {
            LastGuidedSeedCommitResult = GuidedEnrollmentSeedCommitResult.NotRequested;
            LastFullHandSaveResult = FullHandEnrollmentSaveResult.NotRequested;
            CountFullHandAttempt();
            if (!ApplyFullHandAttemptCapMessage())
                LastFullHandCollectionMessage = FullHandEnrollmentCopy.UnexpectedResult;
            consecutiveTechnicalFailureCounts[currentIndex]++;
            if (consecutiveTechnicalFailureCounts[currentIndex] >= 3)
                enrollmentStates[currentIndex] =
                    PetRobotPersonalizationEnrollmentState.TechnicalStop;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "this attempt carried no full-hand result; a full-hand session writes no " +
                "16-point personalization candidate", repository.Counts);
        }

        /// <summary>
        /// F28. Re-derives the decision from THIS capture's own evidence and requires it to be
        /// an accepted result of the pinned participant, model and gate whose ranked winner is
        /// the prompted word. Nothing is re-scored: the frozen evaluator re-reads the gated
        /// outcome that was already produced for this capture.
        /// </summary>
        bool FullHandEvidenceMatchesBinding(PracticeRecognitionAttempt attempt, string word)
        {
            if (attempt == null || attempt.FullHandEvidence == null ||
                attempt.FullHandGameplay == null) return false;
            // all THREE identities must name ONE capture: this attempt, the evidence it
            // carries and the decision carried beside it. An earlier practice capture can
            // therefore not be wrapped as a new NPC attempt.
            if (string.IsNullOrEmpty(attempt.CaptureSessionId) ||
                !string.Equals(attempt.CaptureSessionId,
                               attempt.FullHandEvidence.CaptureSessionId,
                               StringComparison.Ordinal) ||
                !string.Equals(attempt.CaptureSessionId,
                               attempt.FullHandGameplay.CaptureSessionId,
                               StringComparison.Ordinal)) return false;
            FullHandGameplayDecision derived;
            try
            {
                derived = FullHandGameplayEvaluator.Decide(attempt.FullHandEvidence, word,
                    attempt.FullHandEvidence.CaptureSessionId, true);
            }
            catch (Exception) { return false; }
            return FullHandParticipantCandidateStore.IsAcceptedForBinding(
                derived, attempt.FullHandGameplay, word, fullHandBinding);
        }

        /// <summary>
        /// Reads the persisted counts on open, so a reopened session continues where the
        /// collection actually stands instead of starting the count again.
        /// </summary>
        void RestoreFullHandCollection()
        {
            fullHandStore.Reload();
            int firstPending = -1;
            for (int index = 0; index < words.Length; index++)
            {
                fullHandRecordCounts[index] = fullHandStore.CountFor(words[index]);
                if (fullHandStore.IsCollectedFor(words[index]))
                    wordStatuses[index] = PetRobotWordLearningStatus.Mastered;
                else if (firstPending < 0)
                    firstPending = index;
            }
            if (firstPending < 0)
            {
                currentIndex = words.Length - 1;
                State = PetRobotPracticeState.Completed;
            }
            else
            {
                currentIndex = firstPending;
            }
        }

        /// <summary>
        /// Moves to the next word whose records are still incomplete, over the pinned round.
        /// The round completes only when no word is left pending.
        /// </summary>
        bool AdvanceFullHandWord()
        {
            // F32: a word deferred during THIS visit is not offered again, so a word that was
            // already ended early cannot reappear after another word succeeds.
            int next = NextEligibleFullHandWord();
            if (next < 0)
            {
                State = PetRobotPracticeState.Completed;
                Notify();
                return true;
            }
            if (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[next]))
                return false;
            currentIndex = next;
            LastAttempt = null;
            LastFullHandSaveResult = FullHandEnrollmentSaveResult.NotRequested;
            LastFullHandCollectionMessage = string.Empty;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "ready for the next word of the current model vocabulary", repository.Counts);
            State = PetRobotPracticeState.Ready;
            Notify();
            return true;
        }

        void HandleQuestOutcome(PracticeRecognitionAttempt attempt)
        {
            if (attempt.FullHandGameplay != null)
            {
                HandleFullHandQuestOutcome(attempt);
                return;
            }

            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "quest mode does not store personalization candidates", repository.Counts);
            legacyHybridOutcome = null;
            QuestHybridResult = null;

            if (frozenHybridSession == null)
            {
                // F28: the full-hand quest route carries no legacy 16-point session
                QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                return;
            }
            if (attempt.NormalizedSequence == null ||
                string.IsNullOrWhiteSpace(attempt.CaptureSessionId))
            {
                // No usable capture identity or sequence is a technical failure of the
                // capture itself, not a recognition disagreement.
                QuestAuthoritativeOutcome = TrustedHybridOutcomeKind.TechnicalFailure;
                return;
            }

            // The canonical World outcome is decided here and never reassigned below.
            TrustedHybridOutcomeKind canonical =
                MapGuidedToCanonicalWorldOutcome(attempt.Outcome.Kind);
            QuestAuthoritativeOutcome = canonical;
            QuestCompleted = canonical == TrustedHybridOutcomeKind.StrictSuccess ||
                canonical == TrustedHybridOutcomeKind.LowConfidenceCorrect;

            // Legacy frozen hybrid, DIAGNOSTIC ONLY from Step 10W onward. Its result is
            // recorded for comparison and can no longer reach the World branch. A null
            // result, a disagreeing result or a throw leaves the canonical value above
            // exactly as it is, which is the whole point of this repair. The global
            // frozen hybrid policy itself is unchanged.
            if (canonical == TrustedHybridOutcomeKind.TechnicalFailure) return;
            try
            {
                TrustedHybridRecognitionResult hybrid = frozenHybridSession.Recognize(
                    attempt.CaptureSessionId, attempt.NormalizedSequence, "GREETING");
                if (hybrid == null) return;
                QuestHybridResult = hybrid;
                legacyHybridOutcome = hybrid.Kind;
            }
            catch (Exception)
            {
                QuestHybridResult = null;
                legacyHybridOutcome = null;
            }
        }

        void HandleTrustedEnrollmentOutcome(PracticeRecognitionAttempt attempt)
        {
            LastGuidedSeedCommitResult = GuidedEnrollmentSeedCommitResult.NotRequested;
            GuidedQuestOutcomeKind kind = attempt.Outcome.Kind;
            if (kind == GuidedQuestOutcomeKind.TechnicalFailure)
            {
                consecutiveTechnicalFailureCounts[currentIndex]++;
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "technical failure is not a learner attempt and is not stored", repository.Counts);
                if (consecutiveTechnicalFailureCounts[currentIndex] >= 3)
                    enrollmentStates[currentIndex] = PetRobotPersonalizationEnrollmentState.TechnicalStop;
                return;
            }

            scoreableAttemptCounts[currentIndex]++;
            consecutiveTechnicalFailureCounts[currentIndex] = 0;
            if (kind == GuidedQuestOutcomeKind.StrictSuccess)
            {
                int before = trustedRepository.TrustedInventory.CountForWord(CurrentWord);
                SaveCandidate(attempt, PersonalizationCandidateEligibility.Eligible);
                if (LastStorageResult.Status == CandidateStorageStatus.Saved)
                {
                    TrustedCandidateInventory after = trustedRepository.ReloadTrustedAndVerify();
                    int afterCount = after.CountForWord(CurrentWord);
                    if (afterCount > before && afterCount >= 3)
                    {
                        enrollmentStates[currentIndex] = PetRobotPersonalizationEnrollmentState.Personalized;
                        wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                    }
                }
            }
            else if (kind == GuidedQuestOutcomeKind.LowConfidenceCorrect)
            {
                SaveCandidate(attempt, PersonalizationCandidateEligibility.ReviewOnly);
                if (LastStorageResult.Status == CandidateStorageStatus.Saved && guidedSeedRepository != null)
                {
                    LastGuidedSeedCommitResult =
                        guidedSeedRepository.TryCommitGuidedEnrollmentSeedQuorum(CurrentWord);
                    if (LastGuidedSeedCommitResult.IsSuccess)
                    {
                        TrustedCandidateInventory after = trustedRepository.ReloadTrustedAndVerify();
                        if (after.CountForWord(CurrentWord) >= 3)
                        {
                            enrollmentStates[currentIndex] = PetRobotPersonalizationEnrollmentState.Personalized;
                            wordStatuses[currentIndex] = PetRobotWordLearningStatus.Mastered;
                        }
                    }
                }
            }
            else if (kind == GuidedQuestOutcomeKind.WrongWord)
            {
                wrongWordAttemptCounts[currentIndex]++;
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "wrong-word result is not stored as a personalization sequence", repository.Counts);
            }
            else
            {
                throw new ArgumentOutOfRangeException();
            }

            if (enrollmentStates[currentIndex] == PetRobotPersonalizationEnrollmentState.Pending &&
                scoreableAttemptCounts[currentIndex] >= 10)
                enrollmentStates[currentIndex] = PetRobotPersonalizationEnrollmentState.PersonalizationIncomplete;
        }

        bool AdvanceStoppedEnrollmentRound()
        {
            int next = FindNextPendingWord(currentIndex + 1);
            if (next < 0)
            {
                State = PetRobotPracticeState.Completed;
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "enrollment round ended without bypassing the overall three-word gate", repository.Counts);
                Notify();
                return true;
            }
            if (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[next])) return false;
            currentIndex = next;
            LastAttempt = null;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty,
                "ready for the next incomplete personalization word", repository.Counts);
            State = PetRobotPracticeState.Ready;
            Notify();
            return true;
        }

        bool AdvanceResolvedWord()
        {
            if (fullHandStore != null) return AdvanceFullHandWord();
            if (trustedRepository != null)
            {
                int next = FindNextPendingWord(currentIndex + 1);
                if (next < 0)
                {
                    State = PetRobotPracticeState.Completed;
                    Notify();
                    return true;
                }
                if (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[next])) return false;
                currentIndex = next;
                LastAttempt = null;
                LastStorageResult = new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "persisted personalized words skipped; ready for the next incomplete word", repository.Counts);
                State = PetRobotPracticeState.Ready;
                Notify();
                return true;
            }
            if (currentIndex == words.Length - 1)
            {
                State = PetRobotPracticeState.Completed;
                Notify();
                return true;
            }
            if (!recognition.ResetCapture() || !recognition.SetExpectedWord(words[currentIndex + 1])) return false;
            currentIndex++;
            LastAttempt = null;
            LastStorageResult = new CandidateSaveResult(
                CandidateStorageStatus.NotRequested, string.Empty, "ready for the next word", repository.Counts);
            State = PetRobotPracticeState.Ready;
            Notify();
            return true;
        }

        int FindNextPendingWord(int start)
        {
            for (int index = Math.Max(0, start); index < words.Length; index++)
                if (enrollmentStates[index] != PetRobotPersonalizationEnrollmentState.Personalized)
                    return index;
            return -1;
        }

        int IndexOfWord(string word)
        {
            int index = IndexOfWordOrMinusOne(word);
            if (index < 0) throw new ArgumentException("unsupported practice word", nameof(word));
            return index;
        }

        int IndexOfWordOrMinusOne(string word)
        {
            for (int index = 0; index < words.Length; index++)
                if (string.Equals(words[index], word, StringComparison.Ordinal)) return index;
            return -1;
        }

        int CountStatus(PetRobotWordLearningStatus status)
        {
            int count = 0;
            for (int index = 0; index < wordStatuses.Length; index++)
                if (wordStatuses[index] == status) count++;
            return count;
        }

        void Notify() => Changed?.Invoke();

        sealed class AlwaysReadyCameraReadiness : IPetRobotCandidateCameraReadiness
        {
            public static readonly AlwaysReadyCameraReadiness Instance = new AlwaysReadyCameraReadiness();
            public event Action Changed { add { } remove { } }
            public CandidateCameraStartupState CameraState => CandidateCameraStartupState.Ready;
            public bool IsReady => true;
            public bool CameraStarted => true;
            public bool HasReceivedFrame => true;
            public bool ProviderObservationReady => true;
            public int CameraFrameCount => 1;
            public int ProviderObservationCount => 1;
            public string LastErrorCode => string.Empty;
            public string LastErrorMessage => string.Empty;
        }
    }
}
