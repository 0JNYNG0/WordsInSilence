using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    public enum PetRobotHandoffMode
    {
        Normal,
        GreetingQuest,
    }

    public sealed class PetRobotGreetingQuestToken
    {
        public string Id { get; }
        public PetRobotGreetingQuestToken()
        {
            Id = Guid.NewGuid().ToString("N");
        }
        internal PetRobotGreetingQuestToken(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("quest token ID must be non-empty", nameof(id));
            Id = id;
        }
    }

    public sealed class PetRobotGreetingQuestResult
    {
        public string TokenId { get; }
        public TrustedHybridOutcomeKind Kind { get; }
        public string FeedbackText { get; }
        public string FailureType { get; }
        public string FailureMessage { get; }

        public bool IsSuccess =>
            Kind == TrustedHybridOutcomeKind.StrictSuccess ||
            Kind == TrustedHybridOutcomeKind.LowConfidenceCorrect;

        public PetRobotGreetingQuestResult(
            string tokenId,
            TrustedHybridOutcomeKind kind,
            string feedbackText,
            string failureType,
            string failureMessage)
        {
            TokenId = tokenId ?? string.Empty;
            Kind = kind;
            FeedbackText = feedbackText ?? string.Empty;
            FailureType = failureType ?? string.Empty;
            FailureMessage = failureMessage ?? string.Empty;
        }
    }

    public interface IQuestHybridRecognitionSession
    {
        long Generation { get; }
        TrustedHybridRecognitionResult Recognize(
            string querySessionId, RecognitionSequence query, string expectedWord);
    }

    public sealed class TrustedHybridRecognitionSessionAdapter : IQuestHybridRecognitionSession
    {
        readonly TrustedHybridRuntimeSession inner;
        public long Generation => inner.ActiveViewGeneration;

        public TrustedHybridRecognitionSessionAdapter(TrustedHybridRuntimeSession session)
        {
            inner = session ?? throw new ArgumentNullException(nameof(session));
        }

        public TrustedHybridRecognitionResult Recognize(
            string querySessionId, RecognitionSequence query, string expectedWord)
        {
            return inner.Recognize(querySessionId, query, expectedWord);
        }
    }

    public sealed class PetRobotLearnerRootIdentity
    {
        public string LearnerId { get; }
        public string ExplicitRoot { get; }
        public bool HasExplicitRoot => !string.IsNullOrWhiteSpace(ExplicitRoot);

        public PetRobotLearnerRootIdentity(string learnerId, string explicitRoot = null)
        {
            LearnerId = learnerId ?? "local_learner";
            ExplicitRoot = explicitRoot ?? string.Empty;
        }
    }

    public interface IQuestGateVerifier
    {
        bool VerifyOverallEnrollmentGateReady();
        string LastFailure { get; }
    }

    public sealed class ProductionQuestGateVerifier : IQuestGateVerifier
    {
        readonly PetRobotLearnerRootIdentity identity;
        public string LastFailure { get; private set; } = string.Empty;

        public ProductionQuestGateVerifier(PetRobotLearnerRootIdentity identity)
        {
            this.identity = identity ?? throw new ArgumentNullException(nameof(identity));
        }

        public bool VerifyOverallEnrollmentGateReady()
        {
            try
            {
                // F28: the full-hand route has its own store-backed gate. It fails closed on a
                // missing participant, a missing or unreadable pipeline configuration, a stale
                // owner/model/gate, an incompletely scanned store and any partially collected
                // vocabulary. The legacy repository below is not consulted on that route.
                if (FullHandEnrollmentRuntime.ModeActive)
                {
                    string fullHandFailure;
                    if (FullHandEnrollmentQuestGate.VerifyForIdentity(identity,
                                                                      out fullHandFailure))
                    {
                        LastFailure = string.Empty;
                        return true;
                    }
                    LastFailure = fullHandFailure;
                    return false;
                }
                PersonalizationCandidateRepository repository;
                if (identity.HasExplicitRoot)
                    repository = new PersonalizationCandidateRepository(
                        identity.LearnerId, identity.ExplicitRoot);
                else
                    repository = new PersonalizationCandidateRepository(identity.LearnerId);
                var trusted = repository as ITrustedPersonalizationCandidateRepository;
                if (trusted == null)
                {
                    LastFailure = "repository does not implement trusted interface";
                    return false;
                }
                TrustedCandidateInventory inventory = trusted.ReloadTrustedAndVerify();
                if (!inventory.OverallEnrollmentGateReady)
                {
                    LastFailure = "OverallEnrollmentGateReady=false G=" +
                        inventory.CountForWord("GREETING") + " L=" +
                        inventory.CountForWord("LOVE") + " T=" +
                        inventory.CountForWord("THANK_YOU");
                    return false;
                }
                LastFailure = string.Empty;
                return true;
            }
            catch (Exception exception)
            {
                LastFailure = exception.GetType().Name + ": " + exception.Message;
                return false;
            }
        }
    }

    public sealed class PetRobotPracticeCompletionSnapshot
    {
        public PetRobotWordLearningStatus Greeting { get; }
        public PetRobotWordLearningStatus Love { get; }
        public PetRobotWordLearningStatus ThankYou { get; }
        public int MasteredCount { get; }
        public int NeedsPracticeCount { get; }
        public int CandidateSavedTotal { get; }
        public bool SessionCompleted => true;
        public string WorldSummaryMessage => IsFullHandRecordRound
            ? (AllRecordsCollected
                ? "연습 기록 완료 · 기록 " + FullHandRecordTotal + "개"
                : "연습 기록 미완료 · " + MasteredCount + "/" + Words.Count + " 표현")
            : NeedsPracticeCount == 0
            ? "연습 완료"
            : "연습 완료 · " + NeedsPracticeCount + "개 복습 필요";

        public PetRobotPracticeCompletionSnapshot(
            PetRobotWordLearningStatus greeting,
            PetRobotWordLearningStatus love,
            PetRobotWordLearningStatus thankYou,
            int masteredCount,
            int needsPracticeCount,
            int candidateSavedTotal)
        {
            ValidateResolved(greeting, nameof(greeting));
            ValidateResolved(love, nameof(love));
            ValidateResolved(thankYou, nameof(thankYou));
            if (masteredCount < 0 || needsPracticeCount < 0 || masteredCount + needsPracticeCount != 3)
                throw new ArgumentOutOfRangeException(nameof(masteredCount), "completed counts must resolve exactly three words");
            if (candidateSavedTotal < 0)
                throw new ArgumentOutOfRangeException(nameof(candidateSavedTotal));
            int derivedMastered = Count(greeting, PetRobotWordLearningStatus.Mastered) +
                Count(love, PetRobotWordLearningStatus.Mastered) +
                Count(thankYou, PetRobotWordLearningStatus.Mastered);
            int derivedNeedsPractice = 3 - derivedMastered;
            if (masteredCount != derivedMastered || needsPracticeCount != derivedNeedsPractice)
                throw new ArgumentException("word statuses and summary counts disagree");
            Greeting = greeting;
            Love = love;
            ThankYou = thankYou;
            MasteredCount = masteredCount;
            NeedsPracticeCount = needsPracticeCount;
            CandidateSavedTotal = candidateSavedTotal;
            Words = Array.AsReadOnly(new[] { "GREETING", "LOVE", "THANK_YOU" });
            statuses = new[] { greeting, love, thankYou };
            IsFullHandRecordRound = false;
            FullHandRecordTotal = 0;
            RequiredRecordsPerWord = 0;
            AllRecordsCollected = false;
        }

        /// <summary>
        /// F28. The words this round ACTUALLY practised, in round order. For the legacy
        /// three-word round it is GREETING / LOVE / THANK_YOU.
        /// </summary>
        public IReadOnlyList<string> Words { get; }
        readonly PetRobotWordLearningStatus[] statuses;
        /// <summary>True when this was a full-hand practice-record round.</summary>
        public bool IsFullHandRecordRound { get; }
        /// <summary>The real number of accepted recordings collected in this round.</summary>
        public int FullHandRecordTotal { get; }
        public int RequiredRecordsPerWord { get; }
        /// <summary>
        /// Store-derived: every word of the round has its full quorum. It is never inferred
        /// from a word status, so a round that ended early can never read as collected.
        /// </summary>
        public bool AllRecordsCollected { get; }

        /// <summary>
        /// F28 constructor for a round whose words are the CURRENT model vocabulary.
        /// </summary>
        public PetRobotPracticeCompletionSnapshot(
            IReadOnlyList<string> words,
            IReadOnlyList<PetRobotWordLearningStatus> wordStatuses,
            int fullHandRecordTotal,
            int requiredRecordsPerWord,
            bool allRecordsCollected)
        {
            if (words == null || wordStatuses == null || words.Count == 0 ||
                words.Count != wordStatuses.Count)
                throw new ArgumentException("a round needs one status per practised word",
                                            nameof(words));
            if (fullHandRecordTotal < 0 || requiredRecordsPerWord <= 0)
                throw new ArgumentOutOfRangeException(nameof(fullHandRecordTotal));
            var wordCopy = new string[words.Count];
            statuses = new PetRobotWordLearningStatus[words.Count];
            int mastered = 0;
            for (int index = 0; index < wordCopy.Length; index++)
            {
                wordCopy[index] = words[index];
                if (string.IsNullOrWhiteSpace(wordCopy[index]))
                    throw new ArgumentException("a practised word must be named", nameof(words));
                for (int other = 0; other < index; other++)
                    if (string.Equals(wordCopy[other], wordCopy[index], StringComparison.Ordinal))
                        throw new ArgumentException("a round repeats a word", nameof(words));
                ValidateResolved(wordStatuses[index], nameof(wordStatuses));
                statuses[index] = wordStatuses[index];
                if (statuses[index] == PetRobotWordLearningStatus.Mastered) mastered++;
            }
            Words = Array.AsReadOnly(wordCopy);
            IsFullHandRecordRound = true;
            FullHandRecordTotal = fullHandRecordTotal;
            RequiredRecordsPerWord = requiredRecordsPerWord;
            AllRecordsCollected = allRecordsCollected;
            MasteredCount = mastered;
            NeedsPracticeCount = wordCopy.Length - mastered;
            CandidateSavedTotal = 0;
            Greeting = StatusOrPending("GREETING");
            Love = StatusOrPending("LOVE");
            ThankYou = StatusOrPending("THANK_YOU");
        }

        PetRobotWordLearningStatus StatusOrPending(string word)
        {
            PetRobotWordLearningStatus status;
            return TryGetWordStatus(word, out status)
                ? status : PetRobotWordLearningStatus.Pending;
        }

        /// <summary>
        /// The status of a word IF this round practised it. False means the word was not part
        /// of this round, which is NOT the same as failed practice.
        /// </summary>
        public bool TryGetWordStatus(string word, out PetRobotWordLearningStatus status)
        {
            status = PetRobotWordLearningStatus.Pending;
            for (int index = 0; index < Words.Count; index++)
                if (string.Equals(Words[index], word, StringComparison.Ordinal))
                {
                    status = statuses[index];
                    return true;
                }
            return false;
        }

        public PetRobotWordLearningStatus GetWordStatus(string word)
        {
            if (IsFullHandRecordRound)
            {
                PetRobotWordLearningStatus status;
                if (TryGetWordStatus(word, out status)) return status;
                throw new ArgumentException("this round did not practise that word", nameof(word));
            }
            if (string.Equals(word, "GREETING", StringComparison.Ordinal)) return Greeting;
            if (string.Equals(word, "LOVE", StringComparison.Ordinal)) return Love;
            if (string.Equals(word, "THANK_YOU", StringComparison.Ordinal)) return ThankYou;
            throw new ArgumentException("unsupported practice word", nameof(word));
        }

        public static bool TryCreate(PetRobotSignPracticeSession session, out PetRobotPracticeCompletionSnapshot snapshot)
        {
            snapshot = null;
            if (session == null) return false;
            if (session.FullHandRecordCollectionActive)
            {
                if (session.State != PetRobotPracticeState.Completed) return false;
                IReadOnlyList<string> roundWords = session.Words;
                var roundStatuses = new PetRobotWordLearningStatus[roundWords.Count];
                for (int index = 0; index < roundWords.Count; index++)
                {
                    PetRobotWordLearningStatus status =
                        session.GetWordLearningStatus(roundWords[index]);
                    roundStatuses[index] = status == PetRobotWordLearningStatus.Pending
                        ? PetRobotWordLearningStatus.NeedsPractice : status;
                }
                snapshot = new PetRobotPracticeCompletionSnapshot(
                    roundWords, roundStatuses, session.FullHandRecordTotal,
                    session.FullHandRequiredRecordCount, session.AllFullHandRecordsCollected);
                return true;
            }
            if (session.State != PetRobotPracticeState.Completed || !session.AllWordsComplete)
                return false;
            snapshot = new PetRobotPracticeCompletionSnapshot(
                session.GetWordLearningStatus("GREETING"),
                session.GetWordLearningStatus("LOVE"),
                session.GetWordLearningStatus("THANK_YOU"),
                session.MasteredCount,
                session.NeedsPracticeCount,
                session.CandidateCounts.TotalCandidateCount);
            return true;
        }

        static void ValidateResolved(PetRobotWordLearningStatus status, string parameter)
        {
            if (status != PetRobotWordLearningStatus.Mastered &&
                status != PetRobotWordLearningStatus.NeedsPractice)
                throw new ArgumentException("completed snapshot cannot contain Pending", parameter);
        }

        static int Count(PetRobotWordLearningStatus value, PetRobotWordLearningStatus expected) =>
            value == expected ? 1 : 0;
    }

    public sealed class PetRobotWorldReturnPayload
    {
        public string WorldSceneName { get; }
        public Vector3 PlayerPosition { get; }
        public PetRobotPracticeCompletionSnapshot Completion { get; }

        public PetRobotWorldReturnPayload(
            string worldSceneName,
            Vector3 playerPosition,
            PetRobotPracticeCompletionSnapshot completion)
        {
            WorldSceneName = worldSceneName ?? string.Empty;
            PlayerPosition = playerPosition;
            Completion = completion;
        }
    }

    public sealed class PetRobotWorldHandoffChannel
    {
        sealed class PendingEntry
        {
            public string WorldSceneName;
            public string PracticeSceneName;
            public Vector3 PlayerPosition;
            public PetRobotPracticeCompletionSnapshot Completion;
            public PetRobotHandoffMode Mode;
            public PetRobotGreetingQuestToken QuestToken;
            public PetRobotPracticeCompletionSnapshot PriorCompletion;
            public PetRobotLearnerRootIdentity LearnerIdentity;
        }

        public static PetRobotWorldHandoffChannel Shared { get; } = new PetRobotWorldHandoffChannel();

        PendingEntry pending;
        PetRobotGreetingQuestResult questResultStore;

        public bool HasPendingEntry => pending != null;
        public bool HasPublishedCompletion => pending != null && pending.Completion != null;
        public bool HasQuestResult => questResultStore != null;
        public PetRobotHandoffMode PendingMode =>
            pending == null ? PetRobotHandoffMode.Normal : pending.Mode;
        public PetRobotGreetingQuestToken PendingQuestToken => pending?.QuestToken;
        public PetRobotPracticeCompletionSnapshot PendingPriorCompletion => pending?.PriorCompletion;
        public PetRobotLearnerRootIdentity PendingLearnerIdentity => pending?.LearnerIdentity;

        public bool Begin(string worldSceneName, string practiceSceneName, Vector3 playerPosition)
        {
            if (pending != null || string.IsNullOrWhiteSpace(worldSceneName) ||
                string.IsNullOrWhiteSpace(practiceSceneName)) return false;
            pending = new PendingEntry
            {
                WorldSceneName = worldSceneName,
                PracticeSceneName = practiceSceneName,
                PlayerPosition = playerPosition,
            };
            return true;
        }

        public bool BeginGreetingQuest(
            string worldSceneName,
            string practiceSceneName,
            Vector3 playerPosition,
            PetRobotGreetingQuestToken token,
            PetRobotPracticeCompletionSnapshot priorCompletion,
            PetRobotLearnerRootIdentity learnerIdentity)
        {
            if (pending != null || string.IsNullOrWhiteSpace(worldSceneName) ||
                string.IsNullOrWhiteSpace(practiceSceneName) ||
                token == null || string.IsNullOrWhiteSpace(token.Id) ||
                priorCompletion == null || !priorCompletion.SessionCompleted ||
                learnerIdentity == null || string.IsNullOrWhiteSpace(learnerIdentity.LearnerId))
                return false;
            pending = new PendingEntry
            {
                WorldSceneName = worldSceneName,
                PracticeSceneName = practiceSceneName,
                PlayerPosition = playerPosition,
                Mode = PetRobotHandoffMode.GreetingQuest,
                QuestToken = token,
                PriorCompletion = priorCompletion,
                LearnerIdentity = learnerIdentity,
            };
            return true;
        }

        public bool PublishCompleted(PetRobotPracticeCompletionSnapshot completion)
        {
            if (pending == null || pending.Completion != null || completion == null || !completion.SessionCompleted)
                return false;
            pending.Completion = completion;
            return true;
        }

        public bool PublishQuestResult(PetRobotGreetingQuestResult result)
        {
            if (pending == null || pending.Mode != PetRobotHandoffMode.GreetingQuest ||
                pending.QuestToken == null || questResultStore != null ||
                result == null || !string.Equals(result.TokenId, pending.QuestToken.Id, StringComparison.Ordinal))
                return false;
            questResultStore = result;
            return true;
        }

        public bool CommitQuestReturn(
            PetRobotGreetingQuestResult result,
            PetRobotPracticeCompletionSnapshot priorCompletion)
        {
            if (pending == null || pending.Mode != PetRobotHandoffMode.GreetingQuest ||
                pending.QuestToken == null || pending.Completion != null ||
                questResultStore != null ||
                result == null || priorCompletion == null || !priorCompletion.SessionCompleted ||
                !string.Equals(result.TokenId, pending.QuestToken.Id, StringComparison.Ordinal))
                return false;
            questResultStore = result;
            pending.Completion = priorCompletion;
            return true;
        }

        public bool TryConsumeCompleted(string worldSceneName, out PetRobotWorldReturnPayload payload)
        {
            payload = null;
            if (pending == null || pending.Completion == null ||
                !string.Equals(pending.WorldSceneName, worldSceneName, StringComparison.Ordinal)) return false;
            payload = new PetRobotWorldReturnPayload(
                pending.WorldSceneName, pending.PlayerPosition, pending.Completion);
            pending = null;
            return true;
        }

        public bool TryConsumeQuestResult(out PetRobotGreetingQuestResult result)
        {
            result = questResultStore;
            if (questResultStore == null) return false;
            questResultStore = null;
            return true;
        }

        public bool TryConsumeIncomplete(string worldSceneName, out Vector3 playerPosition)
        {
            playerPosition = default(Vector3);
            if (pending == null || pending.Completion != null ||
                !string.Equals(pending.WorldSceneName, worldSceneName, StringComparison.Ordinal)) return false;
            playerPosition = pending.PlayerPosition;
            pending = null;
            return true;
        }

        /// <summary>Drops the pending entry and the stored quest result so a previous participant's
        /// completion or result cannot reappear after the world scene reloads for a new participant.
        /// No other gate behaviour changes.</summary>
        public void ResetForNewParticipant()
        {
            pending = null;
            questResultStore = null;
        }

        public bool CancelUncommitted()
        {
            if (pending == null || pending.Completion != null) return false;
            questResultStore = null;
            pending = null;
            return true;
        }
    }

    public interface IPetRobotWorldSceneTransition
    {
        int RequestCount { get; }
        bool Request(string sceneName);
    }

    public sealed class UnityPetRobotWorldSceneTransition : IPetRobotWorldSceneTransition
    {
        int requestCount;
        public int RequestCount => requestCount;

        public bool Request(string sceneName)
        {
            if (requestCount != 0 || string.IsNullOrWhiteSpace(sceneName)) return false;
            requestCount++;
            try
            {
                // release the camera owner while this scene is still alive, so the
                // device is stopped and its texture destruction is already pending
                // before the replacement scene opens a new device
                PetRobotCandidateCameraStartupRoute.ReleaseAllForSceneTransition();
                SceneManager.LoadScene(sceneName, LoadSceneMode.Single);
                return true;
            }
            catch
            {
                requestCount--;
                throw;
            }
        }
    }

    public sealed class PetRobotWorldEntryGate
    {
        readonly string worldSceneName;
        readonly string practiceSceneName;
        readonly PetRobotWorldHandoffChannel channel;
        readonly IPetRobotWorldSceneTransition transition;
        readonly Action prepareReturnWatcher;

        public bool PlayerInRange { get; private set; }
        public bool TransitionRequested { get; private set; }
        public string LastError { get; private set; } = string.Empty;

        public PetRobotWorldEntryGate(
            string worldSceneName,
            string practiceSceneName,
            PetRobotWorldHandoffChannel channel,
            IPetRobotWorldSceneTransition transition,
            Action prepareReturnWatcher)
        {
            if (string.IsNullOrWhiteSpace(worldSceneName)) throw new ArgumentException(nameof(worldSceneName));
            if (string.IsNullOrWhiteSpace(practiceSceneName)) throw new ArgumentException(nameof(practiceSceneName));
            this.worldSceneName = worldSceneName;
            this.practiceSceneName = practiceSceneName;
            this.channel = channel ?? throw new ArgumentNullException(nameof(channel));
            this.transition = transition ?? throw new ArgumentNullException(nameof(transition));
            this.prepareReturnWatcher = prepareReturnWatcher ?? throw new ArgumentNullException(nameof(prepareReturnWatcher));
        }

        public void SetPlayerInRange(bool value) => PlayerInRange = value;

        public bool TryEnter(bool interactionPressed, Vector3 playerPosition)
        {
            if (!interactionPressed || !PlayerInRange || TransitionRequested) return false;
            if (!channel.Begin(worldSceneName, practiceSceneName, playerPosition))
            {
                LastError = "handoff channel already has a pending transition";
                return false;
            }
            TransitionRequested = true;
            try
            {
                prepareReturnWatcher();
                if (transition.Request(practiceSceneName)) return true;
                LastError = "practice scene transition request was rejected";
            }
            catch (Exception exception)
            {
                LastError = exception.GetType().Name + ": " + exception.Message;
            }
            channel.CancelUncommitted();
            TransitionRequested = false;
            return false;
        }
    }

    public sealed class PetRobotPracticeReturnGate
    {
        readonly string worldSceneName;
        readonly PetRobotWorldHandoffChannel channel;
        readonly IPetRobotWorldSceneTransition transition;
        PetRobotPracticeCompletionSnapshot completedSnapshot;

        public bool HasCompletedSnapshot => completedSnapshot != null;
        public bool ReturnRequested { get; private set; }

        public PetRobotPracticeReturnGate(
            string worldSceneName,
            PetRobotWorldHandoffChannel channel,
            IPetRobotWorldSceneTransition transition)
        {
            if (string.IsNullOrWhiteSpace(worldSceneName)) throw new ArgumentException(nameof(worldSceneName));
            this.worldSceneName = worldSceneName;
            this.channel = channel ?? throw new ArgumentNullException(nameof(channel));
            this.transition = transition ?? throw new ArgumentNullException(nameof(transition));
        }

        public bool ObserveCompleted(PetRobotPracticeCompletionSnapshot snapshot)
        {
            if (completedSnapshot != null || snapshot == null || !snapshot.SessionCompleted) return false;
            completedSnapshot = snapshot;
            return true;
        }

        public bool TryReturn(bool sessionExited)
        {
            if (!sessionExited || ReturnRequested || completedSnapshot == null) return false;
            if (!channel.PublishCompleted(completedSnapshot)) return false;
            ReturnRequested = true;
            return transition.Request(worldSceneName);
        }
    }
}
