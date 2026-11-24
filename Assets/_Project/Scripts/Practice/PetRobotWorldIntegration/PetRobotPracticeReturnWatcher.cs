using UnityEngine;
using UnityEngine.SceneManagement;
using WordsInSilence.Practice.PetRobotVerticalSlice;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    public sealed class PetRobotPracticeReturnWatcher : MonoBehaviour
    {
        string worldSceneName;
        string practiceSceneName;
        PetRobotWorldHandoffChannel channel;
        IPetRobotWorldSceneTransition transition;
        PetRobotPracticeVerticalSliceBootstrap bootstrap;
        PetRobotSignPracticeSession session;
        PetRobotPracticeReturnGate returnGate;

        bool questMode;
        PetRobotGreetingQuestToken questToken;
        PetRobotPracticeCompletionSnapshot priorCompletion;
        PetRobotGreetingQuestResult committedQuestResult;
        bool questReturnTriggered;

        public static PetRobotPracticeReturnWatcher Create(
            string worldSceneName,
            string practiceSceneName,
            PetRobotWorldHandoffChannel channel)
        {
            var carrier = new GameObject("Step10B_PracticeReturnWatcher");
            DontDestroyOnLoad(carrier);
            var watcher = carrier.AddComponent<PetRobotPracticeReturnWatcher>();
            watcher.Initialize(
                worldSceneName,
                practiceSceneName,
                channel,
                new UnityPetRobotWorldSceneTransition());
            return watcher;
        }

        public static PetRobotPracticeReturnWatcher CreateForGreetingQuest(
            string worldSceneName,
            string practiceSceneName,
            PetRobotWorldHandoffChannel channel,
            PetRobotGreetingQuestToken token,
            PetRobotPracticeCompletionSnapshot priorCompletion)
        {
            var carrier = new GameObject("Step10D_QuestPracticeReturnWatcher");
            DontDestroyOnLoad(carrier);
            var watcher = carrier.AddComponent<PetRobotPracticeReturnWatcher>();
            watcher.InitializeQuest(
                worldSceneName,
                practiceSceneName,
                channel,
                new UnityPetRobotWorldSceneTransition(),
                token,
                priorCompletion);
            return watcher;
        }

        public void Initialize(
            string worldScene,
            string practiceScene,
            PetRobotWorldHandoffChannel handoffChannel,
            IPetRobotWorldSceneTransition sceneTransition)
        {
            worldSceneName = worldScene;
            practiceSceneName = practiceScene;
            channel = handoffChannel;
            transition = sceneTransition;
            returnGate = new PetRobotPracticeReturnGate(worldSceneName, channel, transition);
        }

        public void InitializeQuest(
            string worldScene,
            string practiceScene,
            PetRobotWorldHandoffChannel handoffChannel,
            IPetRobotWorldSceneTransition sceneTransition,
            PetRobotGreetingQuestToken token,
            PetRobotPracticeCompletionSnapshot prior)
        {
            worldSceneName = worldScene;
            practiceSceneName = practiceScene;
            channel = handoffChannel;
            transition = sceneTransition;
            questMode = true;
            questToken = token;
            priorCompletion = prior;
        }

        void OnEnable()
        {
            SceneManager.sceneLoaded += OnSceneLoaded;
        }

        void Update()
        {
            if (!string.Equals(SceneManager.GetActiveScene().name, practiceSceneName, System.StringComparison.Ordinal))
                return;
            TryBindApprovedPracticeSession();
            if (questMode)
                CheckQuestCompletion();
            else
                CaptureCompletedSnapshot();
        }

        void OnDisable()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            UnbindSession();
        }

        void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            UnbindSession();
        }

        void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (string.Equals(scene.name, worldSceneName, System.StringComparison.Ordinal))
                Destroy(gameObject);
        }

        void TryBindApprovedPracticeSession()
        {
            if (session != null) return;
            if (bootstrap == null)
                bootstrap = FindFirstObjectByType<PetRobotPracticeVerticalSliceBootstrap>();
            if (bootstrap == null || bootstrap.Session == null) return;
            session = bootstrap.Session;
            session.Changed += OnSessionChanged;
            if (!questMode)
                session.ExitRequested += OnSessionExitRequested;
        }

        void OnSessionChanged()
        {
            if (questMode)
                CheckQuestCompletion();
            else
                CaptureCompletedSnapshot();
        }

        void CheckQuestCompletion()
        {
            if (questReturnTriggered) return;
            if (committedQuestResult != null)
            {
                RetryQuestTransition();
                return;
            }
            if (bootstrap != null && !string.IsNullOrEmpty(bootstrap.QuestInitializationFailure))
            {
                TriggerQuestReturn(new PetRobotGreetingQuestResult(
                    questToken.Id,
                    TrustedHybridOutcomeKind.TechnicalFailure,
                    bootstrap.QuestInitializationFailure,
                    "InitializationFailure",
                    bootstrap.QuestInitializationFailure));
                return;
            }
            if (session == null || !session.QuestCompleted || !session.QuestAuthoritativeOutcome.HasValue)
                return;
            TrustedHybridOutcomeKind kind = session.QuestAuthoritativeOutcome.Value;
            if (kind != TrustedHybridOutcomeKind.StrictSuccess &&
                kind != TrustedHybridOutcomeKind.LowConfidenceCorrect)
                return;
            string feedback = kind == TrustedHybridOutcomeKind.StrictSuccess
                ? "GREETING Quest 성공"
                : "GREETING Quest 성공 (확신도 낮음)";
            TriggerQuestReturn(new PetRobotGreetingQuestResult(
                questToken.Id, kind, feedback, string.Empty, string.Empty));
        }

        void TriggerQuestReturn(PetRobotGreetingQuestResult result)
        {
            if (questReturnTriggered) return;
            if (committedQuestResult == null)
            {
                if (!channel.CommitQuestReturn(result, priorCompletion))
                    return;
                committedQuestResult = result;
            }
            RetryQuestTransition();
        }

        void RetryQuestTransition()
        {
            if (questReturnTriggered) return;
            try
            {
                if (transition.Request(worldSceneName))
                    questReturnTriggered = true;
            }
            catch { /* transition failed, will retry next frame */ }
        }

        void CaptureCompletedSnapshot()
        {
            if (returnGate == null || returnGate.HasCompletedSnapshot || session == null) return;
            PetRobotPracticeCompletionSnapshot value;
            if (PetRobotPracticeCompletionSnapshot.TryCreate(session, out value))
                returnGate.ObserveCompleted(value);
        }

        void OnSessionExitRequested()
        {
            if (session == null || returnGate == null) return;
            returnGate.TryReturn(session.State == PetRobotPracticeState.Exited);
        }

        void UnbindSession()
        {
            if (session == null) return;
            session.Changed -= OnSessionChanged;
            if (!questMode)
                session.ExitRequested -= OnSessionExitRequested;
            session = null;
        }
    }
}
