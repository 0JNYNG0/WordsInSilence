using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public sealed class PetRobotPortfolioQuestController : MonoBehaviour
    {
        [SerializeField] PetRobotWorldEntryController entryController;
        [SerializeField] Behaviour worldTechnicalUi;
        [SerializeField] Transform playerTransform;
        [SerializeField] Transform educationNpcTransform;
        [SerializeField] float interactionRange = 2.5f;
        [SerializeField] string practiceSceneName = "PetRobotSignPracticeVerticalSlice";

        PetRobotPortfolioQuestFlow flow;
        PetRobotWorldHandoffChannel channel;
        IPetRobotWorldSceneTransition transition;
        bool questTransitionRequested;
        bool questResultConsumed;
        bool lastConsumedResultWasTechnicalFailure;
        string lastInitializationFailureDetail = string.Empty;
        IQuestGateVerifier gateVerifier;
        PetRobotLearnerRootIdentity productionIdentity;
        string lastQuestTransitionError = string.Empty;

        public PetRobotPortfolioQuestFlow Flow => flow;
        public PetRobotPortfolioQuestState State => flow == null
            ? PetRobotPortfolioQuestState.AwaitingPractice : flow.State;
        public PetRobotPortfolioQuestSnapshot Completion => flow == null ? null : flow.Snapshot;
        public string Objective => flow == null ? string.Empty : flow.Objective;
        public bool NpcInRange => flow != null && flow.NpcInRange;
        public bool CanOpenNpcDialogue => flow != null && flow.CanReportToNpc;
        public bool PracticeEntryBlocked => flow != null && !flow.CanEnterPractice;
        public bool IsAwaitingGreetingAttempt => flow != null &&
            flow.State == PetRobotPortfolioQuestState.AwaitingGreetingAttempt;
        public bool QuestTransitionRequested => questTransitionRequested;
        public string QuestFeedbackText => flow == null ? string.Empty : flow.QuestFeedbackText;
        public string LastQuestTransitionError => lastQuestTransitionError;
        public PetRobotLearnerRootIdentity ProductionIdentity => productionIdentity;
        public bool CanRetryAfterInitializationFailure =>
            lastConsumedResultWasTechnicalFailure &&
            flow != null &&
            flow.State == PetRobotPortfolioQuestState.AwaitingGreetingAttempt;

        void Awake()
        {
            flow = new PetRobotPortfolioQuestFlow();
            channel = PetRobotWorldHandoffChannel.Shared;
            transition = new UnityPetRobotWorldSceneTransition();
            productionIdentity = BuildProductionIdentity();
#if !UNITY_EDITOR
            if (productionIdentity == null)
            {
                PetRobotVerticalSlice.PetRobotM3RouteNotice.ShowBlocked();
                enabled = false;
                return;
            }
#endif
            gateVerifier = new ProductionQuestGateVerifier(productionIdentity);
            if (worldTechnicalUi != null) worldTechnicalUi.enabled = false;
        }

        void Update()
        {
            DriveUpdateLogic();
            if (Input.GetKeyDown(KeyCode.E) || Input.GetKeyDown(KeyCode.Return) ||
                Input.GetKeyDown(KeyCode.Space))
                TryOpenNpcDialogue(true);
        }

        public void DriveUpdateLogic()
        {
            TryConsumeQuestResultOnReturn();
            CaptureCompletionFromWorldOnce();
            RefreshNpcRange();
        }

        public bool CaptureCompletionFromWorldOnce()
        {
            if (flow == null) flow = new PetRobotPortfolioQuestFlow();
            if (entryController == null ||
                flow.State != PetRobotPortfolioQuestState.AwaitingPractice ||
                flow.Snapshot != null)
                return false;

            PetRobotPracticeCompletionSnapshot completion = entryController.LastCompletion;
            if (completion == null) return false;
            if (!flow.ObserveCompleted(completion)) return false;
            entryController.enabled = false;
            return true;
        }

        public void RefreshNpcRange()
        {
            if (flow == null || playerTransform == null || educationNpcTransform == null) return;
            flow.SetNpcInRange(Vector3.Distance(
                playerTransform.position, educationNpcTransform.position) <= interactionRange);
        }

        public bool TryOpenNpcDialogue(bool interactionPressed)
        {
            return flow != null && flow.TryOpenDialogue(interactionPressed);
        }

        public bool CompleteDialogue(bool explicitConfirmation)
        {
            if (flow == null || !flow.TryCompleteDialogue(explicitConfirmation))
                return false;
            return TryBeginGreetingQuest();
        }

        public bool CompleteGreetingQuest()
        {
            return flow != null && flow.TryCompleteGreetingQuest();
        }

        public bool TryRetryAfterInitializationFailure()
        {
            if (!CanRetryAfterInitializationFailure) return false;
            lastConsumedResultWasTechnicalFailure = false;
            questResultConsumed = false;
            questTransitionRequested = false;
            lastQuestTransitionError = !string.IsNullOrEmpty(lastInitializationFailureDetail)
                ? "initialization failure: " + lastInitializationFailureDetail
                : "initialization failure";
            flow.RollbackToDialogueOpen();
            return true;
        }

        bool TryBeginGreetingQuest()
        {
            if (channel == null || flow == null || flow.Snapshot == null) return false;
            PetRobotPracticeCompletionSnapshot prior = entryController != null
                ? entryController.LastCompletion : null;
            if (prior == null || !prior.SessionCompleted) return false;

            if (gateVerifier == null)
            {
                lastQuestTransitionError = "Gate not ready: no verifier configured (fail-closed)";
                flow.RollbackToDialogueOpen();
                return false;
            }
            if (!gateVerifier.VerifyOverallEnrollmentGateReady())
            {
                lastQuestTransitionError = "Gate not ready: " + gateVerifier.LastFailure;
                flow.RollbackToDialogueOpen();
                return false;
            }

            string worldScene = entryController != null
                ? GetWorldSceneName() : string.Empty;
            if (string.IsNullOrWhiteSpace(worldScene) ||
                string.IsNullOrWhiteSpace(practiceSceneName))
                return false;

            var token = new PetRobotGreetingQuestToken();
            Vector3 playerPos = playerTransform != null ? playerTransform.position : Vector3.zero;
            if (!channel.BeginGreetingQuest(worldScene, practiceSceneName, playerPos, token, prior, productionIdentity))
            {
                flow.RollbackToDialogueOpen();
                return false;
            }

            PetRobotPracticeReturnWatcher watcher = null;
            try
            {
                watcher = PetRobotPracticeReturnWatcher.CreateForGreetingQuest(
                    worldScene, practiceSceneName, channel, token, prior);
                if (!transition.Request(practiceSceneName))
                {
                    lastQuestTransitionError = "scene transition request rejected";
                    if (watcher != null) Object.Destroy(watcher.gameObject);
                    channel.CancelUncommitted();
                    flow.RollbackToDialogueOpen();
                    return false;
                }
                questTransitionRequested = true;
                lastQuestTransitionError = string.Empty;
                return true;
            }
            catch (System.Exception exception)
            {
                lastQuestTransitionError = exception.GetType().Name + ": " + exception.Message;
                if (watcher != null) Object.Destroy(watcher.gameObject);
                channel.CancelUncommitted();
                flow.RollbackToDialogueOpen();
                return false;
            }
        }

        PetRobotLearnerRootIdentity BuildProductionIdentity()
        {
            if (PetRobotVerticalSlice.DemoParticipantSession.IsActive)
            {
                var demo = PetRobotVerticalSlice.DemoParticipantSession.Current;
                return new PetRobotLearnerRootIdentity(demo.LearnerId, demo.CandidateRoot);
            }

            string learnerId = "local_learner";
            if (entryController != null)
            {
                var bootstrap = entryController.GetComponent<PetRobotPracticeVerticalSliceBootstrap>();
                if (bootstrap != null && !string.IsNullOrWhiteSpace(bootstrap.LearnerProfileId))
                    learnerId = bootstrap.LearnerProfileId;
            }
            string explicitRoot = null;
#if UNITY_EDITOR
            string cleanRoot = PetRobotDeveloperCleanCandidateRootOverride.ResolveForBootstrap();
            if (!string.IsNullOrWhiteSpace(cleanRoot))
                explicitRoot = cleanRoot;
#else
            // same resolver, same learner id, so the seams cannot disagree silently
            if (!PetRobotVerticalSlice.PetRobotPlayerCandidateRootOverride
                    .TryResolveForPlayer(learnerId, out explicitRoot))
                return null;
#endif
            return new PetRobotLearnerRootIdentity(learnerId, explicitRoot);
        }

        bool TryConsumeQuestResultOnReturn()
        {
            if (questResultConsumed || channel == null || flow == null) return false;
            if (!channel.HasQuestResult) return false;
            PetRobotPracticeCompletionSnapshot prior = entryController != null
                ? entryController.LastCompletion : null;
            if (prior == null || !prior.SessionCompleted) return false;
            PetRobotGreetingQuestResult result;
            if (!channel.TryConsumeQuestResult(out result)) return false;
            questResultConsumed = true;
            flow.RestoreAwaitingGreetingAttempt(prior);
            flow.ObserveQuestResult(result);
            if (result.Kind == TrustedHybridOutcomeKind.TechnicalFailure)
            {
                lastConsumedResultWasTechnicalFailure = true;
                lastInitializationFailureDetail = result.FailureMessage;
            }
            if (result.IsSuccess)
                flow.TryCompleteGreetingQuest();
            return true;
        }

        string GetWorldSceneName()
        {
            return gameObject.scene.name;
        }

        public void ConfigureForTests(
            PetRobotWorldEntryController worldEntry,
            Behaviour technicalUi,
            Transform player,
            Transform educationNpc,
            float range,
            PetRobotWorldHandoffChannel testChannel,
            IPetRobotWorldSceneTransition testTransition,
            string testPracticeScene,
            IQuestGateVerifier testGateVerifier = null)
        {
            entryController = worldEntry;
            worldTechnicalUi = technicalUi;
            playerTransform = player;
            educationNpcTransform = educationNpc;
            interactionRange = range;
            channel = testChannel ?? PetRobotWorldHandoffChannel.Shared;
            transition = testTransition ?? new UnityPetRobotWorldSceneTransition();
            practiceSceneName = testPracticeScene ?? "PetRobotSignPracticeVerticalSlice";
            gateVerifier = testGateVerifier;
            flow = new PetRobotPortfolioQuestFlow();
            questTransitionRequested = false;
            questResultConsumed = false;
            lastConsumedResultWasTechnicalFailure = false;
            lastInitializationFailureDetail = string.Empty;
            if (worldTechnicalUi != null) worldTechnicalUi.enabled = false;
        }

        public void ConfigureForTests(
            PetRobotWorldEntryController worldEntry,
            Behaviour technicalUi,
            Transform player,
            Transform educationNpc,
            float range)
        {
            ConfigureForTests(worldEntry, technicalUi, player, educationNpc, range, null, null, null);
        }
    }
}
