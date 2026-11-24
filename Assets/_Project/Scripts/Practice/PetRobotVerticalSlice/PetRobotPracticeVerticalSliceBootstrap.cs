using System;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;
using WordsInSilence.Practice.FullHandEnrollment;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public sealed class PetRobotPracticeVerticalSliceBootstrap : MonoBehaviour
    {
        [SerializeField] string learnerProfileId = "local_learner";
        [SerializeField] string candidateHarnessResource =
            "Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical";
        [SerializeField] bool hideTechnicalPanel = true;
        [SerializeField] string returnSceneName = string.Empty;
        [SerializeField] PetRobotPracticePresenter robotPresenter;
        [SerializeField] PetRobotPracticeUiPresenter uiPresenter;
        [SerializeField] GuidedSignPracticePresentationController guidedPresentation;

        GameObject candidateHarnessInstance;
        PetRobotSignPracticeSession session;
        PetRobotCandidateCameraStartupRoute candidateCameraRoute;
        PetRobotNeutralAutoCaptureCoordinator neutralAutoCaptureCoordinator;

        public PetRobotSignPracticeSession Session => session;
        public PetRobotNeutralAutoCaptureCoordinator NeutralAutoCaptureCoordinator =>
            neutralAutoCaptureCoordinator;
        public string LearnerProfileId => learnerProfileId;
        public bool ActivePersonalProfileMutationEnabled => false;
        public bool OverallPersonalizationGateReady =>
            session != null && session.OverallEnrollmentGateReady;
        public bool IsQuestMode { get; private set; }
        public string QuestInitializationFailure { get; private set; }

        public void ConfigureCandidateCameraRoute(PetRobotCandidateCameraStartupRoute route)
        {
            if (session != null)
                throw new InvalidOperationException("candidate camera route cannot change after session initialization");
            candidateCameraRoute = route;
        }

        void Start()
        {
            try
            {
                if (robotPresenter == null) robotPresenter = GetComponent<PetRobotPracticePresenter>();
                if (uiPresenter == null) uiPresenter = GetComponent<PetRobotPracticeUiPresenter>();
                if (guidedPresentation == null)
                    guidedPresentation = GetComponent<GuidedSignPracticePresentationController>();

                if (DemoParticipantSession.IsActive)
                    learnerProfileId = DemoParticipantSession.Current.LearnerId;

                var channel = PetRobotWorldHandoffChannel.Shared;
                PetRobotLearnerRootIdentity questIdentity = null;
                if (channel.PendingMode == PetRobotHandoffMode.GreetingQuest &&
                    channel.PendingQuestToken != null)
                {
                    IsQuestMode = true;
                    questIdentity = channel.PendingLearnerIdentity;
                    if (questIdentity == null || string.IsNullOrWhiteSpace(questIdentity.LearnerId))
                        throw new InvalidOperationException("quest mode requires a non-null learner identity from handoff channel");
                    // a handoff from a previous participant must not be inherited: fail here,
                    // before any repository, harness, camera or recognition work is started
                    if (DemoParticipantSession.IsActive &&
                        !string.Equals(questIdentity.LearnerId,
                            DemoParticipantSession.Current.LearnerId,
                            System.StringComparison.Ordinal))
                        throw new InvalidOperationException(
                            "quest handoff learner identity does not match the active participant");
                    learnerProfileId = questIdentity.LearnerId;
                }

#if !UNITY_EDITOR
                // validate BEFORE anything is instantiated: no harness, no camera, no
                // recognition and no candidate write may start in a blocked state
                string playerRoot;
                if (!PetRobotPlayerCandidateRootOverride.TryResolveForPlayer(
                        learnerProfileId, out playerRoot))
                {
                    PetRobotM3RouteNotice.ShowBlocked();
                    enabled = false;
                    return;
                }
#endif
                GameObject harnessPrefab = Resources.Load<GameObject>(candidateHarnessResource);
                if (harnessPrefab == null) throw new InvalidOperationException("approved candidate harness prefab is missing");
                candidateHarnessInstance = Instantiate(harnessPrefab);
                candidateHarnessInstance.name = "ApprovedCandidateRecognitionRoute";
                PersonalLiveRecognitionHarness harness =
                    candidateHarnessInstance.GetComponent<PersonalLiveRecognitionHarness>();
                if (harness == null) throw new InvalidOperationException("approved candidate harness component is missing");
                if (candidateCameraRoute == null || !candidateCameraRoute.isActiveAndEnabled)
                    throw new InvalidOperationException("the active candidate-only camera owner is not explicitly wired");
                harness.SetTechnicalPanelVisible(!hideTechnicalPanel);
                candidateCameraRoute.BindObservationBridge(harness.ObservationBridge);

                var adapter = new PersonalLivePracticeRecognitionAdapter(harness);
#if UNITY_EDITOR
                string resolvedDeveloperRoot = null;
#endif
                PersonalizationCandidateRepository repository;
                if (DemoParticipantSession.IsActive)
                {
                    // active demo participant wins: a fresh GUID namespace, empty by construction
                    repository = new PersonalizationCandidateRepository(
                        learnerProfileId, DemoParticipantSession.Current.CandidateRoot);
                }
                else if (IsQuestMode)
                {
                    if (questIdentity.HasExplicitRoot)
                        repository = new PersonalizationCandidateRepository(
                            learnerProfileId, questIdentity.ExplicitRoot);
                    else
                        repository = new PersonalizationCandidateRepository(learnerProfileId);
                }
                else
                {
#if UNITY_EDITOR
                    resolvedDeveloperRoot =
                        PetRobotDeveloperCleanCandidateRootOverride.ResolveForBootstrap();
                    if (!string.IsNullOrWhiteSpace(resolvedDeveloperRoot))
                        repository = new PersonalizationCandidateRepository(
                            learnerProfileId, resolvedDeveloperRoot);
                    else
                        repository = new PersonalizationCandidateRepository(learnerProfileId);
#else
                    // the root the flag selected, if any; absent flag keeps the default
                    repository = string.IsNullOrEmpty(playerRoot)
                        ? new PersonalizationCandidateRepository(learnerProfileId)
                        : new PersonalizationCandidateRepository(learnerProfileId, playerRoot);
#endif
                }
                FullHandEnrollmentBinding fullHandBinding = null;
                FullHandParticipantCandidateStore fullHandStore = null;
                if (FullHandEnrollmentRuntime.ModeActive)
                {
                    string storeCode, storeDetail;
                    if (!FullHandEnrollmentRuntime.TryOpenForCurrentParticipant(
                            questIdentity, out fullHandBinding, out fullHandStore,
                            out storeCode, out storeDetail))
                        throw new InvalidOperationException(
                            "the full-hand route is enabled but unavailable: " + storeCode +
                            (string.IsNullOrEmpty(storeDetail) ? string.Empty : " " + storeDetail));
                }

                if (IsQuestMode && fullHandStore != null)
                {
                    // F28: no legacy profiles, no legacy 3/3/3 inventory and no frozen 16-point
                    // hybrid session on this route; the store-backed gate decides entry.
                    string questGateFailure;
                    if (!FullHandEnrollmentQuestGate.Verify(fullHandStore, out questGateFailure))
                        throw new InvalidOperationException(
                            "quest mode requires the collected full-hand practice records: " +
                            questGateFailure);
                    session = new PetRobotSignPracticeSession(
                        adapter, repository, channel.PendingQuestToken, fullHandBinding,
                        fullHandStore, candidateCameraRoute);
                }
                else if (IsQuestMode)
                {
                    TrustedHybridRuntimeSession rawSession =
                        CreateQuestHybridSession(repository);
                    var hybridSession = new TrustedHybridRecognitionSessionAdapter(rawSession);
                    session = new PetRobotSignPracticeSession(
                        adapter, repository, hybridSession,
                        channel.PendingQuestToken, candidateCameraRoute);
                }
                else if (fullHandStore != null)
                {
                    session = new PetRobotSignPracticeSession(
                        adapter, repository, fullHandBinding, fullHandStore, null,
                        candidateCameraRoute);
                }
                else
                {
                    session = new PetRobotSignPracticeSession(adapter, repository, null, candidateCameraRoute);
                }
#if UNITY_EDITOR
                if (!string.IsNullOrWhiteSpace(resolvedDeveloperRoot))
                {
                    TrustedCandidateInventory inventory = repository.TrustedInventory;
                    Debug.Log(
                        PetRobotDeveloperCleanCandidateRootOverride.LogPrefix +
                        " bootstrap root=" + repository.RootDirectory +
                        " learner=" + repository.LearnerId +
                        " candidate_total=" + repository.Counts.TotalCandidateCount +
                        " trusted=" + inventory.CountForWord("GREETING") + "/" +
                        inventory.CountForWord("LOVE") + "/" +
                        inventory.CountForWord("THANK_YOU"));
                }
#endif
                session.ExitRequested += OnExitRequested;
                var segmentationSettings = PetRobotNeutralAutoSegmentationSettings.Default;
                var profileService = new PetRobotNeutralCalibrationProfileService(
                    learnerProfileId, DemoParticipantSession.ProfilesRootOrNull);
                var segmenter = new RawMediaPipeNeutralAutoSegmenter(
                    learnerProfileId, segmentationSettings);
                var ring = new RawMediaPipeObservationRingBuffer(segmentationSettings.PreRollSeconds);
                var autoObservationSource = new Step9CObservationSourceAdapter(harness.ObservationBridge);
                neutralAutoCaptureCoordinator = new PetRobotNeutralAutoCaptureCoordinator(
                    session,
                    profileService,
                    segmenter,
                    ring,
                    autoObservationSource,
                    candidateCameraRoute);
                if (guidedPresentation != null)
                    guidedPresentation.Bind(
                        session,
                        robotPresenter,
                        candidateCameraRoute,
                        harness.ObservationBridge,
                        neutralAutoCaptureCoordinator);
                else
                    uiPresenter?.Bind(session, robotPresenter, candidateCameraRoute);
            }
            catch (Exception exception)
            {
                string message = exception.GetType().Name + ": " + exception.Message;
                // private local player log only: full type, message and stack. Nothing
                // here is exported, and the UI message below is unchanged.
                Debug.LogException(exception, this);
                if (IsQuestMode)
                    QuestInitializationFailure = message;
                guidedPresentation?.ShowInitializationFailure(message);
                uiPresenter?.ShowInitializationFailure(message);
            }
        }

        TrustedHybridRuntimeSession CreateQuestHybridSession(
            PersonalizationCandidateRepository repository)
        {
            ITrustedPersonalizationCandidateRepository trusted =
                repository as ITrustedPersonalizationCandidateRepository;
            if (trusted == null)
                throw new InvalidOperationException("quest mode requires a trusted candidate repository");
            TrustedCandidateInventory inventory = trusted.ReloadTrustedAndVerify();
            if (!inventory.OverallEnrollmentGateReady)
                throw new InvalidOperationException(
                    "quest mode requires OverallEnrollmentGateReady but Gate is " +
                    inventory.CountForWord("GREETING") + "/" +
                    inventory.CountForWord("LOVE") + "/" +
                    inventory.CountForWord("THANK_YOU"));
            string templateDirectory = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "Templates");
            string thresholdPath = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "RuntimeGate",
                "frozen_threshold_config.json");
            RuntimeRecognitionService globalService =
                RuntimeRecognitionService.Load(templateDirectory, thresholdPath);
            if (DemoParticipantSession.IsActive)
            {
                // Participant-only prototypes over the same public global service; the fixed
                // personal profile below is never read in this route.
                return WordsInSilence.Recognition.LiveInput.DemoParticipantLiveRuntimeFactory
                    .CreateQuestSession(globalService, trusted, DemoParticipantSession.Current);
            }
            string fixedProfileDirectory = Path.Combine(
                Application.streamingAssetsPath, "RecognitionCoreV1", "PersonalProfiles",
                ApprovedPersonalMultiPrototypeLiveRuntimeFactory.ProfileDirectoryName);
            var viewService = new TrustedPersonalizationActiveViewService(
                globalService, fixedProfileDirectory, trusted);
            TrustedActiveViewBuildResult refreshResult = viewService.TryRefresh();
            if (!refreshResult.Succeeded)
                throw new InvalidOperationException(
                    "quest active view refresh failed: " +
                    refreshResult.FailureType + ": " + refreshResult.FailureMessage);
            TrustedPersonalizationActiveView view = refreshResult.ActiveView;
            foreach (string word in new[] { "GREETING", "LOVE", "THANK_YOU" })
            {
                if (!view.CandidateCaptureIds.ContainsKey(word) ||
                    view.CandidateCaptureIds[word].Count < 3)
                    throw new InvalidOperationException(
                        "quest active view missing qualified candidates for " + word);
            }
            return viewService.AcquireSession();
        }

        void OnDestroy()
        {
            neutralAutoCaptureCoordinator?.Dispose();
            neutralAutoCaptureCoordinator = null;
            if (session != null)
            {
                session.ExitRequested -= OnExitRequested;
                session.Dispose();
            }
        }

        void OnExitRequested()
        {
            if (!string.IsNullOrWhiteSpace(returnSceneName))
            {
                PetRobotCandidateCameraStartupRoute.ReleaseAllForSceneTransition();
                SceneManager.LoadScene(returnSceneName);
            }
            else
                robotPresenter?.ShowCompleted();
        }
    }
}
