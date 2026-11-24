using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    public sealed class PetRobotWorldEntryController : MonoBehaviour
    {
        [SerializeField] Transform playerTransform;
        [SerializeField] Transform petRobotTransform;
        [SerializeField] string worldSceneName = "PetRobotWorldEntryReturnVerticalSlice";
        [SerializeField] string practiceSceneName = "PetRobotSignPracticeVerticalSlice";
        [SerializeField] float interactionRange = 2.5f;

        PetRobotWorldHandoffChannel channel;
        IPetRobotWorldSceneTransition transition;
        PetRobotWorldEntryGate gate;
        Action prepareWatcher;

        public bool PlayerInRange => gate != null && gate.PlayerInRange;
        public bool CanEnterPractice => gate != null && gate.PlayerInRange && !gate.TransitionRequested;
        public bool TransitionRequested => gate != null && gate.TransitionRequested;
        public string LastTransitionError => gate == null ? string.Empty : gate.LastError;

        /// <summary>
        /// Why the last interaction press did NOT enter practice, empty when it was not
        /// suppressed. Diagnostics only: it carries no personal value and changes no
        /// gameplay.
        /// </summary>
        public string LastSuppressionReason { get; private set; } = string.Empty;
        public PetRobotPracticeCompletionSnapshot LastCompletion { get; private set; }
        public string CompletionMessage => LastCompletion == null ? string.Empty : LastCompletion.WorldSummaryMessage;

        void Awake()
        {
            ConfigureRuntimeServices();
        }

        void Start()
        {
            RestoreWorldReturn();
        }

        void Update()
        {
            if (gate == null || playerTransform == null || petRobotTransform == null) return;
            gate.SetPlayerInRange(Vector3.Distance(playerTransform.position, petRobotTransform.position) <= interactionRange);
            if (Input.GetKeyDown(KeyCode.E) || Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.Space))
                TryEnterPractice(true);
        }

        public bool TryEnterPractice(bool interactionPressed)
        {
            if (gate == null || playerTransform == null) return false;
            if (interactionPressed)
            {
                // deterministic arbitration: another context may own this key right now.
                // This is a state query plus a one-frame consumption rule, not an input
                // latch and not a delay, so it does not depend on which Update ran first.
                // An empty registry reproduces the previous behaviour exactly, so other
                // scenes are unaffected.
                string claimReason;
                if (WorldInteractionArbiter.TryGetClaim(out claimReason))
                {
                    LastSuppressionReason = claimReason;
                    return false;
                }
                LastSuppressionReason = string.Empty;
            }
            return gate.TryEnter(interactionPressed, playerTransform.position);
        }

        public void RefreshInteractionRange()
        {
            if (gate == null || playerTransform == null || petRobotTransform == null) return;
            gate.SetPlayerInRange(Vector3.Distance(playerTransform.position, petRobotTransform.position) <= interactionRange);
        }

        public void ConfigureForTests(
            Transform player,
            Transform petRobot,
            string worldScene,
            string practiceScene,
            float range,
            PetRobotWorldHandoffChannel handoffChannel,
            IPetRobotWorldSceneTransition sceneTransition,
            Action watcherPreparation)
        {
            playerTransform = player;
            petRobotTransform = petRobot;
            worldSceneName = worldScene;
            practiceSceneName = practiceScene;
            interactionRange = range;
            channel = handoffChannel;
            transition = sceneTransition;
            prepareWatcher = watcherPreparation;
            BuildGate();
        }

        public bool RestoreWorldReturn()
        {
            if (channel == null) channel = PetRobotWorldHandoffChannel.Shared;
            PetRobotWorldReturnPayload payload;
            if (channel.TryConsumeCompleted(worldSceneName, out payload))
            {
                if (playerTransform != null) playerTransform.position = payload.PlayerPosition;
                LastCompletion = payload.Completion;
                return true;
            }
            Vector3 position;
            if (channel.TryConsumeIncomplete(worldSceneName, out position))
            {
                if (playerTransform != null) playerTransform.position = position;
                LastCompletion = null;
            }
            return false;
        }

        void ConfigureRuntimeServices()
        {
            channel = PetRobotWorldHandoffChannel.Shared;
            transition = new UnityPetRobotWorldSceneTransition();
            prepareWatcher = () => PetRobotPracticeReturnWatcher.Create(
                worldSceneName, practiceSceneName, channel);
            BuildGate();
        }

        void BuildGate()
        {
            if (channel == null || transition == null || prepareWatcher == null) return;
            gate = new PetRobotWorldEntryGate(
                worldSceneName, practiceSceneName, channel, transition, prepareWatcher);
        }
    }
}
