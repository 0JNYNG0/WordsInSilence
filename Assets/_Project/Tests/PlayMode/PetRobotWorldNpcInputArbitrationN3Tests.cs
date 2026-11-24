// M3_NPC_INPUT_N3 - PlayMode tests for interaction-key ownership and the greeting relay.
//
// WHY THIS FIXTURE EXISTS. The N2T EditMode fixture failed all 16 of its tests in SetUp
// because OnEnable does not run for a non-ExecuteAlways MonoBehaviour in EditMode, so
// ClaimantCount was 0 and no test body ever ran. The same limitation failed six original
// Relay_* tests, whose SendMessage calls assert ShouldRunBehaviour() and whose OnDisable
// never fired. The correct answer is to assert lifecycle behaviour in real PlayMode, NOT to
// add ExecuteAlways to production, NOT to register claimants by hand to fake the coverage,
// and NOT to change the old tests. The installed N2 production files and the installed N2T
// EditMode fixture, including its first failure, stay exactly as they are.
//
// The same 16 input and lifecycle assertions are preserved here and now run with real
// OnEnable, OnDisable, OnDestroy and real Unity object state.
//
// Yielding runs Start and Update. That is intended, and it is also why configuration must
// be complete before the first yield: ConfigureForTests is called immediately after
// AddComponent, in the same frame, so PetRobotWorldEntryController.Start restores from the
// test channel rather than the shared one, and the recording transition is in place long
// before any press. No real scene is loaded, no real camera is opened, no learner
// repository is queried, and nothing here builds a quest success into the playable game.
//
// Learner data: ConfigureForTests is given a NULL identity on purpose, so the existing
// GetCurrentCandidateCounts returns CandidateCountSnapshot.Empty and no
// PersonalizationCandidateRepository is constructed or read while the input-only lifecycle
// is driven. One honest caveat, unchanged from the EditMode fixture: Awake still runs the
// production BuildIdentity() before ConfigureForTests replaces it. That is pre-existing
// production behaviour shared with the current suites, it is not introduced here, and
// nothing in this file reads or asserts any learner identifier, profile, count or recording.
//
// Outcome kinds: LowConfidenceCorrect is deliberately quest-completing in the installed
// ApplyOutcome (Step 10T), so the retryable branch uses WrongWord.
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.PlayMode
{
    [TestFixture]
    public sealed class PetRobotWorldNpcInputArbitrationN3Tests
    {
        readonly List<GameObject> owned = new List<GameObject>();

        GameObject npcRoot;
        GameObject npcTarget;
        GameObject entryRoot;
        GameObject player;
        GameObject pet;
        InWorldGreetingNpcDialogueController npcController;
        PetRobotWorldEntryController entryController;
        RecordingTransition transition;
        FakeGateVerifier gate;
        FakeRuntimeOwner runtime;
        FakeCameraController cameraController;
        FakePlayerLock playerLock;
        FakeNpcPresenter npcPresenter;
        float testTime;
        int testFrame;

        GameObject Own(string name)
        {
            var value = new GameObject(name);
            owned.Add(value);
            return value;
        }

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            WorldInteractionArbiter.ResetForTests();
            testTime = 0f;
            testFrame = 100;

            player = Own("Player");
            pet = Own("Pet");
            pet.transform.position = player.transform.position;   // pet gate open

            npcRoot = Own("NpcRoot");
            npcTarget = Own("Npc");
            npcController = npcRoot.AddComponent<InWorldGreetingNpcDialogueController>();
            gate = new FakeGateVerifier();
            runtime = new FakeRuntimeOwner();
            cameraController = new FakeCameraController();
            playerLock = new FakePlayerLock();
            npcPresenter = new FakeNpcPresenter();
            npcController.ConfigureForTests(
                npcTarget.transform, player.transform, 5f, gate,
                (PetRobotLearnerRootIdentity)null,     // Empty candidate counts, no disk read
                runtime, cameraController, playerLock, npcPresenter);
            npcController.SetTimeSourceForTest(() => testTime);
            npcController.SetFrameSourceForTest(() => testFrame);
            npcController.SetPlayerInsideGreetingTrigger(true);

            entryRoot = Own("EntryRoot");
            entryController = entryRoot.AddComponent<PetRobotWorldEntryController>();
            transition = new RecordingTransition();
            // configured before the first yield, so Start cannot use the shared channel or
            // the real scene transition
            entryController.ConfigureForTests(
                player.transform, pet.transform, "TestWorld", "TestPractice", 2.5f,
                new PetRobotWorldHandoffChannel(), transition, delegate { });
            entryController.RefreshInteractionRange();

            yield return null;                       // real Start and one real Update

            Assert.IsTrue(entryController.PlayerInRange,
                "precondition: the pet gate must be open, otherwise suppression is untested");
            Assert.AreEqual(1, WorldInteractionArbiter.ClaimantCount,
                "precondition: real PlayMode OnEnable must have registered the NPC");
            Assert.AreEqual(0, transition.RequestCount,
                "precondition: no scene load may happen just from running frames");
        }

        [UnityTearDown]
        public IEnumerator TearDown()
        {
            for (int index = 0; index < owned.Count; index++)
                if (owned[index] != null) Object.Destroy(owned[index]);
            owned.Clear();
            yield return null;                       // let real OnDisable and OnDestroy run
            WorldInteractionArbiter.ResetForTests();
        }

        // ---- helpers that drive the REAL lifecycle ---------------------------

        void BeginRecognition()
        {
            if (npcController.State == InWorldDialogueState.QuestPrompt)
            {
                testTime += npcController.QuestPromptSeconds + 0.01f;
                npcController.DriveUpdate();
            }
        }

        void SimulateOutcome(TrustedHybridOutcomeKind kind)
        {
            npcController.InjectOutcomeForTest(System.Guid.NewGuid().ToString("N"), kind);
            npcController.DriveUpdate();
            testTime += npcController.AnalysisDisplaySeconds + 0.01f;
            npcController.DriveUpdate();
        }

        void CompleteQuest()
        {
            gate.Ready = true;
            Assert.IsTrue(npcController.TryStartDialogue());
            BeginRecognition();
            SimulateOutcome(TrustedHybridOutcomeKind.StrictSuccess);
            Assert.AreEqual(InWorldDialogueState.QuestComplete, npcController.State);
            Assert.IsTrue(npcController.ItemGranted);
        }

        void AssertPracticeSuppressed(string because)
        {
            Assert.IsFalse(entryController.TryEnterPractice(true), because);
            Assert.AreEqual(0, transition.RequestCount,
                "a suppressed press must request no scene load");
            Assert.IsFalse(entryController.TransitionRequested);
        }

        void AssertPracticeEntered()
        {
            Assert.IsTrue(entryController.TryEnterPractice(true));
            Assert.AreEqual(1, transition.RequestCount);
            Assert.AreEqual("TestPractice", transition.LastScene);
        }

        // ==== the 16 preserved N2T assertions, now in real PlayMode ==========

        [Test]
        public void OpenDialogue_AtStart_OwnsTheKey()
        {
            gate.Ready = true;
            Assert.IsTrue(npcController.TryStartDialogue());
            Assert.AreEqual(InWorldDialogueState.QuestPrompt, npcController.State);
            Assert.IsTrue(npcController.DialogueOpen);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("an open dialogue must own the interaction key");
        }

        [Test]
        public void ActiveRecognition_OwnsTheKey()
        {
            gate.Ready = true;
            npcController.TryStartDialogue();
            BeginRecognition();
            Assert.AreEqual(InWorldDialogueState.TrackingPreparation, npcController.State);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("an active recognition must own the interaction key");
        }

        [Test]
        public void RetryFeedback_OwnsTheKey()
        {
            gate.Ready = true;
            npcController.TryStartDialogue();
            BeginRecognition();
            // WrongWord is the retryable branch; LowConfidenceCorrect is deliberately
            // quest-completing in the installed ApplyOutcome (Step 10T)
            SimulateOutcome(TrustedHybridOutcomeKind.WrongWord);
            Assert.AreEqual(InWorldDialogueState.Feedback, npcController.State);
            Assert.IsTrue(npcController.CanRetry);
            Assert.IsFalse(npcController.ItemGranted);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("a retryable feedback state must own the key");
        }

        [Test]
        public void CompletedButStillOpen_OwnsTheKey()
        {
            CompleteQuest();
            Assert.IsTrue(npcController.DialogueOpen,
                "QuestComplete has not been closed yet, so the interaction is still open");
            Assert.IsFalse(npcController.InteractionFinished);
            AssertPracticeSuppressed("an unclosed QuestComplete must own the key");
        }

        [Test]
        public void CloseThenEntry_InTheSameFrame_CannotEnterPractice()
        {
            CompleteQuest();
            npcController.CloseDialogue();                 // the NPC handler ran first
            Assert.IsFalse(npcController.DialogueOpen);
            Assert.IsTrue(npcController.ConsumedInteractionThisFrame);
            Assert.AreEqual("npc_close_consumed_frame", npcController.ClaimReason);
            AssertPracticeSuppressed("the press that closed the dialogue must not also "
                                     + "enter practice in the same frame");
            Assert.AreEqual("npc_close_consumed_frame",
                            entryController.LastSuppressionReason);
        }

        [Test]
        public void EntryThenClose_InTheSameFrame_CannotEnterPractice()
        {
            CompleteQuest();
            AssertPracticeSuppressed("the entry controller ran first, while the dialogue "
                                     + "was still open");
            npcController.CloseDialogue();
            Assert.AreEqual(0, transition.RequestCount);
        }

        [Test]
        public void ClosedCompletedAndAway_OnALaterFrame_EntersPracticeNormally()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            npcController.SetPlayerInsideGreetingTrigger(false);
            testFrame++;

            Assert.AreEqual(InWorldDialogueState.QuestComplete, npcController.State,
                "the completed quest state must NOT be reset to release the key");
            Assert.IsTrue(npcController.ItemGranted, "the granted item must be preserved");
            Assert.IsTrue(npcController.InteractionFinished);
            Assert.IsFalse(npcController.ClaimsInteractionInput,
                "a finished, closed interaction must release the interaction key");
            AssertPracticeEntered();
        }

        [Test]
        public void ClosedCompletedButStillAtTheNpc_AlsoReleasesTheKey()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            testFrame++;                                   // player never left the trigger

            Assert.IsTrue(npcController.NpcInRange);
            Assert.IsFalse(npcController.ClaimsInteractionInput,
                "a finished NPC has nothing left to do with the key");
            AssertPracticeEntered();
        }

        [Test]
        public void RepeatedCloseWhileAlreadyClosed_DoesNotRenewSuppression()
        {
            CompleteQuest();
            npcController.CloseDialogue();
            testFrame++;
            npcController.CloseDialogue();                 // the DriveInput repeat path

            Assert.IsFalse(npcController.ConsumedInteractionThisFrame,
                "only a real close may consume a frame");
            Assert.IsFalse(npcController.ClaimsInteractionInput);
            AssertPracticeEntered();
        }

        [Test]
        public void IdleAtTheNpc_OwnsTheKeyAndNeverFabricatesSuccess()
        {
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
            Assert.IsTrue(npcController.NpcInRange);
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("standing at the NPC gives the key to the NPC");
            Assert.IsFalse(npcController.ItemGranted);
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
        }

        [Test]
        public void IdleAtTheNpcWithAnUnreadyGate_StillSuppressesAndRefusesTruthfully()
        {
            gate.Ready = false;
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            AssertPracticeSuppressed("a blocked NPC must not fall through to practice");
            Assert.IsFalse(npcController.TryStartDialogue(),
                "the enrollment gate must still refuse");
            Assert.AreEqual(InWorldDialogueState.Idle, npcController.State);
            Assert.IsFalse(npcController.ItemGranted);
        }

        [Test]
        public void IdleAwayFromTheNpc_EntersPracticeNormally()
        {
            npcController.SetPlayerInsideGreetingTrigger(false);
            Assert.IsFalse(npcController.ClaimsInteractionInput);
            AssertPracticeEntered();
        }

        [Test]
        public void NoClaimantRegistered_OrdinaryPetEntry()
        {
            WorldInteractionArbiter.ResetForTests();
            Assert.AreEqual(0, WorldInteractionArbiter.ClaimantCount);
            AssertPracticeEntered();
            Assert.AreEqual(string.Empty, entryController.LastSuppressionReason);
        }

        [UnityTest]
        public IEnumerator DestroyedNpc_IsPrunedAndDoesNotBlockEntry()
        {
            Assert.IsTrue(npcController.ClaimsInteractionInput);
            Object.Destroy(npcRoot);                 // real deferred destroy
            yield return null;                       // real OnDisable and OnDestroy

            string reason;
            Assert.IsFalse(WorldInteractionArbiter.TryGetClaim(out reason),
                "a destroyed NPC must never hold the interaction key");
            AssertPracticeEntered();
        }

        [Test]
        public void DisabledNpc_UnregistersAndDoesNotBlockEntry()
        {
            npcRoot.SetActive(false);                // real OnDisable, synchronously
            Assert.AreEqual(0, WorldInteractionArbiter.ClaimantCount,
                "OnDisable must unregister the claimant");
            AssertPracticeEntered();
        }

        [Test]
        public void MappingReleasesTheKeyOnlyWhenNothingOwnsIt()
        {
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(true, false, false, false));
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, false, false));
            Assert.IsTrue(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, true, true),
                "a real close consumes its frame even after the interaction finished");
            Assert.IsFalse(WorldInteractionArbiter.NpcOwnsInteractionKey(false, true, true, false),
                "a finished interaction releases the key even inside the trigger");
            Assert.IsFalse(WorldInteractionArbiter.NpcOwnsInteractionKey(false, false, false, false));
        }

        // ==== greeting relay lifecycle ======================================
        //
        // The tests below marked "callback invocation" use SendMessage to invoke the real
        // private OnTriggerEnter / OnTriggerExit with a specific Collider argument. That is
        // a CALLBACK INVOCATION, not an observation of an actual physics collision: it
        // proves the relay's argument handling and bookkeeping, not that Unity physics
        // would deliver that callback. Only RealPhysics_TriggerEnterAndExit below observes
        // an actual engine-delivered trigger.

        GreetingNpcTriggerRelay relay;

        GreetingNpcTriggerRelay CreateWiredRelay()
        {
            GameObject host = Own("GreetingTriggerRelayHost");
            var value = host.AddComponent<GreetingNpcTriggerRelay>();
            FieldInfo field = typeof(GreetingNpcTriggerRelay).GetField(
                "dialogueController", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field,
                "GreetingNpcTriggerRelay.dialogueController was not found; the serialized "
                + "field name changed and this test must be updated rather than skipped");
            field.SetValue(value, npcController);
            npcController.SetPlayerInsideGreetingTrigger(false);   // start outside
            return value;
        }

        Collider CreatePlayerCollider(string rootName, bool tagRoot, bool colliderOnChild)
        {
            GameObject root = Own(rootName);
            if (tagRoot) root.tag = "Player";
            GameObject colliderOwner = root;
            if (colliderOnChild)
            {
                colliderOwner = new GameObject(rootName + "Child");
                colliderOwner.transform.SetParent(root.transform, false);
            }
            var box = colliderOwner.AddComponent<BoxCollider>();
            box.isTrigger = false;
            return box;
        }

        [Test]
        public void Relay_ChildColliderUnderPlayerRoot_TakesOwnership()
        {
            relay = CreateWiredRelay();
            Collider child = CreatePlayerCollider("PlayerRootA", true, true);

            relay.gameObject.SendMessage("OnTriggerEnter", child,
                                         SendMessageOptions.DontRequireReceiver);

            Assert.IsTrue(npcController.NpcInRange,
                "the Player tag is on the ROOT, and IsPlayer walks parents");
            Assert.IsTrue(npcController.ClaimsInteractionInput);
        }

        [Test]
        public void Relay_UntaggedCollider_IsIgnored()
        {
            relay = CreateWiredRelay();
            Collider stranger = CreatePlayerCollider("NotThePlayer", false, false);

            relay.gameObject.SendMessage("OnTriggerEnter", stranger,
                                         SendMessageOptions.DontRequireReceiver);

            Assert.IsFalse(npcController.NpcInRange,
                "an untagged collider must never fabricate in-range state");
            Assert.IsFalse(npcController.ClaimsInteractionInput);
        }

        [Test]
        public void Relay_DuplicateEnterThenOneExit_LeavesRange()
        {
            relay = CreateWiredRelay();
            Collider one = CreatePlayerCollider("PlayerRootB", true, false);

            relay.gameObject.SendMessage("OnTriggerEnter", one,
                                         SendMessageOptions.DontRequireReceiver);
            relay.gameObject.SendMessage("OnTriggerEnter", one,
                                         SendMessageOptions.DontRequireReceiver);
            Assert.IsTrue(npcController.NpcInRange);

            relay.gameObject.SendMessage("OnTriggerExit", one,
                                         SendMessageOptions.DontRequireReceiver);
            Assert.IsFalse(npcController.NpcInRange,
                "a duplicate enter is tracked by collider id, so one exit clears it");
        }

        [Test]
        public void Relay_TwoColliders_OneExit_KeepsOwnership()
        {
            relay = CreateWiredRelay();
            Collider first = CreatePlayerCollider("PlayerRootC", true, false);
            Collider second = CreatePlayerCollider("PlayerRootD", true, true);

            relay.gameObject.SendMessage("OnTriggerEnter", first,
                                         SendMessageOptions.DontRequireReceiver);
            relay.gameObject.SendMessage("OnTriggerEnter", second,
                                         SendMessageOptions.DontRequireReceiver);
            relay.gameObject.SendMessage("OnTriggerExit", first,
                                         SendMessageOptions.DontRequireReceiver);

            Assert.IsTrue(npcController.NpcInRange,
                "one of two tracked colliders left, so the player is still inside");

            relay.gameObject.SendMessage("OnTriggerExit", second,
                                         SendMessageOptions.DontRequireReceiver);
            Assert.IsFalse(npcController.NpcInRange);
        }

        [Test]
        public void Relay_ExitWithoutEnter_DoesNotChangeOwnership()
        {
            relay = CreateWiredRelay();
            Collider tracked = CreatePlayerCollider("PlayerRootE", true, false);
            Collider unknown = CreatePlayerCollider("PlayerRootF", true, false);

            relay.gameObject.SendMessage("OnTriggerEnter", tracked,
                                         SendMessageOptions.DontRequireReceiver);
            relay.gameObject.SendMessage("OnTriggerExit", unknown,
                                         SendMessageOptions.DontRequireReceiver);

            Assert.IsTrue(npcController.NpcInRange,
                "an exit for a collider that never entered must be ignored");
        }

        [Test]
        public void Relay_Disabled_ClearsOwnership()
        {
            relay = CreateWiredRelay();
            Collider inside = CreatePlayerCollider("PlayerRootG", true, false);
            relay.gameObject.SendMessage("OnTriggerEnter", inside,
                                         SendMessageOptions.DontRequireReceiver);
            Assert.IsTrue(npcController.NpcInRange);

            relay.gameObject.SetActive(false);       // real OnDisable in PlayMode

            Assert.IsFalse(npcController.NpcInRange,
                "a disabled relay must release ownership rather than strand it");
            Assert.IsFalse(npcController.ClaimsInteractionInput);
        }

        /// <summary>
        /// The only test here that observes an ACTUAL engine-delivered trigger callback.
        /// Synthetic and self-owned: a kinematic trigger box and a Player-tagged collider
        /// that starts overlapping it, then is moved clearly outside. No game scene, no
        /// prefab and no project physics setting is touched.
        /// </summary>
        [UnityTest]
        public IEnumerator RealPhysics_TriggerEnterAndExit()
        {
            relay = CreateWiredRelay();
            relay.transform.position = new Vector3(1000f, 0f, 1000f);   // away from anything
            var trigger = relay.gameObject.AddComponent<BoxCollider>();
            trigger.isTrigger = true;
            trigger.size = new Vector3(3f, 2f, 3f);
            var body = relay.gameObject.AddComponent<Rigidbody>();
            body.isKinematic = true;
            body.useGravity = false;

            GameObject mover = Own("PhysicsPlayerRoot");
            mover.tag = "Player";
            var moverBox = mover.AddComponent<BoxCollider>();
            moverBox.size = new Vector3(0.5f, 0.5f, 0.5f);
            mover.transform.position = relay.transform.position;        // overlapping

            for (int frame = 0; frame < 8 && !npcController.NpcInRange; frame++)
                yield return new WaitForFixedUpdate();
            Assert.IsTrue(npcController.NpcInRange,
                "an actual physics trigger enter must set ownership");

            mover.transform.position = relay.transform.position + new Vector3(50f, 0f, 0f);
            for (int frame = 0; frame < 8 && npcController.NpcInRange; frame++)
                yield return new WaitForFixedUpdate();
            Assert.IsFalse(npcController.NpcInRange,
                "an actual physics trigger exit must release ownership");
        }

        // --- test doubles, shaped like the existing V2 suite's ---

        sealed class RecordingTransition : IPetRobotWorldSceneTransition
        {
            public int RequestCount { get; private set; }
            public string LastScene { get; private set; }

            public bool Request(string sceneName)
            {
                RequestCount++;
                LastScene = sceneName;
                return true;
            }
        }

        sealed class FakeGateVerifier : IQuestGateVerifier
        {
            public bool Ready;
            public string Failure = string.Empty;
            public string LastFailure => Failure;
            public bool VerifyOverallEnrollmentGateReady() => Ready;
        }

        sealed class FakeRuntimeOwner : IInWorldRecognitionRuntimeOwner
        {
            public long Generation = 1;
            public bool Disposed;
            public bool FirstObservationReady;

            public bool IsActive => !Disposed;
            public long FrozenGeneration => Generation;
            public PetRobotSignPracticeSession Session => null;
            public PetRobotNeutralAutoCaptureCoordinator Coordinator => null;
            public string InitializationFailure => string.Empty;

            public bool RequestAttempt() => true;
            public bool RetrySession() => true;

            public InWorldRecognitionReadinessSnapshot SnapshotReadiness()
            {
                return new InWorldRecognitionReadinessSnapshot(
                    true, true, false, FirstObservationReady,
                    RawMediaPipeNeutralAutoSegmenterState.Inactive, string.Empty);
            }

            public void Dispose() { Disposed = true; }
        }

        sealed class FakeCameraController : IInWorldDialogueCameraController
        {
            public bool Saved;
            public bool Restored;
            public void SaveState() { Saved = true; Restored = false; }
            public void EnterDialogueCamera() { }
            public void RestoreState() { Restored = true; }
        }

        sealed class FakePlayerLock : IInWorldDialoguePlayerLock
        {
            public bool Locked;
            public void Lock() { Locked = true; }
            public void Unlock() { Locked = false; }
        }

        sealed class FakeNpcPresenter : IInWorldNpcPresenter
        {
            public int GreetingCount;
            public int ConfusedCount;
            public int GiveItemCount;
            public void TriggerGreeting() { GreetingCount++; }
            public void TriggerConfused() { ConfusedCount++; }
            public void TriggerGiveItem() { GiveItemCount++; }
        }
    }
}
