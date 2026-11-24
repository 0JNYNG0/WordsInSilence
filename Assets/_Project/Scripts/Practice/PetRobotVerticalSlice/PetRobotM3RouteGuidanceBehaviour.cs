// M3_ROUTE_SOURCE_PREP_A1 (final pre-install fix) - draft, NOT installed.
//
// Its own file, matching the class name, per the Unity script naming rule
// (https://docs.unity3d.com/kr/6000.0/Manual/naming-scripts.html): a file whose name
// matches a static class cannot reliably carry a MonoBehaviour for AddComponent.
//
// Read-only stage display plus a state-change-only log. It reads the canonical NPC
// controller and the world entry controller and writes NOTHING back: no session state,
// no outcome, no item, no progress, no camera, no input. Completion is displayed only
// when the canonical controller itself reports QuestComplete AND ItemGranted.
//
// Honest wording: the practice session restores existing enrollment
// (PetRobotSignPracticeSession.RestorePersistedEnrollment, lines 478-497 - a trusted
// 3/3/3 working copy makes it Completed at once). That is continuation of existing
// training, never new training, and no text here claims otherwise.
using System.Globalization;
using UnityEngine;
using UnityEngine.SceneManagement;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Sequencing, scoped to the M3 route variant only.
    ///
    /// PetRobotCandidateCameraStartupRoute.Start (line 112) calls StartCameraOnceAsync
    /// unconditionally, and it is a separate MonoBehaviour that the NPC's blocked Awake
    /// guard cannot stop. Unity runs every active Awake before any Start, and this
    /// behaviour declares an explicit execution order while the camera route declares
    /// none (default 0), so this Start runs first and can disable those components
    /// before their own Start would open a device. Their existing OnDisable owns
    /// StopAndDisposeOnce; nothing here disposes a camera itself.
    ///
    /// This is a claim about THIS variant scene, whose InWorldDialogueSystem root and
    /// candidateCameraRoute are active and whose serialized learner id was verified the
    /// same. It is not a guarantee for arbitrary scenes or for a route that some other
    /// script enables later.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    public sealed class PetRobotM3RouteGuidanceBehaviour : MonoBehaviour
    {
        const string StageStart =
            "로봇 근처에서 E를 눌러 기존 학습 상태를 확인하세요.";
        const string StageReturned =
            "연습에서 돌아왔습니다. NPC에게 다가가 인사 동작을 해 보세요.";
        const string StageRetry =
            "이번 시도는 인정되지 않았습니다. R을 눌러 다시 시도할 수 있습니다.";
        const string StageComplete =
            "퀘스트 완료. 아이템을 받았습니다.";

        InWorldGreetingNpcDialogueController npc;
        PetRobotWorldEntryController entry;
        string lastSignature;
        bool practiceReturnSeen;
        bool halted;

        void Start()
        {
            npc = FindFirstObjectByType<InWorldGreetingNpcDialogueController>(
                FindObjectsInactive.Exclude);
            entry = FindFirstObjectByType<PetRobotWorldEntryController>(
                FindObjectsInactive.Exclude);
            if (PetRobotPlayerCandidateRootOverride.Blocked) { Halt(); return; }
            PetRobotM3RouteNotice.ShowStage(StageStart);
            Log("start");
        }

        void Update()
        {
            if (halted) return;
            if (PetRobotPlayerCandidateRootOverride.Blocked) { Halt(); return; }
            if (entry != null && entry.LastCompletion != null) practiceReturnSeen = true;

            string signature = Signature();
            if (signature == lastSignature) return;      // state-change only
            lastSignature = signature;
            PetRobotM3RouteNotice.ShowStage(StageText());
            Log("state_change");
        }

        /// <summary>
        /// Fail closed at any point, including after a later learner-identity mismatch:
        /// close an already-running dialogue through its own normal path, stop the entry
        /// affordance, show the error and do nothing further.
        /// </summary>
        void Halt()
        {
            if (halted) return;
            halted = true;
            PetRobotM3RouteNotice.ShowBlocked();
            if (npc != null)
            {
                npc.CloseDialogue();                     // existing normal disposal
                npc.enabled = false;
            }
            if (entry != null) entry.enabled = false;
            int cameraRoutesDisabled = DisableCameraRoutes();
            Log("blocked_halt cameras_disabled=" + cameraRoutesDisabled);
            enabled = false;
        }

        /// <summary>
        /// Disables the scene's existing camera routes. Called from Start, which runs
        /// before their own Start because of this class's execution order, so a blocked
        /// entry stops the device from being opened at all. Disposal stays with their
        /// existing OnDisable path; no camera state is faked, read or reported here.
        /// </summary>
        int DisableCameraRoutes()
        {
            int disabled = 0;
            PetRobotCandidateCameraStartupRoute[] routes =
                FindObjectsByType<PetRobotCandidateCameraStartupRoute>(
                    FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int index = 0; index < routes.Length; index++)
            {
                if (routes[index] == null || !routes[index].enabled) continue;
                routes[index].enabled = false;
                disabled++;
            }
            return disabled;
        }

        string StageText()
        {
            if (npc == null) return StageStart;
            if (npc.State == InWorldDialogueState.QuestComplete && npc.ItemGranted)
                return StageComplete;
            if (npc.State == InWorldDialogueState.Feedback && npc.CanRetry)
                return StageRetry;
            if (practiceReturnSeen) return StageReturned;
            return StageStart;
        }

        string Signature()
        {
            return (npc == null ? "no-npc" : npc.State + "|" + npc.ItemGranted + "|"
                    + npc.CanRetry) + "|" + practiceReturnSeen;
        }

        void Log(string reason)
        {
            // scene, frame, time and the canonical read-only fields. No attempt id, no
            // root path, no array, no learner value.
            Debug.Log(string.Format(CultureInfo.InvariantCulture,
                "[M3Route] {0} scene={1} frame={2} t={3:0.###} state={4} item={5} "
                + "retry={6} practice_return_seen={7} blocked={8}",
                reason, SceneManager.GetActiveScene().name, Time.frameCount,
                Time.realtimeSinceStartup,
                npc == null ? "none" : npc.State.ToString(),
                npc != null && npc.ItemGranted, npc != null && npc.CanRetry,
                practiceReturnSeen, PetRobotPlayerCandidateRootOverride.Blocked));
        }
    }
}
