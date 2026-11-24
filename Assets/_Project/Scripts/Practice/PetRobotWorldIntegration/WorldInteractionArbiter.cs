// M3_NPC_INPUT_N2 - deterministic arbitration of the shared world interaction key.
// Prepared in the workspace. NOT installed. Supersedes the N1 proposal, which is preserved.
//
// The failure being repaired: PetRobotWorldEntryController.Update and
// InWorldGreetingNpcDialogueController.DriveInput BOTH handle E / Return / Space in
// Update, and neither consumes the press. They are not racing - both run in the same
// frame - so standing at the NPC and pressing E opened the practice scene whatever the
// script execution order was.
//
// WHY N2 EXISTS. N1 modelled ownership as "!dialogueIdle || npcInRange". That is wrong,
// because CloseDialogue deliberately RETAINS state == QuestComplete
// (InWorldGreetingNpcDialogueController.cs:635-636) while hiding the presentation and
// restoring the player and camera. Under N1 the NPC would therefore have claimed the key
// forever after the quest completed, blocking every later pet practice entry even far away
// from the NPC. N2 models the OPEN INTERACTION LIFECYCLE instead, and never resets the
// completed quest or item state to hide the problem.
//
//   dialogueOpen        an interaction is actually open right now
//   interactionFinished QuestComplete has been reached AND its dialogue has been closed
//   npcInRange          the player is inside the REAL greeting trigger
//   consumedThisFrame   a real close already consumed this frame's press
//
// Order independence is structural, not timed:
//   * NPC handler first: a real CloseDialogue consumes the frame, so the entry controller
//     sees consumedThisFrame even though the dialogue is no longer open;
//   * entry controller first: the dialogue is still open, so dialogueOpen already holds.
// Both orders suppress, and nothing depends on who ran first.
//
// The frame marker is an INPUT-CONSUMPTION rule, not a delay. It is set only by a real
// close, it is scoped to exactly one frame, it is never used to postpone anything, and it
// is testable through an injected frame source with no sleeps and no future-frame waits.
//
// What this deliberately does NOT do: it never fabricates in-range state, never substitutes
// distance for the real trigger, never bypasses the enrollment gate, never forces or
// fabricates a quest success, never rebinds a key, and never moves or rewires a scene
// object.
using System.Collections.Generic;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    /// <summary>
    /// Implemented by a component that owns the shared interaction key while its own
    /// interaction context is open.
    /// </summary>
    public interface IWorldInteractionClaimant
    {
        /// <summary>True while this claimant owns E / Return / Space.</summary>
        bool ClaimsInteractionInput { get; }

        /// <summary>Short, non-personal reason, for truthful diagnostics only.</summary>
        string ClaimReason { get; }
    }

    /// <summary>
    /// A tiny shared registry. No Unity type, no scene lookup, no coroutine, no timer.
    /// </summary>
    public static class WorldInteractionArbiter
    {
        static readonly List<IWorldInteractionClaimant> claimants =
            new List<IWorldInteractionClaimant>();

        public static int ClaimantCount { get { return claimants.Count; } }

        public static void Register(IWorldInteractionClaimant claimant)
        {
            if (claimant == null) return;
            if (!claimants.Contains(claimant)) claimants.Add(claimant);
        }

        public static void Unregister(IWorldInteractionClaimant claimant)
        {
            if (claimant == null) return;
            claimants.Remove(claimant);
        }

        /// <summary>
        /// True when any live claimant owns the interaction key. Entries whose Unity object
        /// has been destroyed are pruned rather than consulted, so a scene change cannot
        /// leave a stale claim blocking practice entry.
        /// </summary>
        public static bool TryGetClaim(out string reason)
        {
            reason = string.Empty;
            for (int index = claimants.Count - 1; index >= 0; index--)
            {
                IWorldInteractionClaimant claimant = claimants[index];
                // an interface reference cannot see Unity's destroyed state, so the
                // overloaded == is applied to the underlying object when there is one
                var unityObject = claimant as UnityEngine.Object;
                bool destroyed = !ReferenceEquals(unityObject, null) && unityObject == null;
                if (ReferenceEquals(claimant, null) || destroyed)
                {
                    claimants.RemoveAt(index);
                    continue;
                }
                if (!claimant.ClaimsInteractionInput) continue;
                reason = claimant.ClaimReason ?? "npc_interaction_context";
                return true;
            }
            return false;
        }

        /// <summary>
        /// The single production mapping from NPC interaction lifecycle to key ownership.
        ///
        /// A real close consumes its own frame, so the press that closed a dialogue cannot
        /// also enter practice. Once the interaction is finished - QuestComplete reached
        /// and closed - the key is RELEASED, even while the player is still standing in the
        /// greeting trigger, because the NPC has nothing left to do with it. Otherwise the
        /// NPC owns the key whenever a dialogue is open or the player is inside the real
        /// greeting trigger.
        ///
        /// Owning the key never asserts that a dialogue may start: CanStartDialogue and
        /// TryStartDialogue still enforce the enrollment gate and may truthfully refuse.
        /// </summary>
        public static bool NpcOwnsInteractionKey(bool dialogueOpen, bool npcInRange,
                                                 bool interactionFinished,
                                                 bool consumedThisFrame)
        {
            if (consumedThisFrame) return true;
            if (interactionFinished) return false;
            return dialogueOpen || npcInRange;
        }

#if UNITY_INCLUDE_TESTS
        /// <summary>Test seam only. Production never calls this.</summary>
        public static void ResetForTests()
        {
            claimants.Clear();
        }
#endif
    }
}
