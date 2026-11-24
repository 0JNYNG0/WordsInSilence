using System.Collections.Generic;
using UnityEngine;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    /// <summary>
    /// F59. Hides the player avatar and the floating MOA companion while an NPC quest
    /// dialogue is open, because both occlude the NPC close-up.
    ///
    /// Only Renderer.forceRenderingOff is touched: no GameObject is deactivated, no
    /// Animator, collider, movement, material or camera state is changed, so collision,
    /// following and recognition behave exactly as before. The previous bool of every
    /// renderer is snapshotted and restored EXACTLY, so a renderer that was already
    /// forced off stays off after the restore.
    ///
    /// The actors are discovered once, at dialogue entry; nothing is searched, allocated
    /// or updated per frame. Each dialogue controller instance owns its own snapshot.
    /// </summary>
    public sealed class InWorldDialogueActorVisibility
    {
        readonly List<Renderer> hiddenRenderers = new List<Renderer>();
        readonly List<bool> previousForceRenderingOff = new List<bool>();
        readonly List<Renderer> scratch = new List<Renderer>();
        readonly HashSet<int> seenRenderers = new HashSet<int>();
        readonly HashSet<int> seenRoots = new HashSet<int>();
        bool hidden;

        /// <summary>True while this snapshot is holding actors hidden.</summary>
        public bool IsHidden => hidden;

        /// <summary>Number of renderers currently held hidden by this snapshot.</summary>
        public int HiddenRendererCount => hiddenRenderers.Count;

        /// <summary>
        /// Hides the player avatar and every MOA companion that follows exactly this player
        /// in the same scene. Repeated calls while already hidden do nothing, so the first
        /// snapshot is never overwritten with already-hidden values.
        /// </summary>
        public void Hide(Transform playerTransform)
        {
            if (hidden) return;
            CollectCompanionAnchors(playerTransform, scratch2);
            Hide(playerTransform, scratch2);
            scratch2.Clear();
        }

        readonly List<Transform> scratch2 = new List<Transform>();

        /// <summary>
        /// Hides the given player and the explicitly supplied extra roots. The test seam of
        /// the method above; production passes discovered companion anchors.
        /// </summary>
        public void Hide(Transform playerTransform, IList<Transform> extraRoots)
        {
            if (hidden) return;

            seenRoots.Clear();
            seenRenderers.Clear();
            hiddenRenderers.Clear();
            previousForceRenderingOff.Clear();

            AddRoot(playerTransform);
            if (extraRoots != null)
                for (int i = 0; i < extraRoots.Count; i++) AddRoot(extraRoots[i]);

            for (int i = 0; i < hiddenRenderers.Count; i++)
            {
                Renderer renderer = hiddenRenderers[i];
                if (renderer == null) continue;
                renderer.forceRenderingOff = true;
            }
            hidden = true;
        }

        /// <summary>
        /// Restores the exact previous forceRenderingOff of every snapshotted renderer.
        /// Safe to call when nothing is hidden and when renderers were destroyed meanwhile.
        /// </summary>
        public void Restore()
        {
            for (int i = 0; i < hiddenRenderers.Count; i++)
            {
                Renderer renderer = hiddenRenderers[i];
                if (renderer == null) continue;   // destroyed meanwhile: nothing to restore
                renderer.forceRenderingOff = previousForceRenderingOff[i];
            }
            hiddenRenderers.Clear();
            previousForceRenderingOff.Clear();
            seenRenderers.Clear();
            seenRoots.Clear();
            hidden = false;
        }

        void AddRoot(Transform root)
        {
            if (root == null) return;
            if (!seenRoots.Add(root.GetInstanceID())) return;

            scratch.Clear();
            root.GetComponentsInChildren(true, scratch);   // inactive children included
            for (int i = 0; i < scratch.Count; i++)
            {
                Renderer renderer = scratch[i];
                if (renderer == null) continue;
                if (!seenRenderers.Add(renderer.GetInstanceID())) continue;
                hiddenRenderers.Add(renderer);
                previousForceRenderingOff.Add(renderer.forceRenderingOff);
            }
            scratch.Clear();
        }

        /// <summary>
        /// One scoped search, at dialogue entry only: the companions that follow exactly this
        /// player and live in the same scene. CompanionAnchor is used rather than the
        /// controller transform, which may also parent unrelated world objects.
        /// </summary>
        public static void CollectCompanionAnchors(Transform playerTransform,
                                                   IList<Transform> into)
        {
            if (into == null || playerTransform == null) return;
            PetRobotFloatingCompanionController[] companions =
                Object.FindObjectsOfType<PetRobotFloatingCompanionController>(true);
            if (companions == null) return;
            for (int i = 0; i < companions.Length; i++)
            {
                PetRobotFloatingCompanionController companion = companions[i];
                if (companion == null) continue;
                if (companion.PlayerTransform != playerTransform) continue;
                if (companion.gameObject.scene != playerTransform.gameObject.scene) continue;
                Transform anchor = companion.CompanionAnchor;
                if (anchor != null) into.Add(anchor);
            }
        }
    }
}
