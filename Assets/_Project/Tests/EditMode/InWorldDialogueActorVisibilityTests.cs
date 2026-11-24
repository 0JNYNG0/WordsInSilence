using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// F59. Only the player avatar and the matching MOA companion are hidden while the NPC
    /// dialogue is open, and the exact previous renderer state comes back afterwards.
    /// No recognition, camera, timing or model behaviour is exercised here.
    /// </summary>
    public sealed class InWorldDialogueActorVisibilityTests
    {
        readonly List<GameObject> spawned = new List<GameObject>();

        [TearDown]
        public void TearDown()
        {
            foreach (GameObject item in spawned)
                if (item != null) Object.DestroyImmediate(item);
            spawned.Clear();
        }

        GameObject NewRendererObject(string name, Transform parent = null)
        {
            var go = new GameObject(name, typeof(MeshRenderer), typeof(MeshFilter));
            if (parent != null) go.transform.SetParent(parent, false);
            else spawned.Add(go);
            return go;
        }

        [Test]
        public void HidesOnlyTheGivenActorsAndLeavesTheNpcUntouched()
        {
            GameObject player = NewRendererObject("Player");
            GameObject playerChild = NewRendererObject("PlayerMesh", player.transform);
            GameObject companion = NewRendererObject("CompanionAnchor");
            GameObject npc = NewRendererObject("Npc");

            var visibility = new InWorldDialogueActorVisibility();
            visibility.Hide(player.transform, new List<Transform> { companion.transform });

            Assert.IsTrue(visibility.IsHidden);
            Assert.IsTrue(player.GetComponent<Renderer>().forceRenderingOff);
            Assert.IsTrue(playerChild.GetComponent<Renderer>().forceRenderingOff);
            Assert.IsTrue(companion.GetComponent<Renderer>().forceRenderingOff);
            Assert.IsFalse(npc.GetComponent<Renderer>().forceRenderingOff,
                "the NPC must stay visible: it is the subject of the close-up");
        }

        [Test]
        public void RestoresTheExactPreviousForceRenderingOffAndEnabledValues()
        {
            GameObject player = NewRendererObject("Player");
            GameObject alreadyHidden = NewRendererObject("HiddenPart", player.transform);
            Renderer hiddenRenderer = alreadyHidden.GetComponent<Renderer>();
            hiddenRenderer.forceRenderingOff = true;
            Renderer playerRenderer = player.GetComponent<Renderer>();
            playerRenderer.enabled = false;   // a disabled renderer must stay disabled

            var visibility = new InWorldDialogueActorVisibility();
            visibility.Hide(player.transform, null);
            visibility.Restore();

            Assert.IsFalse(visibility.IsHidden);
            Assert.IsTrue(hiddenRenderer.forceRenderingOff,
                "an originally forced-off renderer must remain forced off");
            Assert.IsFalse(playerRenderer.forceRenderingOff);
            Assert.IsFalse(playerRenderer.enabled,
                "Renderer.enabled is never written by the visibility helper");
        }

        [Test]
        public void RepeatedHideAndRestoreAreIdempotent()
        {
            GameObject player = NewRendererObject("Player");
            Renderer renderer = player.GetComponent<Renderer>();

            var visibility = new InWorldDialogueActorVisibility();
            visibility.Hide(player.transform, null);
            int count = visibility.HiddenRendererCount;
            visibility.Hide(player.transform, null);   // must not re-snapshot the hidden value

            Assert.AreEqual(count, visibility.HiddenRendererCount);
            Assert.IsTrue(renderer.forceRenderingOff);

            visibility.Restore();
            visibility.Restore();
            Assert.IsFalse(renderer.forceRenderingOff);
            Assert.AreEqual(0, visibility.HiddenRendererCount);
        }

        [Test]
        public void DeduplicatesRootsAndSurvivesDestroyedRenderers()
        {
            GameObject player = NewRendererObject("Player");
            GameObject child = NewRendererObject("Child", player.transform);

            var visibility = new InWorldDialogueActorVisibility();
            visibility.Hide(player.transform,
                new List<Transform> { player.transform, child.transform });
            Assert.AreEqual(2, visibility.HiddenRendererCount,
                "the same renderer must be snapshotted once");

            Object.DestroyImmediate(child);
            Assert.DoesNotThrow(() => visibility.Restore());
            Assert.IsFalse(player.GetComponent<Renderer>().forceRenderingOff);
        }

        [Test]
        public void CollectsOnlyCompanionsBoundToThisPlayer()
        {
            GameObject player = NewRendererObject("Player");
            GameObject otherPlayer = NewRendererObject("OtherPlayer");

            var mineHost = new GameObject("Companion");
            spawned.Add(mineHost);
            GameObject mineAnchor = NewRendererObject("Anchor", mineHost.transform);
            mineHost.AddComponent<PetRobotFloatingCompanionController>()
                .ConfigureForTests(player.transform, mineAnchor.transform);

            var otherHost = new GameObject("OtherCompanion");
            spawned.Add(otherHost);
            GameObject otherAnchor = NewRendererObject("Anchor", otherHost.transform);
            otherHost.AddComponent<PetRobotFloatingCompanionController>()
                .ConfigureForTests(otherPlayer.transform, otherAnchor.transform);

            var anchors = new List<Transform>();
            InWorldDialogueActorVisibility.CollectCompanionAnchors(player.transform, anchors);

            // compared by instance id, not by collection equality: Transform implements
            // IEnumerable over its children, so two childless anchors compare structurally
            // equal and a containment assert cannot tell them apart
            Assert.AreEqual(1, anchors.Count,
                "exactly the companion bound to this player must be collected");
            Assert.AreEqual(mineAnchor.transform.GetInstanceID(), anchors[0].GetInstanceID(),
                "the collected anchor must be this player's companion anchor");
            Assert.AreNotEqual(otherAnchor.transform.GetInstanceID(), anchors[0].GetInstanceID(),
                "another player's companion anchor must never be collected");
        }

        [Test]
        public void ControllerRestoresVisibilityWhenDisabledWithoutAClose()
        {
            GameObject player = NewRendererObject("Player");
            GameObject npc = NewRendererObject("Npc");
            Renderer playerRenderer = player.GetComponent<Renderer>();

            var host = new GameObject("Dialogue");
            spawned.Add(host);
            var controller = host.AddComponent<InWorldGreetingNpcDialogueController>();
            controller.ConfigureForTests(npc.transform, player.transform, 3f,
                new AlwaysReadyGate(), null, null, null, null, null);
            controller.SetPlayerInsideGreetingTrigger(true);

            Assert.IsTrue(controller.TryStartDialogue());
            Assert.IsTrue(playerRenderer.forceRenderingOff);
            Assert.IsFalse(npc.GetComponent<Renderer>().forceRenderingOff);

            // minimal reflection seam: the normal close is deliberately NOT called
            typeof(InWorldGreetingNpcDialogueController)
                .GetMethod("OnDisable", BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(controller, null);

            Assert.IsFalse(playerRenderer.forceRenderingOff,
                "OnDisable must restore the avatar even without a normal close");
        }

        sealed class AlwaysReadyGate : IQuestGateVerifier
        {
            public bool VerifyOverallEnrollmentGateReady() => true;
            public string LastFailure => string.Empty;
        }
    }
}
