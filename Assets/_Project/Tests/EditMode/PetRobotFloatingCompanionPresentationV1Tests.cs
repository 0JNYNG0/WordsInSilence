using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotPortfolioQuest;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotFloatingCompanionPresentationV1Tests
    {
        const string CompanionSource =
            "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotFloatingCompanionController.cs";
        const string ContractsSource =
            "Assets/_Project/Scripts/Practice/PetRobotPortfolioQuest/PetRobotPortfolioQuestContracts.cs";
        const string WorldEntrySource =
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldEntryController.cs";
        const string PlaceholderSource =
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlaceholderVisuals.cs";
        const string PresentationPrefab =
            "Assets/_Project/Resources/VerticalSlice/PetRobotPortfolioQuestPresentation.prefab";
        const string AnimatedWrapper =
            "Assets/_Project/Resources/VerticalSlice/PetRobotAnimatedVisualWrapper.prefab";
        const string PortfolioScene =
            "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity";

        [Test]
        public void PlayerRelativeOffset_PositiveSidePlacesAnchorOnRight()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor);
                Vector3 target = controller.TargetPositionForTests();
                Assert.AreEqual(1.15f, target.x, 1e-6f);
                Assert.AreEqual(1.25f, target.y, 1e-6f);
                Assert.AreEqual(-0.15f, target.z, 1e-6f);
            });
        }

        [Test]
        public void PlayerRelativeOffset_NegativeSidePlacesAnchorOnLeft()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor, -1.15f);
                Assert.Less(controller.TargetPositionForTests().x, player.position.x);
            });
        }

        [Test]
        public void AnchorHeight_IsFixedPlayerRelativeOffset()
        {
            WithCompanion((controller, player, anchor) =>
            {
                player.position = new Vector3(3f, 4f, 5f);
                controller.ConfigureForTests(player, anchor, 1.15f, 1.25f);
                Assert.AreEqual(5.25f, controller.TargetPositionForTests().y, 1e-6f);
            });
        }

        [Test]
        public void CompanionSource_HasNoProceduralHoverFieldOrYOscillation()
        {
            string source = ReadProject(CompanionSource);
            Assert.IsFalse(source.Contains("hoverAmplitude"));
            Assert.IsFalse(source.Contains("hoverFrequency"));
            Assert.IsFalse(source.Contains("bobAmplitude"));
            Assert.IsFalse(source.Contains("bobFrequency"));
            Assert.IsFalse(source.Contains("Mathf.Sin"));
            Assert.IsFalse(source.Contains("localPosition"));
        }

        [Test]
        public void NormalDistance_UsesSmoothFollowWithoutTeleport()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor);
                Vector3 target = controller.TargetPositionForTests();
                anchor.position = target + Vector3.left;
                Vector3 before = anchor.position;
                Assert.IsTrue(controller.StepForTests(0.1f));
                Assert.Greater(anchor.position.x, before.x);
                Assert.Less(anchor.position.x, target.x);
            });
        }

        [Test]
        public void BeyondSnapDistance_ReturnsImmediatelyToTarget()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor);
                anchor.position = new Vector3(20f, 20f, 20f);
                Assert.IsTrue(controller.StepForTests(0.1f));
                AssertVector(controller.TargetPositionForTests(), anchor.position);
            });
        }

        [Test]
        public void FrameAfterSnap_ReturnsToSmoothFollow()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor);
                anchor.position = new Vector3(20f, 20f, 20f);
                Assert.IsTrue(controller.StepForTests(0.1f));
                float snappedX = anchor.position.x;
                player.position = Vector3.right;
                float nextTargetX = controller.TargetPositionForTests().x;
                Assert.IsTrue(controller.StepForTests(0.1f));
                Assert.Greater(anchor.position.x, snappedX);
                Assert.Less(anchor.position.x, nextTargetX);
            });
        }

        [Test]
        public void NullPlayerReference_IsSafeNoOp()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(null, anchor);
                Vector3 before = anchor.position;
                Assert.IsFalse(controller.StepForTests(0.1f));
                AssertVector(before, anchor.position);
            });
        }

        [Test]
        public void ZeroInvalidDeltaAndSettings_NeverWriteNonFiniteTransform()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor, float.NaN, float.PositiveInfinity,
                    float.NegativeInfinity, float.NaN, -8f, float.NaN);
                Vector3 before = anchor.position;
                Assert.IsFalse(controller.StepForTests(0f));
                Assert.IsFalse(controller.StepForTests(float.NaN));
                AssertVector(before, anchor.position);
                Assert.IsTrue(controller.StepForTests(0.1f));
                Assert.IsFalse(float.IsNaN(anchor.position.x));
                Assert.IsFalse(float.IsInfinity(anchor.position.y));
                Assert.IsFalse(float.IsNaN(anchor.position.z));
            });
        }

        [Test]
        public void DefaultAnchorOffset_IsInsideInteractionRangeTwoPointFive()
        {
            float magnitude = new Vector3(1.15f, 1.25f, -0.15f).magnitude;
            Assert.Less(magnitude, 2.5f);
        }

        [Test]
        public void ExistingPetRobotInteractionAnchorItself_IsMoved()
        {
            WithCompanion((controller, player, anchor) =>
            {
                controller.ConfigureForTests(player, anchor);
                controller.transform.position = new Vector3(9f, 9f, 9f);
                anchor.position = new Vector3(20f, 20f, 20f);
                Assert.IsTrue(controller.StepForTests(0.1f));
                AssertVector(controller.TargetPositionForTests(), anchor.position);
                AssertVector(new Vector3(9f, 9f, 9f), controller.transform.position);
            });
        }

        [Test]
        public void SceneAndPrefab_HaveOneCompanionOwnerAndExplicitPlayerAnchorWiring()
        {
            string prefab = ReadProject(PresentationPrefab);
            string scene = ReadProject(PortfolioScene).Replace("\r\n", "\n");
            string placeholder = ReadProject(PlaceholderSource).Replace("\r\n", "\n");
            Assert.AreEqual(1, Count(prefab, "guid: 10d00000000000000000000000000001"));
            Assert.AreEqual(2, Count(scene, "propertyPath: playerTransform"));
            Assert.AreEqual(1, Count(scene, "propertyPath: companionAnchor"));
            Assert.AreEqual(3, Count(scene, "objectReference: {fileID: 100100010}"));
            Assert.AreEqual(2, Count(scene, "objectReference: {fileID: 100100011}"));
            Assert.AreEqual(1, Count(placeholder,
                "void Awake()\n        {\n            if (!enabled) return;\n"));
            Assert.AreEqual(1, Count(scene,
                "- target: {fileID: 11400002, guid: 10b00000000000000000000000000010, type: 3}\n" +
                "      propertyPath: m_Enabled\n" +
                "      value: 0\n"));
        }

        [Test]
        public void FbxAnimatorControllerClipAndVisualChildHierarchy_AreIntact()
        {
            string prefab = ReadProject(PresentationPrefab);
            string wrapper = ReadProject(AnimatedWrapper);
            string controller = ReadProject("Assets/GameAssets/petrobot_model_animation/Terrain.controller");
            string clip = ReadProject(
                "Assets/GameAssets/petrobot_model_animation/Meshy_Cosmic Companion_Mesh_0_Anim_Blink.anim");
            Assert.AreEqual(1, Count(prefab, "guid: 72d9e61f9f2f7ff45880ba5850543e6b"));
            Assert.AreEqual(1, Count(prefab, "guid: 10d00000000000000000000000000010"));
            Assert.GreaterOrEqual(Count(wrapper, "guid: 50573fbff0bc00c418eb7319a40093d3"), 1);
            Assert.AreEqual(1, Count(wrapper, "guid: a414e66cfd19823419a0f9a1954db8d3"));
            Assert.AreEqual(0, Count(wrapper, "--- !u!1001 &100100000"));
            Assert.AreEqual(1, Count(wrapper, "--- !u!1001 &100200000"));
            Assert.AreEqual(1, Count(wrapper, "  m_PrefabInstance: {fileID: 100200000}"));
            Assert.AreEqual(1, Count(wrapper,
                "m_SourcePrefab: {fileID: 100100000, guid: 50573fbff0bc00c418eb7319a40093d3, type: 3}"));
            StringAssert.Contains("propertyPath: m_Controller", wrapper);
            StringAssert.Contains("propertyPath: m_ApplyRootMotion", wrapper);
            StringAssert.Contains("value: 0", wrapper);
            StringAssert.Contains("guid: 8a56dbc85c0c9b74381607d7d13f5e52", controller);
            StringAssert.Contains("attribute: m_LocalPosition.y", clip);
        }

        [Test]
        public void ForbiddenReferencesAreAbsentAndObjectiveAndManualInputRemainExplicit()
        {
            string source = ReadProject(CompanionSource);
            string contracts = ReadProject(ContractsSource);
            string entry = ReadProject(WorldEntrySource);
            foreach (string token in new[] { "Input", "SceneManager", "WebCam", "MediaPipe",
                         "RecognitionCore", "MaskAwareDtw", "CandidateRepository", "NavMesh" })
                Assert.IsFalse(source.Contains(token), token);
            StringAssert.Contains(
                "동행 애완 로봇과 대화해 수어 연습을 시작하세요 (E / Enter / Space)", contracts);
            StringAssert.Contains("KeyCode.E", entry);
            StringAssert.Contains("KeyCode.Return", entry);
            StringAssert.Contains("KeyCode.Space", entry);
        }

        static void WithCompanion(Action<PetRobotFloatingCompanionController, Transform, Transform> body)
        {
            var owner = new GameObject("Step10DCompanionOwner");
            var player = new GameObject("Step10DPlayer");
            var anchor = new GameObject("Step10DInteractionAnchor");
            try
            {
                body(owner.AddComponent<PetRobotFloatingCompanionController>(),
                    player.transform, anchor.transform);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(anchor);
                UnityEngine.Object.DestroyImmediate(player);
                UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        static void AssertVector(Vector3 expected, Vector3 actual)
        {
            Assert.AreEqual(expected.x, actual.x, 1e-5f);
            Assert.AreEqual(expected.y, actual.y, 1e-5f);
            Assert.AreEqual(expected.z, actual.z, 1e-5f);
        }

        static int Count(string source, string token)
        {
            int count = 0;
            int offset = 0;
            while ((offset = source.IndexOf(token, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += token.Length;
            }
            return count;
        }

        static string ReadProject(string relativePath) => File.ReadAllText(
            Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath))
            .Replace("\r\n", "\n");
    }
}
