using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PlayerRunAnimationV1Tests
    {
        const string ControllerPath =
            "Assets/GameAssets/Player_model_animation_meshy/Player.controller";
        const string PlayerVisualPath =
            "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Wave_One_Hand_withSkin.fbx";
        const string IdleClipPath =
            "Assets/GameAssets/Player_model_animation_meshy/Standing W_Briefcase Idle.fbx";
        const string RunClipPath =
            "Assets/GameAssets/Player_model_animation_meshy/Meshy_AI_OriginalVer_Player_biped_Animation_Running_withSkin.fbx";
        const string WorldScenePath =
            "Assets/_Project/Scenes/VerticalSlice/PetRobotPortfolioQuestVerticalSlice.unity";
        const string PracticeScenePath =
            "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity";
        const string MovementSourcePath =
            "Assets/_Project/Scripts/Practice/PetRobotWorldIntegration/PetRobotWorldPlayerController.cs";

        [Test]
        public void CurrentWorldPlayer_ResolvesExactlyOneAnimatorAndApprovedControllerBinding()
        {
            GameObject playerVisual = AssetDatabase.LoadAssetAtPath<GameObject>(PlayerVisualPath);
            Assert.IsNotNull(playerVisual);
            Assert.AreEqual(1, playerVisual.GetComponentsInChildren<Animator>(true).Length);

            string scene = ReadProject(WorldScenePath);
            StringAssert.Contains(
                "objectReference: {fileID: 9100000, guid: 5824cec5f72de9e459e864865536c381, type: 2}",
                scene);
            StringAssert.Contains(
                "m_SourcePrefab: {fileID: 100100000, guid: f44a2ab6c6a52dd41b44e781d7f0c07a, type: 3}",
                scene);
            StringAssert.Contains("propertyPath: m_ApplyRootMotion\n      value: 0", scene);
        }

        [Test]
        public void PlayerController_DeclaresExactlyOneBoolIsRunParameter()
        {
            AnimatorController controller = LoadController();
            AnimatorControllerParameter[] parameters =
                controller.parameters.Where(value => value.name == "IsRun").ToArray();
            Assert.AreEqual(1, parameters.Length);
            Assert.AreEqual(AnimatorControllerParameterType.Bool, parameters[0].type);
            Assert.IsFalse(parameters[0].defaultBool);
        }

        [Test]
        public void NonZeroMovementInput_SetsIsRunTrueInTheSameMoveCall()
        {
            WithPlayer((player, animator, host) =>
            {
                Vector3 positionBefore = host.transform.position;
                player.Move(Vector2.right, 0f);
                Assert.IsTrue(animator.GetBool("IsRun"));
                Assert.AreEqual(positionBefore, host.transform.position);
            });
        }

        [Test]
        public void ZeroMovementInput_SetsIsRunFalseInTheSameMoveCall()
        {
            WithPlayer((player, animator, host) =>
            {
                player.Move(Vector2.up, 0f);
                Assert.IsTrue(animator.GetBool("IsRun"));
                player.Move(Vector2.zero, 0f);
                Assert.IsFalse(animator.GetBool("IsRun"));
            });
        }

        [Test]
        public void IdleRunTransitions_UseIsRunWithoutExitTimeAndPreserveBlend()
        {
            AnimatorController controller = LoadController();
            AnimatorState idle = StateForMotion(controller, IdleClipPath);
            AnimatorState run = StateForMotion(controller, RunClipPath);

            AssertTransition(idle, run, AnimatorConditionMode.If);
            AssertTransition(run, idle, AnimatorConditionMode.IfNot);
        }

        [Test]
        public void IdleAndRunMotions_KeepApprovedIdentityAndLoopingImport()
        {
            AnimatorController controller = LoadController();
            AnimatorState idle = StateForMotion(controller, IdleClipPath);
            AnimatorState run = StateForMotion(controller, RunClipPath);

            Assert.AreEqual("mixamo.com", idle.motion.name);
            Assert.AreEqual("Armature|Armature|running|baselayer", run.motion.name);
            Assert.IsTrue(((AnimationClip)idle.motion).isLooping);
            Assert.IsTrue(((AnimationClip)run.motion).isLooping);
            AssertImporterLoop(IdleClipPath, idle.motion.name);
            AssertImporterLoop(RunClipPath, run.motion.name);
        }

        [Test]
        public void MovementSource_IsRunWriteAndMovementSemanticsRemainFrozen()
        {
            Assert.AreEqual(
                "dea546acf35c345ac3d33e3ce8df2d322b1d14e9643425eddae6f6a70067fb4d",
                Sha256(ProjectPath(MovementSourcePath)));
            string source = ReadProject(MovementSourcePath);
            StringAssert.Contains("input.sqrMagnitude > 0f", source);
            StringAssert.Contains("playerAnimator.SetBool(\"IsRun\", value);", source);
            StringAssert.Contains("characterController.Move(direction * moveSpeed * Mathf.Max(0f, deltaTime));", source);
            StringAssert.Contains("transform.forward = direction;", source);
            Assert.AreEqual(1, Count(source, "SetBool(\"IsRun\""));
        }

        [Test]
        public void PlayerMoaPlacementAndPracticeScene_RemainFrozen()
        {
            Assert.AreEqual(
                "2bbc0a28973c8d3b4fb53d4d35e65dfb605c10d220b28ab1cc05be6e239535ff",
                Sha256(ProjectPath(WorldScenePath)));
            Assert.AreEqual(
                "5282810b9dd49194d0b175e54360ce98aab867e8e9065ec48a09abc74a213303",
                Sha256(ProjectPath(PracticeScenePath)));

            string scene = ReadProject(WorldScenePath);
            StringAssert.Contains("propertyPath: m_LocalPosition.x\n      value: -3.5579896", scene);
            StringAssert.Contains("propertyPath: m_LocalPosition.z\n      value: -2.7420344", scene);
            StringAssert.Contains("propertyPath: m_LocalPosition.x\n      value: -2.855", scene);
            StringAssert.Contains("propertyPath: m_LocalPosition.y\n      value: 1.63", scene);
            StringAssert.Contains("propertyPath: m_LocalPosition.z\n      value: -3.032", scene);
        }

        static AnimatorController LoadController()
        {
            AnimatorController controller =
                AssetDatabase.LoadAssetAtPath<AnimatorController>(ControllerPath);
            Assert.IsNotNull(controller);
            Assert.AreEqual(1, controller.layers.Length);
            return controller;
        }

        static AnimatorState StateForMotion(AnimatorController controller, string motionPath)
        {
            AnimatorState[] matches = controller.layers[0].stateMachine.states
                .Select(value => value.state)
                .Where(value => value.motion != null &&
                    AssetDatabase.GetAssetPath(value.motion) == motionPath)
                .ToArray();
            Assert.AreEqual(1, matches.Length, "Expected one state for " + motionPath);
            return matches[0];
        }

        static void AssertTransition(
            AnimatorState source,
            AnimatorState destination,
            AnimatorConditionMode expectedMode)
        {
            AnimatorStateTransition[] transitions = source.transitions
                .Where(value => value.destinationState == destination)
                .ToArray();
            Assert.AreEqual(1, transitions.Length);
            AnimatorStateTransition transition = transitions[0];
            Assert.IsFalse(transition.hasExitTime);
            Assert.IsTrue(transition.hasFixedDuration);
            Assert.AreEqual(0.25f, transition.duration, 0.0001f);
            Assert.AreEqual(1, transition.conditions.Length);
            Assert.AreEqual("IsRun", transition.conditions[0].parameter);
            Assert.AreEqual(expectedMode, transition.conditions[0].mode);
        }

        static void AssertImporterLoop(string path, string clipName)
        {
            ModelImporter importer = AssetImporter.GetAtPath(path) as ModelImporter;
            Assert.IsNotNull(importer);
            Assert.AreEqual(ModelImporterAnimationType.Human, importer.animationType);
            ModelImporterClipAnimation[] clips = importer.clipAnimations;
            ModelImporterClipAnimation clip = clips.Single(value => value.name == clipName);
            Assert.IsTrue(clip.loopTime);
        }

        static void WithPlayer(Action<PetRobotWorldPlayerController, Animator, GameObject> assertion)
        {
            var host = new GameObject("Step10H_PlayerRun_TestHost");
            host.SetActive(false);
            try
            {
                host.AddComponent<CharacterController>();

                GameObject playerVisualAsset =
                    AssetDatabase.LoadAssetAtPath<GameObject>(PlayerVisualPath);
                Assert.IsNotNull(playerVisualAsset, PlayerVisualPath);

                Animator[] sourceAnimators =
                    playerVisualAsset.GetComponentsInChildren<Animator>(true);
                Assert.AreEqual(1, sourceAnimators.Length,
                    "The approved Player visual must contain exactly one Animator.");

                GameObject visual =
                    UnityEngine.Object.Instantiate(playerVisualAsset, host.transform);
                visual.name = "PlayerVisual";

                Animator[] animators = visual.GetComponentsInChildren<Animator>(true);
                Assert.AreEqual(1, animators.Length,
                    "The instantiated Player visual must contain exactly one Animator.");

                Animator animator = animators[0];
                Assert.AreSame(sourceAnimators[0].avatar, animator.avatar,
                    "The imported Player Avatar must be preserved.");

                RuntimeAnimatorController controller = LoadController();
                animator.runtimeAnimatorController = controller;
                animator.applyRootMotion = false;
                animator.enabled = true;

                host.SetActive(true);
                Assert.IsTrue(host.activeInHierarchy);
                Assert.IsTrue(visual.activeInHierarchy);
                Assert.IsTrue(animator.isActiveAndEnabled);
                Assert.AreSame(controller, animator.runtimeAnimatorController);
                Assert.IsFalse(animator.applyRootMotion);

                animator.Rebind();
                animator.Update(0f);

                PetRobotWorldPlayerController player =
                    host.AddComponent<PetRobotWorldPlayerController>();
                MethodInfo awakeMethod = typeof(PetRobotWorldPlayerController)
                    .GetMethod("Awake", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.IsNotNull(awakeMethod);
                awakeMethod.Invoke(player, null);

                FieldInfo playerAnimatorField = typeof(PetRobotWorldPlayerController)
                    .GetField("playerAnimator", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.IsNotNull(playerAnimatorField);
                Assert.AreSame(animator, playerAnimatorField.GetValue(player),
                    "The production lifecycle must resolve the actual imported Player Animator.");

                assertion(player, animator, host);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        static string ProjectPath(string relativePath) =>
            Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath);

        static string ReadProject(string relativePath) =>
            File.ReadAllText(ProjectPath(relativePath)).Replace("\r\n", "\n");

        static string Sha256(string path)
        {
            using (SHA256 algorithm = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
            {
                return BitConverter.ToString(algorithm.ComputeHash(stream))
                    .Replace("-", string.Empty).ToLowerInvariant();
            }
        }

        static int Count(string text, string value)
        {
            int count = 0;
            int offset = 0;
            while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += value.Length;
            }

            return count;
        }
    }
}
