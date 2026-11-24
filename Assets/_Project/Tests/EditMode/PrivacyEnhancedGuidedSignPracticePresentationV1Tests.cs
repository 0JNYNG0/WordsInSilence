using System;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PrivacyEnhancedGuidedSignPracticePresentationV1Tests
    {
        const string RuntimeDirectory = "Assets/_Project/Scripts/Practice/PetRobotVerticalSlice";
        const string PresentationSource = RuntimeDirectory + "/GuidedSignPracticePresentationController.cs";
        const string StorySource = RuntimeDirectory + "/MoaStoryGuidedPracticeFlow.cs";
        const string DemonstrationSource = RuntimeDirectory + "/SignDemonstrationAvatarPresenter.cs";
        const string OverlaySource = RuntimeDirectory + "/WebcamLandmarkOverlayPresenter.cs";
        const string TrackingGraphicSource = RuntimeDirectory + "/WebcamTrackingGraphic.cs";
        const string BootstrapSource = RuntimeDirectory + "/PetRobotPracticeVerticalSliceBootstrap.cs";
        const string PracticePrefab =
            "Assets/_Project/Resources/VerticalSlice/PetRobotSignPracticeVerticalSlice.prefab";
        const string PracticeScene =
            "Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity";
        const string ResourceDirectory = "Assets/_Project/Resources/VerticalSlice";
        const string AnimationDirectory = "Assets/_Project/Animations/Multiview";

        static readonly string[] WrapperPaths =
        {
            ResourceDirectory + "/SignDemonstrationGreetingWrapper.prefab",
            ResourceDirectory + "/SignDemonstrationLoveWrapper.prefab",
            ResourceDirectory + "/SignDemonstrationThankYouWrapper.prefab",
        };

        [Test]
        public void CharacterSlots_ReferenceProjectOwnedWrappersNotExternalFbxMainAssets()
        {
            string prefab = Read(PracticePrefab);
            StringAssert.Contains("greetingCharacter: {fileID: 100000, guid: 10f12000000000000000000000000001", prefab);
            StringAssert.Contains("loveCharacter: {fileID: 100000, guid: 10f12000000000000000000000000002", prefab);
            StringAssert.Contains("thankYouCharacter: {fileID: 100000, guid: 10f12000000000000000000000000003", prefab);
            Assert.Zero(Count(prefab, "greetingCharacter: {fileID: 100100000"));
            Assert.Zero(Count(prefab, "loveCharacter: {fileID: 100100000"));
            Assert.Zero(Count(prefab, "thankYouCharacter: {fileID: 100100000"));
        }

        [Test]
        public void WrapperGraph_KeepsEachExternalFbxOnlyAsNestedSourcePrefab()
        {
            string[] expectedGuids =
            {
                "d9c7a496c14c2544a83e75fd761e82a9",
                "92330f8673ab23a499d49ba5166de7b1",
                "3a282cc5617691046baa68cccdaab55b",
            };
            for (int index = 0; index < WrapperPaths.Length; index++)
            {
                string wrapper = Read(WrapperPaths[index]);
                StringAssert.Contains("PrefabInstance:", wrapper);
                StringAssert.Contains("m_SourcePrefab: {fileID: 100100000, guid: " + expectedGuids[index], wrapper);
                Assert.AreEqual(1, Count(wrapper, "m_SourcePrefab: {fileID: 100100000"));
            }
        }

        [Test]
        public void WrapperLocalDocuments_DoNotReuseReserved100100000()
        {
            foreach (string path in WrapperPaths)
                Assert.Zero(Regex.Matches(Read(path), @"^--- !u!\d+ &100100000\s*$", RegexOptions.Multiline).Count, path);
        }

        [Test]
        public void WordAnimationMapping_PreservesTheThreeObservedEmbeddedClipReferences()
        {
            string prefab = Read(PracticePrefab);
            StringAssert.Contains("greetingClip: {fileID: -623603649343465320, guid: d9c7a496c14c2544a83e75fd761e82a9", prefab);
            StringAssert.Contains("loveClip: {fileID: -623603649343465320, guid: 92330f8673ab23a499d49ba5166de7b1", prefab);
            StringAssert.Contains("thankYouClip: {fileID: -623603649343465320, guid: 3a282cc5617691046baa68cccdaab55b", prefab);
        }

        [Test]
        public void WordChange_SelectsTheMatchingWrapperAndClip()
        {
            string source = Read(DemonstrationSource);
            foreach (string token in new[]
                     {
                         "word == GreetingWord", "word == LoveWord", "word == ThankYouWord",
                         "character = greetingCharacter;", "character = loveCharacter;",
                         "character = thankYouCharacter;", "clip = greetingClip;",
                         "clip = loveClip;", "clip = thankYouClip;"
                     })
                StringAssert.Contains(token, source);
        }

        [Test]
        public void GreetingAssetGraph_ResolvesWrapperEmbeddedClipAndPresentationAnimator()
        {
            GameObject sourcePrefab = Resources.Load<GameObject>(
                "VerticalSlice/PetRobotSignPracticeVerticalSlice");
            Assert.NotNull(sourcePrefab);
            SignDemonstrationAvatarPresenter sourcePresenter =
                sourcePrefab.GetComponent<SignDemonstrationAvatarPresenter>();
            Assert.NotNull(sourcePresenter);
            GameObject wrapper = GetField<GameObject>(sourcePresenter, "greetingCharacter");
            AnimationClip clip = GetField<AnimationClip>(sourcePresenter, "greetingClip");
            Assert.NotNull(wrapper);
            Assert.NotNull(clip);
            Assert.AreEqual("SignDemonstrationGreetingWrapper", wrapper.name);
            Assert.AreEqual("Scene", clip.name);
            Assert.Greater(clip.length, 0f);

            GameObject host = new GameObject("Step10F_v1_3_DemonstrationTestHost");
            try
            {
                SignDemonstrationAvatarPresenter presenter =
                    host.AddComponent<SignDemonstrationAvatarPresenter>();
                SetField(presenter, "greetingCharacter", wrapper);
                SetField(presenter, "greetingClip", clip);
                presenter.ShowDemonstration(SignDemonstrationAvatarPresenter.GreetingWord);
                Assert.NotNull(presenter.DemonstrationRoot);
                Assert.NotNull(presenter.DemonstrationAnimator);
                Assert.AreEqual(0.0, presenter.CurrentPlayableTime, 0.000001);
                Assert.IsTrue(presenter.DemonstrationGraphValid);
                Assert.IsTrue(presenter.DemonstrationAnimator.enabled);
                Assert.IsFalse(presenter.DemonstrationAnimator.applyRootMotion);
                Assert.AreEqual(AnimatorCullingMode.AlwaysAnimate,
                    presenter.DemonstrationAnimator.cullingMode);
                Assert.AreEqual(AnimatorUpdateMode.UnscaledTime,
                    presenter.DemonstrationAnimator.updateMode);

                Transform[] animated =
                    presenter.DemonstrationAnimator.GetComponentsInChildren<Transform>(true);
                Vector3[] positions = new Vector3[animated.Length];
                Quaternion[] rotations = new Quaternion[animated.Length];
                for (int index = 0; index < animated.Length; index++)
                {
                    positions[index] = animated[index].localPosition;
                    rotations[index] = animated[index].localRotation;
                }
                presenter.AdvanceDemonstrationPresentation(Mathf.Min(0.5f, clip.length * 0.25f));
                Assert.Greater(presenter.CurrentPlayableTime, 0.0);
                bool changed = false;
                for (int index = 0; index < animated.Length; index++)
                {
                    if ((animated[index].localPosition - positions[index]).sqrMagnitude > 0.00000001f ||
                        Quaternion.Angle(animated[index].localRotation, rotations[index]) > 0.001f)
                    {
                        changed = true;
                        break;
                    }
                }
                Assert.IsTrue(changed, "GREETING must visibly change between t0 and nonzero time");
                presenter.AdvanceDemonstrationPresentation(clip.length + 0.1f);
                Assert.Less(presenter.CurrentPlayableTime, clip.length);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        [Test]
        public void DemonstrationPlayable_UsesManualUnscaledProgressTimeZeroAndLoop()
        {
            string source = Read(DemonstrationSource);
            StringAssert.Contains("DirectorUpdateMode.Manual", source);
            StringAssert.Contains("Time.unscaledDeltaTime", source);
            StringAssert.Contains("demonstrationPlayable.SetSpeed(1.0)", source);
            StringAssert.Contains("output.SetWeight(1f)", source);
            StringAssert.Contains("demonstrationGraph.Evaluate(0f)", source);
            StringAssert.Contains("demonstrationPlayable.SetTime(0.0)", source);
            StringAssert.Contains("demonstrationPlayable.GetTime() >= duration", source);
            StringAssert.Contains("demonstrationPlayable.GetTime() % duration", source);
            StringAssert.Contains("SetDone(false)", source);
            Assert.GreaterOrEqual(Count(source, "applyRootMotion = false"), 2);
            Assert.Zero(Count(source, "DirectorUpdateMode.GameTime"));
            Assert.Zero(Count(source, "RecognitionCore"));
            Assert.Zero(Count(source, "MaskAwareDtw"));
        }

        [Test]
        public void YawHelper_RotatesOnlyDemonstrationWrapperRootAndKeyboardDelegates()
        {
            string source = Read(DemonstrationSource);
            string inputBody = MethodBody(source, "void UpdateDemonstrationYawInput()");
            string yawBody = MethodBody(source,
                "public void ApplyDemonstrationYawDelta(float yawDeltaDegrees)");
            StringAssert.Contains("KeyCode.LeftArrow", inputBody);
            StringAssert.Contains("KeyCode.RightArrow", inputBody);
            StringAssert.Contains("if (Input.GetKey(KeyCode.LeftArrow)) direction += 1f;", inputBody);
            StringAssert.Contains("if (Input.GetKey(KeyCode.RightArrow)) direction -= 1f;", inputBody);
            Assert.AreEqual(1, Count(inputBody, "ApplyDemonstrationYawDelta("));
            Assert.Zero(Count(inputBody, "localRotation ="));
            StringAssert.Contains("currentDemonstrationYaw", yawBody);
            StringAssert.Contains("demonstrationInstance.transform.localRotation", yawBody);
            foreach (string forbidden in new[]
                     {
                         "StartExpression", "FinishExpression", "BeginCapture", "SetTime(", "session",
                         "robotInstance", "currentWord ="
                     })
                Assert.Zero(Count(yawBody, forbidden), forbidden);

            GameObject sourcePrefab = Resources.Load<GameObject>(
                "VerticalSlice/PetRobotSignPracticeVerticalSlice");
            SignDemonstrationAvatarPresenter sourcePresenter =
                sourcePrefab.GetComponent<SignDemonstrationAvatarPresenter>();
            GameObject host = new GameObject("Step10F_v1_3_YawTestHost");
            try
            {
                SignDemonstrationAvatarPresenter presenter =
                    host.AddComponent<SignDemonstrationAvatarPresenter>();
                SetField(presenter, "greetingCharacter",
                    GetField<GameObject>(sourcePresenter, "greetingCharacter"));
                SetField(presenter, "greetingClip",
                    GetField<AnimationClip>(sourcePresenter, "greetingClip"));
                presenter.ShowDemonstration(SignDemonstrationAvatarPresenter.GreetingWord);
                Quaternion before = presenter.DemonstrationRoot.localRotation;
                presenter.ApplyDemonstrationYawDelta(35f);
                Assert.Greater(Quaternion.Angle(before,
                    presenter.DemonstrationRoot.localRotation), 1f);
                Assert.AreEqual(215f, presenter.CurrentDemonstrationYaw, 0.001f);
                Assert.AreEqual(SignDemonstrationAvatarPresenter.GreetingWord,
                    presenter.CurrentWord);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        [Test]
        public void WorldStage_HasNoSecondaryCameraOrRenderTexture()
        {
            string source = Read(DemonstrationSource);
            StringAssert.Contains("UsesMainSceneCamera => true", source);
            StringAssert.Contains("UsesPresentationOnlyCameras => false", source);
            StringAssert.Contains("Step10F_v1_3b_FuturisticPracticeChamber", source);
            StringAssert.Contains("DarkNavyChamberWall", source);
            StringAssert.Contains("CentralCircularTrainingFloor", source);
            StringAssert.Contains("CentralFloorCyanRing", source);
            StringAssert.Contains("RearChamberCyanArc", source);
            StringAssert.Contains("Universal Render Pipeline/Unlit", source);
            StringAssert.Contains("ownedStageMaterials", source);
            Assert.Zero(Count(source, "RenderTexture"));
            Assert.Zero(Count(source, "new GameObject(\"Step10F_DemonstrationCamera"));
            Assert.Zero(Count(source, "targetTexture"));
        }

        [Test]
        public void RobotVisual_ResolvesIntendedControllerAnimatorAndInitializesPresentationSettings()
        {
            string prefab = Read(PracticePrefab);
            StringAssert.Contains("robotVisualPrefab: {fileID: 100000, guid: 10d00000000000000000000000000010", prefab);
            StringAssert.Contains("presentationRobotController: {fileID: 9100000, guid: a414e66cfd19823419a0f9a1954db8d3, type: 2}", prefab);
            StringAssert.Contains("demonstrationPosition: {x: -6.65, y: -3.2, z: -0.65}", prefab);
            StringAssert.Contains("demonstrationScale: {x: 3.25, y: 3.25, z: 3.25}", prefab);
            StringAssert.Contains("robotPosition: {x: 5.85, y: 1.95, z: 0.35}", prefab);
            StringAssert.Contains("robotScale: {x: 0.68, y: 0.68, z: 0.68}", prefab);
            StringAssert.Contains("robotYaw: 220", prefab);
            StringAssert.Contains("Step10F_v1_2_PresentationOnly_MOA_robot", Read(DemonstrationSource));
            StringAssert.DoesNotContain("Animator.Play(", Read(DemonstrationSource));
            StringAssert.DoesNotContain("proceduralHover", Read(DemonstrationSource));

            GameObject wrapper = Resources.Load<GameObject>(
                "VerticalSlice/PetRobotAnimatedVisualWrapper");
            Assert.NotNull(wrapper);
            RuntimeAnimatorController expectedController =
                UnityEditor.AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(
                    "Assets/GameAssets/petrobot_model_animation/Terrain.controller");
            Assert.NotNull(expectedController);
            GameObject instance = UnityEngine.Object.Instantiate(wrapper);
            try
            {
                Animator[] before = instance.GetComponentsInChildren<Animator>(true);
                Assert.LessOrEqual(before.Length, 1);
                if (before.Length == 0) Assert.AreEqual(1, instance.transform.childCount);

                Animator animator =
                    SignDemonstrationAvatarPresenter.InitializePresentationRobotAnimator(
                        instance, expectedController);
                Assert.NotNull(animator);
                Animator[] after = instance.GetComponentsInChildren<Animator>(true);
                Assert.AreEqual(1, after.Length);
                Assert.AreSame(animator, after[0]);
                if (before.Length == 0)
                    Assert.AreSame(instance.transform.GetChild(0).gameObject, animator.gameObject);
                else
                    Assert.AreSame(before[0], animator);
                Assert.AreSame(expectedController, animator.runtimeAnimatorController);
                Assert.IsTrue(animator.enabled);
                Assert.AreEqual(1f, animator.speed);
                Assert.IsFalse(animator.applyRootMotion);
                Assert.AreEqual(AnimatorCullingMode.AlwaysAnimate, animator.cullingMode);
                Assert.AreEqual(AnimatorUpdateMode.UnscaledTime, animator.updateMode);

                GameObject ambiguousRoot = new GameObject("Step10F_v1_3d_AmbiguousRobotRoot");
                try
                {
                    new GameObject("RobotA").transform.SetParent(ambiguousRoot.transform, false);
                    new GameObject("RobotB").transform.SetParent(ambiguousRoot.transform, false);
                    ambiguousRoot.transform.GetChild(0).gameObject.AddComponent<Animator>();
                    ambiguousRoot.transform.GetChild(1).gameObject.AddComponent<Animator>();
                    Assert.IsNull(
                        SignDemonstrationAvatarPresenter.InitializePresentationRobotAnimator(
                            ambiguousRoot, expectedController));
                    Assert.AreEqual(2, ambiguousRoot.GetComponentsInChildren<Animator>(true).Length);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(ambiguousRoot);
                }

                GameObject conflictingRoot = new GameObject("Step10F_v1_3d_ConflictingRobotRoot");
                AnimatorOverrideController conflictingController =
                    new AnimatorOverrideController(expectedController);
                try
                {
                    GameObject conflictingChild = new GameObject("Robot");
                    conflictingChild.transform.SetParent(conflictingRoot.transform, false);
                    Animator conflictingAnimator = conflictingChild.AddComponent<Animator>();
                    conflictingAnimator.runtimeAnimatorController = conflictingController;
                    Assert.IsNull(
                        SignDemonstrationAvatarPresenter.InitializePresentationRobotAnimator(
                            conflictingRoot, expectedController));
                    Assert.AreSame(
                        conflictingController, conflictingAnimator.runtimeAnimatorController);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(conflictingRoot);
                    UnityEngine.Object.DestroyImmediate(conflictingController);
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(instance);
            }
        }

        [Test]
        public void RobotVisual_HasNoGameplayFollowInteractionOrProviderOwner()
        {
            string combined = Read(DemonstrationSource) + Read(ResourceDirectory + "/PetRobotAnimatedVisualWrapper.prefab");
            foreach (string forbidden in new[]
                     {
                         "PetRobotWorldEntryController", "PetRobotFloatingCompanionController",
                         "PetRobotInteractionController", "UnityWebCamFrameSource",
                         "PoseLandmarkProvider", "MediaPipeLandmarkProvider", "RecognitionCore",
                         "PersonalizationCandidateRepository", "PetRobotCandidateCameraStartupRoute"
                     })
                Assert.Zero(Count(combined, forbidden), forbidden);
        }

        [Test]
        public void FlatFullscreenImguiPresentation_IsNoLongerTheDefault()
        {
            string source = Read(PresentationSource);
            Assert.Zero(Count(source, "void OnGUI()"));
            Assert.Zero(Count(source, "DrawBackdrop"));
            Assert.Zero(Count(source, "DemonstrationColumn"));
            Assert.Zero(Count(source, "RecognitionColumn"));
            Assert.Zero(Count(source, "RobotColumn"));
            Assert.Zero(Count(source, "DemonstrationStageFrame"));
            Assert.Zero(Count(source, "RobotStageFrame"));
            Assert.Zero(Count(source, "StageGlassColor"));
            StringAssert.Contains("CompactPracticeStatusHeader", source);
            StringAssert.Contains("CompactDemonstrationHint", source);
            StringAssert.Contains("WebcamCardAccent", source);
            StringAssert.Contains("MoaRobotSpeechBubble", source);
            StringAssert.Contains("return \"진행 \" + step + \"/3\";", source);
            Rect webcam = GuidedSignPracticePresentationController.WebcamPanelRect;
            Rect hud = GuidedSignPracticePresentationController.BottomHudRect;
            Assert.LessOrEqual(webcam.width, 820f);
            Assert.LessOrEqual(webcam.height, 520f);
            Assert.Less(webcam.xMin, ReferenceCenterX());
            Assert.Greater(webcam.xMax, ReferenceCenterX());
            Assert.Greater(hud.xMin, 100f);
            Assert.Less(hud.xMax, GuidedSignPracticePresentationController.ReferenceWidth - 100f);
            Assert.LessOrEqual(hud.height, 200f);
            Assert.GreaterOrEqual(hud.yMin, 820f);
        }

        [Test]
        public void UguiCanvas_UsesThe1920By1080ScalerContract()
        {
            string source = Read(PresentationSource);
            StringAssert.Contains("typeof(Canvas)", source);
            StringAssert.Contains("typeof(CanvasScaler)", source);
            StringAssert.Contains("CanvasScaler.ScaleMode.ScaleWithScreenSize", source);
            StringAssert.Contains("new Vector2(ReferenceWidth, ReferenceHeight)", source);
            Assert.AreEqual(1920f, ReadFloatConstant(source, "ReferenceWidth"));
            Assert.AreEqual(1080f, ReadFloatConstant(source, "ReferenceHeight"));
        }

        [Test]
        public void RoundedPanelsAndButtons_UseOwnedRuntimeSlicedSpriteAndCleanup()
        {
            string source = Read(PresentationSource);
            Assert.Zero(Count(source, "UI/Skin/UISprite.psd"));
            Assert.Zero(Count(source, "Resources.GetBuiltinResource<Sprite>"));
            StringAssert.Contains("Sprite.Create(", source);
            StringAssert.Contains("Vector4 border", source);
            StringAssert.Contains("new Texture2D(", source);
            StringAssert.Contains("ReleaseRoundedGraphicResources();", source);
            StringAssert.Contains("Destroy(runtimeRoundedSprite)", source);
            StringAssert.Contains("Destroy(runtimeRoundedTexture)", source);
            Assert.GreaterOrEqual(Count(source, "Image.Type.Sliced"), 2);
            Assert.GreaterOrEqual(Count(source, "AddComponent<Outline>()"), 2);
            StringAssert.Contains("RoundedWebcamPanel", source);
            StringAssert.Contains("RoundedProgressAndControlsHud", source);

            GameObject host = new GameObject("Step10F_v1_3_RoundedUiTestHost");
            try
            {
                GuidedSignPracticePresentationController controller =
                    host.AddComponent<GuidedSignPracticePresentationController>();
                Sprite roundedSprite = InvokePrivate(
                    controller, "EnsureRoundedGraphicSource") as Sprite;
                Assert.NotNull(roundedSprite);
                Assert.IsTrue(controller.OwnsRuntimeRoundedResources);
                Assert.AreSame(roundedSprite, controller.RoundedGraphicSource);
                Assert.NotNull(controller.RoundedTextureSource);
                Vector4 border = roundedSprite.border;
                Assert.Greater(border.x, 0f);
                Assert.Greater(border.y, 0f);
                Assert.Greater(border.z, 0f);
                Assert.Greater(border.w, 0f);
                Color transparentCorner = controller.RoundedTextureSource.GetPixel(0, 0);
                Color opaqueCenter = controller.RoundedTextureSource.GetPixel(
                    GuidedSignPracticePresentationController.RoundedTextureSize / 2,
                    GuidedSignPracticePresentationController.RoundedTextureSize / 2);
                Assert.Less(transparentCorner.a, opaqueCenter.a);
                InvokePrivate(controller, "ReleaseRoundedGraphicResources");
                Assert.IsFalse(controller.OwnsRuntimeRoundedResources);
                Assert.IsNull(controller.RoundedGraphicSource);
                Assert.IsNull(controller.RoundedTextureSource);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        [Test]
        public void ExistingStep10AButtonActions_AreCalledDirectly()
        {
            string source = Read(PresentationSource);
            foreach (string call in new[]
                     {
                         "session.StartExpression()", "session.FinishExpression()", "session.Retry()",
                         "session.ContinueWithNeedsPractice()", "session.ContinueToNextWord()",
                         "session.ExitPractice()"
                     })
                StringAssert.Contains(call, source);
        }

        [Test]
        public void ExistingManualButtonStateGates_ArePreserved()
        {
            string source = Read(PresentationSource);
            foreach (string gate in new[]
                     {
                         "session.State == PetRobotPracticeState.Ready && session.CameraReady",
                         "session.State == PetRobotPracticeState.Capturing", "session.CanRetry",
                         "session.CanContinueWithNeedsPractice",
                         "session.State == PetRobotPracticeState.Feedback && session.IsCurrentWordComplete",
                         "session.State == PetRobotPracticeState.Completed"
                     })
                StringAssert.Contains(gate, source);
        }

        [Test]
        public void FourOutcomesAndCompletedSummary_AreDisplayedWithoutReclassification()
        {
            string source = Read(PresentationSource);
            StringAssert.Contains("outcome.Kind.ToString()", source);
            StringAssert.Contains("GuidedQuestOutcomeKind.StrictSuccess", source);
            StringAssert.Contains("GuidedQuestOutcomeKind.LowConfidenceCorrect", source);
            StringAssert.Contains("GuidedQuestOutcomeKind.WrongWord", source);
            StringAssert.Contains("PetRobotPracticeState.Completed", source);
            StringAssert.Contains("session.LastStorageResult.Status", source);
            StringAssert.Contains("counts.Eligible[\"GREETING\"]", source);
            StringAssert.Contains("counts.Eligible[\"LOVE\"]", source);
            StringAssert.Contains("counts.Eligible[\"THANK_YOU\"]", source);
            Assert.Zero(Count(source, "GuidedQuestOutcomeEvaluator"));
            Assert.Zero(Count(source, "new GuidedQuestOutcome"));
            Assert.Zero(Count(source, "PersonalizationCandidateRepository"));
        }

        [Test]
        public void WebcamSource_IsReadOnlyAndBlurIsDisplayMaterialOnly()
        {
            string source = Read(OverlaySource);
            string shader = Read(ResourceDirectory + "/DisplayOnlyMildBlur.shader");
            string material = Read(ResourceDirectory + "/DisplayOnlyMildBlur.mat");
            string prefab = Read(PracticePrefab);
            StringAssert.Contains("cameraRoute.PreviewTexture", source);
            StringAssert.Contains("new Material(displayBlurMaterial)", source);
            StringAssert.Contains("WebcamRgbDisplayOnlyBlur", source);
            StringAssert.Contains("DisplayToSource", shader);
            Assert.GreaterOrEqual(Count(shader, "tex2D(_MainTex"), 5);
            StringAssert.Contains("- _BlurRadius: 2.6", material);
            StringAssert.Contains("BodyAndHandsTrackingOverlay", source);
            Assert.Less(source.IndexOf("WebcamRgbDisplayOnlyBlur", StringComparison.Ordinal),
                source.IndexOf("BodyAndHandsTrackingOverlay", StringComparison.Ordinal));
            StringAssert.Contains("displayBlurMaterial: {fileID: 2100000, guid: 10f12000000000000000000000000012", prefab);
        }

        [Test]
        public void FacePointsConnectionsAndMosaic_AreAbsentFromDisplayCode()
        {
            string combined = Read(OverlaySource) + Read(TrackingGraphicSource);
            foreach (string forbidden in new[]
                     {
                         "FaceSupportIndices", "DrawFacePrivacy", "Pixelation", "pixelation",
                         "Mosaic", "mosaic", "face bounds"
                     })
                Assert.Zero(Count(combined, forbidden), forbidden);
            StringAssert.Contains("FirstBodyPoseIndex = 11", combined);
        }

        [Test]
        public void PoseAndBothHands_UseOneSerializedTrackingColor()
        {
            string source = Read(OverlaySource);
            string prefab = Read(PracticePrefab);
            Assert.AreEqual(1, Regex.Matches(source, @"\[SerializeField\]\s+Color trackingColor").Count);
            Assert.Zero(Count(source, "poseColor"));
            Assert.Zero(Count(source, "leftHandColor"));
            Assert.Zero(Count(source, "rightHandColor"));
            Assert.AreEqual(1, Count(prefab, "trackingColor:"));
        }

        [Test]
        public void TrackingGraphic_DrawsBodyPoseAndTwoActualTwentyOnePointHands()
        {
            string source = Read(TrackingGraphicSource);
            StringAssert.Contains("BodyPoseConnections", source);
            StringAssert.Contains("BodyPosePoints", source);
            StringAssert.Contains("observation.LeftHand", source);
            StringAssert.Contains("observation.RightHand", source);
            StringAssert.Contains("RawMediaPipeObservation.HandLandmarkCount", source);
            StringAssert.Contains("public const int FirstBodyPoseIndex = 11", source);
        }

        [Test]
        public void MirrorRotationAspectFitAndMaskClipping_ShareTheDisplayTransform()
        {
            string source = Read(OverlaySource);
            foreach (string token in new[]
                     {
                         "WebcamDisplayTransform", "SourceToDisplay", "DisplayToSource", "IsQuarterTurn",
                         "FitTexture(texture, displayTransform)", "ApplyDisplayOnlyMaterialTransform(displayTransform)",
                         "videoVerticallyMirrored", "videoRotationAngle", "MaskableGraphic"
                     })
                StringAssert.Contains(token, source + Read(TrackingGraphicSource));
            StringAssert.Contains("AddComponent<Mask>()", Read(PresentationSource));
            StringAssert.Contains("displayTransform.ForOverlayGeometry()", source);
            StringAssert.Contains("if (correctFinalOverlayY) display.y = 1f - display.y;", source);

            Type transformType = typeof(GuidedSignPracticePresentationController).Assembly.GetType(
                "WordsInSilence.Practice.PetRobotVerticalSlice." +
                "WebcamLandmarkOverlayPresenter+WebcamDisplayTransform");
            Assert.NotNull(transformType);
            MethodInfo sourceToDisplay = transformType.GetMethod("SourceToDisplay");
            MethodInfo displayToSource = transformType.GetMethod("DisplayToSource");
            MethodInfo forOverlay = transformType.GetMethod("ForOverlayGeometry");
            Assert.NotNull(sourceToDisplay);
            Assert.NotNull(displayToSource);
            Assert.NotNull(forOverlay);

            Vector2[] displaySamples =
            {
                new Vector2(0.42f, 0.92f), // head / top
                new Vector2(0.42f, 0.72f), // shoulder
                new Vector2(0.42f, 0.38f), // hip
                new Vector2(0.42f, 0.08f), // bottom
            };
            foreach (bool horizontalMirror in new[] { false, true })
            foreach (bool verticalMirror in new[] { false, true })
            foreach (int rotation in new[] { 0, 90, 180, 270 })
            {
                object displayTransform = Activator.CreateInstance(
                    transformType, horizontalMirror, verticalMirror, rotation);
                object overlayTransform = forOverlay.Invoke(displayTransform, null);
                float[] guiY = new float[displaySamples.Length];
                for (int index = 0; index < displaySamples.Length; index++)
                {
                    Vector2 sourcePoint = InvokeVector2(
                        displayToSource, displayTransform, displaySamples[index]);
                    Vector2 immutableCopy = sourcePoint;
                    Vector2 overlayPoint = InvokeVector2(
                        sourceToDisplay, overlayTransform, sourcePoint);
                    Assert.AreEqual(immutableCopy, sourcePoint);
                    guiY[index] = 1f - overlayPoint.y;
                    Vector2 roundTrip = InvokeVector2(
                        displayToSource, overlayTransform, overlayPoint);
                    Assert.Less((roundTrip - sourcePoint).sqrMagnitude, 0.000001f);
                }
                Assert.Greater(guiY[0], guiY[1]);
                Assert.Greater(guiY[1], guiY[2]);
                Assert.Greater(guiY[2], guiY[3]);
                Assert.Greater(guiY[0], 0.85f);
                Assert.Less(guiY[3], 0.15f);
            }

            object noMirror = Activator.CreateInstance(transformType, false, false, 0);
            object horizontal = Activator.CreateInstance(transformType, true, false, 0);
            Vector2 sample = new Vector2(0.2f, 0.7f);
            Vector2 plain = InvokeVector2(sourceToDisplay, forOverlay.Invoke(noMirror, null), sample);
            Vector2 mirrored = InvokeVector2(sourceToDisplay, forOverlay.Invoke(horizontal, null), sample);
            Assert.AreEqual(1f - plain.x, mirrored.x, 0.000001f);
            Assert.AreEqual(plain.y, mirrored.y, 0.000001f);
        }

        [Test]
        public void ObservationLoss_ClearsStaleTrackingGeometry()
        {
            string source = Read(OverlaySource);
            StringAssert.Contains("staleObservationSeconds", source);
            StringAssert.Contains("TrackingTemporarilyLost", source);
            StringAssert.Contains("TrackingState == PracticeTrackingDisplayState.TrackingActive", source);
            StringAssert.Contains("trackingGraphic?.ClearTrackingData()", source);
        }

        [Test]
        public void Presentation_AddsNoCameraProviderOrMediaStorageOwner()
        {
            AssertForbiddenAcrossPresentationSources(
                "UnityWebCamFrameSource", "WebCamTexture.Play", "PoseLandmarkProvider",
                "MediaPipeLandmarkProvider", "PoseLandmarker.Create", "File.Write",
                "persistentDataPath", "EncodeToPNG", "AudioClip", "VideoClip");
        }

        [Test]
        public void Observation_IsNotMutatedReprocessedOrFedBackToRecognition()
        {
            AssertForbiddenAcrossPresentationSources(
                "ApprovedLivePreprocessingAdapter", "RecognitionSequenceInputSession",
                "Interpolate(", "Normalize(", "SetExpectedWord(", "RecognitionCore", "MaskAwareDtw");
            string source = Read(OverlaySource);
            StringAssert.Contains("public bool WritesRecognitionInput => false", source);
            StringAssert.Contains("public bool StoresRgbFrames => false", source);
        }

        [Test]
        public void DiagnosticsRemainHiddenByDefaultAndTechnicalCountsStayInsideDiagnostics()
        {
            string prefab = Read(PracticePrefab);
            string source = Read(PresentationSource);
            string camera = MethodBody(source, "string CameraText()");
            string diagnostics = MethodBody(source, "string DiagnosticsText()");
            StringAssert.Contains("diagnosticsVisibleByDefault: 0", prefab);
            StringAssert.Contains("legacyTechnicalUi.enabled = false", source);
            Assert.Zero(Count(camera, "CameraFrameCount"));
            Assert.Zero(Count(camera, "ProviderObservationCount"));
            StringAssert.Contains("CameraFrameCount", diagnostics);
            StringAssert.Contains("ProviderObservationCount", diagnostics);
            StringAssert.Contains("DemonstrationAnimatorPath", diagnostics);
            StringAssert.Contains("CurrentClipName", diagnostics);
            StringAssert.Contains("CurrentPlayableTime", diagnostics);
            StringAssert.Contains("CurrentDemonstrationYaw", diagnostics);
            StringAssert.Contains("RobotAnimatorPath", diagnostics);
            StringAssert.Contains("RobotNormalizedTime", diagnostics);
        }

        [Test]
        public void AnimationImportsRemainHumanoidAndPracticeSceneKeepsItsMainCamera()
        {
            string[] names =
            {
                "SIGN001_GREETING.fbx.meta", "SIGN002_LOVE.fbx.meta", "SIGN003_THANKS.fbx.meta"
            };
            string[] guids =
            {
                "d9c7a496c14c2544a83e75fd761e82a9",
                "92330f8673ab23a499d49ba5166de7b1",
                "3a282cc5617691046baa68cccdaab55b"
            };
            for (int index = 0; index < names.Length; index++)
            {
                string meta = Read(AnimationDirectory + "/" + names[index]);
                StringAssert.Contains("guid: " + guids[index], meta);
                StringAssert.Contains("importAnimation: 1", meta);
                StringAssert.Contains("animationType: 2", meta);
            }
            string scene = Read(PracticeScene);
            StringAssert.Contains("m_Name: Main Camera", scene);
            StringAssert.Contains("m_LocalPosition: {x: 0, y: 1, z: -10}", scene);
            StringAssert.Contains("field of view: 60", scene);
        }

        [Test]
        public void ApprovedRecognitionStorageAndWorldHandoffRoutesRemainOutsidePresentation()
        {
            AssertForbiddenAcrossPresentationSources(
                "PersonalRuntimeOptInSession", "MultiPrototypeCandidate", "GuidedQuestOutcomeEvaluator",
                "WISRCB01", "thresholds[");
            string bootstrap = Read(BootstrapSource);
            StringAssert.Contains("new PersonalLivePracticeRecognitionAdapter(harness)", bootstrap);
            StringAssert.Contains("new PersonalizationCandidateRepository(learnerProfileId)", bootstrap);
        }

        [Test]
        public void StoryFlow_StartsAtWelcomeWithoutTouchingSession()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            Assert.AreEqual(MoaStoryBeat.Welcome, flow.Beat);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.WelcomeCopy, flow.Current.BodyText);
            Assert.AreEqual("다음", flow.Current.ActionLabel);
            Assert.AreEqual(MoaStoryAction.AdvanceStory, flow.Current.Action);
            Assert.IsFalse(flow.ExpressionStartUnlocked);
            Assert.Zero(Count(Read(StorySource), "PetRobotSignPracticeSession"));
        }

        [Test]
        public void StoryFlow_IntroRequiresExplicitAdvanceForEveryBeat()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryBeat.Welcome, flow.Beat);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.LessonOverview, flow.Beat);
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryBeat.LessonOverview, flow.Beat);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.HowToUse, flow.Beat);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.WordIntroduction, flow.Beat);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.AreEqual(MoaStoryBeat.Practice, flow.Beat);
        }

        [Test]
        public void StoryFlow_UsesGreetingLoveThankYouIntroductionOrder()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            CollectionAssert.AreEqual(
                new[] { "GREETING", "LOVE", "THANK_YOU" }, flow.Words);
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.GreetingIntroductionCopy,
                flow.Current.BodyText);
            flow.AdvanceStory();
            flow.Synchronize("LOVE", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryBeat.WordIntroduction, flow.Beat);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.LoveIntroductionCopy,
                flow.Current.BodyText);
            flow.AdvanceStory();
            flow.Synchronize("THANK_YOU", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.ThankYouIntroductionCopy,
                flow.Current.BodyText);
        }

        [Test]
        public void StoryFlow_DoesNotEnableExpressionBeforeWordIntroAccepted()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, false);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            Assert.AreEqual(MoaStoryBeat.WordIntroduction, flow.Beat);
            Assert.AreEqual("직접 해볼게", flow.Current.ActionLabel);
            Assert.AreEqual(MoaStoryAction.BeginAutoAttempt, flow.Current.Action);
            Assert.IsFalse(flow.Current.ActionEnabled);
            Assert.IsFalse(flow.AdvanceStory());
            Assert.IsFalse(flow.ExpressionStartUnlocked);
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            Assert.IsTrue(flow.AdvanceStory());
            Assert.IsTrue(flow.ExpressionStartUnlocked);
        }

        [Test]
        public void StoryFlow_CapturingMirrorsSessionWithoutAutoAdvance()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.Synchronize("GREETING", PetRobotPracticeState.Capturing,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryBeat.Practice, flow.Beat);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.CapturingCopy, flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.None, flow.Current.Action);
            Assert.IsFalse(flow.Current.ActionVisible);
            flow.Synchronize("GREETING", PetRobotPracticeState.Capturing,
                MoaStoryFeedbackKind.None, false, true);
            Assert.AreEqual(MoaStoryBeat.Practice, flow.Beat);
        }

        [Test]
        public void StoryFlow_StrictAndLowConfidenceReuseExistingNextWordRoute()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.StrictSuccess, false, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.StrictSuccessCopy, flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.ContinueToNextWord, flow.Current.Action);
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.LowConfidenceCorrect, false, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.LowConfidenceCorrectCopy,
                flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.ContinueToNextWord, flow.Current.Action);
        }

        [Test]
        public void StoryFlow_WrongWordAndTechnicalFailureNeverAutoAdvance()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.WrongWord, false, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.WrongWordCopy, flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.ReviewWrongWord, flow.Current.Action);
            Assert.AreEqual("시범 다시 보기", flow.Current.ActionLabel);
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.TechnicalFailure, false, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.TechnicalFailureCopy,
                flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.RetryTechnicalFailure, flow.Current.Action);
            Assert.AreEqual("다시 준비할게", flow.Current.ActionLabel);
        }

        [Test]
        public void StoryFlow_NeedsPracticeReusesExistingContinueRoute()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("GREETING");
            flow.Synchronize("GREETING", PetRobotPracticeState.Ready,
                MoaStoryFeedbackKind.None, false, true);
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.AdvanceStory();
            flow.Synchronize("GREETING", PetRobotPracticeState.Feedback,
                MoaStoryFeedbackKind.WrongWord, true, true);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.NeedsPracticeCopy,
                flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.ContinueWithNeedsPractice, flow.Current.Action);
            Assert.AreEqual("다음 표현", flow.Current.ActionLabel);
        }

        [Test]
        public void StoryFlow_CompletedReusesExistingExitRoute()
        {
            MoaStoryGuidedPracticeFlow flow = new MoaStoryGuidedPracticeFlow();
            flow.Reset("THANK_YOU");
            flow.Synchronize("THANK_YOU", PetRobotPracticeState.Completed,
                MoaStoryFeedbackKind.StrictSuccess, false, true);
            Assert.AreEqual(MoaStoryBeat.Completed, flow.Beat);
            Assert.AreEqual(MoaStoryGuidedPracticeFlow.CompletedCopy, flow.Current.BodyText);
            Assert.AreEqual(MoaStoryAction.ExitPractice, flow.Current.Action);
            Assert.AreEqual("연습 공간 나가기", flow.Current.ActionLabel);
            flow.Synchronize("THANK_YOU", PetRobotPracticeState.Exited,
                MoaStoryFeedbackKind.StrictSuccess, false, true);
            Assert.AreEqual(MoaStoryAction.None, flow.Current.Action);
            Assert.IsFalse(flow.Current.ActionVisible);
        }

        [Test]
        public void StoryUi_HidesDuplicateLegacyProgressionAndPreservesCaptureActions()
        {
            string source = Read(PresentationSource);
            string story = Read(StorySource);
            StringAssert.Contains("MoaRobotNameLabel", source);
            StringAssert.Contains("MoaStoryActionButton", source);
            StringAssert.Contains("MoaStoryAction.BeginAutoAttempt", source);
            StringAssert.Contains("MoaStoryAction.RetryPreCapture", source);
            StringAssert.Contains("MoaStoryAction.RetryTechnicalFailure", source);
            StringAssert.Contains("MoaStoryAction.ReviewWrongWord", source);
            StringAssert.Contains("HideLegacyNormalStoryControls();", source);
            StringAssert.Contains("startButton.gameObject.SetActive(false);", source);
            StringAssert.Contains("finishButton.gameObject.SetActive(false);", source);
            StringAssert.Contains("retryButton.gameObject.SetActive(false);", source);
            StringAssert.Contains("nextButton.gameObject.SetActive(false);", source);
            StringAssert.Contains("needsPracticeButton.gameObject.SetActive(false);", source);
            StringAssert.Contains("exitButton.gameObject.SetActive(false);", source);
            string dispatcher = MethodBody(source, "void HandleStoryAction()");
            StringAssert.Contains("HandleNextWord();", dispatcher);
            StringAssert.Contains("HandleNeedsPracticeContinue();", dispatcher);
            StringAssert.Contains("HandleExitPractice();", dispatcher);
            StringAssert.Contains("HandleBeginAutoAttempt();", dispatcher);
            StringAssert.Contains("HandleWrongWordReview();", dispatcher);
            StringAssert.Contains("session.StartExpression()", source);
            StringAssert.Contains("session.FinishExpression()", source);
            StringAssert.Contains("session.Retry()", source);
            foreach (string forbidden in new[]
                     {
                         "RecognitionCore", "MaskAwareDtw", "PersonalizationCandidateRepository",
                         "SaveCandidate", "SetExpectedWord", "BeginCapture", "FinishExpression"
                     })
                Assert.Zero(Count(story, forbidden), forbidden);
        }

        static void AssertForbiddenAcrossPresentationSources(params string[] forbidden)
        {
            string combined = Read(PresentationSource) + Read(DemonstrationSource) +
                Read(OverlaySource) + Read(TrackingGraphicSource);
            foreach (string token in forbidden) Assert.Zero(Count(combined, token), token);
        }

        static string Read(string relativePath)
        {
            string root = Directory.GetParent(Application.dataPath).FullName;
            return File.ReadAllText(Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar)));
        }

        static T GetField<T>(object target, string name) where T : class
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            return field.GetValue(target) as T;
        }

        static void SetField(object target, string name, object value)
        {
            FieldInfo field = target.GetType().GetField(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(field, name);
            field.SetValue(target, value);
        }

        static object InvokePrivate(object target, string name)
        {
            MethodInfo method = target.GetType().GetMethod(
                name, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.NotNull(method, name);
            return method.Invoke(target, null);
        }

        static string MethodBody(string source, string signature)
        {
            int signatureOffset = source.IndexOf(signature, StringComparison.Ordinal);
            Assert.GreaterOrEqual(signatureOffset, 0, signature);
            int openBrace = source.IndexOf('{', signatureOffset);
            Assert.GreaterOrEqual(openBrace, 0, signature);
            int depth = 0;
            for (int index = openBrace; index < source.Length; index++)
            {
                if (source[index] == '{') depth++;
                else if (source[index] == '}' && --depth == 0)
                    return source.Substring(openBrace + 1, index - openBrace - 1);
            }
            Assert.Zero(1, "Method body is not closed: " + signature);
            return string.Empty;
        }

        static int Count(string text, string token)
        {
            int count = 0;
            int offset = 0;
            while ((offset = text.IndexOf(token, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += token.Length;
            }
            return count;
        }

        static float ReadFloatConstant(string source, string name)
        {
            Match match = Regex.Match(source, @"const float " + name + @"\s*=\s*([0-9.]+)f");
            Assert.True(match.Success, name);
            return float.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        }

        static float ReferenceCenterX()
        {
            return GuidedSignPracticePresentationController.ReferenceWidth * 0.5f;
        }

        static Vector2 InvokeVector2(MethodInfo method, object target, Vector2 value)
        {
            return (Vector2)method.Invoke(target, new object[] { value });
        }
    }
}
