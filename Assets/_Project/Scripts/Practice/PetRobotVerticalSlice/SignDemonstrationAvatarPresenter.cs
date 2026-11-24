using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Presentation-only world stage rendered by the practice Scene's Main Camera. Each sign
    /// keeps its model and embedded clip paired through a project-owned nested-FBX wrapper.
    /// </summary>
    public sealed class SignDemonstrationAvatarPresenter : MonoBehaviour, ISignDemonstrationSlot
    {
        public const string GreetingWord = "GREETING";
        public const string LoveWord = "LOVE";
        public const string ThankYouWord = "THANK_YOU";

        [Header("Project-owned nested-FBX wrappers + existing embedded clips")]
        [SerializeField] GameObject greetingCharacter;
        [SerializeField] AnimationClip greetingClip;
        [SerializeField] GameObject loveCharacter;
        [SerializeField] AnimationClip loveClip;
        [SerializeField] GameObject thankYouCharacter;
        [SerializeField] AnimationClip thankYouClip;

        [Header("Main Camera world-stage placement")]
        [SerializeField] GameObject robotVisualPrefab;
        [SerializeField] RuntimeAnimatorController presentationRobotController;
        [SerializeField] Vector3 demonstrationPosition = new Vector3(-6.65f, -3.2f, -0.65f);
        [SerializeField] Vector3 demonstrationScale = new Vector3(3.25f, 3.25f, 3.25f);
        [SerializeField] float demonstrationYaw = 180f;
        [SerializeField] float demonstrationYawSpeed = 80f;
        [SerializeField] Vector3 robotPosition = new Vector3(5.85f, 1.95f, 0.35f);
        [SerializeField] Vector3 robotScale = new Vector3(0.68f, 0.68f, 0.68f);
        [SerializeField] float robotYaw = 220f;

        GameObject stageRoot;
        GameObject demonstrationInstance;
        GameObject robotInstance;
        Animator demonstrationAnimator;
        Animator robotAnimator;
        PlayableGraph demonstrationGraph;
        AnimationClipPlayable demonstrationPlayable;
        AnimationClip currentClip;
        readonly List<Material> ownedStageMaterials = new List<Material>();
        string currentWord = string.Empty;
        string status = "시범 준비 중";
        float currentDemonstrationYaw;
        bool demonstrationPlaybackPaused;

        public string CurrentWord => currentWord;
        public string Status => status;
        public float CurrentDemonstrationYaw => currentDemonstrationYaw;
        public Transform DemonstrationRoot =>
            demonstrationInstance == null ? null : demonstrationInstance.transform;
        public Animator DemonstrationAnimator => demonstrationAnimator;
        public Animator RobotAnimator => robotAnimator;
        public bool DemonstrationGraphValid => demonstrationGraph.IsValid();
        public double CurrentPlayableTime =>
            demonstrationPlayable.IsValid() ? demonstrationPlayable.GetTime() : 0.0;
        public string CurrentClipName => currentClip == null ? string.Empty : currentClip.name;
        public float CurrentClipLength => currentClip == null ? 0f : currentClip.length;
        public bool DemonstrationPlaybackPaused => demonstrationPlaybackPaused;
        public string DemonstrationAnimatorPath { get; private set; } = string.Empty;
        public bool DemonstrationAnimatorRuntimeCreated { get; private set; }
        public string RobotAnimatorPath { get; private set; } = string.Empty;
        public bool RobotControllerAssigned =>
            robotAnimator != null && robotAnimator.runtimeAnimatorController != null;
        public float RobotNormalizedTime
        {
            get
            {
                if (robotAnimator == null || !robotAnimator.isActiveAndEnabled ||
                    robotAnimator.layerCount <= 0) return 0f;
                return robotAnimator.GetCurrentAnimatorStateInfo(0).normalizedTime;
            }
        }
        public Vector3 DemonstrationWorldPosition => demonstrationPosition;
        public Vector3 RobotWorldPosition => robotPosition;
        public bool DemonstrationRootMotionEnabled { get; private set; }
        public bool RobotRootMotionEnabled { get; private set; }
        public bool UsesMainSceneCamera => true;
        public bool UsesPresentationOnlyCameras => false;
        public int StageDecorativeElementCount { get; private set; }

        void Awake()
        {
            currentDemonstrationYaw = demonstrationYaw;
            CreateWorldStage();
        }

        void Update()
        {
            UpdateDemonstrationYawInput();
            AdvanceDemonstrationPresentation(Time.unscaledDeltaTime);
        }

        void OnDestroy()
        {
            DestroyDemonstration();
            DestroyOwned(stageRoot);
            foreach (Material material in ownedStageMaterials) DestroyOwned(material);
            ownedStageMaterials.Clear();
        }

        public void ShowDemonstration(string word)
        {
            if (word == currentWord && demonstrationInstance != null) return;

            GameObject character;
            AnimationClip clip;
            if (!TryResolve(word, out character, out clip))
            {
                ClearDemonstration();
                status = "지원하지 않는 시범 단어";
                return;
            }

            CreateWorldStage();
            DestroyDemonstration();
            currentDemonstrationYaw = demonstrationYaw;
            demonstrationInstance = Instantiate(character, stageRoot.transform, false);
            demonstrationInstance.name = "SignDemonstration_" + word;
            demonstrationInstance.transform.localPosition = demonstrationPosition;
            demonstrationInstance.transform.localRotation = Quaternion.Euler(0f, currentDemonstrationYaw, 0f);
            demonstrationInstance.transform.localScale = demonstrationScale;

            bool animatorWasCreated;
            Animator animator = ResolveOrCreateDemonstrationAnimator(
                demonstrationInstance, out animatorWasCreated);
            if (animator == null)
            {
                status = "시범 캐릭터 Animator를 찾지 못했습니다";
                return;
            }

            demonstrationAnimator = animator;
            DemonstrationAnimatorRuntimeCreated = animatorWasCreated;
            DemonstrationAnimatorPath = RelativeHierarchyPath(
                demonstrationInstance.transform, animator.transform);
            animator.enabled = true;
            animator.applyRootMotion = false;
            animator.speed = 1f;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            animator.updateMode = AnimatorUpdateMode.UnscaledTime;
            DemonstrationRootMotionEnabled = animator.applyRootMotion;
            currentClip = clip;
            demonstrationGraph = PlayableGraph.Create("Step10F_v1_3_" + word + "_PresentationLoop");
            demonstrationGraph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
            demonstrationPlayable = AnimationClipPlayable.Create(demonstrationGraph, clip);
            demonstrationPlayable.SetApplyFootIK(false);
            demonstrationPlayable.SetSpeed(1.0);
            demonstrationPlayable.SetTime(0.0);
            demonstrationPlayable.SetDone(false);
            AnimationPlayableOutput output = AnimationPlayableOutput.Create(
                demonstrationGraph, "Step10F v1.3 presentation demonstration", animator);
            output.SetSourcePlayable(demonstrationPlayable);
            output.SetWeight(1f);
            currentWord = word;
            demonstrationGraph.Play();
            demonstrationGraph.Evaluate(0f);
            demonstrationPlaybackPaused = false;
            ApplyCurrentClipPoseForRuntimeCreatedAnimator();
            status = KoreanWord(word) + " 시범 반복 재생";
        }

        public void ClearDemonstration()
        {
            DestroyDemonstration();
            currentWord = string.Empty;
            demonstrationPlaybackPaused = false;
            status = "시범 완료";
        }

        public void SetDemonstrationPlaybackPaused(bool paused, bool resetToStartPose)
        {
            if (!demonstrationGraph.IsValid() || !demonstrationPlayable.IsValid()) return;
            if (resetToStartPose)
            {
                demonstrationPlayable.SetTime(0.0);
                demonstrationPlayable.SetDone(false);
                demonstrationGraph.Evaluate(0f);
                ApplyCurrentClipPoseForRuntimeCreatedAnimator();
            }
            demonstrationPlaybackPaused = paused;
            demonstrationPlayable.SetSpeed(paused ? 0.0 : 1.0);
            status = paused
                ? KoreanWord(currentWord) + " 시범 일시 정지"
                : KoreanWord(currentWord) + " 시범 반복 재생";
        }

        public static string KoreanWord(string word)
        {
            if (word == GreetingWord) return "인사";
            if (word == LoveWord) return "사랑";
            if (word == ThankYouWord) return "감사합니다";
            return word ?? string.Empty;
        }

        void CreateWorldStage()
        {
            if (stageRoot != null) return;
            stageRoot = new GameObject("Step10F_v1_3b_FuturisticPracticeChamber");
            stageRoot.transform.SetParent(transform, false);
            CreatePracticeChamber();

            if (robotVisualPrefab == null) return;
            robotInstance = Instantiate(robotVisualPrefab, stageRoot.transform, false);
            robotInstance.name = "Step10F_v1_2_PresentationOnly_MOA_robot";
            robotInstance.transform.localPosition = robotPosition;
            robotInstance.transform.localRotation = Quaternion.Euler(0f, robotYaw, 0f);
            robotInstance.transform.localScale = robotScale;
            robotAnimator = InitializePresentationRobotAnimator(
                robotInstance, presentationRobotController);
            if (robotAnimator == null) return;
            RobotAnimatorPath = RelativeHierarchyPath(robotInstance.transform, robotAnimator.transform);
            RobotRootMotionEnabled = robotAnimator.applyRootMotion;
        }

        void CreatePracticeChamber()
        {
            StageDecorativeElementCount = 0;
            Material backdrop = CreateStageMaterial(
                "Step10F_v1_3b_DeepNavy", new Color(0.012f, 0.025f, 0.055f, 1f));
            Material floor = CreateStageMaterial(
                "Step10F_v1_3b_TrainingFloor", new Color(0.018f, 0.085f, 0.12f, 1f));
            Material cyan = CreateStageMaterial(
                "Step10F_v1_3b_CyanAccent", new Color(0.08f, 0.72f, 0.9f, 1f));
            if (backdrop == null || floor == null || cyan == null) return;

            CreateStagePrimitive("DarkNavyChamberWall", PrimitiveType.Cube,
                new Vector3(0f, 1.25f, 4.4f), new Vector3(20f, 11f, 0.2f),
                Quaternion.identity, backdrop);
            CreateStagePrimitive("CentralCircularTrainingFloor", PrimitiveType.Cylinder,
                new Vector3(0f, -3.05f, 0.7f), new Vector3(11.2f, 0.035f, 6.7f),
                Quaternion.identity, floor);
            CreateStagePrimitive("LeftCyanChamberPillar", PrimitiveType.Cube,
                new Vector3(-7.65f, 0.75f, 4.15f), new Vector3(0.075f, 6.6f, 0.14f),
                Quaternion.identity, cyan);
            CreateStagePrimitive("RightCyanChamberPillar", PrimitiveType.Cube,
                new Vector3(7.65f, 0.75f, 4.15f), new Vector3(0.075f, 6.6f, 0.14f),
                Quaternion.identity, cyan);
            CreateStageRing("CentralFloorCyanRing", new Vector3(0f, -2.96f, 0.7f),
                new Vector2(5.25f, 3.05f), Quaternion.Euler(90f, 0f, 0f), cyan, 0.055f);
            CreateStageRing("RearChamberCyanArc", new Vector3(0f, 1.15f, 4.18f),
                new Vector2(7.2f, 4.7f), Quaternion.identity, cyan, 0.045f);
        }

        Material CreateStageMaterial(string name, Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) return null;
            Material material = new Material(shader)
            {
                name = name,
                hideFlags = HideFlags.HideAndDontSave
            };
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            else if (material.HasProperty("_Color")) material.SetColor("_Color", color);
            if (material.HasProperty("_Cull")) material.SetFloat("_Cull", 0f);
            ownedStageMaterials.Add(material);
            return material;
        }

        void CreateStagePrimitive(
            string name,
            PrimitiveType primitiveType,
            Vector3 localPosition,
            Vector3 localScale,
            Quaternion localRotation,
            Material material)
        {
            GameObject value = GameObject.CreatePrimitive(primitiveType);
            value.name = name;
            value.transform.SetParent(stageRoot.transform, false);
            value.transform.localPosition = localPosition;
            value.transform.localRotation = localRotation;
            value.transform.localScale = localScale;
            Renderer renderer = value.GetComponent<Renderer>();
            if (renderer != null) renderer.sharedMaterial = material;
            Collider collider = value.GetComponent<Collider>();
            if (collider != null) DestroyOwned(collider);
            StageDecorativeElementCount++;
        }

        void CreateStageRing(
            string name,
            Vector3 localPosition,
            Vector2 radius,
            Quaternion localRotation,
            Material material,
            float width)
        {
            GameObject value = new GameObject(name, typeof(LineRenderer));
            value.transform.SetParent(stageRoot.transform, false);
            value.transform.localPosition = localPosition;
            value.transform.localRotation = localRotation;
            LineRenderer ring = value.GetComponent<LineRenderer>();
            ring.useWorldSpace = false;
            ring.loop = true;
            ring.positionCount = 64;
            ring.startWidth = ring.endWidth = width;
            ring.numCornerVertices = 2;
            ring.numCapVertices = 2;
            ring.sharedMaterial = material;
            for (int index = 0; index < ring.positionCount; index++)
            {
                float angle = Mathf.PI * 2f * index / ring.positionCount;
                ring.SetPosition(index,
                    new Vector3(Mathf.Cos(angle) * radius.x, Mathf.Sin(angle) * radius.y, 0f));
            }
            StageDecorativeElementCount++;
        }

        void UpdateDemonstrationYawInput()
        {
            if (demonstrationInstance == null) return;
            float direction = 0f;
            if (Input.GetKey(KeyCode.LeftArrow)) direction += 1f;
            if (Input.GetKey(KeyCode.RightArrow)) direction -= 1f;
            if (Mathf.Approximately(direction, 0f)) return;
            ApplyDemonstrationYawDelta(direction * demonstrationYawSpeed * Time.unscaledDeltaTime);
        }

        public void ApplyDemonstrationYawDelta(float yawDeltaDegrees)
        {
            if (demonstrationInstance == null || float.IsNaN(yawDeltaDegrees) ||
                float.IsInfinity(yawDeltaDegrees)) return;
            currentDemonstrationYaw = Mathf.Repeat(currentDemonstrationYaw + yawDeltaDegrees, 360f);
            demonstrationInstance.transform.localRotation =
                Quaternion.Euler(0f, currentDemonstrationYaw, 0f);
        }

        public void AdvanceDemonstrationPresentation(float unscaledDeltaSeconds)
        {
            if (!demonstrationGraph.IsValid() || !demonstrationPlayable.IsValid()) return;
            if (demonstrationPlaybackPaused) return;
            if (unscaledDeltaSeconds <= 0f || float.IsNaN(unscaledDeltaSeconds) ||
                float.IsInfinity(unscaledDeltaSeconds)) return;
            demonstrationGraph.Evaluate(unscaledDeltaSeconds);
            double duration = currentClip == null ? 0.0 : currentClip.length;
            if (duration > 0.0 && demonstrationPlayable.GetTime() >= duration)
            {
                demonstrationPlayable.SetTime(demonstrationPlayable.GetTime() % duration);
                demonstrationPlayable.SetDone(false);
                demonstrationGraph.Evaluate(0f);
            }
            ApplyCurrentClipPoseForRuntimeCreatedAnimator();
        }

        void ApplyCurrentClipPoseForRuntimeCreatedAnimator()
        {
            if (!DemonstrationAnimatorRuntimeCreated || demonstrationAnimator == null ||
                currentClip == null || !demonstrationPlayable.IsValid()) return;
            currentClip.SampleAnimation(
                demonstrationAnimator.gameObject, (float)demonstrationPlayable.GetTime());
        }

        public static Animator InitializePresentationRobotAnimator(
            GameObject robotRoot,
            RuntimeAnimatorController expectedController)
        {
            if (robotRoot == null) return null;
            if (expectedController == null) return null;

            Animator[] animators = robotRoot.GetComponentsInChildren<Animator>(true);
            if (animators.Length > 1) return null;

            Animator selected;
            if (animators.Length == 1)
            {
                selected = animators[0];
            }
            else
            {
                if (robotRoot.transform.childCount != 1) return null;
                selected = robotRoot.transform.GetChild(0).gameObject.AddComponent<Animator>();
            }

            if (selected.runtimeAnimatorController == null)
                selected.runtimeAnimatorController = expectedController;
            else if (selected.runtimeAnimatorController != expectedController)
                return null;

            selected.enabled = true;
            selected.speed = 1f;
            selected.applyRootMotion = false;
            selected.cullingMode = AnimatorCullingMode.AlwaysAnimate;
            selected.updateMode = AnimatorUpdateMode.UnscaledTime;
            selected.Rebind();
            selected.Update(0f);
            return selected;
        }

        static Animator ResolveOrCreateDemonstrationAnimator(
            GameObject wrapperRoot,
            out bool runtimeCreated)
        {
            runtimeCreated = false;
            if (wrapperRoot == null) return null;
            Animator[] animators = wrapperRoot.GetComponentsInChildren<Animator>(true);
            if (animators.Length == 1) return animators[0];
            if (animators.Length != 0 || wrapperRoot.transform.childCount != 1) return null;

            runtimeCreated = true;
            return wrapperRoot.transform.GetChild(0).gameObject.AddComponent<Animator>();
        }

        static string RelativeHierarchyPath(Transform root, Transform target)
        {
            if (root == null || target == null) return string.Empty;
            if (root == target) return "<wrapper-root>";
            string value = target.name;
            Transform cursor = target.parent;
            while (cursor != null && cursor != root)
            {
                value = cursor.name + "/" + value;
                cursor = cursor.parent;
            }
            return cursor == root ? value : string.Empty;
        }

        void DestroyDemonstration()
        {
            if (demonstrationGraph.IsValid()) demonstrationGraph.Destroy();
            demonstrationPlayable = default;
            DestroyOwned(demonstrationInstance);
            demonstrationInstance = null;
            demonstrationAnimator = null;
            currentClip = null;
            DemonstrationAnimatorPath = string.Empty;
            DemonstrationAnimatorRuntimeCreated = false;
            DemonstrationRootMotionEnabled = false;
            demonstrationPlaybackPaused = false;
        }

        static void DestroyOwned(Object value)
        {
            if (value == null) return;
            if (Application.isPlaying) Destroy(value);
            else DestroyImmediate(value);
        }

        bool TryResolve(string word, out GameObject character, out AnimationClip clip)
        {
            if (word == GreetingWord)
            {
                character = greetingCharacter;
                clip = greetingClip;
            }
            else if (word == LoveWord)
            {
                character = loveCharacter;
                clip = loveClip;
            }
            else if (word == ThankYouWord)
            {
                character = thankYouCharacter;
                clip = thankYouClip;
            }
            else
            {
                character = null;
                clip = null;
            }
            return character != null && clip != null;
        }
    }
}
