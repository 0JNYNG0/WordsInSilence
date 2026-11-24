using System;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// World-scene only demo host. It activates the fresh-participant storage identity before any
    /// other scene component's Awake runs, and offers a single reset control.
    ///
    /// Reset semantics: switching the active participant to a new empty GUID namespace. Nothing is
    /// copied, backed up or deleted - previous personal records stay on disk and simply stop being
    /// used by the current participant. AIHub / training / three-view source data is never touched.
    ///
    /// Failure policy: if storage initialization fails, the host must not let the game continue on
    /// the legacy identity. It therefore blocks interaction by releasing the candidate camera route,
    /// disabling the scene's dialogue/world-entry input behaviours and setting Time.timeScale to 0
    /// (restored only if this host set it, and only on a successful reset or scene exit).
    /// </summary>
    [DefaultExecutionOrder(-32000)]
    [DisallowMultipleComponent]
    public sealed class DemoParticipantSessionHost : MonoBehaviour
    {
        const string StorageRootFolder = "DemoParticipantsV1";

        const string ResetButtonLabel = "개인화 초기화 · 새 참가자";
        const string HelpText =
            "새 참가자는 개인화 기록 0개로 시작합니다. 이전 기록은 보관되며 현재 참가자에게 사용하지 않습니다.";
        const string StatusReady = "현재: 새 참가자 세션으로 식별 중";
        const string StatusBusy = "대화 또는 이동 중에는 초기화할 수 없습니다.";
        const string StatusStorageFailed = "저장소 오류로 초기화하지 못했습니다. 현재 참가자는 그대로입니다.";
        const string StatusBlocked = "개인화 세션을 시작하지 못했습니다. 게임을 계속할 수 없습니다.";
        const string StatusTransitionFailed =
            "새 참가자로 전환했지만 장면을 다시 불러오지 못했습니다. 게임을 다시 시작해 주세요.";

        [SerializeField] Font portfolioFont;

        string statusText = StatusReady;
        bool blocked;
        bool hostChangedTimeScale;
        float previousTimeScale = 1f;
        GUIStyle panelStyle;
        GUIStyle labelStyle;
        GUIStyle buttonStyle;
        Texture2D panelTexture;

        WordsInSilence.Practice.PetRobotPortfolioQuest.InWorldGreetingNpcDialogueController[] dialogues =
            Array.Empty<WordsInSilence.Practice.PetRobotPortfolioQuest.InWorldGreetingNpcDialogueController>();
        WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldEntryController[] entries =
            Array.Empty<WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldEntryController>();

        void Awake()
        {
            try
            {
                // already active (e.g. additive reload): keep the identity we are running on
                DemoParticipantSession.Activate(
                    Path.Combine(Application.persistentDataPath, StorageRootFolder));
                statusText = StatusReady;
            }
            catch (Exception)
            {
                blocked = true;
                statusText = StatusBlocked;
                BlockInteractions();
            }
            RefreshSceneBehaviours();
        }

        // no per-frame or timed scene-wide search: the cache is filled at Awake and refreshed
        // once more at the moment the button is pressed, which is when the state must be exact
        void RefreshSceneBehaviours()
        {
            dialogues = UnityEngine.Object
                .FindObjectsOfType<WordsInSilence.Practice.PetRobotPortfolioQuest.InWorldGreetingNpcDialogueController>();
            entries = UnityEngine.Object
                .FindObjectsOfType<WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldEntryController>();
        }

        bool CanReset()
        {
            if (blocked) return false;
            foreach (var dialogue in dialogues)
                if (dialogue != null && dialogue.DialogueOpen) return false;
            foreach (var entry in entries)
                if (entry != null && entry.TransitionRequested) return false;
            return Application.CanStreamedLevelBeLoaded(SceneManager.GetActiveScene().name);
        }

        void EnsureStyles()
        {
            if (labelStyle != null) return;
            panelTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false)
            {
                name = "F30DemoHostPanel",
                hideFlags = HideFlags.HideAndDontSave
            };
            panelTexture.SetPixel(0, 0, new Color(0.012f, 0.03f, 0.055f, 0.94f));
            panelTexture.Apply();
            panelStyle = new GUIStyle(GUI.skin.box);
            panelStyle.normal.background = panelTexture;
            panelStyle.border = new RectOffset(0, 0, 0, 0);
            labelStyle = new GUIStyle(GUI.skin.label)
            {
                font = portfolioFont,
                fontSize = 19,
                wordWrap = true,
                richText = false,
                alignment = TextAnchor.UpperLeft,
            };
            labelStyle.normal.textColor = Color.white;
            buttonStyle = new GUIStyle(GUI.skin.button)
            {
                font = portfolioFont,
                fontSize = 19,
                wordWrap = true,
                alignment = TextAnchor.MiddleCenter,
            };
            buttonStyle.normal.textColor = Color.white;
            buttonStyle.hover.textColor = Color.white;
            buttonStyle.active.textColor = Color.white;
        }

        // reads the same bounded cache the reset guard reads; no scene search happens here
        bool AnyDialogueOpen()
        {
            foreach (var dialogue in dialogues)
                if (dialogue != null && dialogue.DialogueOpen) return true;
            return false;
        }

        void OnGUI()
        {
            EnsureStyles();

            // the bottom NPC dialogue owns that corner while it is open, and a reset is refused
            // in that state anyway; a blocked or failed status is never hidden
            if (!blocked && AnyDialogueOpen()) return;

            const float margin = 16f;
            const float padding = 12f;
            float width = Mathf.Clamp(Screen.width - 2f * margin, 240f, 430f);
            float inner = width - 2f * padding;
            float statusHeight = labelStyle.CalcHeight(new GUIContent(statusText), inner);
            float helpHeight = labelStyle.CalcHeight(new GUIContent(HelpText), inner);
            float buttonHeight = Mathf.Max(
                44f, buttonStyle.CalcHeight(new GUIContent(ResetButtonLabel), inner) + 14f);
            float height = Mathf.Min(
                2f * padding + statusHeight + 8f + helpHeight + 12f + buttonHeight,
                Mathf.Max(120f, Screen.height - 2f * margin));
            var panel = new Rect(Screen.width - width - margin,
                                 Screen.height - height - margin, width, height);

            Font previousFont = GUI.skin.font;
            Color previousColor = GUI.color;
            bool wasEnabled = GUI.enabled;
            if (portfolioFont != null) GUI.skin.font = portfolioFont;
            try
            {
                GUI.color = Color.white;
                GUI.Box(panel, GUIContent.none, panelStyle);
                GUILayout.BeginArea(new Rect(panel.x + padding, panel.y + padding,
                                             inner, panel.height - 2f * padding));
                // always the real status, so a scene/storage error is never hidden by a generic one
                GUILayout.Label(statusText, labelStyle, GUILayout.Width(inner));
                GUILayout.Space(8f);
                GUILayout.Label(HelpText, labelStyle, GUILayout.Width(inner));
                GUILayout.Space(12f);
                GUI.enabled = wasEnabled && CanReset();
                if (GUILayout.Button(ResetButtonLabel, buttonStyle,
                                     GUILayout.Width(inner), GUILayout.Height(buttonHeight)))
                    PerformReset();
                GUI.enabled = wasEnabled;
                GUILayout.EndArea();
            }
            finally
            {
                GUI.skin.font = previousFont;
                GUI.color = previousColor;
                GUI.enabled = wasEnabled;
            }
        }

        void PerformReset()
        {
            RefreshSceneBehaviours();
            if (!CanReset()) { statusText = blocked ? StatusBlocked : StatusBusy; return; }

            try
            {
                DemoParticipantSession.StartNew();
            }
            catch (Exception)
            {
                // nothing was committed: session, controllers and game state are unchanged
                statusText = StatusStorageFailed;
                return;
            }

            try
            {
                PetRobotCandidateCameraStartupRoute.ReleaseAllForSceneTransition();
                WordsInSilence.Practice.PetRobotWorldIntegration.PetRobotWorldHandoffChannel.Shared
                    .ResetForNewParticipant();
                RestoreTimeScaleIfHostChangedIt();
                SceneManager.LoadScene(SceneManager.GetActiveScene().name);
                statusText = StatusReady;
            }
            catch (Exception)
            {
                // the identity did switch, but the running controllers were not rebuilt:
                // block instead of pretending the reset reached them
                blocked = true;
                statusText = StatusTransitionFailed;
                BlockInteractions();
            }
        }

        void BlockInteractions()
        {
            try { PetRobotCandidateCameraStartupRoute.ReleaseAllForSceneTransition(); }
            catch (Exception) { /* blocking must not throw */ }

            RefreshSceneBehaviours();
            foreach (var dialogue in dialogues) if (dialogue != null) dialogue.enabled = false;
            foreach (var entry in entries) if (entry != null) entry.enabled = false;

            if (!hostChangedTimeScale)
            {
                previousTimeScale = Time.timeScale;
                hostChangedTimeScale = true;
            }
            Time.timeScale = 0f;
        }

        void RestoreTimeScaleIfHostChangedIt()
        {
            if (!hostChangedTimeScale) return;
            Time.timeScale = previousTimeScale;
            hostChangedTimeScale = false;
        }

        void OnDestroy()
        {
            RestoreTimeScaleIfHostChangedIt();
            if (panelTexture == null) return;
            if (Application.isPlaying) Destroy(panelTexture);
            else DestroyImmediate(panelTexture);
            panelTexture = null;
        }
    }
}
