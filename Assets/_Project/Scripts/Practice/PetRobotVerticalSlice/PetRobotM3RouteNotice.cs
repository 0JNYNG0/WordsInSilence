// M3_ROUTE_SOURCE_PREP_A1 (review fix) - draft, NOT installed.
//
// A minimal on-screen notice for the M3 route: the fail-closed error, and the route
// guidance line. It only displays text. It never sets a quest state, an outcome, an item
// grant or any progress value, and it reads nothing from the recognition path.
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public static class PetRobotM3RouteNotice
    {
        // The practice session restores existing enrollment: with a trusted 3/3/3 working
        // copy PetRobotSignPracticeSession.RestorePersistedEnrollment sets State.Completed
        // immediately (lines 478-497). That is continuation of existing training, not new
        // learning, and the wording below must never imply otherwise.
        public const string GuidanceText =
            "로봇 근처에서 E를 눌러 기존 학습 상태를 확인한 뒤, 연습을 나와 NPC에게 인사하세요.";

        static string homeScene;

        /// <summary>
        /// Informational guidance belongs to the scene that created it. A fail-closed
        /// blocked notice stays visible everywhere and is never hidden here.
        /// </summary>
        static void OnActiveSceneChanged(Scene previous, Scene next)
        {
            if (host == null) return;
            if (blockedShown) { host.SetActive(true); return; }
            host.SetActive(string.Equals(next.name, homeScene, System.StringComparison.Ordinal));
        }

        static GameObject host;
        static TextMeshProUGUI label;
        static bool blockedShown;

        /// <summary>Fail-closed error. Shown once; safe to call from any consumer.</summary>
        public static void ShowBlocked()
        {
            if (blockedShown) return;
            blockedShown = true;
            if (host != null) host.SetActive(true);   // a blocked notice is never hidden
            string message = PetRobotPlayerCandidateRootOverride.PublicFailureMessage;
            if (string.IsNullOrEmpty(message))
                message = "데모 데이터 검증에 실패했습니다. 실행을 중단합니다.";
            Show(message, new Color(1f, 0.72f, 0.72f, 1f));
            Debug.LogError("[M3Route] blocked: " + PetRobotPlayerCandidateRootOverride.Failure);
        }

        /// <summary>Route guidance. Purely informational.</summary>
        public static void ShowGuidance()
        {
            if (blockedShown) return;              // never hide an error with guidance
            Show(GuidanceText, Color.white);
        }

        static void Show(string message, Color colour)
        {
            if (!EnsureHost())
            {
                Debug.LogError("[M3Route] " + message);   // text asset unavailable
                return;
            }
            label.text = message;
            label.color = colour;
        }

        static bool EnsureHost()
        {
            if (label != null) return true;
            TMP_FontAsset font = TMP_Settings.defaultFontAsset;
            if (font == null) return false;

            host = new GameObject("M3RouteNotice");
            Object.DontDestroyOnLoad(host);
            homeScene = SceneManager.GetActiveScene().name;
            SceneManager.activeSceneChanged -= OnActiveSceneChanged;
            SceneManager.activeSceneChanged += OnActiveSceneChanged;
            var canvas = host.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 5000;
            var scaler = host.AddComponent<UnityEngine.UI.CanvasScaler>();
            scaler.uiScaleMode = UnityEngine.UI.CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);

            var textHost = new GameObject("Text");
            textHost.transform.SetParent(host.transform, false);
            label = textHost.AddComponent<TextMeshProUGUI>();
            label.font = font;
            label.fontSize = 28f;
            label.alignment = TextAlignmentOptions.Top;
            label.enableWordWrapping = true;
            var rect = (RectTransform)textHost.transform;
            rect.anchorMin = new Vector2(0.5f, 1f);
            rect.anchorMax = new Vector2(0.5f, 1f);
            rect.pivot = new Vector2(0.5f, 1f);
            rect.sizeDelta = new Vector2(1200f, 120f);
            rect.anchoredPosition = new Vector2(0f, -40f);
            return true;
        }

        /// <summary>Sets the visible line. Display only; no state is written anywhere.</summary>
        public static void ShowStage(string message)
        {
            if (blockedShown) return;
            Show(message, Color.white);
        }
    }
}
