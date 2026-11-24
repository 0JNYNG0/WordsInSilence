using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    public sealed class PetRobotWorldUiPresenter : MonoBehaviour
    {
        [SerializeField] PetRobotWorldEntryController entryController;
        [SerializeField] Font portfolioFont;
        [SerializeField] Rect panelRect = new Rect(24f, 24f, 560f, 430f);

        GUIStyle panelStyle;
        GUIStyle titleStyle;
        GUIStyle bodyStyle;
        Texture2D panelTexture;

        // bounded: filled once here, never per frame and never from OnGUI
        WordsInSilence.Practice.PetRobotPortfolioQuest.InWorldGreetingNpcDialogueController[]
            dialogues = new WordsInSilence.Practice.PetRobotPortfolioQuest
                .InWorldGreetingNpcDialogueController[0];

        void Awake()
        {
            dialogues = UnityEngine.Object.FindObjectsOfType<
                WordsInSilence.Practice.PetRobotPortfolioQuest
                    .InWorldGreetingNpcDialogueController>();
        }

        bool AnyDialogueOpen()
        {
            foreach (var dialogue in dialogues)
                if (dialogue != null && dialogue.DialogueOpen) return true;
            return false;
        }

        void EnsureStyles()
        {
            if (bodyStyle != null) return;
            panelTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false)
            {
                name = "F30WorldUiPanel",
                hideFlags = HideFlags.HideAndDontSave
            };
            panelTexture.SetPixel(0, 0, new Color(0.012f, 0.03f, 0.055f, 0.94f));
            panelTexture.Apply();
            panelStyle = new GUIStyle(GUI.skin.box);
            panelStyle.normal.background = panelTexture;
            panelStyle.border = new RectOffset(0, 0, 0, 0);
            panelStyle.padding = new RectOffset(14, 14, 12, 12);
            titleStyle = new GUIStyle(GUI.skin.label)
            {
                font = portfolioFont,
                fontSize = 24,
                // F30: BMJUA is already heavy; simulated bold only blurs it
                fontStyle = FontStyle.Normal,
                wordWrap = true,
            };
            titleStyle.normal.textColor = Color.white;
            bodyStyle = new GUIStyle(GUI.skin.label)
            {
                font = portfolioFont,
                fontSize = 19,
                wordWrap = true,
            };
            bodyStyle.normal.textColor = Color.white;
        }

        void OnDestroy()
        {
            if (panelTexture == null) return;
            if (Application.isPlaying) Destroy(panelTexture);
            else DestroyImmediate(panelTexture);
            panelTexture = null;
        }

        void OnGUI()
        {
            EnsureStyles();
            // the NPC dialogue owns the screen while it is open; this panel is informational
            if (AnyDialogueOpen()) return;

            GUILayout.BeginArea(Clamp(panelRect), panelStyle);
            GUILayout.Label("애완 로봇이 있는 연습 마당", titleStyle);
            GUILayout.Label("이동: WASD / 방향키", bodyStyle);
            if (entryController == null)
            {
                GUILayout.Label("World integration 준비 실패", bodyStyle);
                GUILayout.EndArea();
                return;
            }
            if (entryController.TransitionRequested)
                GUILayout.Label("연습 공간으로 이동 중...", bodyStyle);
            else if (entryController.PlayerInRange)
                GUILayout.Label("E / Enter / Space: 애완 로봇과 수어 연습", bodyStyle);
            else
                GUILayout.Label("애완 로봇에게 가까이 가 보세요.", bodyStyle);
            if (!string.IsNullOrEmpty(entryController.LastTransitionError))
                GUILayout.Label("이동 오류: " + entryController.LastTransitionError, bodyStyle);

            PetRobotPracticeCompletionSnapshot result = entryController.LastCompletion;
            if (result != null)
            {
                GUILayout.Space(12f);
                GUILayout.Label(entryController.CompletionMessage, bodyStyle);
                foreach (string word in result.Words)
                {
                    PetRobotWordLearningStatus status;
                    if (result.TryGetWordStatus(word, out status))
                        GUILayout.Label(word + ": " + status, bodyStyle);
                }
                GUILayout.Label("익힌 표현: " + result.MasteredCount + " / " + result.Words.Count,
                    bodyStyle);
                GUILayout.Label("연습이 더 필요한 표현: " + result.NeedsPracticeCount, bodyStyle);
                if (result.IsFullHandRecordRound)
                {
                    GUILayout.Label("연습 기록: " + result.FullHandRecordTotal, bodyStyle);
                    GUILayout.Label("기록을 모두 모았는지: " +
                        (result.AllRecordsCollected ? "예" : "아니요"), bodyStyle);
                }
                else
                {
                    GUILayout.Label("저장된 후보 기록: " + result.CandidateSavedTotal, bodyStyle);
                }
            }
            GUILayout.EndArea();
        }

        // keeps the panel on screen at any window size instead of shrinking the text
        static Rect Clamp(Rect requested)
        {
            float width = Mathf.Min(requested.width, Mathf.Max(280f, Screen.width - 48f));
            float height = Mathf.Min(requested.height,
                Mathf.Max(180f, Screen.height - requested.y - 24f));
            float x = Mathf.Min(requested.x, Mathf.Max(0f, Screen.width - width));
            return new Rect(x, requested.y, width, height);
        }
    }
}
