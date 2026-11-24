using UnityEngine;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public sealed class PetRobotPortfolioQuestUiPresenter : MonoBehaviour
    {
        [SerializeField] PetRobotPortfolioQuestController controller;
        [SerializeField] Font portfolioFont;
        [SerializeField] Rect objectiveRect = new Rect(32f, 28f, 760f, 108f);
        [SerializeField] Rect dialogueRect = new Rect(280f, 590f, 1360f, 420f);

        GUIStyle titleStyle;
        GUIStyle bodyStyle;
        GUIStyle buttonStyle;

        void OnGUI()
        {
            if (controller == null) return;
            EnsureStyles();

            GUILayout.BeginArea(Clamp(objectiveRect), GUI.skin.box);
            GUILayout.Label("\uD604\uC7AC \uBAA9\uD45C", bodyStyle);
            GUILayout.Label(controller.Objective, titleStyle);
            if (controller.State == PetRobotPortfolioQuestState.AwaitingNpcReport)
            {
                GUILayout.Label(controller.NpcInRange
                    ? "E / Enter / Space : \uAD50\uC721 \uB2F4\uB2F9\uC790\uC5D0\uAC8C \uACB0\uACFC \uBCF4\uACE0"
                    : "\uAD50\uC721 \uB2F4\uB2F9\uC790\uC5D0\uAC8C \uAC00\uAE4C\uC774 \uAC00\uC138\uC694", bodyStyle);
            }
            GUILayout.EndArea();

            if (controller.State == PetRobotPortfolioQuestState.DialogueOpen)
                DrawDialogue();
            else if (controller.State == PetRobotPortfolioQuestState.AwaitingGreetingAttempt)
                DrawGreetingQuest();
            else if (controller.State == PetRobotPortfolioQuestState.DemoCompleted)
                DrawCompleted();
        }

        void DrawDialogue()
        {
            PetRobotPortfolioQuestSnapshot result = controller.Completion;
            if (result == null) return;
            GUILayout.BeginArea(Clamp(dialogueRect), GUI.skin.box);
            GUILayout.Label("\uAD50\uC721 \uB2F4\uB2F9\uC790", titleStyle);
            GUILayout.Label(result.ResultHeadline, bodyStyle);
            GUILayout.Label(result.ResultFollowUp, bodyStyle);
            DrawResult(result);
            string transitionError = controller.LastQuestTransitionError;
            if (!string.IsNullOrEmpty(transitionError))
                GUILayout.Label(transitionError, bodyStyle);
            if (GUILayout.Button("\uBCF4\uACE0 \uB9C8\uCE58\uAE30", buttonStyle))
                controller.CompleteDialogue(true);
            GUILayout.EndArea();
        }

        void DrawGreetingQuest()
        {
            GUILayout.BeginArea(Clamp(dialogueRect), GUI.skin.box);
            GUILayout.Label("GREETING Quest", titleStyle);
            if (controller.QuestTransitionRequested)
            {
                GUILayout.Label("\uC5F0\uC2B5 \uACF5\uAC04\uC73C\uB85C \uC774\uB3D9 \uC911...", bodyStyle);
            }
            else if (controller.CanRetryAfterInitializationFailure)
            {
                string feedback = controller.QuestFeedbackText;
                if (!string.IsNullOrEmpty(feedback))
                    GUILayout.Label(feedback, bodyStyle);
                GUILayout.Label("\uCD08\uAE30\uD654 \uC2E4\uD328\uB85C \uC5F0\uC2B5\uC744 \uC2DC\uC791\uD560 \uC218 \uC5C6\uC5C8\uC2B5\uB2C8\uB2E4.", bodyStyle);
                if (GUILayout.Button("\uB2E4\uC2DC \uC2DC\uB3C4", buttonStyle))
                    controller.TryRetryAfterInitializationFailure();
            }
            else
            {
                string feedback = controller.QuestFeedbackText;
                if (!string.IsNullOrEmpty(feedback))
                    GUILayout.Label(feedback, bodyStyle);
                else
                    GUILayout.Label("GREETING \uC218\uC5B4\uB85C NPC\uC5D0\uAC8C \uC778\uC0AC\uD574 \uBCF4\uC138\uC694.", bodyStyle);
            }
            GUILayout.EndArea();
        }

        void DrawCompleted()
        {
            PetRobotPortfolioQuestSnapshot result = controller.Completion;
            if (result == null) return;
            GUILayout.BeginArea(Clamp(dialogueRect), GUI.skin.box);
            GUILayout.Label("\uB370\uBAA8 \uC644\uB8CC", titleStyle);
            GUILayout.Label("\uC560\uC644 \uB85C\uBD07 \uC218\uC5B4 \uC5F0\uC2B5\uACFC \uAD50\uC721 \uB2F4\uB2F9\uC790 \uBCF4\uACE0\uB97C \uC644\uB8CC\uD588\uC2B5\uB2C8\uB2E4.", bodyStyle);
            DrawResult(result);
            GUILayout.Label("\uD3EC\uD2B8\uD3F4\uB9AC\uC624 \uC218\uC9C1 \uB370\uBAA8\uAC00 \uB05D\uB0AC\uC2B5\uB2C8\uB2E4.", bodyStyle);
            GUILayout.EndArea();
        }

        void DrawResult(PetRobotPortfolioQuestSnapshot result)
        {
            GUILayout.Label("GREETING: " + result.Greeting, bodyStyle);
            GUILayout.Label("LOVE: " + result.Love, bodyStyle);
            GUILayout.Label("THANK_YOU: " + result.ThankYou, bodyStyle);
            GUILayout.Label("Mastered " + result.MasteredCount +
                " / NeedsPractice " + result.NeedsPracticeCount, bodyStyle);
            GUILayout.Label("Candidate saved " + result.CandidateSavedTotal, bodyStyle);
        }

        void EnsureStyles()
        {
            if (titleStyle != null) return;
            titleStyle = new GUIStyle(GUI.skin.label)
            {
                font = portfolioFont,
                fontSize = 28,
                fontStyle = FontStyle.Bold,
                wordWrap = true,
            };
            bodyStyle = new GUIStyle(GUI.skin.label)
            {
                font = portfolioFont,
                fontSize = 21,
                wordWrap = true,
            };
            buttonStyle = new GUIStyle(GUI.skin.button)
            {
                font = portfolioFont,
                fontSize = 22,
                fixedHeight = 52f,
            };
        }

        static Rect Clamp(Rect requested)
        {
            float width = Mathf.Min(requested.width, Mathf.Max(320f, Screen.width - 64f));
            float x = Mathf.Min(requested.x, Mathf.Max(0f, Screen.width - width));
            float height = Mathf.Min(requested.height, Mathf.Max(180f, Screen.height - requested.y - 24f));
            return new Rect(x, requested.y, width, height);
        }
    }
}
