using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    /// <summary>
    /// Minimal MOA/webcam feedback presenter for the in-world greeting NPC dialogue.
    /// Shows feedback text and webcam preview state during recognition.
    /// Wire to moaDialogueRoot children in Inspector.
    /// </summary>
    public sealed class InWorldGreetingNpcDialogueFeedbackPresenter : MonoBehaviour
    {
        [SerializeField] TextMeshProUGUI feedbackLabel;
        [SerializeField] TextMeshProUGUI statusLabel;
        [SerializeField] InWorldGreetingNpcDialogueController dialogueController;

        void LateUpdate()
        {
            if (dialogueController == null) return;

            if (feedbackLabel != null)
                feedbackLabel.text = dialogueController.LastFeedbackText;

            if (statusLabel != null)
            {
                switch (dialogueController.State)
                {
                    case InWorldDialogueState.Idle:
                        statusLabel.text = string.Empty;
                        break;
                    case InWorldDialogueState.QuestPrompt:
                        statusLabel.text = string.Empty;
                        break;
                    case InWorldDialogueState.DialogueActive:
                        statusLabel.text = "\uc900\ube44 \uc911\u2026";
                        break;
                    case InWorldDialogueState.TrackingPreparation:
                        statusLabel.text = "\uce74\uba54\ub77c\ub97c \ud655\uc778\ud558\uace0 \uc788\uc5b4\uc694\u2026";
                        break;
                    case InWorldDialogueState.NeutralPreparation:
                        statusLabel.text = "\uc190\uc744 \ub0b4\ub9ac\uace0 \ud3b8\uc548\ud558\uac8c \uc11c \uc8fc\uc138\uc694.";
                        break;
                    case InWorldDialogueState.ReadyForGreeting:
                        statusLabel.text = "\uc900\ube44 \ub410! \uc778\uc0ac \ub3d9\uc791\uc744 \ud574 \ubcf4\uc138\uc694.";
                        break;
                    case InWorldDialogueState.Capturing:
                        statusLabel.text = "\ub3d9\uc791\uc744 \uc778\uc2dd\ud558\uace0 \uc788\uc5b4\uc694\u2026";
                        break;
                    case InWorldDialogueState.Analyzing:
                        statusLabel.text = "\ubd84\uc11d \uc911\u2026";
                        break;
                    case InWorldDialogueState.RecognitionActive:
                        statusLabel.text = "\uc778\uc2dd \uc900\ube44 \uc911\u2026";
                        break;
                    case InWorldDialogueState.Feedback:
                        statusLabel.text = dialogueController.CanRetry
                            ? "R\uc744 \ub20c\ub7ec \ub2e4\uc2dc \uc2dc\ub3c4\ud558\uc138\uc694"
                            : string.Empty;
                        break;
                    case InWorldDialogueState.QuestComplete:
                        statusLabel.text = "\ud034\uc2a4\ud2b8 \uc644\ub8cc! E\ub97c \ub20c\ub7ec \ub2eb\uc73c\uc138\uc694.";
                        break;
                }
            }
        }
    }
}
