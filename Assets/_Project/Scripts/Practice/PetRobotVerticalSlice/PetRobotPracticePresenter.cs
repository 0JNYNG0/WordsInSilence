using UnityEngine;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>Replaceable pet-robot presentation anchor; no sign animation is synthesized here.</summary>
    public sealed class PetRobotPracticePresenter : MonoBehaviour
    {
        [SerializeField] MonoBehaviour demonstrationSlot;
        string speech = "안녕! 세 가지 수어를 한 동작씩 연습해 보자.";

        public string Speech => speech;
        public ISignDemonstrationSlot DemonstrationSlot => demonstrationSlot as ISignDemonstrationSlot;

        public void ShowWord(string word)
        {
            speech = word + " 동작을 보고 준비되면 ‘표현 시작’을 눌러 줘.";
            DemonstrationSlot?.ShowDemonstration(word);
        }

        public void ShowOutcome(GuidedQuestOutcome outcome)
        {
            if (outcome == null)
                speech = "기술 결과를 확인하지 못했어. 다시 시도해 줘.";
            else if (outcome.Kind == GuidedQuestOutcomeKind.StrictSuccess)
                speech = "정확히 인식했어! 다음 표현으로 갈 수 있어.";
            else if (outcome.Kind == GuidedQuestOutcomeKind.LowConfidenceCorrect)
                speech = "정답 방향은 맞아. 신뢰도가 낮아 복습용으로만 저장했어.";
            else if (outcome.Kind == GuidedQuestOutcomeKind.WrongWord)
                speech = "다른 단어로 인식됐어. 천천히 다시 표현해 보자.";
            else
                speech = "학습자 오답이 아닌 기술 오류야. 준비되면 수동으로 다시 시도해 줘.";
        }

        public void ShowCompleted()
        {
            speech = "세 단어 연습 완료! 개인 프로필 갱신은 아직 활성화하지 않았어.";
            DemonstrationSlot?.ClearDemonstration();
        }

        public void ShowNeedsPractice(string word)
        {
            speech = word + ": 이번 단어는 연습 필요로 남겼어. 인식 결과는 바꾸지 않고 다음 단계로 갈게.";
        }

        public void ShowCompleted(int masteredCount, int needsPracticeCount)
        {
            speech = "연습 세션 완료. Mastered " + masteredCount +
                ", NeedsPractice " + needsPracticeCount +
                ". 개인 프로필은 변경하지 않았어.";
            DemonstrationSlot?.ClearDemonstration();
        }

        public void DrawPlaceholder(Rect rect)
        {
            GUI.Box(rect, GUIContent.none);
            float cx = rect.x + rect.width * 0.5f;
            GUI.Box(new Rect(cx - 54f, rect.y + 26f, 108f, 78f), "PET BOT");
            GUI.Box(new Rect(cx - 34f, rect.y + 50f, 18f, 18f), "•");
            GUI.Box(new Rect(cx + 16f, rect.y + 50f, 18f, 18f), "•");
            GUI.Box(new Rect(cx - 68f, rect.y + 112f, 136f, 70f), "SIGN\nPRACTICE");
        }
    }
}
