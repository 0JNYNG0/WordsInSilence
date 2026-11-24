namespace WordsInSilence.Recognition.ExpectedSign
{
    /// <summary>
    /// 수어 인식에 사용할 알고리즘 모드.
    /// 기능 플래그로 동작하며 기본값은 LegacyDtw.
    /// </summary>
    public enum RecognitionMode
    {
        /// <summary>
        /// 기존 DTW + MotionScore 알고리즘 (HandPoseEvaluator.Compare).
        /// 20클래스 폐쇄형. 정답 수어 ID 없이도 동작.
        /// </summary>
        LegacyDtw,

        /// <summary>
        /// 기존 LinearSVM 알고리즘 (SVMClassifier.Classify).
        /// 26피처, StreamingAssets/svm_model.json 필요.
        /// </summary>
        LegacySvm,

        /// <summary>
        /// 새 expected-sign 프로토타입.
        /// 정답 수어 ID를 미리 알고 있다는 전제.
        /// 좌표 표준화 → 구간 검출 → 해당 수어 템플릿과 비교.
        /// </summary>
        ExpectedSignPrototype,
    }
}
