namespace WordsInSilence.Core
{
    /// <summary>
    /// 수어 동작 평가 인터페이스 (Phase 3에서 구현체 추가 예정).
    /// SignMotionSample을 수어 사전과 비교하여 인식 결과를 반환한다.
    /// </summary>
    public interface IMotionEvaluator
    {
        /// <summary>
        /// 녹화된 동작 샘플을 평가한다.
        /// </summary>
        /// <param name="sample">녹화된 수어 동작 샘플.</param>
        /// <returns>평가 결과. 인식 실패 시 null 또는 IsRecognized == false.</returns>
        MotionEvaluationResult Evaluate(SignMotionSample sample);
    }

    /// <summary>
    /// 수어 동작 평가 결과 (Phase 3에서 상세 정의 예정).
    /// </summary>
    public sealed class MotionEvaluationResult
    {
        /// <summary>인식 성공 여부.</summary>
        public bool IsRecognized;

        /// <summary>인식된 수어 ID (수어 사전 키). 실패 시 null.</summary>
        public string SignId;

        /// <summary>유사도 점수 (0~1).</summary>
        public float Confidence;

        /// <summary>평가 실패 오류 코드 (ERROR_CODES.md 기준). 성공 시 null.</summary>
        public string ErrorCode;
    }
}
