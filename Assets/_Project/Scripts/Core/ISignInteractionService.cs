using System.Threading;
using System.Threading.Tasks;

namespace WordsInSilence.Core
{
    /// <summary>
    /// 수어 상호작용 전체 흐름을 조율하는 서비스 인터페이스 (Phase 3~4에서 구현 예정).
    /// 카운트다운 → 녹화 → 평가의 전체 파이프라인을 실행한다.
    /// </summary>
    public interface ISignInteractionService
    {
        SignInteractionState CurrentState { get; }

        /// <summary>
        /// 수어 인식 시퀀스를 실행한다.
        /// </summary>
        /// <param name="expectedSignId">
        /// 예상하는 수어 ID. null이면 모든 수어를 인식 대상으로 한다.
        /// </param>
        /// <param name="ct">취소 토큰.</param>
        /// <returns>상호작용 결과.</returns>
        Task<SignInteractionResult> RunInteractionAsync(string expectedSignId, CancellationToken ct);

        void Cancel();
    }

    /// <summary>
    /// 수어 상호작용 상태.
    /// </summary>
    public enum SignInteractionState
    {
        Idle,
        Countdown,
        Recording,
        Evaluating,
        Success,
        Failure,
        Cancelled
    }

    /// <summary>
    /// 수어 상호작용 결과.
    /// </summary>
    public sealed class SignInteractionResult
    {
        public SignInteractionState FinalState;
        public MotionEvaluationResult EvaluationResult;
        public string ErrorCode;
    }
}
