using System.Threading;
using System.Threading.Tasks;

namespace WordsInSilence.Core
{
    /// <summary>
    /// 수어 동작 녹화 인터페이스 (Phase 3에서 구현체 추가 예정).
    /// 랜드마크 프레임 시계열을 수집하여 SignMotionSample로 변환한다.
    /// </summary>
    public interface IMotionRecorder
    {
        bool IsRecording { get; }

        /// <summary>
        /// 동작 녹화를 시작하고 완료되면 SignMotionSample을 반환한다.
        /// </summary>
        /// <param name="maxDurationSeconds">최대 녹화 시간 (초). 초과 시 자동 종료.</param>
        /// <param name="ct">취소 토큰.</param>
        Task<SignMotionSample> RecordAsync(float maxDurationSeconds, CancellationToken ct);

        void StopRecording();
    }

    /// <summary>
    /// 수어 동작 시계열 샘플 (Phase 3에서 상세 정의 예정).
    /// </summary>
    public sealed class SignMotionSample
    {
        /// <summary>녹화에 포함된 랜드마크 프레임 목록.</summary>
        public LandmarkFrame[] Frames;

        /// <summary>총 녹화 시간 (초).</summary>
        public float DurationSeconds;

        /// <summary>유효 프레임 비율 (랜드마크가 감지된 프레임 수 / 전체 프레임 수).</summary>
        public float ValidFrameRatio;
    }
}
