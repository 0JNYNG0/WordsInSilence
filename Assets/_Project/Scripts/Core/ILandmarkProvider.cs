using System;
using System.Threading;
using System.Threading.Tasks;

namespace WordsInSilence.Core
{
    /// <summary>
    /// 손 랜드마크 공급자 인터페이스 (Phase 2에서 구현체 추가 예정).
    /// </summary>
    public interface ILandmarkProvider
    {
        bool IsRunning { get; }

        /// <summary>
        /// 새 랜드마크 프레임이 도착했을 때 발생.
        /// </summary>
        event Action<LandmarkFrame> FrameArrived;

        Task StartAsync(CancellationToken ct);
        Task StopAsync(CancellationToken ct);
    }

    /// <summary>
    /// 손 랜드마크 프레임 (Phase 2에서 상세 정의 예정).
    /// </summary>
    public sealed class LandmarkFrame
    {
        /// <summary>캡처 타임스탬프 (Time.realtimeSinceStartup 기준).</summary>
        public float Timestamp;

        /// <summary>감지된 손 목록 (최대 2개).</summary>
        public HandLandmarks[] Hands;
    }

    /// <summary>
    /// 단일 손의 랜드마크 데이터 (Phase 2에서 상세 정의 예정).
    /// </summary>
    public sealed class HandLandmarks
    {
        /// <summary>true = 왼손, false = 오른손 (MediaPipe 기준, 미러 없는 원본 이미지 기준).</summary>
        public bool IsLeftHand;

        /// <summary>21개 랜드마크 좌표 (0~1 정규화, 원본 이미지 공간).</summary>
        public LandmarkPoint[] Points;

        /// <summary>감지 신뢰도 (0~1).</summary>
        public float Confidence;
    }

    /// <summary>
    /// 단일 랜드마크 좌표.
    /// </summary>
    public sealed class LandmarkPoint
    {
        public float X;
        public float Y;
        public float Z;
    }
}
