using System;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// 저장된 HandPoseRecording 프레임을 실시간 provider와 동일한 인터페이스로 재생한다.
    /// 실시간 웹캠 없이 상태 머신과 판정기를 테스트할 수 있게 한다.
    /// </summary>
    public interface IFrameReplayProvider
    {
        /// <summary>재생할 녹화 데이터를 설정한다.</summary>
        void Load(HandPoseRecording recording);

        /// <summary>재생을 시작한다.</summary>
        void Play();

        /// <summary>재생을 중지한다.</summary>
        void Stop();

        /// <summary>재생 중인지 여부.</summary>
        bool IsPlaying { get; }

        /// <summary>현재 재생 위치 (0~1).</summary>
        float Progress { get; }

        /// <summary>현재 프레임 인덱스.</summary>
        int CurrentFrameIndex { get; }

        /// <summary>전체 프레임 수.</summary>
        int TotalFrames { get; }

        /// <summary>프레임이 재생될 때 발행된다. HandPoseIntegrator.UnifiedFrameArrived와 호환.</summary>
        event Action<HandPoseFrame> FrameReplayed;
    }
}
