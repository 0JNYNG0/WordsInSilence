using System;
using UnityEngine;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// HandPoseRecording의 프레임을 원본 타임스탬프 간격에 맞춰 재생한다.
    /// MonoBehaviour가 아닌 순수 C# 클래스 — Update는 외부에서 호출한다.
    /// 테스트에서도 시간 주입이 가능하다.
    /// </summary>
    public sealed class FrameReplayProvider : IFrameReplayProvider
    {
        HandPoseRecording _recording;
        int _currentIndex;
        float _elapsedMs;
        bool _isPlaying;

        public bool IsPlaying => _isPlaying;
        public float Progress => _recording == null || _recording.TotalFrames <= 0
            ? 0f
            : (float)_currentIndex / _recording.TotalFrames;
        public int CurrentFrameIndex => _currentIndex;
        public int TotalFrames => _recording?.TotalFrames ?? 0;

        public event Action<HandPoseFrame> FrameReplayed;

        public void Load(HandPoseRecording recording)
        {
            if (recording == null)
                throw new ArgumentNullException(nameof(recording));
            if (recording.Frames == null || recording.Frames.Length == 0)
                throw new ArgumentException("Recording has no frames.", nameof(recording));

            // Validate timestamp ordering
            for (int i = 1; i < recording.Frames.Length; i++)
            {
                if (recording.Frames[i].TimestampMs < recording.Frames[i - 1].TimestampMs)
                    throw new ArgumentException(
                        $"Timestamp not monotonic at frame {i}: {recording.Frames[i].TimestampMs} < {recording.Frames[i - 1].TimestampMs}",
                        nameof(recording));
            }

            _recording = recording;
            _currentIndex = 0;
            _elapsedMs = 0f;
            _isPlaying = false;
        }

        public void Play()
        {
            if (_recording == null)
                throw new InvalidOperationException("No recording loaded.");
            _currentIndex = 0;
            _elapsedMs = 0f;
            _isPlaying = true;
        }

        public void Stop()
        {
            _isPlaying = false;
        }

        /// <summary>
        /// 매 프레임 호출한다. deltaTimeMs만큼 시간을 진행하고 해당 프레임을 발행한다.
        /// </summary>
        public void Update(float deltaTimeMs)
        {
            if (!_isPlaying || _recording == null)
                return;

            var frames = _recording.Frames;
            if (_currentIndex >= frames.Length)
            {
                _isPlaying = false;
                return;
            }

            _elapsedMs += deltaTimeMs;

            long baseTimestamp = frames[0].TimestampMs;
            while (_currentIndex < frames.Length)
            {
                long frameRelativeMs = frames[_currentIndex].TimestampMs - baseTimestamp;
                if (frameRelativeMs > _elapsedMs)
                    break;

                EmitFrame(frames[_currentIndex]);
                _currentIndex++;
            }

            if (_currentIndex >= frames.Length)
                _isPlaying = false;
        }

        /// <summary>
        /// 특정 인덱스의 프레임을 즉시 발행한다. 테스트용.
        /// </summary>
        public void EmitFrameAt(int index)
        {
            if (_recording == null)
                throw new InvalidOperationException("No recording loaded.");
            if (index < 0 || index >= _recording.Frames.Length)
                throw new ArgumentOutOfRangeException(nameof(index));

            EmitFrame(_recording.Frames[index]);
        }

        void EmitFrame(HandPoseRecordingFrame recFrame)
        {
            var frame = new HandPoseFrame
            {
                TimestampMs = recFrame.TimestampMs,
                Timestamp = recFrame.TimestampMs / 1000f,
                HasLeftHand = recFrame.HasLeftHand,
                HasRightHand = recFrame.HasRightHand,
                HasPose = recFrame.HasPose,
                HasValidShoulders = recFrame.HasValidShoulders,
                ShoulderCenter = recFrame.ShoulderCenter,
                ShoulderWidth = recFrame.ShoulderWidth,
                LeftWristBodyRelative = recFrame.LeftWristBodyRelative,
                RightWristBodyRelative = recFrame.RightWristBodyRelative,
                Quality = recFrame.Quality,
                IsValidForBodyRelativeEvaluation =
                    recFrame.HasPose && recFrame.HasValidShoulders &&
                    recFrame.ShoulderWidth > 0.01f,
            };

            FrameReplayed?.Invoke(frame);
        }
    }
}
