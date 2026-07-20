using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Landmarks;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// IMotionRecorder 구현체.
    /// ILandmarkProvider.FrameArrived를 구독해 지정 시간 동안 프레임을 수집한 뒤
    /// SignMotionSample로 반환한다.
    /// </summary>
    public sealed class MotionRecorder : MonoBehaviour, IMotionRecorder
    {
        [SerializeField] MediaPipeLandmarkProvider _provider;

        bool _isRecording;
        float _startTime;
        float _endTime;
        int _totalFrames;
        List<LandmarkFrame> _validFrames;
        TaskCompletionSource<SignMotionSample> _tcs;
        CancellationTokenRegistration _cancelReg;

        // ─── IMotionRecorder ─────────────────────────────────────────────────

        public bool IsRecording => _isRecording;

        public Task<SignMotionSample> RecordAsync(float maxDurationSeconds, CancellationToken ct)
        {
            if (_isRecording)
                throw new InvalidOperationException("[MotionRecorder] Already recording.");

            if (_provider == null)
            {
                Debug.LogError("[MotionRecorder] Provider is not assigned.");
                return Task.FromResult<SignMotionSample>(null);
            }

            _validFrames = new List<LandmarkFrame>();
            _totalFrames = 0;
            _startTime = Time.realtimeSinceStartup;
            _endTime = _startTime + maxDurationSeconds;
            _tcs = new TaskCompletionSource<SignMotionSample>();
            _isRecording = true;

            _cancelReg = ct.Register(CancelRecording, useSynchronizationContext: false);
            _provider.FrameArrived += OnFrameArrived;

            Debug.Log($"[MotionRecorder] 녹화 시작 ({maxDurationSeconds}s)");
            return _tcs.Task;
        }

        public void StopRecording()
        {
            if (!_isRecording) return;
            FinishRecording();
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void OnFrameArrived(LandmarkFrame frame)
        {
            if (!_isRecording) return;

            _totalFrames++;

            if (frame.Hands != null && frame.Hands.Length > 0)
                _validFrames.Add(frame);

            if (frame.Timestamp >= _endTime)
                FinishRecording();
        }

        void FinishRecording()
        {
            if (!_isRecording) return;

            _provider.FrameArrived -= OnFrameArrived;
            _isRecording = false;
            _cancelReg.Dispose();

            float duration = Time.realtimeSinceStartup - _startTime;
            float validRatio = _totalFrames > 0 ? (float)_validFrames.Count / _totalFrames : 0f;

            var sample = new SignMotionSample
            {
                Frames = _validFrames.ToArray(),
                DurationSeconds = duration,
                ValidFrameRatio = validRatio
            };

            Debug.Log($"[MotionRecorder] 녹화 완료 — {_validFrames.Count}/{_totalFrames} 유효 프레임 ({duration:F2}s)");
            _tcs.TrySetResult(sample);
        }

        void CancelRecording()
        {
            // 취소 토큰 콜백 — 스레드풀에서 호출될 수 있음
            if (!_isRecording) return;

            _provider.FrameArrived -= OnFrameArrived;
            _isRecording = false;
            _cancelReg.Dispose();
            _tcs.TrySetCanceled();
        }
    }
}
