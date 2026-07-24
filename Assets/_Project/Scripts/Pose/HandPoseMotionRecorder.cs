using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// HandPoseIntegrator.UnifiedFrameArrived를 구독해 지정 시간 동안 프레임을 수집한 뒤
    /// HandPoseRecording으로 반환한다.
    /// </summary>
    public sealed class HandPoseMotionRecorder : MonoBehaviour
    {
        [SerializeField] HandPoseIntegrator _integrator;

        bool _isRecording;
        float _startTime;
        float _endTime;
        List<HandPoseRecordingFrame> _frames;
        TaskCompletionSource<HandPoseRecording> _tcs;
        CancellationTokenRegistration _cancelReg;

        public bool IsRecording => _isRecording;

        public Task<HandPoseRecording> RecordAsync(float maxDurationSeconds, CancellationToken ct)
        {
            if (_isRecording)
                throw new InvalidOperationException("[HandPoseMotionRecorder] Already recording.");

            if (_integrator == null)
            {
                Debug.LogError("[HandPoseMotionRecorder] Integrator is not assigned.");
                return Task.FromResult<HandPoseRecording>(null);
            }

            _frames = new List<HandPoseRecordingFrame>();
            _startTime = Time.realtimeSinceStartup;
            _endTime = _startTime + maxDurationSeconds;
            _tcs = new TaskCompletionSource<HandPoseRecording>();
            _isRecording = true;

            _cancelReg = ct.Register(CancelRecording, useSynchronizationContext: false);
            _integrator.UnifiedFrameArrived += OnFrame;

            Debug.Log($"[HandPoseMotionRecorder] 녹화 시작 ({maxDurationSeconds}s)");
            return _tcs.Task;
        }

        public void StopRecording()
        {
            if (!_isRecording) return;
            FinishRecording();
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void OnFrame(HandPoseFrame f)
        {
            if (!_isRecording) return;

            _frames.Add(ConvertFrame(f));

            if (Time.realtimeSinceStartup >= _endTime)
                FinishRecording();
        }

        void FinishRecording()
        {
            if (!_isRecording) return;

            _integrator.UnifiedFrameArrived -= OnFrame;
            _isRecording = false;
            _cancelReg.Dispose();

            float duration = Time.realtimeSinceStartup - _startTime;
            var recording = BuildRecording(_frames, duration);

            Debug.Log($"[HandPoseMotionRecorder] 녹화 완료 — Total:{recording.TotalFrames} Hand:{recording.ValidHandFrames} Pose:{recording.ValidPoseFrames} Combined:{recording.ValidCombinedFrames} ({duration:F2}s)");
            _tcs.TrySetResult(recording);
        }

        void CancelRecording()
        {
            if (!_isRecording) return;

            _integrator.UnifiedFrameArrived -= OnFrame;
            _isRecording = false;
            _cancelReg.Dispose();
            _tcs.TrySetCanceled();
        }

        HandPoseRecordingFrame ConvertFrame(HandPoseFrame f)
        {
            Vector3[] leftPoints = null;
            Vector3[] rightPoints = null;

            if (f.HandFrame?.Hands != null)
            {
                foreach (var hand in f.HandFrame.Hands)
                {
                    if (hand.Points == null) continue;
                    var pts = new Vector3[hand.Points.Length];
                    for (int i = 0; i < hand.Points.Length; i++)
                        pts[i] = new Vector3(hand.Points[i].X, hand.Points[i].Y, hand.Points[i].Z);

                    if (hand.IsLeftHand) leftPoints = pts;
                    else rightPoints = pts;
                }
            }

            return new HandPoseRecordingFrame
            {
                TimestampMs            = f.TimestampMs,
                HasLeftHand            = f.HasLeftHand,
                HasRightHand           = f.HasRightHand,
                HasPose                = f.HasPose,
                HasValidShoulders      = f.HasValidShoulders,
                LeftHandPoints         = leftPoints,
                RightHandPoints        = rightPoints,
                PosePoints             = f.PoseFrame?.AllPoints,
                ShoulderCenter         = f.ShoulderCenter,
                ShoulderWidth          = f.ShoulderWidth,
                LeftWristBodyRelative  = f.LeftWristBodyRelative,
                RightWristBodyRelative = f.RightWristBodyRelative,
                Quality                = f.Quality
            };
        }

        HandPoseRecording BuildRecording(List<HandPoseRecordingFrame> frames, float duration)
        {
            int validHand = 0, validPose = 0, validCombined = 0;
            foreach (var fr in frames)
            {
                bool hasHand = fr.HasLeftHand || fr.HasRightHand;
                if (hasHand) validHand++;
                if (fr.HasPose) validPose++;
                if (hasHand && fr.HasPose) validCombined++;
            }

            return new HandPoseRecording
            {
                MotionId            = Guid.NewGuid().ToString("N").Substring(0, 8),
                DurationMs          = duration * 1000f,
                SourceType          = validPose > 0 ? "hand_pose" : "hand_only",
                Frames              = frames.ToArray(),
                TotalFrames         = frames.Count,
                ValidHandFrames     = validHand,
                ValidPoseFrames     = validPose,
                ValidCombinedFrames = validCombined
            };
        }
    }
}
