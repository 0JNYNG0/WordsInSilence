using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// Phase 3.5 진단 패널 — FPS / 처리시간 / 품질 / 신체 좌표를 표시한다.
    /// 매 프레임 Console 출력 금지.
    /// </summary>
    public sealed class PoseIntegrationTestUI : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] HandPoseIntegrator  _integrator;
        [SerializeField] PoseOverlayRenderer _poseOverlay;
        [SerializeField] TMP_Text            _textDiagnostics;

        [Header("Layer Toggles")]
        [SerializeField] Toggle _toggleShowPose;
        [SerializeField] Toggle _toggleShowShoulder;
        [SerializeField] Toggle _toggleShowAssociation;

        readonly Queue<HandPoseFrame> _recentFrames = new Queue<HandPoseFrame>();
        const int MaxQueueSize = 60;

        HandPoseFrame _lastFrame;
        float _nextUpdateTime;
        const float UpdateInterval = 1f;

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void OnEnable()
        {
            if (_integrator != null)
                _integrator.UnifiedFrameArrived += OnUnifiedFrame;

            if (_toggleShowPose != null)
                _toggleShowPose.onValueChanged.AddListener(OnTogglePose);
            if (_toggleShowShoulder != null)
                _toggleShowShoulder.onValueChanged.AddListener(OnToggleShoulder);
            if (_toggleShowAssociation != null)
                _toggleShowAssociation.onValueChanged.AddListener(OnToggleAssociation);
        }

        void OnDisable()
        {
            if (_integrator != null)
                _integrator.UnifiedFrameArrived -= OnUnifiedFrame;

            if (_toggleShowPose != null)
                _toggleShowPose.onValueChanged.RemoveListener(OnTogglePose);
            if (_toggleShowShoulder != null)
                _toggleShowShoulder.onValueChanged.RemoveListener(OnToggleShoulder);
            if (_toggleShowAssociation != null)
                _toggleShowAssociation.onValueChanged.RemoveListener(OnToggleAssociation);
        }

        void Update()
        {
            if (Time.realtimeSinceStartup < _nextUpdateTime) return;
            _nextUpdateTime = Time.realtimeSinceStartup + UpdateInterval;
            RefreshDiagnosticsText();
        }

        // ─── Event handlers ──────────────────────────────────────────────────

        void OnUnifiedFrame(HandPoseFrame frame)
        {
            _lastFrame = frame;
            _recentFrames.Enqueue(frame);
            if (_recentFrames.Count > MaxQueueSize)
                _recentFrames.Dequeue();
        }

        void OnTogglePose(bool val)
        {
            if (_poseOverlay != null) _poseOverlay.ShowPoseLandmarks = val;
        }

        void OnToggleShoulder(bool val)
        {
            if (_poseOverlay != null) _poseOverlay.ShowShoulderCenter = val;
        }

        void OnToggleAssociation(bool val)
        {
            if (_poseOverlay != null) _poseOverlay.ShowAssociationLines = val;
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void RefreshDiagnosticsText()
        {
            if (_textDiagnostics == null) return;

            var frames = _recentFrames.ToArray();
            int total = frames.Length;
            if (total == 0)
            {
                _textDiagnostics.text = "Waiting for frames...";
                return;
            }

            // FPS 계산 (최근 60프레임 기간 기반)
            int goodCount = 0, handCount = 0, poseCount = 0;
            float diffSum = 0f;
            int diffCount = 0;
            float minT = float.MaxValue, maxT = float.MinValue;

            foreach (var f in frames)
            {
                if (f.Quality == FrameDataQuality.Good) goodCount++;
                if (f.HasLeftHand || f.HasRightHand) handCount++;
                if (f.HasPose) poseCount++;
                if (f.HandPoseTimestampDiffMs > 0f) { diffSum += f.HandPoseTimestampDiffMs; diffCount++; }
                if (f.Timestamp < minT) minT = f.Timestamp;
                if (f.Timestamp > maxT) maxT = f.Timestamp;
            }

            float period = maxT - minT;
            float combinedFps = period > 0.01f ? goodCount / period : 0f;
            float handFps     = period > 0.01f ? handCount  / period : 0f;
            float poseFps     = period > 0.01f ? poseCount  / period : 0f;
            float avgDiff     = diffCount > 0 ? diffSum / diffCount : 0f;

            var f2 = _lastFrame;
            string qualityStr = f2 != null ? f2.Quality.ToString() : "N/A";

            string sb = $"=== Pose Integration ===\n" +
                        $"Combined FPS : {combinedFps:F1}\n" +
                        $"Hand FPS     : {handFps:F1}\n" +
                        $"Pose FPS     : {poseFps:F1}\n" +
                        $"Quality      : {qualityStr}\n";

            if (f2 != null && f2.HasValidShoulders)
            {
                sb += $"Shoulder W   : {f2.ShoulderWidth:F3}\n" +
                      $"Shoulder Ctr : ({f2.ShoulderCenter.x:F3}, {f2.ShoulderCenter.y:F3})\n";
            }
            else
            {
                sb += "Shoulder     : N/A\n";
            }

            if (f2 != null && f2.IsValidForBodyRelativeEvaluation)
            {
                string lwStr = PoseBodyMath.HasNaNOrInfinity(f2.LeftWristBodyRelative)
                    ? "N/A"
                    : $"({f2.LeftWristBodyRelative.x:F2}, {f2.LeftWristBodyRelative.y:F2})";
                string rwStr = PoseBodyMath.HasNaNOrInfinity(f2.RightWristBodyRelative)
                    ? "N/A"
                    : $"({f2.RightWristBodyRelative.x:F2}, {f2.RightWristBodyRelative.y:F2})";

                sb += $"L Wrist Body : {lwStr}\n" +
                      $"R Wrist Body : {rwStr}\n";
            }
            else
            {
                sb += "Wrist Body   : N/A\n";
            }

            sb += $"TS Diff Avg  : {avgDiff:F1} ms";

            _textDiagnostics.text = sb;
        }
    }
}
