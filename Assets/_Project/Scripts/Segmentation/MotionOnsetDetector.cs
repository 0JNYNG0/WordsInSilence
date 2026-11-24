using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Segmentation
{
    /// <summary>
    /// 몸 기준 손목 변위/속도로 실제 움직임 시작을 감지한다.
    /// MotionOnsetMinFrames 연속 프레임이 임계값을 넘으면 onset.
    /// </summary>
    public sealed class MotionOnsetDetector
    {
        readonly float _threshold;
        readonly int _minFrames;

        int _consecutiveCount;
        Vector2 _prevLeftWrist;
        Vector2 _prevRightWrist;
        bool _hasPrev;

        public bool IsMoving => _consecutiveCount >= _minFrames;

        public MotionOnsetDetector(RecognitionSettings settings = null)
        {
            var s = settings ?? RecognitionSettings.Default;
            _threshold = s.MotionOnsetThreshold;
            _minFrames = s.MotionOnsetMinFrames;
        }

        public void Reset()
        {
            _consecutiveCount = 0;
            _hasPrev = false;
        }

        /// <summary>
        /// 프레임의 body-relative 손목 변위를 확인한다.
        /// </summary>
        public bool Update(HandPoseFrame frame)
        {
            if (frame == null || !frame.IsValidForBodyRelativeEvaluation)
            {
                // 추적 실패 → 연속 카운트 유지 (짧은 gap 허용)
                return IsMoving;
            }

            if (!_hasPrev)
            {
                _prevLeftWrist = frame.LeftWristBodyRelative;
                _prevRightWrist = frame.RightWristBodyRelative;
                _hasPrev = true;
                return false;
            }

            float leftDisp = Vector2.Distance(frame.LeftWristBodyRelative, _prevLeftWrist);
            float rightDisp = Vector2.Distance(frame.RightWristBodyRelative, _prevRightWrist);
            float maxDisp = Mathf.Max(leftDisp, rightDisp);

            _prevLeftWrist = frame.LeftWristBodyRelative;
            _prevRightWrist = frame.RightWristBodyRelative;

            if (maxDisp >= _threshold)
            {
                _consecutiveCount++;
            }
            else
            {
                _consecutiveCount = 0;
            }

            return IsMoving;
        }
    }
}
