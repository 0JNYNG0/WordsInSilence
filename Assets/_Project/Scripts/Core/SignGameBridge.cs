using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;

namespace WordsInSilence.Core
{
    /// <summary>
    /// 수어 판정 결과를 게임 상태(대화, 퀘스트)에 연결한다.
    /// - attempt ID + cooldown으로 중복 이벤트 방지
    /// - 성공 시 정확히 1회 게임 상태 변화
    /// - 오답/uncertain은 퀘스트를 진행시키지 않음
    /// </summary>
    public sealed class SignGameBridge
    {
        readonly float _cooldownSeconds;
        readonly HashSet<string> _processedAttempts = new HashSet<string>();
        readonly Dictionary<string, float> _lastSignPassedTime = new Dictionary<string, float>();

        // 현재 장면에서 인식 대상인 수어 목록
        readonly HashSet<string> _allowedSignIds = new HashSet<string>();

        // 게임 상태 변경 콜백
        public event Action<string> OnSignSuccess;    // signId
        public event Action<string> OnSignRetry;      // signId
        public event Action<string> OnCameraRecovery; // signId

        public SignGameBridge(float cooldownSeconds = 3.0f)
        {
            _cooldownSeconds = cooldownSeconds;
        }

        /// <summary>
        /// 현재 장면의 인식 대상 수어를 설정한다.
        /// </summary>
        public void SetAllowedSigns(IEnumerable<string> signIds)
        {
            _allowedSignIds.Clear();
            foreach (var id in signIds)
                _allowedSignIds.Add(id);
        }

        public bool IsSignAllowed(string signId)
        {
            return _allowedSignIds.Contains(signId);
        }

        /// <summary>
        /// 판정 결과를 처리한다. 중복 방지 포함.
        /// </summary>
        public SignGameAction ProcessResult(SignAttemptEvent evt, float currentTime)
        {
            if (evt == null) return SignGameAction.None;

            // 중복 방지: 같은 attemptId는 1번만
            if (_processedAttempts.Contains(evt.AttemptId))
                return SignGameAction.None;
            _processedAttempts.Add(evt.AttemptId);

            // 허용 수어 확인
            if (!string.IsNullOrEmpty(evt.SignId) && !_allowedSignIds.Contains(evt.SignId))
                return SignGameAction.None;

            // cooldown 확인
            if (!string.IsNullOrEmpty(evt.SignId) &&
                _lastSignPassedTime.TryGetValue(evt.SignId, out float lastTime) &&
                currentTime - lastTime < _cooldownSeconds)
            {
                return SignGameAction.None;
            }

            switch (evt.Status)
            {
                case AttemptResultStatus.Passed:
                    _lastSignPassedTime[evt.SignId] = currentTime;
                    OnSignSuccess?.Invoke(evt.SignId);
                    SignPracticeEvents.EmitSignPassed(evt.SignId);
                    return SignGameAction.Success;

                case AttemptResultStatus.Incorrect:
                    OnSignRetry?.Invoke(evt.SignId);
                    return SignGameAction.Retry;

                case AttemptResultStatus.Uncertain:
                    OnCameraRecovery?.Invoke(evt.SignId);
                    SignPracticeEvents.EmitSignUncertain(evt.SignId);
                    return SignGameAction.CameraRecovery;

                case AttemptResultStatus.Invalid:
                    return SignGameAction.None;

                default:
                    return SignGameAction.None;
            }
        }

        /// <summary>
        /// 장면 전환 시 상태 초기화.
        /// </summary>
        public void Reset()
        {
            _processedAttempts.Clear();
            _lastSignPassedTime.Clear();
            _allowedSignIds.Clear();
        }
    }

    public enum SignGameAction
    {
        None,
        Success,
        Retry,
        CameraRecovery,
    }
}
