using System.Collections.Generic;
using WordsInSilence.Segmentation;

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// 기술적 피드백 코드를 사용자 친화적인 안내 문구로 변환한다.
    /// 카메라 문제와 수어 수행 문제를 분리한다.
    /// </summary>
    public static class SignFeedbackMapper
    {
        // ─── 수어 수행 피드백 ──────────────────────────────────────────────

        static readonly Dictionary<string, string> PerformanceFeedback = new Dictionary<string, string>
        {
            { "HANDSHAPE_MISMATCH", "손 모양이 다릅니다. 손가락 위치를 확인해 주세요." },
            { "HANDSHAPE_RIGHT_THUMB_CLOSED", "오른손 엄지를 더 펴 주세요." },
            { "POSITION_MISMATCH", "손 위치가 다릅니다. 동작 위치를 확인해 주세요." },
            { "POSITION_RIGHT_HAND_TOO_LOW", "오른손이 너무 낮습니다." },
            { "POSITION_RIGHT_HAND_TOO_HIGH", "오른손이 너무 높습니다." },
            { "ORIENTATION_MISMATCH", "손바닥 방향이 다릅니다." },
            { "ORIENTATION_LEFT_PALM_INWARD", "왼손 손바닥을 바깥으로 향해 주세요." },
            { "TRAJECTORY_MISMATCH", "동작 경로가 다릅니다. 움직임을 확인해 주세요." },
            { "TRAJECTORY_RANGE_TOO_SMALL", "동작 범위가 작습니다. 더 크게 해 주세요." },
            { "TRAJECTORY_RANGE_TOO_LARGE", "동작 범위가 큽니다. 더 작게 해 주세요." },
            { "TWO_HAND_MISMATCH", "양손 관계가 다릅니다. 양손 위치를 확인해 주세요." },
            { "TWO_HAND_TIMING_EARLY", "한 손이 너무 일찍 움직입니다." },
            { "AMBIGUOUS_MATCH", "비슷한 수어가 여러 개 있습니다. 더 정확하게 해 주세요." },
        };

        // ─── 카메라 문제 피드백 ────────────────────────────────────────────

        static readonly Dictionary<string, string> CameraFeedback = new Dictionary<string, string>
        {
            { "CAMERA_OUT_OF_FRAME", "손이 화면 밖으로 나갔습니다. 카메라 앞에서 수행해 주세요." },
            { "CAMERA_HAND_NOT_VISIBLE", "손이 잘 보이지 않습니다. 조명이나 카메라 위치를 확인해 주세요." },
            { "CAMERA_LEFT_HAND_OUT_OF_FRAME", "왼손이 화면 밖입니다." },
            { "CAMERA_RIGHT_HAND_OUT_OF_FRAME", "오른손이 화면 밖입니다." },
            { "CAMERA_TRACKING_LOST", "추적이 끊겼습니다. 카메라 앞에서 다시 시도해 주세요." },
            { "CAMERA_OCCLUDED", "손이 가려져 있습니다. 손이 잘 보이도록 해 주세요." },
        };

        /// <summary>
        /// 피드백 코드가 카메라 문제인지 확인한다.
        /// </summary>
        public static bool IsCameraIssue(string code)
        {
            return code != null && code.StartsWith("CAMERA_");
        }

        /// <summary>
        /// 피드백 코드를 사용자 문구로 변환한다.
        /// </summary>
        public static string GetUserMessage(string code)
        {
            if (string.IsNullOrEmpty(code)) return "";
            if (CameraFeedback.TryGetValue(code, out var camMsg)) return camMsg;
            if (PerformanceFeedback.TryGetValue(code, out var perfMsg)) return perfMsg;
            return code; // 알 수 없는 코드는 그대로 표시
        }

        /// <summary>
        /// 인식 결과의 상태를 사용자 문구로 변환한다.
        /// </summary>
        public static string GetStatusMessage(AttemptResultStatus status)
        {
            return status switch
            {
                AttemptResultStatus.Passed => "잘했습니다!",
                AttemptResultStatus.Incorrect => "다시 한번 해 볼까요?",
                AttemptResultStatus.Uncertain => "잘 보이지 않았습니다. 다시 시도해 주세요.",
                AttemptResultStatus.Invalid => "동작을 인식하지 못했습니다.",
                _ => "",
            };
        }

        /// <summary>
        /// 피드백 코드 목록에서 카메라/수행 분리하여 사용자 메시지를 생성한다.
        /// 가장 큰 문제 1~2개만 표시.
        /// </summary>
        public static FeedbackMessages BuildMessages(SignRecognitionResult result)
        {
            var msg = new FeedbackMessages();
            msg.StatusMessage = GetStatusMessage(result.Status);

            foreach (var code in result.FeedbackCodes)
            {
                if (IsCameraIssue(code))
                    msg.CameraMessages.Add(GetUserMessage(code));
                else
                    msg.PerformanceMessages.Add(GetUserMessage(code));
            }

            return msg;
        }
    }

    /// <summary>
    /// 분리된 피드백 메시지.
    /// </summary>
    public sealed class FeedbackMessages
    {
        public string StatusMessage;
        public List<string> CameraMessages = new List<string>();
        public List<string> PerformanceMessages = new List<string>();

        public bool HasCameraIssues => CameraMessages.Count > 0;
        public bool HasPerformanceIssues => PerformanceMessages.Count > 0;
    }
}
