using UnityEngine;

namespace WordsInSilence.Settings
{
    /// <summary>
    /// 카메라 캡처 설정 ScriptableObject.
    /// Assets/_Project/Settings/ 에서 Create > SilentSign > CaptureConfig 로 인스턴스 생성.
    /// </summary>
    [CreateAssetMenu(fileName = "CaptureConfig", menuName = "SilentSign/CaptureConfig")]
    public class CaptureConfig : ScriptableObject
    {
        [Header("[임시 설정 — 프로토타입용, 검증된 수치 아님]")]
        [Tooltip("요청 해상도 (가로)")]
        public int requestedWidth = 640;

        [Tooltip("요청 해상도 (세로)")]
        public int requestedHeight = 480;

        [Tooltip("요청 프레임레이트")]
        public int requestedFps = 30;

        [Tooltip("화면 표시 시 좌우 반전 (셀카 방향)")]
        public bool mirrorDisplay = true;

        [Header("[Phase 3용 — 현재 미사용]")]
        [Tooltip("수어 인식 시작 전 카운트다운 시간 (초)")]
        public float countdownSeconds = 3f;

        [Tooltip("최대 녹화 시간 (초)")]
        public float maxRecordingSeconds = 10f;

        [Tooltip("유효 프레임 최소 비율 (0~1). 이 비율 미만이면 인식 실패 처리")]
        [Range(0f, 1f)]
        public float minValidFrameRatio = 0.7f;
    }
}
