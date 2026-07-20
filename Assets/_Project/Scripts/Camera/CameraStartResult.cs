namespace WordsInSilence.Camera
{
    /// <summary>
    /// 카메라 시작 결과 데이터.
    /// </summary>
    public sealed class CameraStartResult
    {
        /// <summary>시작 성공 여부.</summary>
        public bool IsSuccess;

        /// <summary>
        /// 실패 시 오류 코드. Docs/Technical/ERROR_CODES.md 기준.
        /// 성공 시 null.
        /// </summary>
        public string ErrorCode;

        /// <summary>오류 상세 설명 (사람이 읽을 수 있는 형태). 성공 시 null.</summary>
        public string ErrorDetail;

        /// <summary>실제 해상도 (가로). 요청 값과 다를 수 있다.</summary>
        public int ActualWidth;

        /// <summary>실제 해상도 (세로).</summary>
        public int ActualHeight;

        /// <summary>실제 프레임레이트. 요청 값과 다를 수 있다.</summary>
        public int ActualFps;

        /// <summary>실제 사용된 장치 이름.</summary>
        public string DeviceName;

        public static CameraStartResult Failure(string errorCode, string detail) =>
            new CameraStartResult { IsSuccess = false, ErrorCode = errorCode, ErrorDetail = detail };

        public static CameraStartResult Success(string deviceName, int width, int height, int fps) =>
            new CameraStartResult
            {
                IsSuccess = true,
                DeviceName = deviceName,
                ActualWidth = width,
                ActualHeight = height,
                ActualFps = fps
            };
    }
}
