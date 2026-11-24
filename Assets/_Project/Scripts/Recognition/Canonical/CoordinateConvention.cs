using System;

namespace WordsInSilence.Recognition.Canonical
{
    /// <summary>
    /// 입력 프레임의 좌표 규약(coordinate convention)을 명시적으로 나타낸다.
    ///
    /// 두 소스의 규약 차이:
    ///   UnityRuntime:
    ///     - LeftWristBodyRelative.x < 0  (왼쪽이 음수)
    ///     - RightWristBodyRelative.x > 0 (오른쪽이 양수)
    ///     - WristBodyRelative.y < 0 (머리 위쪽이 대체로 음수)
    ///     - Hand landmark (0~1 normalized screen): y 아래로 증가
    ///   OfflineMediaPipe (extract_landmarks.py 출력):
    ///     - LeftWristBodyRelative.x > 0  (반전)
    ///     - RightWristBodyRelative.x < 0 (반전)
    ///     - WristBodyRelative.y > 0      (반전)
    ///     - 동일한 랜드마크 스크린 좌표지만 body-relative 부호가 다름
    ///
    /// 표준(Canonical) 규약은 UnityRuntime을 기준으로 한다.
    /// </summary>
    public enum CoordinateConvention
    {
        /// <summary>
        /// Unity WebCamTexture 런타임에서 직접 녹화된 데이터.
        /// body-relative: Left.x < 0, Right.x > 0, Y 아래 증가.
        /// 표준 규약과 동일 — 변환 불필요.
        /// </summary>
        UnityRuntime,

        /// <summary>
        /// extract_landmarks.py (오프라인 MediaPipe)로 전문가 MP4에서 추출된 데이터.
        /// body-relative: Left.x > 0, Right.x < 0, Y 아래 증가(but sign 반전).
        /// 표준 규약으로 변환 시 body-relative X·Y 부호를 반전해야 한다.
        /// Hand landmark 스크린 좌표 자체는 동일하다.
        /// </summary>
        OfflineMediaPipe,
    }

    /// <summary>
    /// CoordinateConvention에 적용되는 변환 기술자.
    /// </summary>
    public readonly struct CoordinateTransformDescriptor
    {
        /// <summary>원본 규약.</summary>
        public readonly CoordinateConvention Source;

        /// <summary>body-relative X 부호 반전 여부.</summary>
        public readonly bool FlipBodyRelativeX;

        /// <summary>body-relative Y 부호 반전 여부.</summary>
        public readonly bool FlipBodyRelativeY;

        /// <summary>hand landmark 스크린 좌표 X 반전 여부.</summary>
        public readonly bool FlipHandScreenX;

        /// <summary>hand landmark 스크린 좌표 Y 반전 여부.</summary>
        public readonly bool FlipHandScreenY;

        /// <summary>
        /// Z 처리 방침 (현재 hand landmark z는 상대적 깊이; 부호 변환 없음).
        /// </summary>
        public readonly string ZNotes;

        public CoordinateTransformDescriptor(
            CoordinateConvention source,
            bool flipBodyRelativeX, bool flipBodyRelativeY,
            bool flipHandScreenX,   bool flipHandScreenY,
            string zNotes)
        {
            Source             = source;
            FlipBodyRelativeX  = flipBodyRelativeX;
            FlipBodyRelativeY  = flipBodyRelativeY;
            FlipHandScreenX    = flipHandScreenX;
            FlipHandScreenY    = flipHandScreenY;
            ZNotes             = zNotes;
        }

        /// <summary>
        /// 소스에 따라 표준 변환 기술자를 반환한다.
        /// </summary>
        public static CoordinateTransformDescriptor For(CoordinateConvention convention)
        {
            switch (convention)
            {
                case CoordinateConvention.UnityRuntime:
                    return new CoordinateTransformDescriptor(
                        convention,
                        flipBodyRelativeX: false,
                        flipBodyRelativeY: false,
                        flipHandScreenX:   false,
                        flipHandScreenY:   false,
                        zNotes: "Hand z: wrist=0, finger tip > 0 (toward camera). No transform needed.");

                case CoordinateConvention.OfflineMediaPipe:
                    return new CoordinateTransformDescriptor(
                        convention,
                        flipBodyRelativeX: true,
                        flipBodyRelativeY: true,
                        flipHandScreenX:   false,
                        flipHandScreenY:   false,
                        zNotes: "Hand z: same as UnityRuntime. No transform needed.");

                default:
                    throw new ArgumentOutOfRangeException(nameof(convention), convention, null);
            }
        }

        public override string ToString()
        {
            return $"Source={Source} FlipBR=({FlipBodyRelativeX},{FlipBodyRelativeY}) " +
                   $"FlipScreen=({FlipHandScreenX},{FlipHandScreenY})";
        }
    }
}
