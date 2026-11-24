// F51 - opt-in upright detector input. Row-order repair only.
//
// The installed providers hand Unity's bottom-up Color32 rows straight to Mediapipe.Image,
// so the detector sees a vertically reflected picture. This helper reverses the rows BEFORE
// inference and maps the predicted y back to the unchanged source-texture grid (y-up), so
// overlays and the neutral segmenter keep the coordinate convention they have today.
//
// It never rewrites a Left/Right label, never mirrors horizontally, never clamps and never
// fabricates a point. It claims no accuracy or performance improvement of any kind, and it
// is refused outright on any camera geometry outside the verified desktop contract
// (videoRotationAngle 0, videoVerticallyMirrored false).
using System;
using UnityEngine;
using WordsInSilence.Recognition.Diagnostics;

namespace WordsInSilence.Core
{
    public static class UprightInferencePixelBridge
    {
        public const string EnvironmentFlagName = "WIS_UPRIGHT_INFERENCE";
        public const string EnabledValue = "1";
        public const string RowOrderTopDown = "top-down";
        public const string RowOrderBottomUp = "bottom-up";
        public const string UnsupportedGeometryCode = "INPUT_GEOMETRY_UNSUPPORTED";

        /// <summary>Opt-in only, exactly "1". Read once per provider at initialization.</summary>
        public static bool ReadOptIn()
        {
            try
            {
                return string.Equals(Environment.GetEnvironmentVariable(EnvironmentFlagName),
                                     EnabledValue, StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }

        /// <summary>The only geometry this repair is verified for: rotation 0 and no mirror.</summary>
        public static bool IsSupportedCameraGeometry(int cameraRotationDegrees,
                                                     bool videoVerticallyMirrored)
        {
            int normalized = cameraRotationDegrees % 360;
            if (normalized < 0) normalized += 360;
            return normalized == 0 && !videoVerticallyMirrored;
        }

        public static string DescribeUnsupportedGeometry(int cameraRotationDegrees,
                                                         bool videoVerticallyMirrored)
        {
            return UnsupportedGeometryCode + ": videoRotationAngle=" + cameraRotationDegrees
                + ", videoVerticallyMirrored=" + videoVerticallyMirrored
                + "; upright inference is verified only for 0 / false and is refused here "
                + "rather than claimed correct";
        }

        /// <summary>A reusable exact-length buffer, so no large per-frame allocation is added.</summary>
        public static Color32[] EnsureBuffer(Color32[] buffer, int length)
        {
            if (length <= 0) throw new ArgumentOutOfRangeException("length");
            return (buffer != null && buffer.Length == length) ? buffer : new Color32[length];
        }

        /// <summary>
        /// Pure row reversal: destination row r receives source row (height - 1 - r). Handles
        /// odd heights and non-square images. Source and destination must not be the same array.
        /// </summary>
        public static void ReversePixelRows(Color32[] source, int width, int height,
                                            Color32[] destination)
        {
            if (source == null) throw new ArgumentNullException("source");
            if (destination == null) throw new ArgumentNullException("destination");
            if (ReferenceEquals(source, destination))
                throw new ArgumentException("source and destination must not alias", "destination");
            if (width <= 0) throw new ArgumentOutOfRangeException("width");
            if (height <= 0) throw new ArgumentOutOfRangeException("height");
            long required = (long)width * height;
            if (required > int.MaxValue)
                throw new ArgumentOutOfRangeException("width", "pixel count overflows Int32");
            if (source.Length != (int)required)
                throw new ArgumentException("source length must equal width * height", "source");
            if (destination.Length != (int)required)
                throw new ArgumentException("destination length must equal width * height",
                                            "destination");
            for (int row = 0; row < height; row++)
                Array.Copy(source, (height - 1 - row) * width, destination, row * width, width);
        }

        /// <summary>
        /// Maps one predicted point from the upright (top-down) inference grid back to the
        /// legacy source-texture grid: (x, 1 - y, z). No clamp, no label rewrite, and a
        /// non-finite input stays non-finite instead of being replaced by an invented value.
        /// </summary>
        public static Vector3 ToSourceTexturePoint(Vector3 inferencePoint)
        {
            return new Vector3(inferencePoint.x, 1f - inferencePoint.y, inferencePoint.z);
        }

        /// <summary>
        /// True only when BOTH provider geometries are available and BOTH declare an applied
        /// pre-inference vertical flip with top-down rows. A mixed or unknown pair returns
        /// false: agreement is never manufactured here, and the existing downstream contract
        /// check remains the authority on refusing such a capture.
        /// </summary>
        public static bool IsUprightTextureGeometry(Step9DMergedInferenceGeometry merged)
        {
            if (merged == null || !merged.BothAvailable) return false;
            return IsUpright(merged.Hand) && IsUpright(merged.Pose);
        }

        static bool IsUpright(Step9DInferenceFrameGeometry geometry)
        {
            return geometry != null
                && geometry.PreInferenceVerticalFlip == Step9DPreInferenceTransformState.Applied
                && string.Equals(geometry.PixelBufferRowOrder, RowOrderTopDown,
                                 StringComparison.Ordinal);
        }
    }
}
