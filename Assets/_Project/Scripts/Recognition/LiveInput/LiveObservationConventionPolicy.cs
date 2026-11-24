// F53 - ONE acceptance rule for live observation conventions.
//
// F51R1 fixed the capture controller only, so the calibration/segmentation entry path
// (ring buffer and neutral auto segmenter) still rejected every upright observation and the
// practice attempt could never start. This policy is the exact F51R1 predicate, moved to a
// shared place so the three call sites cannot drift again.
//
// It changes no score, model, gate, threshold, hand side or camera lifecycle behaviour.
using WordsInSilence.Core;
using WordsInSilence.Recognition.Diagnostics;

namespace WordsInSilence.Recognition.LiveInput
{
    public static class LiveObservationConventionPolicy
    {
        /// <summary>
        /// Either the unchanged approved live convention, or the F51 upright convention and
        /// ONLY when the observation's own recorded geometry backs it: both provider
        /// geometries available and fully contract compatible, both declaring the applied
        /// pre-inference vertical flip with top-down rows, and both reporting an actual
        /// camera geometry of rotation 0 with no vertical mirror. Unknown rotation or mirror
        /// metadata is refused rather than assumed, and no other normalized-image convention
        /// is ever accepted. Within-sequence convention stability and timestamp ordering stay
        /// the caller's own unchanged checks.
        /// </summary>
        public static bool IsAccepted(RawMediaPipeObservation observation)
        {
            if (observation == null) return false;
            RawObservationConvention convention = observation.Convention;
            if (RawObservationConvention.LiveMediaPipe.Equals(convention)) return true;
            if (!RawObservationConvention.LiveMediaPipeUprightTexture.Equals(convention))
                return false;
            Step9DMergedInferenceGeometry merged = observation.MergedGeometry;
            if (merged == null || !merged.BothAvailable || !merged.FullContractCompatible)
                return false;
            if (!UprightInferencePixelBridge.IsUprightTextureGeometry(merged)) return false;
            return IsSupportedActualCameraGeometry(merged.Hand)
                && IsSupportedActualCameraGeometry(merged.Pose);
        }

        public static bool IsSupportedActualCameraGeometry(Step9DInferenceFrameGeometry geometry)
        {
            if (geometry == null) return false;
            if (!geometry.ObservedVideoRotationAngleOrNull.HasValue) return false;
            if (!geometry.ObservedVideoVerticallyMirroredOrNull.HasValue) return false;
            return UprightInferencePixelBridge.IsSupportedCameraGeometry(
                geometry.ObservedVideoRotationAngleOrNull.Value,
                geometry.ObservedVideoVerticallyMirroredOrNull.Value);
        }

        /// <summary>The one rejection wording the live entry points share.</summary>
        public const string RejectionMessage =
            "observation convention is not an accepted live MediaPipe convention, or its " +
            "upright inference geometry is missing, mixed or unsupported";
    }
}
