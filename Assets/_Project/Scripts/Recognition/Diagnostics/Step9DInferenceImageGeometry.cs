namespace WordsInSilence.Recognition.Diagnostics
{
    /// <summary>
    /// Immutable record of the actual inference image geometry at the moment
    /// pixels were submitted to MediaPipe. Populated once per diagnostic capture
    /// from live WebCamTexture properties.
    /// </summary>
    public sealed class Step9DInferenceImageGeometry
    {
        /// <summary>WebCamTexture.width after hardware negotiation.</summary>
        public int WebCamTextureWidth { get; }

        /// <summary>WebCamTexture.height after hardware negotiation.</summary>
        public int WebCamTextureHeight { get; }

        /// <summary>WebCamTexture.videoRotationAngle (degrees, CW).</summary>
        public int VideoRotationAngle { get; }

        /// <summary>WebCamTexture.videoVerticallyMirrored.</summary>
        public bool VideoVerticallyMirrored { get; }

        /// <summary>Texture2D width passed to Mediapipe.Image constructor.</summary>
        public int InferenceTextureWidth { get; }

        /// <summary>Texture2D height passed to Mediapipe.Image constructor.</summary>
        public int InferenceTextureHeight { get; }

        public Step9DInferenceImageGeometry(
            int webCamTextureWidth,
            int webCamTextureHeight,
            int videoRotationAngle,
            bool videoVerticallyMirrored,
            int inferenceTextureWidth,
            int inferenceTextureHeight)
        {
            if (webCamTextureWidth <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "WebCamTexture width must be positive");
            if (webCamTextureHeight <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "WebCamTexture height must be positive");
            if (inferenceTextureWidth <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference texture width must be positive");
            if (inferenceTextureHeight <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference texture height must be positive");

            WebCamTextureWidth = webCamTextureWidth;
            WebCamTextureHeight = webCamTextureHeight;
            VideoRotationAngle = videoRotationAngle;
            VideoVerticallyMirrored = videoVerticallyMirrored;
            InferenceTextureWidth = inferenceTextureWidth;
            InferenceTextureHeight = inferenceTextureHeight;
        }

        internal Step9DInferenceImageGeometry Clone() =>
            new Step9DInferenceImageGeometry(
                WebCamTextureWidth, WebCamTextureHeight,
                VideoRotationAngle, VideoVerticallyMirrored,
                InferenceTextureWidth, InferenceTextureHeight);
    }
}
