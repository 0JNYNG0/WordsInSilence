using System;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum PracticeTrackingDisplayState
    {
        CameraStarting,
        ObservationWaiting,
        TrackingActive,
        TrackingTemporarilyLost,
        Failed,
    }

    /// <summary>
    /// Reads the existing camera PreviewTexture and immutable Step 9C observations for uGUI
    /// display only. It never changes the source texture or recognition observations.
    /// </summary>
    public sealed class WebcamLandmarkOverlayPresenter : MonoBehaviour
    {
        [SerializeField] bool mirrorDisplay = true;
        [SerializeField] float staleObservationSeconds = 0.45f;
        [SerializeField] Color trackingColor = new Color(0.55f, 1f, 0.22f, 0.98f);
        [SerializeField] Material displayBlurMaterial;

        PetRobotCandidateCameraStartupRoute cameraRoute;
        Step9CHandIdentityObservationBridge observationBridge;
        RawMediaPipeObservation latestObservation;
        float latestObservationAt = float.NegativeInfinity;
        bool receivedObservation;
        bool bridgeFailed;
        RectTransform roundedViewport;
        RectTransform contentRect;
        RawImage webcamImage;
        WebcamTrackingGraphic trackingGraphic;
        Material runtimeDisplayMaterial;

        internal readonly struct WebcamDisplayTransform
        {
            public readonly bool MirrorHorizontally;
            public readonly bool MirrorVertically;
            public readonly int RotationDegrees;
            readonly bool correctFinalOverlayY;

            public WebcamDisplayTransform(bool mirrorHorizontally, bool mirrorVertically, int rotationDegrees)
                : this(mirrorHorizontally, mirrorVertically, rotationDegrees, false)
            {
            }

            WebcamDisplayTransform(
                bool mirrorHorizontally,
                bool mirrorVertically,
                int rotationDegrees,
                bool correctFinalOverlayY)
            {
                MirrorHorizontally = mirrorHorizontally;
                MirrorVertically = mirrorVertically;
                RotationDegrees = rotationDegrees;
                this.correctFinalOverlayY = correctFinalOverlayY;
            }

            public bool IsQuarterTurn => RotationDegrees == 90 || RotationDegrees == 270;

            public WebcamDisplayTransform ForOverlayGeometry()
            {
                return new WebcamDisplayTransform(
                    MirrorHorizontally, MirrorVertically, RotationDegrees, true);
            }

            public Vector2 SourceToDisplay(Vector2 source)
            {
                if (MirrorHorizontally) source.x = 1f - source.x;
                if (MirrorVertically) source.y = 1f - source.y;
                Vector2 display;
                if (RotationDegrees == 90) display = new Vector2(1f - source.y, source.x);
                else if (RotationDegrees == 180) display = new Vector2(1f - source.x, 1f - source.y);
                else if (RotationDegrees == 270) display = new Vector2(source.y, 1f - source.x);
                else display = source;
                if (correctFinalOverlayY) display.y = 1f - display.y;
                return display;
            }

            public Vector2 DisplayToSource(Vector2 display)
            {
                if (correctFinalOverlayY) display.y = 1f - display.y;
                Vector2 source;
                if (RotationDegrees == 90) source = new Vector2(display.y, 1f - display.x);
                else if (RotationDegrees == 180) source = new Vector2(1f - display.x, 1f - display.y);
                else if (RotationDegrees == 270) source = new Vector2(1f - display.y, display.x);
                else source = display;
                if (MirrorHorizontally) source.x = 1f - source.x;
                if (MirrorVertically) source.y = 1f - source.y;
                return source;
            }
        }

        public PracticeTrackingDisplayState TrackingState
        {
            get
            {
                if (cameraRoute == null || bridgeFailed ||
                    cameraRoute.CameraState == CandidateCameraStartupState.Failed)
                    return PracticeTrackingDisplayState.Failed;
                if (!cameraRoute.CameraStarted || !cameraRoute.HasReceivedFrame)
                    return PracticeTrackingDisplayState.CameraStarting;
                if (!receivedObservation)
                    return PracticeTrackingDisplayState.ObservationWaiting;
                return IsObservationFresh
                    ? PracticeTrackingDisplayState.TrackingActive
                    : PracticeTrackingDisplayState.TrackingTemporarilyLost;
            }
        }

        public bool IsObservationFresh => receivedObservation &&
            Time.unscaledTime - latestObservationAt <= Mathf.Max(0.1f, staleObservationSeconds);
        public int AvailablePoseLandmarks => RawMediaPipeObservation.PoseLandmarkCount;
        public int AvailableHandLandmarksPerHand => RawMediaPipeObservation.HandLandmarkCount;
        public bool HasFaceMesh => false;
        public bool HasSegmentationMask => false;
        public bool WritesRecognitionInput => false;
        public bool StoresRgbFrames => false;
        public bool DisplayOnlyBlurActive => runtimeDisplayMaterial != null;
        public Color TrackingColor => trackingColor;

        public void AttachView(RectTransform viewport)
        {
            roundedViewport = viewport;
            if (roundedViewport == null || contentRect != null) return;

            GameObject content = new GameObject("WebcamAspectFitContent", typeof(RectTransform));
            contentRect = content.GetComponent<RectTransform>();
            contentRect.SetParent(roundedViewport, false);
            contentRect.anchorMin = contentRect.anchorMax = new Vector2(0.5f, 0.5f);
            contentRect.pivot = new Vector2(0.5f, 0.5f);
            contentRect.anchoredPosition = Vector2.zero;

            GameObject rgb = new GameObject(
                "WebcamRgbDisplayOnlyBlur", typeof(RectTransform), typeof(CanvasRenderer), typeof(RawImage));
            RectTransform rgbRect = rgb.GetComponent<RectTransform>();
            Stretch(rgbRect, contentRect);
            webcamImage = rgb.GetComponent<RawImage>();
            webcamImage.raycastTarget = false;
            if (displayBlurMaterial != null)
            {
                runtimeDisplayMaterial = new Material(displayBlurMaterial)
                {
                    name = "Step10F_v1_2_DisplayOnlyMildBlur_Runtime"
                };
                webcamImage.material = runtimeDisplayMaterial;
            }

            GameObject overlay = new GameObject(
                "BodyAndHandsTrackingOverlay", typeof(RectTransform), typeof(CanvasRenderer));
            RectTransform overlayRect = overlay.GetComponent<RectTransform>();
            Stretch(overlayRect, contentRect);
            trackingGraphic = overlay.AddComponent<WebcamTrackingGraphic>();
            trackingGraphic.raycastTarget = false;
            trackingGraphic.color = Color.white;
        }

        public void Bind(
            PetRobotCandidateCameraStartupRoute route,
            Step9CHandIdentityObservationBridge bridge)
        {
            Unbind();
            cameraRoute = route;
            observationBridge = bridge;
            if (observationBridge != null)
            {
                observationBridge.ObservationArrived += OnObservationArrived;
                observationBridge.BridgeFailed += OnBridgeFailed;
            }
        }

        public void Unbind()
        {
            if (observationBridge != null)
            {
                observationBridge.ObservationArrived -= OnObservationArrived;
                observationBridge.BridgeFailed -= OnBridgeFailed;
            }
            cameraRoute = null;
            observationBridge = null;
            latestObservation = null;
            receivedObservation = false;
            bridgeFailed = false;
            latestObservationAt = float.NegativeInfinity;
            trackingGraphic?.ClearTrackingData();
        }

        void Update()
        {
            RefreshDisplay();
        }

        void OnDestroy()
        {
            Unbind();
            if (runtimeDisplayMaterial != null) Destroy(runtimeDisplayMaterial);
        }

        void OnObservationArrived(RawMediaPipeObservation observation)
        {
            if (observation == null) return;
            bridgeFailed = false;
            latestObservation = observation;
            receivedObservation = true;
            latestObservationAt = Time.unscaledTime;
        }

        void OnBridgeFailed(Exception exception)
        {
            bridgeFailed = true;
            latestObservation = null;
            receivedObservation = false;
            trackingGraphic?.ClearTrackingData();
        }

        void RefreshDisplay()
        {
            if (webcamImage == null || contentRect == null || roundedViewport == null) return;
            Texture texture = cameraRoute == null ? null : cameraRoute.PreviewTexture;
            WebcamDisplayTransform displayTransform = BuildDisplayTransform(texture);
            webcamImage.texture = texture;
            FitTexture(texture, displayTransform);
            ApplyDisplayOnlyMaterialTransform(displayTransform);

            if (TrackingState == PracticeTrackingDisplayState.TrackingActive && latestObservation != null)
                trackingGraphic?.SetTrackingData(
                    latestObservation, displayTransform.ForOverlayGeometry(), trackingColor);
            else
                trackingGraphic?.ClearTrackingData();
        }

        void FitTexture(Texture texture, WebcamDisplayTransform displayTransform)
        {
            Rect viewport = roundedViewport.rect;
            if (texture == null || texture.width <= 0 || texture.height <= 0)
            {
                contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, viewport.width);
                contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, viewport.height);
                return;
            }
            float sourceWidth = displayTransform.IsQuarterTurn ? texture.height : texture.width;
            float sourceHeight = displayTransform.IsQuarterTurn ? texture.width : texture.height;
            float scale = Mathf.Min(viewport.width / sourceWidth, viewport.height / sourceHeight);
            contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, sourceWidth * scale);
            contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, sourceHeight * scale);
        }

        void ApplyDisplayOnlyMaterialTransform(WebcamDisplayTransform displayTransform)
        {
            if (runtimeDisplayMaterial == null) return;
            runtimeDisplayMaterial.SetFloat("_MirrorX", displayTransform.MirrorHorizontally ? 1f : 0f);
            runtimeDisplayMaterial.SetFloat("_MirrorY", displayTransform.MirrorVertically ? 1f : 0f);
            runtimeDisplayMaterial.SetFloat("_Rotation", displayTransform.RotationDegrees);
        }

        public static string TrackingLabel(PracticeTrackingDisplayState state)
        {
            if (state == PracticeTrackingDisplayState.CameraStarting) return "카메라 시작 중";
            if (state == PracticeTrackingDisplayState.ObservationWaiting) return "카메라 준비됨 · 추적 대기 중";
            if (state == PracticeTrackingDisplayState.TrackingActive) return "몸과 손 추적 중";
            if (state == PracticeTrackingDisplayState.TrackingTemporarilyLost) return "추적이 잠시 끊겼습니다";
            return "카메라 또는 관측 오류";
        }

        WebcamDisplayTransform BuildDisplayTransform(Texture texture)
        {
            return new WebcamDisplayTransform(
                mirrorDisplay,
                IsVerticallyMirrored(texture),
                VideoRotation(texture));
        }

        static bool IsVerticallyMirrored(Texture texture)
        {
            WebCamTexture webcam = texture as WebCamTexture;
            return webcam != null && webcam.videoVerticallyMirrored;
        }

        static int VideoRotation(Texture texture)
        {
            WebCamTexture webcam = texture as WebCamTexture;
            if (webcam == null) return 0;
            int value = webcam.videoRotationAngle % 360;
            return value < 0 ? value + 360 : value;
        }

        static void Stretch(RectTransform child, RectTransform parent)
        {
            child.SetParent(parent, false);
            child.anchorMin = Vector2.zero;
            child.anchorMax = Vector2.one;
            child.offsetMin = Vector2.zero;
            child.offsetMax = Vector2.zero;
        }
    }
}
