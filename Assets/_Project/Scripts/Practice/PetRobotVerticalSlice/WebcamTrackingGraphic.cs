using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>Maskable uGUI geometry for body pose (indices 11+) and two actual 21-point hands.</summary>
    public sealed class WebcamTrackingGraphic : MaskableGraphic
    {
        public const int FirstBodyPoseIndex = 11;

        static readonly int[] BodyPoseConnections =
        {
            11, 12, 11, 13, 13, 15, 12, 14, 14, 16,
            11, 23, 12, 24, 23, 24,
        };

        static readonly int[] BodyPosePoints = { 11, 12, 13, 14, 15, 16, 23, 24 };

        static readonly int[] HandConnections =
        {
            0, 1, 1, 2, 2, 3, 3, 4,
            0, 5, 5, 6, 6, 7, 7, 8,
            5, 9, 9, 10, 10, 11, 11, 12,
            9, 13, 13, 14, 14, 15, 15, 16,
            13, 17, 17, 18, 18, 19, 19, 20,
            0, 17,
        };

        RawMediaPipeObservation observation;
        WebcamLandmarkOverlayPresenter.WebcamDisplayTransform displayTransform;
        Color trackingColor;

        internal void SetTrackingData(
            RawMediaPipeObservation value,
            WebcamLandmarkOverlayPresenter.WebcamDisplayTransform transform,
            Color unifiedTrackingColor)
        {
            observation = value;
            displayTransform = transform;
            trackingColor = unifiedTrackingColor;
            SetVerticesDirty();
        }

        public void ClearTrackingData()
        {
            if (observation == null) return;
            observation = null;
            SetVerticesDirty();
        }

        protected override void OnPopulateMesh(VertexHelper vertexHelper)
        {
            vertexHelper.Clear();
            if (observation == null) return;

            Color glow = trackingColor;
            glow.a *= 0.22f;
            AddConnections(vertexHelper, observation.Pose, BodyPoseConnections, glow, 10f);
            AddConnections(vertexHelper, observation.LeftHand, HandConnections, glow, 7f);
            AddConnections(vertexHelper, observation.RightHand, HandConnections, glow, 7f);

            AddConnections(vertexHelper, observation.Pose, BodyPoseConnections, trackingColor, 3f);
            AddConnections(vertexHelper, observation.LeftHand, HandConnections, trackingColor, 2.5f);
            AddConnections(vertexHelper, observation.RightHand, HandConnections, trackingColor, 2.5f);
            AddIndexedPoints(vertexHelper, observation.Pose, BodyPosePoints, trackingColor, 5f);
            AddAllHandPoints(vertexHelper, observation.LeftHand, trackingColor, 6f);
            AddAllHandPoints(vertexHelper, observation.RightHand, trackingColor, 6f);
        }

        void AddConnections(
            VertexHelper vertexHelper,
            RawLandmarkObservationSet landmarks,
            int[] connections,
            Color value,
            float width)
        {
            for (int index = 0; index < connections.Length; index += 2)
            {
                int first = connections[index];
                int second = connections[index + 1];
                if (!landmarks.IsValid(first) || !landmarks.IsValid(second)) continue;
                AddLine(vertexHelper, Map(landmarks, first), Map(landmarks, second), value, width);
            }
        }

        void AddIndexedPoints(
            VertexHelper vertexHelper,
            RawLandmarkObservationSet landmarks,
            int[] indices,
            Color value,
            float size)
        {
            foreach (int index in indices)
                if (landmarks.IsValid(index)) AddPoint(vertexHelper, Map(landmarks, index), value, size);
        }

        void AddAllHandPoints(
            VertexHelper vertexHelper,
            RawLandmarkObservationSet landmarks,
            Color value,
            float size)
        {
            for (int index = 0; index < RawMediaPipeObservation.HandLandmarkCount; index++)
                if (landmarks.IsValid(index)) AddPoint(vertexHelper, Map(landmarks, index), value, size);
        }

        Vector2 Map(RawLandmarkObservationSet landmarks, int index)
        {
            Vector2 source = new Vector2(
                (float)landmarks.Coordinate(index, 0),
                (float)landmarks.Coordinate(index, 1));
            Vector2 display = displayTransform.SourceToDisplay(source);
            Rect rect = rectTransform.rect;
            return new Vector2(
                rect.xMin + display.x * rect.width,
                rect.yMax - display.y * rect.height);
        }

        static void AddLine(
            VertexHelper vertexHelper,
            Vector2 start,
            Vector2 end,
            Color value,
            float width)
        {
            Vector2 delta = end - start;
            if (delta.sqrMagnitude <= 0.0001f) return;
            Vector2 normal = new Vector2(-delta.y, delta.x).normalized * (width * 0.5f);
            AddQuad(vertexHelper, start - normal, start + normal, end + normal, end - normal, value);
        }

        static void AddPoint(
            VertexHelper vertexHelper,
            Vector2 center,
            Color value,
            float size)
        {
            float radius = size * 0.5f;
            AddQuad(vertexHelper,
                center + new Vector2(-radius, -radius),
                center + new Vector2(-radius, radius),
                center + new Vector2(radius, radius),
                center + new Vector2(radius, -radius), value);
        }

        static void AddQuad(
            VertexHelper vertexHelper,
            Vector2 bottomLeft,
            Vector2 topLeft,
            Vector2 topRight,
            Vector2 bottomRight,
            Color value)
        {
            int start = vertexHelper.currentVertCount;
            Color32 color32 = value;
            vertexHelper.AddVert(bottomLeft, color32, Vector2.zero);
            vertexHelper.AddVert(topLeft, color32, Vector2.up);
            vertexHelper.AddVert(topRight, color32, Vector2.one);
            vertexHelper.AddVert(bottomRight, color32, Vector2.right);
            vertexHelper.AddTriangle(start, start + 1, start + 2);
            vertexHelper.AddTriangle(start + 2, start + 3, start);
        }
    }
}
