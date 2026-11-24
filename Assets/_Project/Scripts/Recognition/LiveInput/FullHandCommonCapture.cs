// F2 - full-hand common-2D candidate capture. Additional API only: no classifier, no
// default call site and no existing behaviour is affected, and nothing here converts to
// the frozen 16-point capture or feeds a model.
using System;

namespace WordsInSilence.Recognition.LiveInput
{
    public enum FullHandComponent { Pose, LeftHand, RightHand }

    /// <summary>One point of the 46-point schema. Numeric ids are authoritative.</summary>
    public sealed class FullHandPointSpec
    {
        public int Index { get; private set; }
        public FullHandComponent Component { get; private set; }
        public int SourceIndex { get; private set; }        // pose id, or hand joint 0..20
        public string Name { get; private set; }

        public FullHandPointSpec(int index, FullHandComponent component, int sourceIndex,
                                 string name)
        {
            Index = index; Component = component; SourceIndex = sourceIndex; Name = name;
        }
    }

    /// <summary>
    /// Schema wis_full_hand_common_v1: pose 13/14/15/16, then left hand 0..20, then right
    /// hand 0..20. The thumb comes FIRST, matching the numeric extraction.
    /// </summary>
    public static class FullHandCommonSchema
    {
        public const string SchemaId = "wis_full_hand_common_v1";
        public const int PointCount = 46;
        public const int CoordinateDimensions = 3;
        public const int HandJointCount = 21;

        static readonly int[] poseSourceIndices = { 13, 14, 15, 16 };
        static readonly string[] poseNames =
            { "left_elbow", "right_elbow", "left_wrist", "right_wrist" };
        static readonly string[] handJointNames =
        {
            "wrist",
            "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
        };

        static readonly FullHandPointSpec[] points = BuildPoints();

        // Positions of the frozen 16-point selection inside this 46-point order. These are
        // INDICES ONLY: no conversion to CommonTwoDimensionalCapture and no model input is
        // produced from them, and no model may consume 46 points under the 16 schema.
        static readonly int[] legacy16PointIndices =
            { 0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 25, 29, 33, 37, 41, 45 };

        static FullHandPointSpec[] BuildPoints()
        {
            var built = new FullHandPointSpec[PointCount];
            int at = 0;
            for (int index = 0; index < poseSourceIndices.Length; index++, at++)
                built[at] = new FullHandPointSpec(at, FullHandComponent.Pose,
                                                  poseSourceIndices[index], poseNames[index]);
            foreach (FullHandComponent hand in new[] { FullHandComponent.LeftHand,
                                                       FullHandComponent.RightHand })
                for (int joint = 0; joint < HandJointCount; joint++, at++)
                    built[at] = new FullHandPointSpec(at, hand, joint, handJointNames[joint]);
            return built;
        }

        public static FullHandPointSpec Point(int index)
        {
            if (index < 0 || index >= PointCount)
                throw new ArgumentOutOfRangeException("index");
            return points[index];
        }

        public static FullHandPointSpec[] CopyPoints() { return (FullHandPointSpec[])points.Clone(); }

        public static int[] CopyLegacy16PointIndices()
        {
            return (int[])legacy16PointIndices.Clone();
        }
    }

    /// <summary>
    /// Immutable 46-point common-2D candidate: flattened [T,46,3] coordinates, the final
    /// normalized validity mask, and the per-point original-valid and interpolated masks.
    ///
    /// A masked point holds NaN in all three channels; a valid point's channel 2 is
    /// structurally 0. OriginalValid and Interpolated describe the SOURCE observation and
    /// are not the final mask: a point can be originally valid while its frame has no
    /// torso transform, in which case the normalized valid mask is false for it.
    /// </summary>
    public sealed class FullHandCommonCapture
    {
        readonly double[] coordinates;
        readonly bool[] validMask;
        readonly bool[] originalValidMask;
        readonly bool[] interpolatedMask;

        public int FrameCount { get; private set; }
        public int PointCount { get { return FullHandCommonSchema.PointCount; } }
        public string SchemaId { get { return FullHandCommonSchema.SchemaId; } }
        public int ValidPointCount { get; private set; }
        public int InterpolatedPointCount { get; private set; }
        public int ImageWidth { get; private set; }
        public int ImageHeight { get; private set; }
        public string ConventionId { get; private set; }
        public string DomainContract
        {
            get { return "raw_post_gap_pixel_plane_z0_torso_normalize_2d_full_hand"; }
        }

        public FullHandCommonCapture(
            int frameCount, double[] coordinates, bool[] validMask,
            bool[] originalValidMask, bool[] interpolatedMask,
            int validPointCount, int interpolatedPointCount,
            int imageWidth, int imageHeight, string conventionId)
        {
            if (frameCount <= 0) throw new ArgumentOutOfRangeException("frameCount");
            int points = frameCount * FullHandCommonSchema.PointCount;
            Require(coordinates, points * FullHandCommonSchema.CoordinateDimensions,
                    "coordinates");
            Require(validMask, points, "validMask");
            Require(originalValidMask, points, "originalValidMask");
            Require(interpolatedMask, points, "interpolatedMask");

            FrameCount = frameCount;
            ValidPointCount = validPointCount;
            InterpolatedPointCount = interpolatedPointCount;
            ImageWidth = imageWidth;
            ImageHeight = imageHeight;
            ConventionId = conventionId ?? string.Empty;
            this.coordinates = (double[])coordinates.Clone();
            this.validMask = (bool[])validMask.Clone();
            this.originalValidMask = (bool[])originalValidMask.Clone();
            this.interpolatedMask = (bool[])interpolatedMask.Clone();
        }

        static void Require(Array values, int expected, string name)
        {
            if (values == null) throw new ArgumentNullException(name);
            if (values.Length != expected)
                throw new ArgumentException(name + " length " + values.Length +
                                            " does not match the expected " + expected, name);
        }

        public double[] CopyCoordinates() { return (double[])coordinates.Clone(); }
        public bool[] CopyValidMask() { return (bool[])validMask.Clone(); }
        public bool[] CopyOriginalValidMask() { return (bool[])originalValidMask.Clone(); }
        public bool[] CopyInterpolatedMask() { return (bool[])interpolatedMask.Clone(); }

        public bool IsValid(int frame, int point) { return validMask[Slot(frame, point)]; }
        public bool WasOriginallyValid(int frame, int point)
        {
            return originalValidMask[Slot(frame, point)];
        }
        public bool WasInterpolated(int frame, int point)
        {
            return interpolatedMask[Slot(frame, point)];
        }

        public double Coordinate(int frame, int point, int axis)
        {
            if (axis < 0 || axis >= FullHandCommonSchema.CoordinateDimensions)
                throw new ArgumentOutOfRangeException("axis");
            return coordinates[Slot(frame, point) * FullHandCommonSchema.CoordinateDimensions
                               + axis];
        }

        int Slot(int frame, int point)
        {
            if (frame < 0 || frame >= FrameCount)
                throw new ArgumentOutOfRangeException("frame");
            if (point < 0 || point >= FullHandCommonSchema.PointCount)
                throw new ArgumentOutOfRangeException("point");
            return frame * FullHandCommonSchema.PointCount + point;
        }
    }
}
