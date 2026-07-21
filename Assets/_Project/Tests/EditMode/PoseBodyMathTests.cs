using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Tests.EditMode
{
    public class PoseBodyMathTests
    {
        // ─── ShoulderCenter ──────────────────────────────────────────────────

        [Test]
        public void ShoulderCenter_SymmetricPoints_ReturnsMidpoint()
        {
            var left  = new Vector3(0.2f, 0.5f, 0f);
            var right = new Vector3(0.8f, 0.5f, 0f);

            var result = PoseBodyMath.ShoulderCenter(left, right);

            Assert.AreEqual(0.5f, result.x, 1e-5f);
            Assert.AreEqual(0.5f, result.y, 1e-5f);
        }

        [Test]
        public void ShoulderCenter_AsymmetricPoints_ReturnsCorrectMidpoint()
        {
            var left  = new Vector3(0.1f, 0.3f, 0f);
            var right = new Vector3(0.5f, 0.7f, 0f);

            var result = PoseBodyMath.ShoulderCenter(left, right);

            Assert.AreEqual(0.3f, result.x, 1e-5f);
            Assert.AreEqual(0.5f, result.y, 1e-5f);
        }

        // ─── ShoulderWidth2D ─────────────────────────────────────────────────

        [Test]
        public void ShoulderWidth2D_KnownPoints_ReturnsCorrectDistance()
        {
            var left  = new Vector3(0.3f, 0.5f, 0f);
            var right = new Vector3(0.7f, 0.5f, 0f);

            float width = PoseBodyMath.ShoulderWidth2D(left, right);

            Assert.AreEqual(0.4f, width, 1e-5f);
        }

        [Test]
        public void ShoulderWidth2D_IdenticalPoints_ReturnsZero()
        {
            var pt = new Vector3(0.5f, 0.5f, 0f);

            float width = PoseBodyMath.ShoulderWidth2D(pt, pt);

            Assert.AreEqual(0f, width, 1e-5f);
        }

        // ─── IsValidShoulderWidth ────────────────────────────────────────────

        [Test]
        public void IsValidShoulderWidth_AboveMinimum_ReturnsTrue()
        {
            Assert.IsTrue(PoseBodyMath.IsValidShoulderWidth(0.3f));
        }

        [Test]
        public void IsValidShoulderWidth_BelowMinimum_ReturnsFalse()
        {
            Assert.IsFalse(PoseBodyMath.IsValidShoulderWidth(0.01f));
        }

        [Test]
        public void IsValidShoulderWidth_NaN_ReturnsFalse()
        {
            Assert.IsFalse(PoseBodyMath.IsValidShoulderWidth(float.NaN));
        }

        // ─── BodyRelative ────────────────────────────────────────────────────

        [Test]
        public void BodyRelative_AtShoulderCenter_ReturnsZero()
        {
            var center = new Vector2(0.5f, 0.4f);
            float width = 0.4f;
            var point = new Vector3(0.5f, 0.4f, 0f);

            var result = PoseBodyMath.BodyRelative(point, center, width);

            Assert.AreEqual(0f, result.x, 1e-5f);
            Assert.AreEqual(0f, result.y, 1e-5f);
        }

        [Test]
        public void BodyRelative_AtLeftShoulder_ReturnsNegativeHalf()
        {
            // left shoulder at center.x - width/2
            float width = 0.4f;
            var center = new Vector2(0.5f, 0.4f);
            var leftShoulder = new Vector3(0.5f - width * 0.5f, 0.4f, 0f);

            var result = PoseBodyMath.BodyRelative(leftShoulder, center, width);

            Assert.AreEqual(-0.5f, result.x, 1e-5f);
            Assert.AreEqual(0f,    result.y, 1e-5f);
        }

        [Test]
        public void BodyRelative_ZeroWidth_ReturnsNaN()
        {
            var center = new Vector2(0.5f, 0.5f);
            var point  = new Vector3(0.6f, 0.5f, 0f);

            var result = PoseBodyMath.BodyRelative(point, center, 0f);

            Assert.IsTrue(float.IsNaN(result.x));
            Assert.IsTrue(float.IsNaN(result.y));
        }

        [Test]
        public void BodyRelative_NaNInput_ReturnsNaN()
        {
            var center = new Vector2(0.5f, 0.5f);
            var point  = new Vector3(float.NaN, 0.5f, 0f);

            // width도 NaN이면 IsValidShoulderWidth → false → NaN 반환
            var result = PoseBodyMath.BodyRelative(point, center, float.NaN);

            Assert.IsTrue(float.IsNaN(result.x));
            Assert.IsTrue(float.IsNaN(result.y));
        }

        // ─── ArmAngleDegrees ─────────────────────────────────────────────────

        [Test]
        public void ArmAngleDegrees_StraightArm_ReturnsApprox180()
        {
            var shoulder = new Vector3(0.5f, 0.3f, 0f);
            var elbow    = new Vector3(0.5f, 0.5f, 0f);
            var wrist    = new Vector3(0.5f, 0.7f, 0f);

            float angle = PoseBodyMath.ArmAngleDegrees(shoulder, elbow, wrist);

            Assert.AreEqual(180f, angle, 1f);
        }

        [Test]
        public void ArmAngleDegrees_NaNInput_ReturnsNaN()
        {
            var shoulder = new Vector3(float.NaN, 0.3f, 0f);
            var elbow    = new Vector3(0.5f, 0.5f, 0f);
            var wrist    = new Vector3(0.5f, 0.7f, 0f);

            float angle = PoseBodyMath.ArmAngleDegrees(shoulder, elbow, wrist);

            Assert.IsTrue(float.IsNaN(angle));
        }

        // ─── HasNaNOrInfinity ────────────────────────────────────────────────

        [Test]
        public void HasNaNOrInfinity_NormalVector_ReturnsFalse()
        {
            Assert.IsFalse(PoseBodyMath.HasNaNOrInfinity(new Vector2(0.3f, 0.7f)));
        }

        [Test]
        public void HasNaNOrInfinity_NaNVector_ReturnsTrue()
        {
            Assert.IsTrue(PoseBodyMath.HasNaNOrInfinity(new Vector2(float.NaN, 0.5f)));
        }

        [Test]
        public void HasNaNOrInfinity_InfinityVector_ReturnsTrue()
        {
            Assert.IsTrue(PoseBodyMath.HasNaNOrInfinity(new Vector2(0.5f, float.PositiveInfinity)));
        }
    }
}
