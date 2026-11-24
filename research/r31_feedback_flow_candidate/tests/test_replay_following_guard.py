"""Synthetic replay payloads through the demonstration_following assembly.

No camera, no model, no contract, no network, no files: _split_result and _replay_following read
plain dicts only, so the two blocks the replay handler assembles can be built here exactly as
POST /api/v1/replays builds them.

The case under test is the one the camera path already refuses. _explicit_comparison only needs
sequence_features to succeed, so it returns axis / joint / location numbers for an observation
predict() rejected - a take whose every reference match fell under the path evidence minimum
returns observation_status INSUFFICIENT with a full explicit comparison beside it. The replay
branch used to publish that as demonstration_following.status = AVAILABLE.

These are structural assertions about what the server may claim. Nothing here measures accuracy,
establishes a threshold, or says anything about how a person performed.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wis_lab.service.app import _replay_following, _split_result   # noqa: E402

CLIP, WORD = "NIKL_JOTA_01", "JOTA"


def binding(status="BOUND"):
    return {"status": status, "demonstration_clip_id": CLIP, "demonstration_word_id": WORD,
            "bound_at": "replay_request", "reference_set_id": "nikl19_uniform32_v1"}


def comparison(word_id=WORD):
    """What _explicit_comparison returns when sequence_features succeeded."""
    return {"status": "OK", "reference_word_id": word_id,
            "axis_deviation": {"status": "OK", "axes": {"handshape": {"distance": 0.31}}},
            "joint_deviation": {"status": "OK"},
            "location_path": {"status": "OK", "steps": []},
            "is_a_verdict": False}


def replay_result(observation_status="OK", insufficient_reason=None):
    return {"observation_status": observation_status,
            "insufficient_reason": insufficient_reason,
            "ranking": [], "origin": "replay", "execution_assessment": "UNCONFIGURED",
            "accepted": None, "counts_as_human_attempt": False,
            "explicit_reference_comparison": comparison()}


def assemble(result, bind, erc):
    """The replay handler's own order: split the result, then apply the explicit comparison."""
    _split_result(result, {"reference_binding": bind})
    _replay_following(result, bind, erc)
    return result["demonstration_following"]


class ReplayFollowingGuard(unittest.TestCase):

    def test_insufficient_observation_is_not_available(self):
        """The trigger: predict() refused the take, the explicit comparison still has numbers."""
        result = replay_result("INSUFFICIENT", "no reference produced an admissible warping path")
        df = assemble(result, binding(), comparison())
        self.assertEqual(df["status"], "UNDETERMINED_OBSERVATION_INSUFFICIENT")
        self.assertEqual(df["observation_status"], "INSUFFICIENT")
        self.assertEqual(df["insufficient_reason"],
                         "no reference produced an admissible warping path")
        # refused, not emptied of its reason, and no number is carried over as feedback
        for field in ("axis_deviation", "joint_deviation", "location_path", "axis_checks"):
            self.assertIsNone(df[field], field)
        self.assertFalse(df["is_a_verdict"])
        # the explicit comparison itself is untouched: it is still readable under its own name
        self.assertEqual(result["explicit_reference_comparison"]["status"], "OK")

    def test_accepted_observation_still_reaches_the_learner(self):
        """The legitimate replay is unchanged: a working comparison must not start disappearing."""
        result = replay_result("OK")
        erc = comparison()
        df = assemble(result, binding(), erc)
        self.assertEqual(df["status"], "AVAILABLE")
        self.assertEqual(df["reference_word_id"], WORD)
        self.assertEqual(df["reference_clip_id"], CLIP)
        self.assertIs(df["axis_deviation"], erc["axis_deviation"])
        self.assertIs(df["joint_deviation"], erc["joint_deviation"])
        self.assertIs(df["location_path"], erc["location_path"])
        self.assertIsNone(df["axis_checks"])

    def test_comparison_about_another_word_is_never_promoted(self):
        """Unchanged behaviour, asserted so the new branch cannot loosen it."""
        df = assemble(replay_result("OK"), binding(), comparison("MOREUDA"))
        self.assertNotEqual(df["status"], "AVAILABLE")
        self.assertIsNone(df["axis_deviation"])

    def test_unbound_demonstration_is_never_promoted(self):
        """No binding means no comparison, whatever the observation status is."""
        bind = {"status": "UNDETERMINED_NO_DEMONSTRATION_SPECIFIED",
                "demonstration_clip_id": None, "demonstration_word_id": None}
        df = assemble(replay_result("OK"), bind, comparison())
        self.assertEqual(df["status"], "UNDETERMINED_NO_DEMONSTRATION_SPECIFIED")
        self.assertIsNone(df["axis_deviation"])
        self.assertIn("NOT used as a stand-in", df["why"])


if __name__ == "__main__":
    unittest.main()
