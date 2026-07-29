"""Unit tests for the elementary-math example's grading processor (no LLM server
required).

Stage 4 decides the reported accuracy, so its comparison has to be exact and its
tolerance has to come from the config rather than from a model's opinion.
"""

import pytest

from examples.elementary_math.extensions import DEFAULT_TOLERANCE, Answer, Verdict, VerdictProcessor


class _StubStageConfig:
    def __init__(self, kwargs=None):
        self.kwargs = kwargs


def processor(kwargs=None) -> VerdictProcessor:
    """Build a VerdictProcessor without a Pipeline: it only reads its stage kwargs."""
    p = VerdictProcessor.__new__(VerdictProcessor)
    p._stage_config = _StubStageConfig(kwargs)
    return p


def grade(answer: float, ground_truth: float, kwargs=None) -> Verdict:
    return processor(kwargs)({"answer": Answer(value=answer), "ground_truth_answer": ground_truth})


class TestGrading:
    def test_exact_match_is_correct(self):
        assert grade(13.8, 13.8).correct is True

    def test_integer_and_float_forms_match(self):
        """Ground truth is stored as 4.0; an answer of 4 must still pass."""
        assert grade(4, 4.0).correct is True

    def test_mismatch_is_incorrect(self):
        assert grade(3.0, 4.0).correct is False

    def test_sign_matters(self):
        assert grade(-4.0, 4.0).correct is False

    @pytest.mark.parametrize(
        "answer,expected",
        [(2.0, True), (2.005, True), (2.01, True), (2.011, False), (1.99, True), (1.9, False)],
    )
    def test_default_tolerance_boundary(self, answer, expected):
        assert grade(answer, 2.0).correct is expected

    def test_returns_verdict_model(self):
        assert isinstance(grade(1.0, 1.0), Verdict)

    def test_reason_reports_both_numbers(self):
        reason = grade(3.0, 4.0).reason
        assert "3" in reason and "4" in reason
        assert reason.strip()


class TestTolerance:
    def test_tolerance_comes_from_stage_kwargs(self):
        assert grade(2.4, 2.0, kwargs={"tolerance": 0.5}).correct is True
        assert grade(2.4, 2.0, kwargs={"tolerance": 0.1}).correct is False

    def test_zero_tolerance_demands_exact_equality(self):
        assert grade(2.0, 2.0, kwargs={"tolerance": 0}).correct is True
        assert grade(2.0001, 2.0, kwargs={"tolerance": 0}).correct is False

    def test_default_applies_when_kwargs_absent(self):
        assert processor(None)._tolerance == pytest.approx(DEFAULT_TOLERANCE)
        assert processor({})._tolerance == pytest.approx(DEFAULT_TOLERANCE)


class TestMissingInputs:
    def test_missing_answer_returns_none(self):
        """An upstream failure must fail this stage too, not score it as wrong."""
        assert processor()({"answer": None, "ground_truth_answer": 4.0}) is None

    def test_missing_ground_truth_returns_none(self):
        assert processor()({"answer": Answer(value=4.0), "ground_truth_answer": None}) is None
