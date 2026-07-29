"""Unit tests for the elementary-math example's expression evaluator (no LLM
server required).

The evaluator parses model-authored text, so the tests cover both the
arithmetic it must get right and the constructs it must refuse.
"""

import pytest

from examples.elementary_math.extensions import CalculatedNumbers, CalculatorProcessor, evaluate


class TestArithmetic:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("3 * 4.6", 13.8),
            ("8 - 2 - 1", 5.0),
            ("2 * (2 + 1)", 6.0),
            ("4 / 2", 2.0),
            ("7 // 2", 3.0),
            ("7 % 2", 1.0),
            ("2 ** 3", 8.0),
            ("-5 + 2", -3.0),
            ("6", 6.0),
        ],
    )
    def test_evaluates(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected)

    def test_returns_float(self):
        assert isinstance(evaluate("2 + 2"), float)

    def test_trailing_equals_is_tolerated(self):
        """Models often append '=' despite the prompt saying not to."""
        assert evaluate("2 + 2 =") == pytest.approx(4.0)

    def test_surrounding_whitespace_is_tolerated(self):
        assert evaluate("  10 / 4  ") == pytest.approx(2.5)


class TestRejects:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",  # code execution
            "open('/etc/passwd').read()",             # attribute access + call
            "x + 1",                                  # variable name
            "3 | 6",                                  # unsupported operator
            "4.6, 3, *",                              # postfix junk a small model emits
            "circle + square = 5",                    # words
            "",                                       # empty
            "   ",                                    # blank
        ],
    )
    def test_raises(self, expression):
        with pytest.raises(Exception):
            evaluate(expression)


class TestProcessor:
    """CalculatorProcessor is exercised without a Pipeline: it only needs the
    `calculations` input to produce its output."""

    class _StubCalculations:
        def __init__(self, expressions):
            self.expressions = expressions

    def _processor(self) -> CalculatorProcessor:
        # bypass Processor.__init__, which wants a Pipeline and a StageConfig
        return CalculatorProcessor.__new__(CalculatorProcessor)

    def test_evaluates_each_expression(self):
        out = self._processor()({"calculations": self._StubCalculations(["1 + 1", "3 * 4.6"])})
        assert isinstance(out, CalculatedNumbers)
        assert [r.expression for r in out.results] == ["1 + 1", "3 * 4.6"]
        assert [r.value for r in out.results] == pytest.approx([2.0, 13.8])
        assert all(r.error is None for r in out.results)

    def test_bad_expression_is_recorded_not_raised(self):
        out = self._processor()({"calculations": self._StubCalculations(["2 + 2", "3 | 6"])})
        assert out.results[0].value == pytest.approx(4.0)
        assert out.results[1].value is None
        assert "unsupported operator" in out.results[1].error

    def test_missing_input_returns_none(self):
        """A None input means the upstream stage failed; the stage must fail too
        rather than store an empty result."""
        assert self._processor()({"calculations": None}) is None

    def test_empty_expression_list(self):
        out = self._processor()({"calculations": self._StubCalculations([])})
        assert out.results == []
