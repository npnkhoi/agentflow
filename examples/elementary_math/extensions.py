"""Domain extensions for the elementary-math example: output types plus the two
processors that do not call a model.

Everything here is registered through agentflow's public registration APIs, so
`agentflow/` itself stays untouched (see docs/extension_points.md).

Two of the four pipeline stages are deterministic:

- `CalculatorProcessor` (stage 2) takes the `Calculations` an LLM wrote and
  evaluates them with Python's `ast`, so the arithmetic is exact and the model
  never has to do mental math.
- `VerdictProcessor` (stage 4) grades the answer against ground truth by
  numeric comparison. Grading two floats is not a judgement call, so it must not
  depend on a model's mood.

Usage:
    from examples.elementary_math.extensions import register
    register()                      # before constructing Pipeline
"""

import ast
import operator
from pathlib import Path

from pydantic import BaseModel

from agentflow.pipeline import Pipeline
from agentflow.processors.base import Processor


class Calculations(BaseModel):
    """Arithmetic an LLM decided is needed to answer the question."""

    reasoning: str
    expressions: list[str]


class CalculatedNumber(BaseModel):
    expression: str
    value: float | None = None
    error: str | None = None


class CalculatedNumbers(BaseModel):
    """Results of evaluating each expression — the calculator's output."""

    results: list[CalculatedNumber]


class Answer(BaseModel):
    value: float


class Verdict(BaseModel):
    correct: bool
    reason: str


# How far an answer may sit from the ground truth and still count as correct.
# Override per stage in the config with `kwargs: {tolerance: ...}`.
DEFAULT_TOLERANCE = 0.01


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression. Only numbers and + - * / // % ** are
    allowed — no names, calls, or attribute access, so this never executes
    model-authored code."""
    cleaned = expression.strip().rstrip("=").strip()
    if not cleaned:
        raise ValueError("empty expression")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"unsupported constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return float(_eval(ast.parse(cleaned, mode="eval")))


class CalculatorProcessor(Processor):
    """`Calculations` → `CalculatedNumbers`, evaluated exactly, no model involved.

    A failed expression is recorded with its error rather than failing the
    stage: the answering stage sees which computations worked and which did not.
    """

    def __call__(self, inputs: dict, logger=None, output_dir: Path | None = None) -> CalculatedNumbers | None:
        calculations = inputs.get("calculations")
        if calculations is None:
            return None

        results = []
        for expression in calculations.expressions:
            try:
                results.append(CalculatedNumber(expression=expression, value=evaluate(expression)))
            except Exception as e:
                results.append(CalculatedNumber(expression=expression, error=f"{type(e).__name__}: {e}"))
        print(f"evaluated {len(results)} expression(s)", file=logger, flush=True)
        return CalculatedNumbers(results=results)


class VerdictProcessor(Processor):
    """`GroundTruthAnswer`, `Answer` → `Verdict`, by numeric comparison.

    Deterministic on purpose: whether two numbers agree is arithmetic, not
    judgement, so the evaluation stage stays reproducible no matter which model
    produced the answer.
    """

    @property
    def _tolerance(self) -> float:
        kwargs = getattr(self, "_stage_config", None) and self._stage_config.kwargs
        return float((kwargs or {}).get("tolerance", DEFAULT_TOLERANCE))

    def __call__(self, inputs: dict, logger=None, output_dir: Path | None = None) -> Verdict | None:
        answer = inputs.get("answer")
        ground_truth = inputs.get("ground_truth_answer")
        if answer is None or ground_truth is None:
            return None

        tolerance = self._tolerance
        difference = abs(answer.value - float(ground_truth))
        # Nudge the bound by a relative hair: in binary floating point
        # |2.01 - 2.0| lands just under 0.01 while |1.99 - 2.0| lands just over,
        # and grading must not depend on which side of the truth an answer fell.
        correct = difference <= tolerance * (1 + 1e-9)
        verdict = Verdict(
            correct=correct,
            reason=(
                f"answer {answer.value:g} vs ground truth {float(ground_truth):g}: "
                f"|difference| = {difference:g}, "
                f"{'within' if correct else 'outside'} tolerance {tolerance:g}"
            ),
        )
        print(verdict.reason, file=logger, flush=True)
        return verdict


def register() -> None:
    """Register this domain's types and processors. Call before building Pipeline."""
    Pipeline.register_type("Calculations", Calculations)
    Pipeline.register_type("CalculatedNumbers", CalculatedNumbers)
    Pipeline.register_type("Answer", Answer)
    Pipeline.register_type("Verdict", Verdict)
    Pipeline.register_processor("CalculatorProcessor", CalculatorProcessor)
    Pipeline.register_processor("VerdictProcessor", VerdictProcessor)
