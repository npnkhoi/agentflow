"""Integration tests for the ``examples/elementary_math`` project (local vision server).

Runs the 4-stage math workflow — Image+QuestionText -> Calculations ->
CalculatedNumbers (calculator) -> Answer -> Verdict (exact comparison) — over
MathVista items. Only stages 1 and 3 call a model. Assumes an OpenAI-compatible
vision server at http://0.0.0.0:8010 (see ``test_captioning`` for how to start
one).

These assert pipeline mechanics, not model quality: which stages ran, that the
calculator's results line up with the expressions it was given, and that each
stage's output validates. A weak model that answers everything wrong still
passes.

Run from the repo root:
    pytest test/integration/test_elementary_math.py -v
"""

import json

import pytest

from examples.elementary_math.extensions import register
from test.integration.harness import EXAMPLES_DIR, make_runner, output_path

register()  # output types + CalculatorProcessor, before any Pipeline is built

run_pipeline = make_runner("elementary_math")
PIPELINE_NAME = "elementary_math"
STAGES = ["Calculations", "CalculatedNumbers", "Answer", "Verdict"]

ITEMS = json.loads((EXAMPLES_DIR / "elementary_math" / "data" / "items.json").read_text())
ITEM_IDS = [item["id"] for item in ITEMS]


def stage_outputs(stage: str) -> dict[str, dict]:
    """Load every ``output.json`` written by ``stage``, keyed by item id."""
    out = output_path(PIPELINE_NAME) / stage
    return {p.parent.name: json.loads(p.read_text()) for p in out.glob("*/output.json")}


class TestPipelineRuns:
    def test_runs_without_error(self):
        run_pipeline("math.yaml")

    def test_all_stage_dirs_created(self):
        run_pipeline("math.yaml")
        out = output_path(PIPELINE_NAME)
        stage_dirs = {d.name for d in out.iterdir() if d.is_dir()}
        assert set(STAGES) <= stage_dirs, f"missing stage dirs: {set(STAGES) - stage_dirs}"

    @pytest.mark.parametrize("stage", STAGES)
    def test_every_item_has_output(self, stage):
        run_pipeline("math.yaml")
        outputs = stage_outputs(stage)
        assert sorted(outputs) == sorted(ITEM_IDS), (
            f"{stage} produced outputs for {sorted(outputs)}, expected {sorted(ITEM_IDS)}"
        )

    def test_second_run_uses_cache(self):
        run_pipeline("math.yaml")
        out = output_path(PIPELINE_NAME) / "Answer"
        mtimes_first = {f: f.stat().st_mtime for f in out.glob("*/output.json")}
        assert mtimes_first, "no Answer outputs to compare"

        run_pipeline("math.yaml")
        mtimes_second = {f: f.stat().st_mtime for f in out.glob("*/output.json")}

        assert mtimes_first == mtimes_second, (
            "Answer outputs were re-written on second run (cache not used)"
        )


class TestCalculatorStage:
    """Stage 2 is a CalculatorProcessor, not a model — its output is checkable exactly."""

    def test_results_align_with_expressions(self):
        run_pipeline("math.yaml")
        calculations = stage_outputs("Calculations")
        calculated = stage_outputs("CalculatedNumbers")
        assert calculated, "no CalculatedNumbers outputs"

        for item_id, numbers in calculated.items():
            expressions = calculations[item_id]["expressions"]
            got = [r["expression"] for r in numbers["results"]]
            assert got == expressions, (
                f"[{item_id}] calculator results do not match the expressions it was given"
            )

    def test_each_result_is_value_or_error(self):
        run_pipeline("math.yaml")
        calculated = stage_outputs("CalculatedNumbers")
        assert calculated, "no CalculatedNumbers outputs"

        for item_id, numbers in calculated.items():
            for result in numbers["results"]:
                has_value = result["value"] is not None
                has_error = result["error"] is not None
                assert has_value != has_error, (
                    f"[{item_id}] {result['expression']!r} must carry exactly one of value/error, got {result}"
                )

    def test_arithmetic_is_exact(self):
        """Re-evaluate the stored expressions and compare — the calculator must not
        drift from Python's own arithmetic."""
        from examples.elementary_math.extensions import evaluate

        run_pipeline("math.yaml")
        for item_id, numbers in stage_outputs("CalculatedNumbers").items():
            for result in numbers["results"]:
                if result["value"] is None:
                    continue
                assert result["value"] == pytest.approx(evaluate(result["expression"])), (
                    f"[{item_id}] stored value for {result['expression']!r} is not the exact result"
                )


class TestAnswerAndVerdict:
    def test_answer_is_a_number(self):
        run_pipeline("math.yaml")
        answers = stage_outputs("Answer")
        assert answers, "no Answer outputs"
        for item_id, answer in answers.items():
            assert isinstance(answer["value"], (int, float)), f"[{item_id}] answer is not numeric: {answer}"

    def test_verdict_is_boolean_with_reason(self):
        run_pipeline("math.yaml")
        verdicts = stage_outputs("Verdict")
        assert verdicts, "no Verdict outputs"
        for item_id, verdict in verdicts.items():
            assert isinstance(verdict["correct"], bool), f"[{item_id}] verdict is not boolean: {verdict}"
            assert verdict["reason"].strip(), f"[{item_id}] verdict has an empty reason"

    def test_verdict_agrees_with_ground_truth(self):
        """Stage 4 grades Answer against the dataset's ground truth by exact
        comparison, so the verdict must match one computed here — no model
        involved, no tolerance for drift."""
        run_pipeline("math.yaml")
        ground_truth = {item["id"]: item["data"]["ground_truth_answer"] for item in ITEMS}
        answers = stage_outputs("Answer")
        verdicts = stage_outputs("Verdict")
        assert verdicts, "no Verdict outputs"

        for item_id, verdict in verdicts.items():
            expected = answers[item_id]["value"] == pytest.approx(ground_truth[item_id], abs=0.01)
            assert verdict["correct"] == expected, (
                f"[{item_id}] judge said correct={verdict['correct']} but "
                f"answer={answers[item_id]['value']} vs ground truth={ground_truth[item_id]}"
            )
