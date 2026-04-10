"""
Unit tests for core.input_formater.InputFormater.
No LLM server required — these are pure unit tests.
"""
import pytest
from pydantic import BaseModel

from agentflow.input_formater import InputFormater


class _MyOutput(BaseModel):
    value: int
    label: str


class TestDefaultFormat:
    """InputFormater with input_format=None: serialize each non-image input."""

    def test_single_text_input(self):
        fmt = InputFormater(None, ["caption"])
        result = fmt.format({"caption": "A funny cat meme."})
        assert result == "caption:\nA funny cat meme."

    def test_image_skipped(self):
        fmt = InputFormater(None, ["image"])
        result = fmt.format({"image": "/some/path.png"})
        assert result == ""

    def test_image_plus_text(self):
        fmt = InputFormater(None, ["image", "caption"])
        result = fmt.format({"image": "/some/path.png", "caption": "Hello"})
        assert result == "caption:\nHello"

    def test_two_text_inputs(self):
        fmt = InputFormater(None, ["title", "body"])
        result = fmt.format({"title": "My Title", "body": "My Body"})
        assert "title:\nMy Title" in result
        assert "body:\nMy Body" in result
        # separated by double newline
        assert "\n\n" in result

    def test_pydantic_model_serialized_as_json(self):
        fmt = InputFormater(None, ["output"])
        obj = _MyOutput(value=42, label="test")
        result = fmt.format({"output": obj})
        assert '"value"' in result
        assert '"label"' in result
        assert "42" in result

    def test_none_input_format_ignores_registry(self):
        """Even if registry has entries, None input_format uses default path."""
        InputFormater.register("dummy", lambda inputs, names: "DUMMY")
        fmt = InputFormater(None, ["caption"])
        result = fmt.format({"caption": "hello"})
        assert result == "caption:\nhello"


class TestRegistry:
    """InputFormater.register() extension mechanism."""

    def setup_method(self):
        # isolate registry changes per test using a backup
        self._original = dict(InputFormater._registry)

    def teardown_method(self):
        InputFormater._registry = self._original

    def test_registered_format_called(self):
        def upper_handler(inputs, names):
            return inputs["text"].upper()

        InputFormater.register("upper", upper_handler)
        fmt = InputFormater("upper", ["text"])
        assert fmt.format({"text": "hello"}) == "HELLO"

    def test_unregistered_format_raises(self):
        fmt = InputFormater("nonexistent_format", ["text"])
        with pytest.raises(NotImplementedError, match="nonexistent_format"):
            fmt.format({"text": "hello"})

    def test_handler_receives_input_names(self):
        received = {}

        def capture_handler(inputs, names):
            received["names"] = names
            return "ok"

        InputFormater.register("capture", capture_handler)
        fmt = InputFormater("capture", ["foo", "bar"])
        fmt.format({"foo": 1, "bar": 2})
        assert received["names"] == ["foo", "bar"]

    def test_register_overwrites(self):
        InputFormater.register("fmt", lambda inputs, names: "first")
        InputFormater.register("fmt", lambda inputs, names: "second")
        fmt = InputFormater("fmt", [])
        assert fmt.format({}) == "second"

    def test_toy_domain_format(self):
        """Toy domain: combine a 'score' and 'comment' into a structured summary."""

        def review_handler(inputs, names):
            score = inputs["score"]
            comment = inputs["comment"]
            stars = "*" * int(score)
            return f"Rating: {stars} ({score}/5)\nComment: {comment}"

        InputFormater.register("review_summary", review_handler)
        fmt = InputFormater("review_summary", ["score", "comment"])
        result = fmt.format({"score": 4, "comment": "Pretty good!"})
        assert "****" in result
        assert "Pretty good!" in result
