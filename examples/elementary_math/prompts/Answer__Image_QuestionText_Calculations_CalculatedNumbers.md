You are given a figure, an elementary-school math question about it, the arithmetic that was planned for it, and the exact results a calculator produced for that arithmetic.

Give the final answer to the question as a single number.

- Trust the calculator's `value` fields over any arithmetic of your own.
- The last successfully evaluated expression is usually the final answer, but check it against what the question actually asks.
- If an expression carries an `error` instead of a `value`, ignore it and rely on the ones that worked.
- Answer with a bare number only — no units, no words.

Respond with JSON matching the schema: {"value": <number>}.
