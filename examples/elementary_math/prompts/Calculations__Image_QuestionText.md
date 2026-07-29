You are given a figure and an elementary-school math question about it. Do NOT compute the answer yourself — a calculator will run your arithmetic for you.

Read the figure to find the numbers the question depends on, then write the arithmetic needed to answer it.

Every string in `expressions` must be a valid infix arithmetic expression that Python could evaluate, for example `16 - 3 - 4` or `(2 + 5) * 3`. Specifically:
- Write it in normal infix order (`4.6 * 3`), never as a list of numbers and symbols (`4.6, 3, *` is wrong).
- Use only digits, `.`, parentheses, and the operators `+ - * / // % **`.
- Never use variable names, words, units, `=`, or commas.
- Make each expression self-contained: substitute actual numbers rather than referring to an earlier result.
- Put the expression that yields the final answer last.
- If the answer can be read straight off the figure with no arithmetic, give one expression that is just that number, e.g. `7`.

Examples of the exact output format:

Question: "Natalie buys 4.6 kilograms of turmeric at $3 per kilogram. What is the total cost?"
{"reasoning": "The table gives turmeric at $3 per kilogram, so multiply the price by 4.6 kilograms.", "expressions": ["3 * 4.6"]}

Question: "Subtract all red things. Subtract all tiny balls. How many objects are left?"
{"reasoning": "The scene has 8 objects; 2 are red and 1 more is a tiny ball, so remove 3 of them.", "expressions": ["8 - 2 - 1"]}

Question: "What is the perimeter of the rectangle?"
{"reasoning": "The rectangle is 2 units wide and 1 unit tall, so the perimeter is twice the sum of the sides.", "expressions": ["2 * (2 + 1)"]}

Respond with JSON matching the schema: {"reasoning": "<how you read the figure and what must be computed>", "expressions": ["<expression>", ...]}.
