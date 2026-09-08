*This project has been created as part of the 42 curriculum by gpecelli.*

# Call Me Maybe — Constrained Decoding for LLMs

## Description
Call Me Maybe implements a lightweight, from-scratch constrained decoding system for a causal language model (Qwen 0.6B). The tool forces the model to produce structured JSON function calls that exactly match predefined schemas, enabling reliable programmatic interpretation of natural-language requests without relying on external constrained-decoding libraries.

This repository contains the constrained decoder, a small LLM wrapper, and a test harness that validates function-calling accuracy across a suite of edge cases (numbers, paths, SQL, templates, encodings).

## Instructions

### Installation
Ensure you have Python and `uv` installed, then set up the environment:
```bash
make install
```
If you are on a 42 Network machine (USER -> $USER | $(whoami) | user_name)
```bash
rm -rf .venv && mkdir -p /goinfre/$USER/{venvs/call_me_maybe,.cache,tmp,uv,huggingface}
```
```bash
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/venvs/call_me_maybe UV_LINK_MODE=copy XDG_CACHE_HOME=/goinfre/$USER/.cache TMPDIR=/goinfre/$USER/tmp UV_CACHE_DIR=/goinfre/$USER/uv HF_HOME=/goinfre/$USER/huggingface
```

### Execution
To run the full test suite and measure execution time:
```bash
make run
```
You can also run a specific test file manually:
```bash
uv run python -m src --input data/input/function_calling_tests.json
```
Run a specific test for the Unconstrained decoding:
```bash
uv run python test_unconstrained.py
```

## Algorithm explanation
At its core the constrained decoder uses a small Finite State Machine (the "Nudger") combined with token-level logit inspection to limit the model's output to schema-compliant continuations.

- Nudger: pre-encodes valid function signatures as token sequences and, at each generation step, computes `allowed_ids` — the exact token IDs that may legally follow the tokens generated so far. When `allowed_ids` contains a single token, the algorithm auto-appends it without consulting the model logits, which speeds deterministic completions.
- Numeric tokens: the implementation precomputes a compact set of tokens that represent numeric characters (digits, decimal point, sign). When a numeric parameter is expected, logits are restricted to that set and the highest-scoring token is chosen.
- String parameters: the decoder permits the full vocabulary until a closing quote is observed. To correctly handle tokens that embed both content and closing quote (token fusion), the decoder decodes the selected token, splits on the quote character, re-encodes the valid portion, and appends it safely.

This approach keeps generation strictly valid by construction and minimizes downstream parsing and validation.

## Design decisions
- Minimal dependencies: implemented from first principles so the behavior is fully observable and debuggable.
- Deterministic bypass: auto-complete deterministic sequences when the Nudger yields one `allowed_id`, greatly improving speed.
- Defensive token handling: re-encode partial tokens when a decoded token contains both payload and delimiter characters (e.g., trailing `\"`).
- No global state: token caches and helper structures are injected rather than global, improving testability.

## Performance analysis
- Accuracy: enforcing the schema at the token level yields high functional correctness for the test suite (most prompts resolve to valid function calls and parameters).
- Speed: short-circuiting deterministic continuations reduces unnecessary LLM forward passes; typical full-suite runs complete in a few minutes on a modest CPU.
- Reliability: the FSM + token-validation approach is robust to common tokenizer edge cases (fused tokens, escaped characters), though some pathological strings may require targeted sanitization.

## Challenges faced
- Token fusion and escaping: tokens sometimes contain punctuation and escape sequences (e.g., `}\"`), which required careful decoding and re-encoding to avoid truncation or double-escaping.
- Truncation & loops: naive fixed-step loops caused truncated parameters; switching to bounded loops with sensible limits prevented hangs while allowing long values.
- Encoding variants: file paths, SQL fragments and Windows-style paths required additional unescape logic to present values in the expected grading format.

## Testing strategy
1. Unit and integration tests are driven by `data/input/function_calling_tests.json`, covering numeric, string, SQL, path and template examples.
2. An `unconstrained` mode isolates the model's natural output to reproduce edge cases before applying constraints.
3. Iterative debugging: token-level debugging printouts were used to inspect generated token IDs and decoded fragments for failing cases.

## Example usage
Run the full harness (recommended):
```bash
make run
```

Run a single prompt set:
```bash
uv run python -m src --input data/input/function_calling_tests.json
```

Example output (successful function call):
```
{'prompt': 'What is the product of 3 and 5?', 'name': 'fn_multiply_numbers', 'parameters': {'a': 3.0, 'b': 5.0}}
```

## Resources
- Hugging Face Tokenizer documentation: https://huggingface.co/docs/tokenizers/index
- Grammar-guided generation (paper): https://arxiv.org/abs/2307.09702

### AI usage disclosure
AI assisted development in these areas:
- Explaining tokenization edge cases (token fusion and escapes).
- Brainstorming and refining the constrained decoding approach.
- Iterative code suggestions and debugging guidance (token-level instrumentation).

## Contributing
Issues and PRs are welcome. If you plan to change the constrained-decoder core, please open an issue first to discuss design trade-offs.

## License
This repository is provided for educational purposes.
