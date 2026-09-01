*This project has been created as part of the 42 curriculum by gpecelli.*

# Call Me Maybe - Constrained Decoding for LLMs

## Description
This project focuses on implementing structured function calling for a lightweight Causal Language Model (Qwen/Qwen3-0.6B) running locally. The goal is to enforce the LLM to strictly output syntactically valid JSON responses matching specific function schemas. This is achieved entirely from scratch without using any external libraries like `transformers`' built-in constrained decoding pipelines, `dspy`, or `outlines`.

## Instructions

### Installation
Ensure you have Python and `uv` installed, then set up the environment:
```bash
make install
```
If you are on a 42 Network machine (USER -> $USER | $(whoami) | user_name)
```bash
mkdir -p /goinfre/$USER/.cache /goinfre/$USER/tmp
chown -R $(id -u):$(id -g)
```
```bash
XDG_CACHE_HOME=/goinfre/$USER/.cache TMPDIR=/goinfre/$USER/tmp make install
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

## Algorithm Explanation
The core of the solution is a custom Finite State Machine (FSM) referred to as the **Nudger**. 
When the LLM generates the function name and the JSON structural boilerplate, the Nudger acts as a guardrail. It checks what has been generated so far against a pre-encoded list of valid function signatures. At each generation step, the Nudger computes the `allowed_ids`—the exact subset of token IDs that are legally permitted next.
For parameter values:
- **Numbers**: The algorithm uses a pre-computed cache of tokens that contain only numeric characters (`0-9.-`), forcing the model to pick the highest probability token from this restricted set.
- **Strings**: The model is allowed to use the full vocabulary until it generates a closing quote (`"`).

## Design Decisions
1. **No Global Variables**: To adhere to strict software engineering standards, the pre-computed token lists (like numeric tokens) are initialized once in the main loop and passed down as arguments (`Dependency Injection`), keeping the functions modular and testable.
2. **Fast C-Level Indexing**: Instead of relying on slow Python `lambda` functions to sort or find the maximum probability in the `logits` list (which has ~150,000 items), the algorithm uses `key=logits.__getitem__`, exploiting Python's underlying C-implementation for massive speedups.
3. **Smart Token Splitting**: Language models often merge characters into single tokens. For example, when generating a string, the model might output a single token `]"` containing both the text and the closing quote. To handle this, the algorithm decodes the token, splits it by `"`, and safely re-encodes the valid part before closing the parameter, ensuring 100% accuracy.

## Performance Analysis
The project successfully processes all test prompts in under 5 minutes on a standard CPU.
- **Initial Execution Time**: ~6 minutes and 30 seconds.
- **Final Execution Time**: ~4 minutes and 15 seconds.
This massive performance gain was achieved primarily by skipping the LLM entirely whenever the choice was deterministic. The accuracy is near-perfect due to the strict FSM enforcement.

## Challenges Faced
1. **CPU Bottlenecks**: A 0.6B parameter model requires billions of floating-point operations per forward pass (`get_logits_from_input_ids`). Calling it for every single character in the JSON boilerplate was too slow.
   **Solution**: Added a check to see if `len(allowed_ids) == 1`. If the Nudger determines there is only one valid continuation (e.g., the rest of a function name), the algorithm bypasses the neural network completely and just auto-completes the token, saving dozens of seconds per prompt.
2. **Infinite Loops & Truncated Text**: Using fixed loops like `for step in range(15)` caused long string parameters to be truncated, breaking the JSON.
   **Solution**: Switched to a robust bounded-loop approach (`for step in range(100)` for values and `50` for structure) which acts as a safe `while True` loop, giving the model enough breathing room to finish long strings without risking an infinite loop.

## Testing Strategy
1. Created an `unconstrained_decoder` script to test the model's raw generative capabilities in a sandbox environment before applying constraints.
2. Used the provided `function_calling_tests.json` to iteratively find edge cases (like negative numbers, decimal points, and complex regex strings).

## Resources
- [Hugging Face Tokenizer Documentation](https://huggingface.co/docs/tokenizers/index)
- [Grammar-Guided Generation Concepts](https://arxiv.org/abs/2307.09702)
- **AI Usage**: AI was heavily utilized as a pair-programming partner during the development of this project. It was primarily used to:
  - Explain tokenization edge cases (like token fusion).
  - Identify Python-specific bottlenecks (like the `lambda` sorting issue).
  - Refactor algorithms for CPU optimization and brainstorm FSM bypass strategies.
