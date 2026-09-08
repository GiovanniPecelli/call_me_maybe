from src.validator import tokens_validator
from typing import Any
try:
    from llm_sdk import Small_LLM_Model
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Module 'Small_LLM_Model' not available\n"
        f"Details: {e}"
    )
import json
import re


def _clean_extracted_string(s: str) -> Any:
    if s is None:
        return s
    s = s.strip()
    # remove eventually trailing quote/comma attached from the decoder
    # (do not strip closing '}' here — it can be part of valid templates)
    s = re.sub(r'["\s,]+$', '', s)
    # try unescape JSON (manage \\ -> \)
    try:
        return json.loads(f'"{s}"')
    except Exception:
        return s.replace('\\\\', '\\')


def generate_value(
        type_parameter: str,
        model: Small_LLM_Model,
        input_ids_list: list[int],
        generated_tokens: list[int]
) -> Any:
    """ Generate and extract the parameter value from the LLM based
        on its expected type.
        __getitem__ - Uses fast C-level native indexing instead of a slow
        lambda. When calling my_list[1], Python internally executes:
        my_list.__getitem__(1).
        Using __getitem__ directly allows us to avoid the slowness of a
        lambda function.
    Args:
        type_parameter (str): Expected parameter type ('number' or 'string').
        model (Small_LLM_Model): The LLM wrapper instance used for decoding
        and logits.
        input_ids_list (list[int]): Accumulated context token IDs.
        generated_tokens (list[int]): Tokens generated specifically for this
        response.
    Returns:
        Any: The extracted value cast to float/int for numbers, or str for
        strings.
    """
    # valid_ids contain validated token -> shortest vocab to improve efficency
    valid_ids = tokens_validator(model)
    value_tokens = []

    for _ in range(100):
        # logit is a list of float
        # the token ID is the index into the logits list.
        logits = model.get_logits_from_input_ids(input_ids_list)

        if type_parameter == "number" or type_parameter == "integer":
            #  max('ID 0' from valid_ids and find it in logit
            # -> logits[0] -> max take the highest logits[i] value)
            best_token_id = max(valid_ids, key=logits.__getitem__)
            clean_word = model.decode([best_token_id]).strip()
            if clean_word in [",", "}"]:
                break

        elif type_parameter == "string":
            best_token_id = logits.index(max(logits))
            word = model.decode([best_token_id])
            idx = -1
            for i, char in enumerate(word):
                if char == '"':
                    # bs is backslash - if bs are even -> quote not excaped
                    bs = 0
                    j = i - 1
                    while j >= 0 and word[j] == '\\':
                        bs += 1
                        j -= 1
                    # real quote found
                    if bs % 2 == 0:
                        idx = i
                        break
            if idx != -1:
                clean_word = word[:idx]
                if clean_word != "":
                    # .tolist() always returns a list (one or more elements).
                    # Why? LLMs use a system called BPE (Byte-Pair Encoding).
                    # A single word can be split into multiple tokens.
                    # To safely handle any tokenizer rule always ret: list.
                    extra_token = model.encode(clean_word).tolist()[0]
                    input_ids_list.extend(extra_token)
                    generated_tokens.extend(extra_token)
                    value_tokens.extend(extra_token)
                break

            clean_word = word.strip()
            # print(
            #     f"[DBG generate_value] word={word!r} "
            #     f"best_token_id={best_token_id} "
            #     f"decode={model.decode([best_token_id])!r} "
            #     f"value_tokens={value_tokens}"
            # )
            if clean_word == "}":
                # safety: ensure the current token decodes to '}'
                if model.decode([best_token_id]).strip() == "}":
                    input_ids_list.append(best_token_id)
                    generated_tokens.append(best_token_id)
                    value_tokens.append(best_token_id)
                break

        input_ids_list.append(best_token_id)
        generated_tokens.append(best_token_id)
        value_tokens.append(best_token_id)

    # if the type_parameter expected is "number"
    # -> cast to "float" or "int" bf return
    raw_val = model.decode(value_tokens).strip()
    if type_parameter == "number":
        return float(raw_val)
    if type_parameter == "integer":
        return int(raw_val)
    return raw_val


def force_string(
        target_string: str,
        model: Small_LLM_Model,
        input_ids_list: list[int],
        generated_tokens: list[int]
) -> None:
    """ Force specific target tokens into the context and generated token
        list.
    Args:
        target_string (str): The exact text sequence to inject.
        model (Small_LLM_Model): The LLM wrapper instance used to encode the
        string.
        input_ids_list (list[int]): Accumulated context token IDs to update.
        generated_tokens (list[int]): List of generated token IDs to update.
    """
    target_tokens = model.encode(target_string).tolist()[0]

    input_ids_list.extend(target_tokens)
    generated_tokens.extend(target_tokens)


def nudger(
        generated_tokens: list[int],
        encoded_functions: list[list[int]]
) -> list[int]:
    """ Manage Finite State Machine (FSM) states for Grammar-Guided
        Generation.
        Tracks the current state of the generated sequence and determines
        the syntactically valid subset of next token IDs from candidate
        function signatures to enforce schema compliance.
    Args:
        generated_tokens (list[int]): Token sequence generated so far.
        encoded_functions (list[list[int]]): Pre-encoded token sequences
        of valid function prefixes.
    Returns:
        list[int]: List of allowed token IDs for the current generation step.
    """
    step = len(generated_tokens)
    allowed_ids = []

    # if func_tokens[:step] == generated_tokens:
    # -> builds the available next steps based on what is already written
    for func_tokens in encoded_functions:
        if func_tokens[:step] == generated_tokens:
            if step < len(func_tokens):
                allowed_ids.append(func_tokens[step])
    return allowed_ids


def constrained_decoder(
        model: Small_LLM_Model,
        input_ids_list: list[int],
        functions_json: list[dict[str, Any]],
        user_question: str
) -> dict[str, Any]:
    """ Perform constrained decoding to pick a function and extract valid
        parameters. Uses token masking against candidate function definitions
        to ensure only schema-compliant function calls are generated by the
        LLM.
    Args:
        model (Small_LLM_Model): The LLM wrapper instance.
        input_ids_list (list[int]): Initial prompt token IDs.
        functions_json (list[dict]): List of available function definitions
        with schemas.
        user_question (str): The original natural-language user query.
    Returns:
        dict: Structured function call containing
        'prompt', 'name', and 'parameters'.
    """
    functions_name = [f['name'] for f in functions_json]

    # ========================================================================
    #    Implement the constrained decoding for the function_name section.
    # ========================================================================
    encoded_functions = []
    for name in functions_name:
        full_string = '"name": "' + name + '", "parameters": {'
        func_tokens = model.encode(full_string).tolist()[0]
        encoded_functions.append(func_tokens)

    generated_tokens: list[int] = []

    for step in range(50):
        allowed_ids = nudger(generated_tokens, encoded_functions)

        if not allowed_ids or allowed_ids == [None]:
            break

        if len(allowed_ids) == 1:
            best_token_id = allowed_ids[0]
        else:
            logits = model.get_logits_from_input_ids(input_ids_list)
            # TAKE the best_token_id in the logits (float_values)
            best_token_id = max(allowed_ids, key=lambda i: logits[i])

        input_ids_list.append(best_token_id)
        generated_tokens.append(best_token_id)

    final_text = model.decode(generated_tokens)

    # After the name has been chosen
    # Extract the chosen function's name
    after_prefix = final_text.split('"name": "')[1]
    function_chosen = after_prefix.split('"')[0]
    extracted_params = {}
    parameters = {}
    for func in functions_json:
        if func["name"] == function_chosen:
            parameters = func.get("parameters", {})
            break
    # Contain the params name extracted from functions_json
    parameters_name = list(parameters.keys())

    # ========================================================================
    #       Implement the constrained decoding for the params section
    # ========================================================================
    for index, name in enumerate(parameters_name):
        target_string = '"' + name + '": '
        # force_string just build the context
        # (input_ids_list and generated_tokens)
        force_string(target_string, model, input_ids_list, generated_tokens)

        type_parameter = parameters[name]["type"]
        if type_parameter == "number" or type_parameter == "integer":
            value = generate_value(
                type_parameter, model,
                input_ids_list, generated_tokens
            )
            extracted_params[name] = value
        elif type_parameter == "string":
            force_string('"', model, input_ids_list, generated_tokens)
            value = generate_value(
                type_parameter, model,
                input_ids_list, generated_tokens
            )
            extracted_params[name] = value
            force_string('"', model, input_ids_list, generated_tokens)

        # Params counter -> if is last close with '}' else ','
        is_last = (index == len(parameters_name) - 1)
        if not is_last:
            force_string(', ', model, input_ids_list, generated_tokens)

    force_string('}', model, input_ids_list, generated_tokens)

    # ========================================================================
    #                           Returning data
    # ========================================================================
    # Clean/unescape any extracted string parameters
    for k, v in list(extracted_params.items()):
        if isinstance(v, str):
            extracted_params[k] = _clean_extracted_string(v)

    return {
        "prompt": user_question,
        "name": function_chosen,
        "parameters": extracted_params
    }


def llm_interaction(
        json_input: list[dict[str, str]],
        functions_json: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """ Format prompts, process queries with the LLM, and collect function
        call results. Formats function signatures by extracting parameter
        names and types from `func.get('parameters', {}).items()` using a
        list comprehension: `[f"{p_name}: {p_info['type']}"
        for p_name, p_info in func.get('parameters', {}).items()]`.
        Constructs a ChatML prompt for each query and runs constrained
        decoding.
    Args:
        json_input (list[dict[str, str]]): List of test objects
        containing natural language prompts.
        functions_json (list[dict]): List of available function definitions
        and metadata.
    Returns:
        list[dict]: List of structured function call result dictionaries.
    """
    model = Small_LLM_Model()
    # print(model._device)

    tools_text = "Available functions:\n"
    for func in functions_json:
        params_desc_list = []
        for p_name, p_type in func.get("parameters", {}).items():
            param = f"{p_name}: {p_type['type']}"
            params_desc_list.append(param)
        params_desc = ", ".join(params_desc_list)
        # === list comprehension method ===
        # params_desc = ", ".join([
        #     f"{p_name}: {p_info['type']}"
        #     for p_name, p_info in func.get("parameters", {}).items()
        # ])
        tools_text += (
            f"- {func['name']}({params_desc}): {func['description']}\n"
        )

    all_results = []
    for quest in json_input:
        # extract user_question:
        # ex: {"prompt": "What is the sum of 2 and 3?"}
        user_question = quest.get("prompt")
        # skip prompts that are None or only whitespace
        if user_question is None or str(user_question).strip() == "":
            continue
        if "\\\\" in user_question:
            try:
                user_question = json.loads(f'"{user_question}"')
            except Exception:
                user_question = user_question.replace("\\\\", "\\")

        # Build the prompt using Qwen's chat format so the model can
        # distinguish system instructions, user input, and assistant
        # output. The opening "{" primes the assistant response to
        # generate the required JSON object.
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant. "
            "Select the correct tool and extract "
            "its parameters to answer the user request.\n"
            f"{tools_text}<|im_end|>\n"
            f"<|im_start|>user\n{user_question}<|im_end|>\n"
            "<|im_start|>assistant\n{"
        )

        # returns the integer ID numbers that represent
        # those fragments in the model's dictionary
        encoded_tensor = model.encode(prompt)
        # input_ids_list for every func_definition contein:
        # === in tools_text ===
        #   f"- {func['name']}        -> Function name
        #   ({params_desc}):          -> params - name: type
        #   {func['description']}\n"  -> Func description
        # === in user_question ===
        #   Current "user_question"
        # Why '.tolist()[0]' ??
        #   The function deliberately takes the flat list of ids
        #   and wraps it in an extra list[[ids...]]
        input_ids_list = encoded_tensor.tolist()[0]

        # Constrained Decoding function:
        result_item = constrained_decoder(
            model,
            input_ids_list,
            functions_json,
            user_question
        )
        # print(f"{result_item}\n")
        all_results.append(result_item)

    return all_results
