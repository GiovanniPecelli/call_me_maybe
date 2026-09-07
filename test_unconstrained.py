from llm_sdk import Small_LLM_Model
from typing import Any
import json
import sys


def unconstrained_decoder(
        model: Small_LLM_Model,
        input_ids_list: list[int],
) -> None:
    generated_tokens = []
    for step in range(30):
        logits = model.get_logits_from_input_ids(input_ids_list)
        best_token_id = max(range(len(logits)), key=logits.__getitem__)
        new_word = model.decode([best_token_id])
        print(new_word, end="", flush=True)
        input_ids_list.append(best_token_id)
        generated_tokens.append(best_token_id)


def free_llm(
        data: list[dict[str, str]],
) -> list[dict[str, Any]]:
    model = Small_LLM_Model()

    for i, quest in enumerate(data, 1):
        user_question = quest["prompt"]
        print(f"Question {i}: {user_question}")
        encoded_tensor = model.encode(user_question)
        input_ids_list = encoded_tensor.tolist()[0]

        unconstrained_decoder(model, input_ids_list)
        print("\n\n")


def get_data(filepath: str):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: Could not find the file at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as json_syntax_error:
        print(f"Error: The file {filepath} contains invalid JSON.")
        print(f"Details: {json_syntax_error}")
        sys.exit(1)


if __name__ == "__main__":
    data = get_data("data/input/test.json")
    # print("Data extracted: ", data)
    free_llm(data)
