from src.file_handler import json_loader, json_output
from src.validator import function_validator
from src.agent import llm_interaction
from src.parser import get_arguments
import time


def main() -> None:
    """
    Main entry point of the application.
    Parses arguments, loads JSON files, validates the schema,
    interacts with the LLM to extract parameters, and saves the output.
    """
    time_start = time.time()

    # parsing & json syntax check
    args = get_arguments()
    functions_json = json_loader(args.functions_definition)

    # Pydantic is used as a strict validator to catch bad schemas early.
    function_validator(functions_json)

    input_json = json_loader(args.input)

    try:
        data = llm_interaction(input_json, functions_json)
    except ModuleNotFoundError as e:
        print(e)
        return None

    if len(data) == 0:
        print("No questions are providing!")

    json_output(data, args.output)

    time_end = time.time()
    execution_time = time_end - time_start
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    print(f"Execution time: {minutes} minutes and {seconds} seconds")


if __name__ == "__main__":
    main()
