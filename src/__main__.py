import time
from src.parser import get_arguments
from src.file_handler import json_loader, json_output
from src.agent import llm_interaction


def main() -> None:
    time_start = time.time()

    # parsing & json syntax check
    args = get_arguments()
    functions_json = json_loader(args.functions_definition)

    functions_name = []
    for func in functions_json:
        name = func["name"]
        functions_name.append(name)

    # default: functions_calling_test.json
    input_json = json_loader(args.input)

    data = llm_interaction(input_json, functions_json)

    json_output(data, args.output)

    time_end = time.time()
    execution_time = time_end - time_start
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    print(f"Execution time: {minutes} minutes and {seconds} seconds")


if __name__ == "__main__":
    main()
