from src.parser import get_arguments
from src.file_handler import json_loader, json_output
from src.agent import llm_interaction


def main() -> None:
    args = get_arguments()
    functions_json = json_loader(args.functions_definition)

    functions_name = []
    for func in functions_json:
        name = func["name"]
        functions_name.append(name)

    input_json = json_loader(args.input)
    data = llm_interaction(input_json, functions_json)
    json_output(data, args.output)


if __name__ == "__main__":
    main()
