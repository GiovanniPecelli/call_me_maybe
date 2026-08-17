from src.call_me_maybe.parser import get_arguments
from src.call_me_maybe.file_handler import json_loader
from src.call_me_maybe.validator import function_validator
from llm_sdk import Small_LLM_Model


def main() -> None:
	args = get_arguments()
	validated_json = json_loader(args.function_definition)
	validated_functions = function_validator(validated_json)
	


if __name__ == "__main__":
	main()