import json
import sys
from typing import Any


def json_loader(filepath: str) -> Any:
	try:
		with open(filepath, 'r') as file_input: 
			data = json.load(file_input)
			return data
	except FileNotFoundError:
		print(f"Error: Could not find the file at {filepath}")
		sys.exit(1)
	except json.JSONDecodeError as json_syntax_error:
		print(f"Error: The file {filepath} contains invalid JSON.")
		print(f"Details: {json_syntax_error}")
		sys.exit(1)


if __name__ == "__main__":
	json_loader()