import json
import os
import sys
from typing import Any


def json_output(data: list[dict[str, Any]], output_filepath: str) -> None:
    """ Save structured data to a file in JSON format.
        Ensures parent directories exist before writing and formats the JSON
        with a 2-space indentation. Exits the program on filesystem errors.
    Args:
        data (list[dict]): The list of structured data objects to serialize.
        output_filepath (str): Target path where the JSON file will be
        written.
    """
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as file_output:
            json.dump(data, file_output, indent=2)
    except OSError as e:
        print(f"Error: Could not write output to {output_filepath}")
        print(f"Details: {e}")
        sys.exit(1)


def json_loader(filepath: str) -> Any:
    """ Load and parse JSON content from a file.
    Args:
        filepath (str): Path to the target JSON file.
        encoding='utf-8' convert bytes in characters
    Returns:
        data can be Any: depend on the file json (curr. list[dict])
        Any: Deserialized Python data structure from the JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file_input:
            # json.load try to interpretate the file with json syntax
            data = json.load(file_input)
            return data
    except FileNotFoundError:
        print(f"Error: Could not find the file at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as json_syntax_error:
        print(f"Error: The file {filepath} contains invalid JSON.")
        print(f"Details: {json_syntax_error}")
        sys.exit(1)
