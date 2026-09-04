from pydantic import BaseModel, ValidationError
from llm_sdk import Small_LLM_Model
from typing import Dict, Any
import sys


# [misc], ignore a miscellaneous (mean: various) error on this line.
class TypeDefinition(BaseModel):  # type: ignore[misc]
    """Data model representing a parameter or return type definition."""
    type: str


class FunctionDefinition(BaseModel):  # type: ignore[misc]
    """ Data model representing the full specification of an available
        function. BaseModel create a data model called FunctionDefinition
    Pydantic's BaseModel provides functionality for:
        1- Validating input and checking type / field
        2- Converting compatible values and object back to dict/JSON
        3- Creating structured objects or producing validation error
    """
    name: str
    description: str
    parameters: Dict[str, TypeDefinition]
    returns: TypeDefinition


def function_validator(raw_functions_list: Any) -> list[FunctionDefinition]:
    """ Validate raw function definition dictionaries against the
        FunctionDefinition schema.
    Args:
        raw_functions_list (Any): Parsed JSON list of raw function
        definitions.
    Returns:
        list[FunctionDefinition]: List of validated FunctionDefinition
        model instances.
    """
    validated_functions = []

    for raw_func in raw_functions_list:
        try:
            # take the raw_func and try to construct a
            # valid FunctionDefinition object from it.
            # "**" is an unpackable operator
            # it unpacks a dictionary into keyword arguments.
            valid_func = FunctionDefinition(**raw_func)
            validated_functions.append(valid_func)
        except ValidationError as e:
            print(
                f"Error in function:\n{raw_func}\n\n"
                "The functions format is invalid!\n"
                "Must countain 4 keys:\n 1- name\n 2- description\n"
                " 3- parameters\n 4- returns\n\n"
                f"Details: {e}"
            )
            sys.exit(1)
    return validated_functions


def tokens_validator(model: Small_LLM_Model) -> list[int]:
    valid_ids = []

    # Normally Digits and special char are in the first side
    for i in range(10000):
        word = model.decode([i]).strip()
        if word != "" and all(char in "0123456789.-" for char in word):
            valid_ids.append(i)
    valid_ids.extend(model.encode(",").tolist()[0])
    valid_ids.extend(model.encode("}").tolist()[0])

    return valid_ids
