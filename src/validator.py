from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import sys


class TypeDefinition(BaseModel):
	"""Data model representing a parameter or return type definition."""
	type: str


class FunctionDefinition(BaseModel):
	"""Data model representing the full specification of an available function."""
	name: str
	description: str
	parameters: Dict[str, TypeDefinition]
	returns: TypeDefinition


def function_validator(raw_functions_list: Any) -> list[FunctionDefinition]:
	"""Validate raw function definition dictionaries against the FunctionDefinition schema.

	Args:
		raw_functions_list (Any): Parsed JSON list of raw function definitions.

	Returns:
		list[FunctionDefinition]: List of validated FunctionDefinition model instances.
	"""
	validated_functions = []

	for raw_func in raw_functions_list:
		try:
			valid_func = FunctionDefinition(**raw_func)
			validated_functions.append(valid_func)
		except ValidationError as e:
			print(f"Error: The JSON structure is invalid!")
			print(f"Details: {e}")
			sys.exit(1)
	return validated_functions