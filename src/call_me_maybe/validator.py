from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import sys


class TypeDefinition(BaseModel):
	type: str


class FunctionDefinition(BaseModel):
	name: str
	description: str
	parameters: Dict[str, TypeDefinition]
	returns: TypeDefinition


def function_validator(raw_functions_list: Any):
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