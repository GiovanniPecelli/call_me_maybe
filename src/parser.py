import argparse


# ArgumentParser return a Namespace variable that holds
# arguments as attributes - accesseble using "dot notation".
# it's basically a wrapper around a standard Python dict (__dict__).
def get_arguments() -> argparse.Namespace:
	"""Builds the argument parser and returns the parsed arguments."""
	parser = argparse.ArgumentParser(
		description='Call Me Maybe - Function Calling in LLMs'
	)

	parser.add_argument(
		"--functions_definition", type=str,
		default="data/input/functions_definition.json",
		help="Path to the JSON file containing the function definitions."
	)

	parser.add_argument(
		"--input", type=str,
		default="data/input/function_calling_tests.json",
		help="Path to the JSON file containing the natural language prompts."
	)

	parser.add_argument(
		"--output", type=str,
		default="data/output/function_calls.json",
		help="Path to the output JSON file where results will be written."
	)

	return parser.parse_args()


if __name__ == "__main__":
	args = get_arguments()
	print("Ready to read form:", args.input)
	print("Ready to write form:", args.output)