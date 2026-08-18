from llm_sdk import Small_LLM_Model
from file_handler import json_loader


def generate_value(
		type_parameter: str,
		model: Small_LLM_Model,
		input_ids_list: list[int],
		generated_tokens: list[int]
) -> None:
	"""Generate te correct value based on parameter type"""
	for step in range(15):
		logits = model.get_logits_from_input_ids(input_ids_list)
		sorted_logits = sorted(
			range(len(logits)),
			key=lambda i: logits[i],
			reverse=True
		)

		for token_id in sorted_logits:
			word = model.decode([token_id])
			clean_word = word.strip()

			if clean_word == "," or clean_word == "}":
				return
			if type_parameter == "number":
				if (
					all(char in "0123456789.-" for char in clean_word)
					and clean_word != ""
				):
					best_token_id = token_id
					break
			elif type_parameter == "string":
				if '"' in clean_word:
					return
				if clean_word != "":
					best_token_id = token_id
					break
		new_word = model.decode([best_token_id])
		print(new_word, end="", flush=True)
		input_ids_list.append(best_token_id)
		generated_tokens.append(best_token_id)

def force_string(
		target_string: str,
		model: Small_LLM_Model,
		input_ids_list: list[int],
		generated_tokens: list[int]
) -> None:
	"""Push the model to generate a correct string"""
	target_tokens = model.encode(target_string).tolist()[0]

	for forced_token in target_tokens:
		logits = model.get_logits_from_input_ids(input_ids_list)

		for i in range(len(logits)):
			if i != forced_token:
				logits[i] = float('-inf')

		best_token_id = logits.index(max(logits))

		new_word = model.decode([best_token_id])
		print(new_word ,end="", flush=True)

		input_ids_list.append(best_token_id)
		generated_tokens.append(best_token_id)


def nudger(
		generated_tokens: list[int],
		encoded_functions: list[list[int]]
) -> list[int]:
	"""
	Acts as a Finite State Machine (FSM) state manager for Grammar-Guided Generation.
	This function tracks the current state of the generation process and determines 
	the syntactically valid subset of tokens required to maintain compliance with 
	the target JSON schema. It returns the allowed token ID to be used by the 
	Logits Processor for masking invalid probabilities.
	Args:
		step (int): The current generation step index.
		vocab_dict (dict): The tokenizer's vocabulary mapping strings to token IDs.
	Returns:
		int: The allowed token ID for the current state.
	"""
	step = len(generated_tokens)
	allowed_ids = []

	for func_tokens in encoded_functions:
		if func_tokens[:step] == generated_tokens:
			if step < len(func_tokens):
				allowed_ids.append(func_tokens[step])
	return allowed_ids	


def Unconstrained_decoder(
		model: Small_LLM_Model,
		input_ids_list: list[int]
) -> None:
	"""
	Unconstrained_decoder is a function that allows the model to
	generate answers without constraints. This function was created
	solely for testing purpose.
	As demonstrated, small language models are notoriously unreliable
	at generating structured output spontaneously. The solution of
	this problem lies in constrained decoding.
	"""
	print("=== Unconstrained model answer ===")
	for step in range(30):
		logits = model.get_logits_from_input_ids(input_ids_list)
		best_token_id = logits.index(max(logits))
		new_token_id = best_token_id
		new_word = model.decode([new_token_id])
		print(new_word ,end="", flush=True)
		input_ids_list.append(new_token_id)
	print("\n\n")


def constrained_decoder(
		model: Small_LLM_Model,
		input_ids_list: list[int],
		functions_json: list[dict]
) -> None:
	functions_name = [f['name'] for f in functions_json]
	encoded_functions = []
	for name in functions_name:
		full_string = '{"name": "' + name + '", "parameters": {'
		func_tokens = model.encode(full_string).tolist()[0]
		encoded_functions.append(func_tokens)

	generated_tokens = []
	
	for step in range(20):
		logits = model.get_logits_from_input_ids(input_ids_list)
		allowed_ids = nudger(generated_tokens, encoded_functions)

		if not allowed_ids or allowed_ids == [None]:
			break

		for i in range(len(logits)):
			if i not in allowed_ids:
				logits[i] = float('-inf')

		best_token_id = logits.index(max(logits))

		new_word = model.decode([best_token_id])
		print(new_word ,end="", flush=True)

		input_ids_list.append(best_token_id)
		generated_tokens.append(best_token_id)

	final_text = model.decode(generated_tokens)

	# function name chosen
	after_prefix = final_text.split('{"name": "')[1]
	function_chosen = after_prefix.split('"')[0]
	parameters = {}
	for func in functions_json:
		if func["name"] == function_chosen:
			parameters = func.get("parameters", {})
			break
	parameters_name = list(parameters.keys())
	for index, name in enumerate(parameters_name):
		target_string = '"' + name + '": '
		force_string(target_string, model, input_ids_list, generated_tokens)

		type_parameter = parameters[name]["type"]
		if type_parameter == "number":
			generate_value(
				type_parameter, model,
				input_ids_list, generated_tokens
			)
		elif type_parameter == "string":
			force_string('"', model, input_ids_list, generated_tokens)
			generate_value(
				type_parameter, model,
				input_ids_list, generated_tokens
			)
			force_string('"', model, input_ids_list, generated_tokens)

		is_last = (index == len(parameters_name) - 1)
		if not is_last:
			force_string(', ', model, input_ids_list, generated_tokens)

	force_string('}', model, input_ids_list, generated_tokens)
	
			

	print("\n\n")


def llm_interaction(
		test_quest_json: list[dict[str, str]],
		functions_json: list[dict]
) -> None:
	"""
	encoded_tensor is: tensor[[sentence_1][sentence_2]]
		- (enhanced splitter algorithm: BPE - Byte Pair Encoding)
		- give an ID for every word in the sentences
	logit a list[fload] 
		- idx = the word's ID
		- value: best word choice in prob.
	"""
	model = Small_LLM_Model()

	tools_text = "Available functions:\n"
	for func in functions_json:
		tools_text += f"- {func['name']}: {func['description']}\n"

	vocab_path = model.get_path_to_vocab_file()
	print("Vocab path is:", vocab_path)
	for quest in test_quest_json:
		user_question = quest["prompt"]
		prompt = f"{tools_text}\nUser request: {user_question}\n"

		encoded_tensor = model.encode(prompt)
		input_ids_list = encoded_tensor.tolist()[0]

		# Constrained Decoding function:
		constrained_decoder(
			model,
			input_ids_list,
			functions_json
		)

		# Unconstrained Decoding function:
		# Unconstrained_decoder(model, input_ids_list)
		