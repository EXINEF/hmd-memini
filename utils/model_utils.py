import json
import logging
from classes.Config import Config, generate
from config.config import ALLOWED_MULTIPLE_INTENTS, TASK_INTENTS
from utils.json_utils import extract_json_from_text
import re


def extract_intents_with_nlu(
    nlu_input: dict,
    config: Config,
):
    nlu_output = format_tokenize_and_generate_response(
        config.prompts["NLU_intent_detection"], json.dumps(nlu_input), config
    )

    # Extract JSON
    nlu_output_json = extract_json_from_text(nlu_output)
    logging.info(f"\nNLU INTENT OUTPUT:\n{nlu_output_json}\n")

    # Return intents or fallback
    if "intents" not in list(nlu_output_json.keys()):
        return ["fallback"]
    nlu_output_json["intents"]
    if not nlu_output_json["intents"]:
        return ["fallback"]
    return nlu_output_json["intents"]


def format_tokenize_and_generate_response(prompt: str, input: str, config: Config):
    input_text = config.chat_template.format(prompt, input)
    tokenized_input_text = config.tokenizer(input_text, return_tensors="pt").to(
        config.model.device
    )
    output = generate(config.model, tokenized_input_text, config.tokenizer, config)
    return output


def extract_slots_with_nlu(
    intent: str,
    nlu_input: dict,
    config: Config,
):
    # Get the appropriate prompt for this intent
    prompt_key = f"NLU_intents_{intent}"
    if prompt_key not in config.prompts:
        logging.critical(
            f"No specific slot extraction prompt found for intent: {intent}"
        )
        return {}

    nlu_output = format_tokenize_and_generate_response(
        config.prompts[prompt_key], json.dumps(nlu_input), config
    )
    nlu_output_json = extract_json_from_text(nlu_output)
    logging.info(f"\nNLU SLOTS OUTPUT:\n{nlu_output_json}\n")

    if "slots" in nlu_output_json:
        return nlu_output_json["slots"]
    else:
        return {}


def generate_nlg_output(nlg_input: dict, prompt: str, config: Config):
    nlg_output = format_tokenize_and_generate_response(
        prompt, json.dumps(nlg_input, indent=2), config
    )

    if nlg_output.strip().startswith("{") and nlg_output.strip().endswith("}"):
        match = re.search(r'"([^"]+)"', nlg_output)
        if match:
            nlg_output = match.group(1)

    return nlg_output


def combine_multiple_nlg_responses_if_more_than_one(
    nlg_responses: list, config: Config
):
    if len(nlg_responses) > 1:
        nlg_input = {"responses": nlg_responses}
        prompt = config.prompts.get("NLG_merge")
        final_response = generate_nlg_output(nlg_input, prompt, config)
    else:
        final_response = nlg_responses[0]
    return final_response


def process_multi_intent_input(intents, user_input, config: Config):
    intents = [intent for intent in intents if intent not in ["fallback"]]

    if not intents:
        return "fallback: I couldn't understand your request.", ["fallback"]

    if len(intents) == 1:
        return {"phrase1": user_input}, intents

    if all(intent in ALLOWED_MULTIPLE_INTENTS for intent in intents):
        # Multi-intent splitting allowed
        prompt_input = {"intents": intents, "user_input": user_input}
        output = format_tokenize_and_generate_response(
            config.prompts["NLU_split_complex_input"],
            json.dumps(prompt_input, indent=2),
            config,
        )

        try:
            phrases = extract_json_from_text(output)

            if not phrases or not any(
                key.startswith("phrase") for key in phrases.keys()
            ):
                logging.error(f"Invalid phrases output: {phrases}")
                return "fallback", ["I couldn't properly divide your request."]

            return phrases, intents

        except Exception as e:
            logging.error(f"Error parsing split phrases: {e}")
            return (
                "fallback: I had trouble understanding your multiple actions.",
                intents,
            )

    for intent in intents:
        if intent in TASK_INTENTS:
            return {"phrase1": user_input}, [intent]

    return "fallback: I couldn't understand your request.", ["fallback"]
