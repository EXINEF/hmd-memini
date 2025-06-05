import json
import torch
import os
import logging

from classes.Config import Config
from classes.Database import Database
from classes.DialogueManager import DialogueManager
from config.config import MAX_HISTORICAL_CONTEXT_LENGTH, MAX_NEW_TOKENS


from utils.nlg_utils import (
    preprocess_nlg_input,
    select_nlg_prompt,
)
from utils.model_utils import (
    extract_intents_with_nlu,
    extract_slots_with_nlu,
    generate_nlg_output,
    combine_multiple_nlg_responses_if_more_than_one,
    process_multi_intent_input,
)

from utils.utils import (
    get_welcome_message,
    remove_newlines_from_text,
    setup_logger,
    set_seed,
    parse_args,
)

# Clear the terminal screen
os.system("clear")

os.makedirs("tmp", exist_ok=True)

PROMPTS_FOLDER = "prompts"
DATABASE_PATH = "tmp/database.json"
COMPLETE_TRANSCRIPT_PATH = "tmp/complete_transcript.txt"
SEED = 42


def main():
    args = parse_args()
    logging_level = getattr(logging, args.logging_level.upper())

    set_seed(SEED)
    setup_logger(logging_level)

    historical_context = []
    complete_transcript = []

    database = Database(DATABASE_PATH)
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    config = Config(
        model_name=args.model_name,
        device=device,
        parallel=False,
        max_new_tokens=MAX_NEW_TOKENS,
        prompts_folder=PROMPTS_FOLDER,
    )
    dialogue_manager = DialogueManager(database, not args.deterministic_dm, config)

    print(get_welcome_message())

    while True:
        user_input = input("USER: ")

        nlu_input = {"user_input": user_input, "historical_context": historical_context}

        # Step 1: Extract intents with NLU
        intents = extract_intents_with_nlu(nlu_input, config)

        phrases_result, intents = process_multi_intent_input(
            intents, user_input, config
        )

        # reduce the size of intents of the same of phrases_result
        intents = intents[: len(phrases_result)]

        # Handle unsupported multi-intent scenario
        if isinstance(phrases_result, tuple) and phrases_result[0] == "fallback":
            intent, error_message = phrases_result
            nlg_responses = [error_message]
        else:
            nlg_responses = []
            for i, intent in enumerate(intents):
                if intent == "fallback":
                    phrase = phrases_result
                else:
                    phrase = phrases_result[f"phrase{i+1}"]
                logging.info(f"\nProcessing intent: {intent} with phrase: {phrase}\n")

                # Step 2: Extract slots for each phrase-intent pair
                phrase_nlu_input = {
                    "user_input": phrase,
                    "historical_context": historical_context,
                }
                slots = extract_slots_with_nlu(intent, phrase_nlu_input, config)
                logging.info(f"\nExtracted slots: {slots}\n")

                # Step 3: Process in Dialogue Manager
                dm_output = dialogue_manager.determine_action_with_dm(intent, slots)
                logging.info(
                    f"\nDM OUTPUT(Required action): {dm_output['action_required']}\n"
                )

                # Step 4: Generate NLG response based on DM output
                nlg_input = preprocess_nlg_input(intent, slots, dm_output, database)
                logging.info(
                    f"\nPreprocessed NLG input:\n{json.dumps(nlg_input, indent=2)}\n"
                )

                prompt_key = select_nlg_prompt(
                    dm_output["action_required"], config.prompts
                )

                nlg_output = generate_nlg_output(
                    nlg_input,
                    config.prompts.get(prompt_key),
                    config,
                )

                nlg_responses.append(nlg_output)

        # Step 5: Combine NLG responses if there are multiple
        final_response = combine_multiple_nlg_responses_if_more_than_one(
            nlg_responses, config
        )

        clean_response = remove_newlines_from_text(final_response)

        print(f"BOT:\n{clean_response}\n")

        log_user_bot_interaction(
            historical_context, complete_transcript, user_input, clean_response
        )

        if len(historical_context) > MAX_HISTORICAL_CONTEXT_LENGTH:
            historical_context = historical_context[-MAX_HISTORICAL_CONTEXT_LENGTH:]

        database.remove_all_completed_tasks()


def log_user_bot_interaction(
    historical_context, complete_transcript, user_input, clean_response
):
    historical_context.append(f"USER: {user_input}\nBOT: {clean_response}")
    complete_transcript.extend([("USER", user_input), ("BOT", clean_response)])
    with open(COMPLETE_TRANSCRIPT_PATH, "w") as f:
        for speaker, text in complete_transcript:
            f.write(f"{speaker}: {text}\n\n")


if __name__ == "__main__":
    main()
