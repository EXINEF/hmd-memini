import logging
from typing import Dict, Any


def select_nlg_prompt(action: str, prompts: Dict[str, str]) -> str:
    prompt_key = f"NLG_{action}"

    if prompt_key not in prompts:
        logging.critical(
            f"Prompt {prompt_key} not found, falling back to generic prompt"
        )
        prompt_key = "NLG_fallback"

    logging.info(f"Selected NLG prompt: {prompt_key}")
    return prompt_key


def preprocess_nlg_input(
    intent: str, slots: Dict[str, Any], dm_output: Dict[str, Any], database
) -> Dict[str, Any]:
    nlg_input = {
        "intent": intent,
        "slots": slots,
        "dm_output": dm_output,
    }

    # Add mixed-initiative suggestions if available
    if "mixed_initiative" in dm_output and dm_output["mixed_initiative"]:
        nlg_input["mixed_initiative"] = dm_output["mixed_initiative"]

    if dm_output.get("action_required") == "complete_task":
        task = dm_output.get("additional_info", {}).get("task", {})
        message = dm_output.get("message", "")

        nlg_input.update(
            {
                "intent": "complete_task",
                "task_name": task.get("task_name", slots.get("task_name", "")),
                "due_date": task.get("due_date"),
                "message": message,
                "task": task,
                "tasks": database.get_tasks(),
            }
        )

    if dm_output.get("action_required") == "task_not_found":
        nlg_input.update(
            {
                "tasks": database.get_tasks(),
            }
        )
    else:
        nlg_input.update(
            {
                "tasks": dm_output.get("tasks", []),
            }
        )

    if "additional_info" in dm_output and dm_output["additional_info"]:
        for key, value in dm_output["additional_info"].items():
            nlg_input[key] = value

    if "message" in dm_output:
        nlg_input["message"] = dm_output["message"]

    if "original_intent" in dm_output:
        nlg_input["original_intent"] = dm_output["original_intent"]

    return nlg_input
