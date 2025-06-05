import json
import random
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
import os
import logging
import argparse
import torch
import time
from datetime import datetime

from classes.Config import Config
from config.config import ALL_INTENTS
from utils.eval_utils import analyze_nlu_test_data_statistics, print_nlu_data_statistics
from utils.model_utils import (
    extract_intents_with_nlu,
    extract_slots_with_nlu,
    process_multi_intent_input,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# SET SEEDS
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Templates for generating test data (single intent)
ADD_TASK_TEMPLATES = [
    "Add a task to {task_name} by {due_date}",
    "Create a task {task_name}",
    "I need to {task_name} by {due_date}",
    "Add {task_name} to my list",
    "Set up a reminder to {task_name} for {due_date}",
    "Don't let me forget to {task_name}",
    "Schedule {task_name} for {due_date}",
    "Add to my to-do list: {task_name}",
    "Create a new task: {task_name} due {due_date}",
    "I want to add {task_name} to my tasks",
]

COMPLETE_TASK_TEMPLATES = [
    "Mark {task_name} as complete",
    "Complete the task: {task_name}",
    "I've finished {task_name}",
    "Done with {task_name}",
    "Task {task_name} is completed",
    "I completed {task_name}",
    "Mark {task_name} as done",
    "Finish {task_name}",
    "Check off {task_name}",
    "{task_name} is finished",
]

DELETE_TASK_TEMPLATES = [
    "Delete the task {task_name}",
    "Remove {task_name} from my list",
    "Cancel the task {task_name}",
    "Delete {task_name}",
    "Get rid of {task_name} from my tasks",
    "Remove the reminder for {task_name}",
    "Drop {task_name} from my to-do list",
    "I want to delete {task_name}",
    "Erase task {task_name}",
    "Discard {task_name} from my tasks",
]

RETRIEVE_TASKS_TEMPLATES = [
    "Show me my tasks",
    "What's on my to-do list?",
    "List my {status} tasks",
    "What do I need to do?",
    "Show all my {status} tasks",
    "What tasks do I have?",
    "Display my to-do list",
    "What's on my agenda?",
    "What are my {status} tasks?",
    "Show me my task list",
]

UPDATE_TASK_TEMPLATES = [
    "Change {task_name} to {new_task_name}",
    "Update the due date for {task_name} to {new_due_date}",
    "Reschedule {task_name} for {new_due_date}",
    "Rename {task_name} to {new_task_name}",
    "Move {task_name} to {new_due_date}",
    "Update {task_name} to be due on {new_due_date}",
    "Change the deadline for {task_name} to {new_due_date}",
    "Edit {task_name} and make it {new_task_name}",
    "I want to change {task_name} to {new_task_name}",
    "Push back {task_name} to {new_due_date}",
]

FALLBACK_TEMPLATES = [
    "What's the weather today?",
    "Tell me a joke",
    "What time is it?",
    "Play music",
    "Who is the president?",
    "Search for restaurants nearby",
    "What's the capital of France?",
    "How tall is Mount Everest?",
    "Tell me about quantum physics",
    "What's the meaning of life?",
]

# Templates for generating multi-intent test data
MULTI_INTENT_TEMPLATES = [
    # Add task + Add task
    "Add two tasks: {task_name1} by {due_date1} and {task_name2} by {due_date2}",
    "Create these tasks: {task_name1} for {due_date1}, and {task_name2} for {due_date2}",
    "I need to {task_name1} by {due_date1} and also {task_name2} by {due_date2}",
    "Add {task_name1} due {due_date1} and {task_name2} due {due_date2} to my list",
    "Add multiple tasks: {task_name1} for {due_date1} and {task_name2} for {due_date2}",
    # Complete task + Add task
    "Mark {task_name1} as complete and add a task to {task_name2} by {due_date2}",
    "Complete {task_name1} and create a new task {task_name2} for {due_date2}",
    "I've finished {task_name1} and now I need to add {task_name2} due {due_date2}",
    "Mark {task_name1} as done and don't let me forget to {task_name2} by {due_date2}",
    "Task {task_name1} is done, now add {task_name2} to my list for {due_date2}",
    # Complete task + Complete task
    "Mark both {task_name1} and {task_name2} as complete",
    "I've finished {task_name1} and {task_name2}",
    "Complete these tasks: {task_name1} and {task_name2}",
    "Mark {task_name1} and {task_name2} as done",
    "Tasks {task_name1} and {task_name2} are completed",
]

# Sample data for slot values
TASK_NAMES = [
    "buy groceries",
    "call mom",
    "finish report",
    "schedule dentist appointment",
    "pay bills",
    "send email to boss",
    "clean the house",
    "prepare presentation",
    "fix the car",
    "book flight tickets",
    "renew subscription",
    "attend meeting",
    "pick up kids from school",
    "submit project proposal",
    "review documents",
]

DUE_DATES = [
    "tomorrow",
    "next week",
    "on Friday",
    "by the end of the day",
    "on Monday",
    "this weekend",
    "next month",
    "in two days",
    "by noon",
    "before the meeting",
    "after lunch",
    "by 5 PM",
    "on March 15",
    "next Tuesday",
    "by the end of the week",
]

STATUS_VALUES = ["pending", "completed", "all"]


def fill_template(template, slots):
    """Fill a template with provided slot values."""
    try:
        return template.format(**slots)
    except KeyError as e:
        return f"Missing slot value for {e}"


def generate_test_data_for_intent(template_list, intent, slots_config, num_samples=10):
    """Generate test data for a specific intent using templates."""
    test_data = []

    for _ in range(num_samples):
        # Randomly select a template
        template = random.choice(template_list)

        # Prepare slot values
        slot_values = {}
        for slot in slots_config:
            if slot == "task_name":
                slot_values[slot] = random.choice(TASK_NAMES)
            elif slot == "new_task_name":
                # Make sure new_task_name is different from task_name
                task_name = slot_values.get("task_name")
                remaining_tasks = [t for t in TASK_NAMES if t != task_name]
                slot_values[slot] = (
                    random.choice(remaining_tasks)
                    if remaining_tasks
                    else random.choice(TASK_NAMES)
                )
            elif slot == "due_date" or slot == "new_due_date":
                slot_values[slot] = random.choice(DUE_DATES)
            elif slot == "status":
                slot_values[slot] = random.choice(STATUS_VALUES)

        # Fill the template
        filled_question = fill_template(template, slot_values)

        # Prepare the answer structure
        answer = {"intent": [intent], "slots": slot_values, "question": filled_question}

        test_data.append(answer)

    return test_data


def generate_multi_intent_test_data(num_samples=10):
    """Generate test data for multiple intents in a single input."""
    test_data = []

    # Define intent combinations to test
    intent_combinations = [
        (["add_task", "add_task"], 0, 4),  # Add task + Add task templates (index 0-4)
        (
            ["complete_task", "add_task"],
            5,
            9,
        ),  # Complete task + Add task templates (index 5-9)
        (
            ["complete_task", "complete_task"],
            10,
            14,
        ),  # Complete task + Complete task templates (index 10-14)
    ]

    for _ in range(num_samples):
        # Randomly select an intent combination and template range
        intent_pair, start_idx, end_idx = random.choice(intent_combinations)
        template_idx = random.randint(start_idx, end_idx)
        template = MULTI_INTENT_TEMPLATES[template_idx]

        # Create slot values
        slot_values = {
            "task_name1": random.choice(TASK_NAMES),
            "task_name2": random.choice(TASK_NAMES),
            "due_date1": random.choice(DUE_DATES),
            "due_date2": random.choice(DUE_DATES),
        }

        # Fill the template
        filled_question = fill_template(template, slot_values)

        # Prepare expected slots for each intent
        expected_slots = []
        expected_phrases = []

        if intent_pair[0] == "add_task":
            expected_slots.append(
                {
                    "task_name": slot_values["task_name1"],
                    "due_date": slot_values["due_date1"],
                }
            )
            expected_phrases.append(
                f"Add a task {slot_values['task_name1']} by {slot_values['due_date1']}"
            )
        elif intent_pair[0] == "complete_task":
            expected_slots.append({"task_name": slot_values["task_name1"]})
            expected_phrases.append(f"Complete the task {slot_values['task_name1']}")

        if intent_pair[1] == "add_task":
            expected_slots.append(
                {
                    "task_name": slot_values["task_name2"],
                    "due_date": slot_values["due_date2"],
                }
            )
            expected_phrases.append(
                f"Add a task {slot_values['task_name2']} by {slot_values['due_date2']}"
            )
        elif intent_pair[1] == "complete_task":
            expected_slots.append({"task_name": slot_values["task_name2"]})
            expected_phrases.append(f"Complete the task {slot_values['task_name2']}")

        # Create test data structure
        answer = {
            "intent": intent_pair,
            "slots": expected_slots,
            "expected_phrases": expected_phrases,
            "question": filled_question,
        }

        test_data.append(answer)

    return test_data


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 score for intents and slots."""

    def calculate_single_metrics(tp: int, fp: int, fn: int) -> Dict:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    # Initialize counters
    intent_counts = Counter()
    slot_counts = Counter()

    for pred in predictions:
        # Intent evaluation
        true_intents = set(pred["intent"])
        if "detected_intent" not in pred:
            continue
        pred_intents = set(pred["detected_intent"])

        intent_counts["tp"] += len(true_intents & pred_intents)  # intersection
        intent_counts["fp"] += len(pred_intents - true_intents)  # false predictions
        intent_counts["fn"] += len(true_intents - pred_intents)  # missed predictions

        # For single intent cases
        if "slots" in pred and isinstance(pred["slots"], dict):
            true_slots = {
                slot: {str(value).lower().replace(" ", "") if value is not None else ""}
                for slot, value in pred["slots"].items()
            }

            pred_slots = {
                slot: {str(value).lower().replace(" ", "") if value is not None else ""}
                for slot, value in pred.get("detected_slots", {}).items()
            }

            # Count slot matches
            for slot in set(true_slots.keys()) | set(pred_slots.keys()):
                true_values = true_slots.get(slot, set())
                pred_values = pred_slots.get(slot, set())

                slot_counts["tp"] += len(true_values & pred_values)  # intersection
                slot_counts["fp"] += len(pred_values - true_values)  # false predictions
                slot_counts["fn"] += len(
                    true_values - pred_values
                )  # missed predictions

    # Calculate metrics
    intent_metrics = calculate_single_metrics(
        intent_counts["tp"], intent_counts["fp"], intent_counts["fn"]
    )

    slot_metrics = calculate_single_metrics(
        slot_counts["tp"], slot_counts["fp"], slot_counts["fn"]
    )

    return {
        "intent_metrics": {**intent_metrics, "counts": dict(intent_counts)},
        "slot_metrics": {**slot_metrics, "counts": dict(slot_counts)},
    }


def evaluate_multi_intent(test_data, config, model_name):
    """Evaluate NLU performance for multi-intent inputs."""
    for item in tqdm(
        test_data, desc=f"Processing multi-intent inputs with {model_name}"
    ):
        user_input = item["question"]
        nlu_input = {"user_input": user_input, "historical_context": []}

        # Detect intents
        detected_intents = extract_intents_with_nlu(nlu_input, config)
        item["detected_intent"] = detected_intents

        # Process multi-intent input
        phrases_result, processed_intents = process_multi_intent_input(
            detected_intents, user_input, config
        )

        if isinstance(phrases_result, tuple) and phrases_result[0] == "fallback":
            item["split_result"] = "fallback"
            item["error_message"] = phrases_result[1]
        else:
            # If phrases_result is not a dictionary, it's an unexpected type
            # This means there was an error in the intent division
            if not isinstance(phrases_result, dict):
                item["split_result"] = "fallback"
                item["error_message"] = (
                    f"Unexpected result type: {type(phrases_result).__name__}"
                )
                # Log the unexpected result for debugging
                logging.warning(
                    f"Unexpected phrases_result type: {type(phrases_result).__name__}, value: {phrases_result}"
                )
                continue

            item["split_result"] = phrases_result
            item["processed_intents"] = processed_intents

            # Extract slots for each phrase
            detected_slots = []
            for i, intent in enumerate(processed_intents[: len(phrases_result)]):
                phrase = phrases_result.get(f"phrase{i+1}")
                if phrase:
                    phrase_nlu_input = {
                        "user_input": phrase,
                        "historical_context": [],
                    }
                    slots = extract_slots_with_nlu(intent, phrase_nlu_input, config)
                    detected_slots.append(slots)

            item["detected_slots"] = detected_slots

    # Calculate intent detection accuracy
    total = len(test_data)
    correct_intent_count = 0
    correct_intent_order_count = 0
    correct_phrase_split_count = 0
    correct_slot_extraction_count = 0

    for item in test_data:
        # Check if all intents were detected (regardless of order)
        true_intents = set(item["intent"])
        detected_intents = set(item.get("detected_intent", []))
        if true_intents == detected_intents:
            correct_intent_count += 1

        # Check if intents were detected in the correct order
        if item.get("intent") == item.get("detected_intent"):
            correct_intent_order_count += 1

        # Check if phrases were split correctly
        split_result = item.get("split_result")
        if split_result and split_result != "fallback":
            phrases_count = len(item["intent"])
            # Only count as correct if split_result is a dictionary with the expected number of phrases
            if isinstance(split_result, dict) and len(split_result) == phrases_count:
                correct_phrase_split_count += 1

        # Check if slots were extracted correctly
        if "detected_slots" in item and isinstance(item["detected_slots"], list):
            expected_slots = item.get("slots", [])
            detected_slots = item.get("detected_slots", [])

            if len(expected_slots) == len(detected_slots):
                all_slots_correct = True
                for i in range(len(expected_slots)):
                    expected = expected_slots[i]
                    detected = detected_slots[i] if i < len(detected_slots) else {}

                    # Check if all expected slots are detected correctly
                    for slot, value in expected.items():
                        if slot not in detected or detected[slot] != value:
                            all_slots_correct = False
                            break

                if all_slots_correct:
                    correct_slot_extraction_count += 1

    metrics = {
        "total_samples": total,
        "intent_detection": {
            "accuracy": round(correct_intent_count / total, 3) if total > 0 else 0,
            "correct": correct_intent_count,
            "total": total,
        },
        "intent_order": {
            "accuracy": (
                round(correct_intent_order_count / total, 3) if total > 0 else 0
            ),
            "correct": correct_intent_order_count,
            "total": total,
        },
        "phrase_splitting": {
            "accuracy": (
                round(correct_phrase_split_count / total, 3) if total > 0 else 0
            ),
            "correct": correct_phrase_split_count,
            "total": total,
        },
        "slot_extraction": {
            "accuracy": (
                round(correct_slot_extraction_count / total, 3) if total > 0 else 0
            ),
            "correct": correct_slot_extraction_count,
            "total": total,
        },
    }

    return metrics


def evaluate_intent(intent, test_data, config, model_name):
    """Evaluate NLU performance for a specific intent."""
    for item in tqdm(test_data, desc=f"Processing {intent} with {model_name}"):
        user_input = item["question"]
        nlu_input = {"user_input": user_input, "historical_context": []}

        detected_intents = extract_intents_with_nlu(nlu_input, config)
        item["detected_intent"] = detected_intents

        # Only extract slots if the intent was correctly detected
        if intent in detected_intents:
            slots = extract_slots_with_nlu(intent, nlu_input, config)
            item["detected_slots"] = slots
        else:
            item["detected_slots"] = {}

    # Calculate metrics
    metrics = calculate_metrics(test_data)
    return metrics


def evaluate_model(model_name, test_data, args):
    """Evaluate a specific model configuration on all intents."""

    logging.info(f"\n\n===== Evaluating model: {model_name} =====\n")

    # Initialize the model configuration
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    config = Config(
        model_name=model_name,
        device=device,
        parallel=False,
        max_new_tokens=args.max_new_tokens,
        prompts_folder=args.prompts_folder,
    )

    # Create a deep copy of test data for this model
    model_test_data = {}
    for intent_name, intent_data in test_data.items():
        model_test_data[intent_name] = []
        for item in intent_data:
            model_test_data[intent_name].append(item.copy())

    # Track evaluation time
    start_time = time.time()

    # Create output directory for this model
    output_dir = f"tmp/nlu_evaluation_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each intent
    model_metrics = {}
    for intent in args.intents:
        if intent not in model_test_data:
            logging.warning(f"No test data for intent {intent}, skipping.")
            continue

        logging.info(f"Evaluating intent: {intent}")
        if intent == "multi_intent":
            metrics = evaluate_multi_intent(model_test_data[intent], config, model_name)
        else:
            metrics = evaluate_intent(
                intent, model_test_data[intent], config, model_name
            )

        with open(f"{output_dir}/test_data_{intent}.json", "w") as f:
            json.dump(model_test_data[intent], f, indent=4)

        model_metrics[intent] = metrics

    evaluation_time = time.time() - start_time
    model_metrics["evaluation_time"] = evaluation_time

    # Save model metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=4)

    logging.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")
    return model_metrics


def create_test_data():
    """Create test data for all intents."""
    test_data = {}

    # Add task (task_name, due_date)
    test_data["add_task"] = generate_test_data_for_intent(
        ADD_TASK_TEMPLATES, "add_task", ["task_name", "due_date"]
    )

    # Complete task (task_name)
    test_data["complete_task"] = generate_test_data_for_intent(
        COMPLETE_TASK_TEMPLATES, "complete_task", ["task_name"]
    )

    # Delete task (task_name)
    test_data["delete_task"] = generate_test_data_for_intent(
        DELETE_TASK_TEMPLATES, "delete_task", ["task_name"]
    )

    # Retrieve tasks (status)
    test_data["retrieve_tasks"] = generate_test_data_for_intent(
        RETRIEVE_TASKS_TEMPLATES, "retrieve_tasks", ["status"]
    )

    # Update task (task_name, new_task_name, new_due_date)
    test_data["update_task"] = generate_test_data_for_intent(
        UPDATE_TASK_TEMPLATES,
        "update_task",
        ["task_name", "new_task_name", "new_due_date"],
    )

    # Fallback (no slots)
    test_data["fallback"] = generate_test_data_for_intent(
        FALLBACK_TEMPLATES, "fallback", []
    )

    # Multi-intent
    test_data["multi_intent"] = generate_multi_intent_test_data()

    return test_data


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NLU performance across multiple models"
    )
    parser.add_argument(
        "--intents",
        nargs="+",
        default=list(ALL_INTENTS.keys()) + ["multi_intent"],
        help="Intents to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3"],
        choices=["llama3", "llama2", "tinyllama", "llama3_8bit", "llama3_4bit"],
        help="Models to evaluate",
    )
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device to use")
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens"
    )
    parser.add_argument(
        "--prompts_folder",
        type=str,
        default="prompts",
        help="Folder containing prompts",
    )
    return parser.parse_args()


def main():
    """Main function to run NLU evaluation across multiple models."""
    args = get_args()

    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create root output directory
    root_output_dir = "tmp/nlu_evaluation_results"
    os.makedirs(root_output_dir, exist_ok=True)

    # Generate or load test data
    test_data_path = f"{root_output_dir}/test_data.json"
    if not os.path.exists(test_data_path):
        logging.info("Generating test data...")
        test_data = create_test_data()
        with open(test_data_path, "w") as f:
            json.dump(test_data, f, indent=4)
        logging.info("Test data generated and saved.")
    else:
        logging.info("Loading existing test data...")
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
        logging.info("Test data loaded.")

    logging.info("Analyzing NLU test data statistics...")
    data_stats = analyze_nlu_test_data_statistics(test_data)
    print_nlu_data_statistics(data_stats)

    # Save statistics to file
    with open(f"{root_output_dir}/eval_nlu_data_statistics.json", "w") as f:
        json.dump(data_stats, f, indent=4, default=str)

    # Evaluate the selected models
    all_model_results = {}
    for model_id in args.models:
        model_metrics = evaluate_model(model_id, test_data, args)
        all_model_results[model_id] = model_metrics

    # Save comparison results
    comparison_results = {
        "timestamp": timestamp,
        "models_compared": args.models,
        "intents_evaluated": args.intents,
        "results": {},
    }

    # Restructure results for easier comparison
    for intent in args.intents:
        comparison_results["results"][intent] = {}
        for model_id in args.models:
            if intent in all_model_results[model_id]:
                if intent == "multi_intent":
                    comparison_results["results"][intent][model_id] = {
                        "intent_detection": all_model_results[model_id][intent][
                            "intent_detection"
                        ]["accuracy"],
                        "intent_order": all_model_results[model_id][intent][
                            "intent_order"
                        ]["accuracy"],
                        "phrase_splitting": all_model_results[model_id][intent][
                            "phrase_splitting"
                        ]["accuracy"],
                        "slot_extraction": all_model_results[model_id][intent][
                            "slot_extraction"
                        ]["accuracy"],
                    }
                else:
                    comparison_results["results"][intent][model_id] = {
                        "intent_precision": all_model_results[model_id][intent][
                            "intent_metrics"
                        ]["precision"],
                        "intent_recall": all_model_results[model_id][intent][
                            "intent_metrics"
                        ]["recall"],
                        "intent_f1": all_model_results[model_id][intent][
                            "intent_metrics"
                        ]["f1"],
                        "slot_precision": all_model_results[model_id][intent][
                            "slot_metrics"
                        ]["precision"],
                        "slot_recall": all_model_results[model_id][intent][
                            "slot_metrics"
                        ]["recall"],
                        "slot_f1": all_model_results[model_id][intent]["slot_metrics"][
                            "f1"
                        ],
                    }

    # Add performance metrics
    comparison_results["performance"] = {}
    for model_id in args.models:
        comparison_results["performance"][model_id] = {
            "evaluation_time": all_model_results[model_id]["evaluation_time"]
        }

    # Save comparison
    with open(f"{root_output_dir}/model_comparison_{timestamp}.json", "w") as f:
        json.dump(comparison_results, f, indent=4)

    # Print summary
    print("\n===== NLU Model Comparison Summary =====")
    for model_id in args.models:
        print(f"\n----- Model: {model_id} -----")
        for intent in args.intents:
            if intent == "multi_intent":
                print(f"\n  Multi-intent Evaluation:")
                print(
                    f"  Intent Detection Accuracy: {all_model_results[model_id][intent]['intent_detection']['accuracy']}"
                )
                print(
                    f"  Intent Order Accuracy: {all_model_results[model_id][intent]['intent_order']['accuracy']}"
                )
                print(
                    f"  Phrase Splitting Accuracy: {all_model_results[model_id][intent]['phrase_splitting']['accuracy']}"
                )
                print(
                    f"  Slot Extraction Accuracy: {all_model_results[model_id][intent]['slot_extraction']['accuracy']}"
                )
            else:
                print(f"\n  Intent: {intent}")
                print(
                    f"  Intent Detection: P={all_model_results[model_id][intent]['intent_metrics']['precision']:.3f}, R={all_model_results[model_id][intent]['intent_metrics']['recall']:.3f}, F1={all_model_results[model_id][intent]['intent_metrics']['f1']:.3f}"
                )
                print(
                    f"  Slot Extraction: P={all_model_results[model_id][intent]['slot_metrics']['precision']:.3f}, R={all_model_results[model_id][intent]['slot_metrics']['recall']:.3f}, F1={all_model_results[model_id][intent]['slot_metrics']['f1']:.3f}"
                )

        print(
            f"\n  Evaluation time: {all_model_results[model_id]['evaluation_time']:.2f} seconds"
        )

    print("\n===== End of Evaluation =====")


if __name__ == "__main__":
    # remove the folder with all the files "nlu_evaluation_results" and run the code
    os.system("rm -rf tmp/nlu_evaluation_results")
    main()

# to run the code
# python eval_nlu.py --models llama3 --cuda_device 0
# python eval_nlu.py --models llama3 llama3_8bit llama3_4bit llama2 tinyllama --cuda_device 0

# to run the code in the background and save the output to a file
# nohup python eval_nlu.py --models llama3 llama3_8bit llama3_4bit llama2 tinyllama --cuda_device 0 > eval_nlu.out 2>&1 &
