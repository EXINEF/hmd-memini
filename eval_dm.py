import json
import random
import torch
import os
import logging
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict
from collections import Counter

from classes.Config import Config
from classes.Database import Database
from classes.DialogueManager import DialogueManager
from classes.classes import Task
from config.config import TaskIntent, TaskAction
from utils.eval_utils import (
    ADD_TASK_SLOTS,
    COMPLETE_TASK_SLOTS,
    DELETE_TASK_SLOTS,
    RETRIEVE_TASKS_SLOTS,
    UPDATE_TASK_SLOTS,
    analyze_dm_test_data_statistics,
    print_dm_data_statistics,
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

# Database setup
DB_TEST_PATH = "db_test.json"


def setup_test_database():
    """Set up a fresh test database with some sample tasks."""
    if os.path.exists(DB_TEST_PATH):
        os.remove(DB_TEST_PATH)

    db = Database(DB_TEST_PATH)

    # Add some sample tasks
    tasks = [
        Task(task_name="buy groceries", due_date="tomorrow", completed=False),
        Task(task_name="call mom", due_date="this weekend", completed=False),
        Task(task_name="submit report", due_date="Friday", completed=False),
        Task(task_name="buy milk", due_date="today", completed=False),
        Task(task_name="buy eggs", due_date="today", completed=True),
    ]

    for task in tasks:
        db.add_task(task)

    return db


def generate_test_data_for_intent(intent: str, slots_list: list) -> List[Dict]:
    """Generate test data for a specific intent using provided slots list."""
    test_data = []

    for slots in slots_list:
        # Define expected actions based on slots
        expected_actions = determine_expected_actions(intent, slots)

        test_item = {
            "intent": intent,
            "slots": slots,
            "expected_actions": expected_actions,
        }

        test_data.append(test_item)

    return test_data


def determine_expected_actions(intent: str, slots: Dict) -> List[str]:
    """Determine the expected DM actions based on intent and slots."""
    actions = []

    if intent == TaskIntent.ADD_TASK.value:
        if slots.get("task_name") is None:
            actions.append(TaskAction.REQ_TASK_NAME.value)
        elif slots.get("task_name") == "buy groceries":  # Simulate existing task
            actions.append(TaskAction.TASK_ALREADY_EXISTS.value)
        else:
            actions.append(TaskAction.ADD_TASK.value)

    elif intent == TaskIntent.COMPLETE_TASK.value:
        if slots.get("task_name") is None:
            actions.append(TaskAction.REQ_TASK_NAME.value)
        elif slots.get("task_name") == "non-existent task":
            actions.append(TaskAction.TASK_NOT_FOUND.value)
        elif slots.get("task_name") == "buy":
            actions.append(TaskAction.MULTIPLE_TASKS_FOUND.value)
        else:
            actions.append(TaskAction.COMPLETE_TASK.value)

    elif intent == TaskIntent.DELETE_TASK.value:
        if slots.get("task_name") is None:
            actions.append(TaskAction.REQ_TASK_NAME.value)
        elif slots.get("task_name") == "non-existent task":
            actions.append(TaskAction.TASK_NOT_FOUND.value)
        elif slots.get("task_name") == "buy":
            actions.append(TaskAction.MULTIPLE_TASKS_FOUND.value)
        elif slots.get("confirmation") is True:
            actions.append(TaskAction.DELETE_TASK.value)
        else:
            actions.append(TaskAction.CONFIRM_DELETE_TASK.value)

    elif intent == TaskIntent.RETRIEVE_TASKS.value:
        actions.append(TaskAction.RETRIEVE_TASKS.value)

    elif intent == TaskIntent.UPDATE_TASK.value:
        if slots.get("task_name") is None:
            actions.append(TaskAction.REQ_TASK_NAME.value)
        elif slots.get("task_name") == "non-existent task":
            actions.append(TaskAction.TASK_NOT_FOUND.value)
        elif slots.get("task_name") == "buy":
            actions.append(TaskAction.MULTIPLE_TASKS_FOUND.value)
        elif slots.get("new_task_name") is None and slots.get("new_due_date") is None:
            actions.append(TaskAction.REQ_TASK_UPDATE_DETAILS.value)
        else:
            actions.append(TaskAction.UPDATE_TASK.value)

    elif intent == "fallback":
        actions.append("fallback")

    return actions


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 score for DM actions."""

    def calculate_metrics_for_single(tp: int, fp: int, fn: int) -> Dict:
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
    action_counts = Counter()

    for pred in predictions:
        # Action evaluation
        true_actions = set(pred["expected_actions"])
        if "predicted_action" not in pred:
            continue
        pred_actions = {pred["predicted_action"]} if pred["predicted_action"] else set()

        action_counts["tp"] += len(true_actions & pred_actions)  # intersection
        action_counts["fp"] += len(pred_actions - true_actions)  # false predictions
        action_counts["fn"] += len(true_actions - pred_actions)  # missed predictions

    # Calculate metrics
    action_metrics = calculate_metrics_for_single(
        action_counts["tp"], action_counts["fp"], action_counts["fn"]
    )

    return {"action_metrics": {**action_metrics, "counts": dict(action_counts)}}


def evaluate_intent(
    intent: str, test_data: List[Dict], dialogue_manager: DialogueManager
) -> Dict:
    """Evaluate DM performance for a specific intent."""

    for item in tqdm(test_data, desc=f"Processing {intent}"):
        # Call the dialogue manager with the given intent and slots
        dm_output = dialogue_manager.determine_action_with_dm(
            item["intent"], item["slots"]
        )

        # Store the action predicted by the DM
        item["predicted_action"] = dm_output.get("action_required")
        item["dm_output"] = dm_output

    # Calculate metrics
    metrics = calculate_metrics(test_data)
    return metrics


def evaluate_model(model_name, args):
    """Evaluate a specific model configuration on all intents."""
    use_deterministic = True if model_name == "deterministic" else False

    model_id = f"{'deterministic' if use_deterministic else 'model'}"

    logging.info(f"\n\n===== Evaluating DM: {model_id} =====\n")

    # Initialize the model configuration
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    if not use_deterministic:
        config = Config(
            model_name=model_name,
            device=device,
            parallel=False,
            max_new_tokens=args.max_new_tokens,
            prompts_folder=args.prompts_folder,
        )

    # Set up test database
    db = setup_test_database()

    # Initialize dialogue manager
    dialogue_manager = DialogueManager(
        database=db,
        use_model=not use_deterministic,
        config=config if not use_deterministic else None,
    )

    # Create output directory for this model
    output_dir = f"tmp/dm_evaluation_results/{model_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Create test data for all intents
    test_data = {}
    test_data["add_task"] = generate_test_data_for_intent(
        TaskIntent.ADD_TASK.value, ADD_TASK_SLOTS
    )
    test_data["complete_task"] = generate_test_data_for_intent(
        TaskIntent.COMPLETE_TASK.value, COMPLETE_TASK_SLOTS
    )
    test_data["delete_task"] = generate_test_data_for_intent(
        TaskIntent.DELETE_TASK.value, DELETE_TASK_SLOTS
    )
    test_data["retrieve_tasks"] = generate_test_data_for_intent(
        TaskIntent.RETRIEVE_TASKS.value, RETRIEVE_TASKS_SLOTS
    )
    test_data["update_task"] = generate_test_data_for_intent(
        TaskIntent.UPDATE_TASK.value, UPDATE_TASK_SLOTS
    )

    # Track evaluation time
    start_time = time.time()

    # Evaluate each intent
    model_metrics = {}
    for intent in args.intents:
        if intent not in test_data:
            logging.warning(f"No test data for intent {intent}, skipping.")
            continue

        logging.info(f"Evaluating intent: {intent}")
        metrics = evaluate_intent(intent, test_data[intent], dialogue_manager)

        # Save the test data with predictions
        with open(f"{output_dir}/test_data_{intent}.json", "w") as f:
            json.dump(test_data[intent], f, indent=4)

        model_metrics[intent] = metrics

    evaluation_time = time.time() - start_time
    model_metrics["evaluation_time"] = evaluation_time

    # Save model metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=4)

    logging.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")

    # Cleanup test database
    if os.path.exists(DB_TEST_PATH):
        os.remove(DB_TEST_PATH)

    return model_metrics


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DM performance with different configurations"
    )
    parser.add_argument(
        "--intents",
        nargs="+",
        default=[
            "add_task",
            "complete_task",
            "delete_task",
            "retrieve_tasks",
            "update_task",
        ],
        help="Intents to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deterministic"],
        choices=[
            "deterministic",
            "llama3",
            "llama3_8bit",
            "llama3_4bit",
            "llama2",
            "tinyllama",
        ],
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
    """Main function to run DM evaluation across multiple configurations."""
    args = get_args()

    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create root output directory
    root_output_dir = "tmp/dm_evaluation_results"
    os.makedirs(root_output_dir, exist_ok=True)

    logging.info("Analyzing DM test data statistics...")
    dm_data_stats = analyze_dm_test_data_statistics()
    print_dm_data_statistics(dm_data_stats)

    # Save statistics to file
    with open(f"{root_output_dir}/eval_dm_data_statistics.json", "w") as f:
        json.dump(dm_data_stats, f, indent=4, default=str)

    # Evaluate the selected models
    all_model_results = {}
    for model_id in args.models:
        model_metrics = evaluate_model(model_id, args)
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
                comparison_results["results"][intent][model_id] = {
                    "action_precision": all_model_results[model_id][intent][
                        "action_metrics"
                    ]["precision"],
                    "action_recall": all_model_results[model_id][intent][
                        "action_metrics"
                    ]["recall"],
                    "action_f1": all_model_results[model_id][intent]["action_metrics"][
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
    print("\n===== DM Model Comparison Summary =====")
    for model_id in args.models:
        print(f"\n----- Model: {model_id} -----")
        for intent in args.intents:
            print(f"\n  Intent: {intent}")
            print(
                f"  Action Prediction: P={all_model_results[model_id][intent]['action_metrics']['precision']:.3f}, "
                f"R={all_model_results[model_id][intent]['action_metrics']['recall']:.3f}, "
                f"F1={all_model_results[model_id][intent]['action_metrics']['f1']:.3f}"
            )

        print(
            f"\n  Evaluation time: {all_model_results[model_id]['evaluation_time']:.2f} seconds"
        )

    print("\n===== End of Evaluation =====")


if __name__ == "__main__":
    # Remove previous evaluation results
    os.system("rm -rf tmp/dm_evaluation_results")
    main()

# to run the code
# python eval_dm.py --models deterministic llama3 --cuda_device 0
# python eval_dm.py --models deterministic llama3 llama2 tinyllama --cuda_device 0

# To run in background and save output to file
# nohup python eval_dm.py --models deterministic llama3 llama2 tinyllama --cuda_device 0 > eval_dm_output.out 2>&1 &
