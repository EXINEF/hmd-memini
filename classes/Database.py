import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from classes.classes import Task
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any

from utils.utils import get_date_range


class Database:
    def __init__(self, filename: str) -> None:
        self.filename: str = filename
        self.data: Dict[str, List[Dict[str, Any]]] = {"tasks": []}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.data = json.load(f)

    def save(self) -> None:
        with open(self.filename, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_task(self, task: Task) -> None:
        self.data["tasks"].append(asdict(task))
        self.save()

    def get_tasks(self) -> List[Dict[str, Any]]:
        return self.data["tasks"]

    def retrieve_tasks_by_name(
        self, task_name: str
    ) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        if not task_name or not self.data["tasks"]:
            return None

        query_lower = task_name.lower()
        tasks = self.data["tasks"]

        if not tasks:
            return None

        # Check for exact matches first
        exact_matches = [
            {k: v for k, v in task.items() if k != "user"}
            for task in tasks
            if task["task_name"].lower() == query_lower
        ]
        if exact_matches:
            return exact_matches[0] if len(exact_matches) == 1 else exact_matches

        # Check for substring matches next
        substring_matches = [
            {k: v for k, v in task.items() if k != "user"}
            for task in tasks
            if query_lower in task["task_name"].lower()
        ]
        if substring_matches:
            return (
                substring_matches[0]
                if len(substring_matches) == 1
                else substring_matches
            )

        # If no exact or substring matches, use fuzzy matching
        best_match = None
        best_score = 0
        min_threshold = 0.4  # Minimum similarity threshold (0-1)

        for task in tasks:
            # Calculate similarity ratio between query and task name
            score = SequenceMatcher(
                None, query_lower, task["task_name"].lower()
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = task

        # Only return if the best match exceeds our threshold
        if best_score >= min_threshold and best_match:
            return {k: v for k, v in best_match.items() if k != "user"}

        return None

    def get_task_by_exact_name(self, task_name: str) -> Optional[Dict[str, Any]]:
        if not task_name or not self.data["tasks"]:
            return None

        for task in self.data["tasks"]:
            if task["task_name"].lower() == task_name.lower():
                return {k: v for k, v in task.items()}

        return None

    def remove_all_completed_tasks(self) -> None:
        self.data["tasks"] = [
            task for task in self.data["tasks"] if not task["completed"]
        ]
        self.save()

    def remove_task_by_exact_name(self, task_name: str) -> None:
        if not task_name or not self.data["tasks"]:
            return

        for task in self.data["tasks"]:
            if task["task_name"].lower() == task_name.lower():
                self.data["tasks"].remove(task)
                self.save()
                return

    def update_task_status(self, task_name, completed=True):
        for task in self.data["tasks"]:
            if task["task_name"].lower() == task_name.lower():
                task["completed"] = completed
                return True
        return False

    def get_tasks_for_period(self, period_type: str) -> List[Dict[str, Any]]:
        logging.info(f"Getting tasks for period: {period_type}")
        tasks = self.get_tasks()

        # If no period filter, return all tasks
        if not period_type:
            return tasks

        # Get date range for the period
        start_date, end_date = get_date_range(period_type)

        logging.info(f"Date range for {period_type}: {start_date} to {end_date}")

        if not start_date or not end_date:
            logging.warning(f"Invalid period type: {period_type}, returning all tasks")
            return tasks

        # Filter tasks by date range
        filtered_tasks = []
        for task in tasks:
            normalized_date = task.get("normalized_due_date")
            if not normalized_date:
                continue

            try:
                # Check if task date is within range
                if start_date <= normalized_date <= end_date:
                    filtered_tasks.append(task)
                    logging.info(
                        f"Task '{task.get('task_name')}' with date {normalized_date} is within range"
                    )
                else:
                    logging.info(
                        f"Task '{task.get('task_name')}' with date {normalized_date} is outside range"
                    )
            except (ValueError, TypeError) as e:
                logging.warning(
                    f"Error comparing dates for task '{task.get('task_name')}': {e}"
                )

        return filtered_tasks
