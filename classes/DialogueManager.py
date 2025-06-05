import logging
from typing import Dict, Any, Tuple
from classes.Config import Config
from classes.Database import Database
from classes.classes import Task
from config.config import (
    EXECUTABLE_ACTIONS,
    TaskAction,
    TaskIntent,
)
from utils.model_utils import format_tokenize_and_generate_response
from utils.json_utils import extract_json_from_text
import json
import logging
from utils.utils import normalize_date, remove_normalized_due_date
from datetime import datetime


class DialogueManager:

    def __init__(
        self,
        database: Database,
        use_model: bool,
        config=Config,
    ) -> None:
        self.db = database
        self.use_model = use_model
        self.config = config
        self.state = {}
        self.active_intents = {}
        self.pending_suggestions = {}

        if self.use_model and not all([self.config.model, self.config.tokenizer]):
            raise ValueError(
                "Model-based DialogueManager requires model, tokenizer, and config."
            )

    def determine_action_with_dm(self, intent: str, slots: dict) -> Dict[str, Any]:
        logging.info(f"\nProcessing intent: {intent} with slots: {slots}\n")

        original_intent = intent

        if intent == "acknowledgment" and self.pending_suggestions:
            logging.info(
                f"Detected response to pending suggestion: {self.pending_suggestions}"
            )
            return self._handle_pending_suggestion_response()

        self._update_state(intent, slots)

        if self.use_model:
            required_action, additional_info = self._determine_action_with_model(
                intent, slots
            )
        else:
            required_action, additional_info = self._determine_action_deterministic(
                intent, slots
            )

        self._track_conversation_context(intent, required_action)

        result = {}
        if required_action in EXECUTABLE_ACTIONS:
            result = self.execute_action(required_action, slots)

        mixed_initiative = self._generate_mixed_initiative_suggestions(
            required_action, intent, slots, additional_info
        )

        if mixed_initiative:
            logging.info(f"Generated mixed-initiative suggestions: {mixed_initiative}")
            self.pending_suggestions = {
                "original_action": required_action,
                "original_intent": intent,
                "original_slots": slots.copy(),
                "suggestions": mixed_initiative,
            }

        response = {
            "action_required": required_action,
            "additional_info": additional_info,
            "original_intent": original_intent,
            "mixed_initiative": mixed_initiative,
        }

        if result:
            response.update(result)

        return response

    def _update_state(self, intent: str, slots: dict) -> None:
        if intent not in self.active_intents:
            self.active_intents[intent] = {}
        if slots:
            for slot_name, slot_value in slots.items():
                if slot_value is not None:
                    self.active_intents[intent][slot_name] = slot_value

    def _determine_action_deterministic(
        self, intent: str, slots: dict
    ) -> Tuple[str, Dict[str, Any]]:

        additional_info = {}

        if intent == TaskIntent.ADD_TASK.value:
            if not slots.get("task_name"):
                return TaskAction.REQ_TASK_NAME.value, additional_info

            task_name = slots.get("task_name")
            matching_tasks = self.db.get_task_by_exact_name(task_name)
            if matching_tasks:
                return TaskAction.TASK_ALREADY_EXISTS.value, additional_info
            else:
                return TaskAction.ADD_TASK.value, additional_info

        if intent == TaskIntent.COMPLETE_TASK.value:
            if not slots.get("task_name"):
                return TaskAction.REQ_TASK_NAME.value, additional_info

            task_name = slots.get("task_name")
            matching_task = self.db.get_task_by_exact_name(task_name)

            if matching_task:
                additional_info["task"] = matching_task
                return TaskAction.COMPLETE_TASK.value, additional_info

            else:
                matching_tasks = self.db.retrieve_tasks_by_name(task_name)

                if not matching_tasks:
                    return TaskAction.TASK_NOT_FOUND.value, additional_info

                if len(matching_tasks) == 1:
                    task = matching_tasks[0]
                    additional_info["task"] = task
                    return TaskAction.COMPLETE_TASK.value, additional_info

                additional_info["matching_tasks"] = matching_tasks
                return TaskAction.MULTIPLE_TASKS_FOUND.value, additional_info

        elif intent == TaskIntent.DELETE_TASK.value:
            if not slots.get("task_name"):
                return TaskAction.REQ_TASK_NAME.value, additional_info

            task_name = slots.get("task_name")
            matching_task = self.db.get_task_by_exact_name(task_name)

            if matching_task:
                additional_info["task"] = matching_task
                if slots.get("confirmation") == True:
                    return TaskAction.DELETE_TASK.value, additional_info
                return TaskAction.CONFIRM_DELETE_TASK.value, additional_info

            else:
                matching_tasks = self.db.retrieve_tasks_by_name(task_name)

                if not matching_tasks:
                    return TaskAction.TASK_NOT_FOUND.value, additional_info

                if len(matching_tasks) == 1:
                    task = matching_tasks[0]
                    additional_info["task"] = task
                    if slots.get("confirmation") == True:
                        return TaskAction.DELETE_TASK.value, additional_info
                    return TaskAction.CONFIRM_DELETE_TASK.value, additional_info

                additional_info["matching_tasks"] = matching_tasks
                return TaskAction.MULTIPLE_TASKS_FOUND.value, additional_info

        elif intent == TaskIntent.RETRIEVE_TASKS.value:

            # Get time period (today, this_week, next_week, this_month, next_month or all)
            time_period = slots.get("time_period")

            additional_info = {}
            if time_period:
                additional_info["time_period"] = time_period
                logging.info(f"Filtering tasks by time period: {time_period}")
                filtered_tasks = self.db.get_tasks_for_period(time_period)
                additional_info["filtered_tasks"] = filtered_tasks

            return TaskAction.RETRIEVE_TASKS.value, additional_info

        elif intent == TaskIntent.UPDATE_TASK.value:
            if not slots.get("task_name"):
                return TaskAction.REQ_TASK_NAME.value, additional_info
            elif not slots.get("new_task_name") and not slots.get("new_due_date"):
                return TaskAction.REQ_TASK_UPDATE_DETAILS.value, additional_info
            else:
                task_name = slots.get("task_name")
                matching_task = self.db.get_task_by_exact_name(task_name)

                if matching_task:
                    additional_info["task"] = matching_task
                    return TaskAction.UPDATE_TASK.value, additional_info

                else:
                    matching_tasks = self.db.retrieve_tasks_by_name(task_name)
                    if not matching_tasks:
                        return TaskAction.TASK_NOT_FOUND.value, additional_info
                    if len(matching_tasks) == 1:
                        task = matching_tasks[0]
                        additional_info["task"] = task
                        return TaskAction.UPDATE_TASK.value, additional_info

                    additional_info["matching_tasks"] = matching_tasks
                    return TaskAction.MULTIPLE_TASKS_FOUND.value, additional_info

        if intent in ["decline_action", "fallback"]:
            return intent, additional_info

        else:
            logging.critical(f"Unknown intent: {intent}")
            return "fallback", additional_info

    def execute_action(self, action: str, slots: dict) -> Dict[str, Any]:
        try:
            if action == TaskAction.ADD_TASK.value:
                task_name, due_date = slots.get("task_name"), slots.get("due_date")

                task = Task(
                    task_name=task_name,
                    due_date=due_date,
                    normalized_due_date=normalize_date(due_date) if due_date else None,
                    completed=False,
                )

                self.db.add_task(task)

                message = f"Task '{task_name}'"
                message += f" with due date: {due_date}" if due_date else ""
                message += " added."
                return {"message": message}

            elif action == TaskAction.COMPLETE_TASK.value:
                task_name = slots.get("task_name")
                matching_task = self.db.get_task_by_exact_name(task_name)

                if not matching_task:
                    logging.critical(
                        f"Task '{task_name}' not found was found in determine action."
                    )
                    return {"error": "Critical error task, was found before now no."}

                self.db.update_task_status(task_name, completed=True)
                self.db.save()

                return {"message": f"Task '{task_name}' marked as completed."}

            elif action == TaskAction.CONFIRM_DELETE_TASK.value:
                task_name = slots.get("task_name")
                return {
                    "message": f"Confirm deletion of task '{task_name}'?",
                    "needs_confirmation": True,
                    "item_to_delete": task_name,
                    "item_type": "task",
                }

            elif action == TaskAction.DELETE_TASK.value:
                task_name = slots.get("task_name")
                matching_task = self.db.get_task_by_exact_name(task_name)

                if not matching_task:
                    logging.critical(
                        f"Task '{task_name}' not found was found in determine action."
                    )
                    return {"error": "Critical error task, was found before now no."}

                self.db.data["tasks"].remove(matching_task)
                self.db.save()
                return {"message": f"Task '{task_name}' deleted."}

            elif action == TaskAction.RETRIEVE_TASKS.value:
                time_period = slots.get("time_period")

                if time_period:
                    tasks = self.db.get_tasks_for_period(time_period)
                    tasks = remove_normalized_due_date(tasks)
                    return {
                        "time_period": time_period,
                        "tasks": tasks,
                    }
                else:
                    tasks = self.db.get_tasks()
                    tasks = remove_normalized_due_date(tasks)
                    return {"tasks": tasks}

            elif action == TaskAction.UPDATE_TASK.value:
                task_name = slots.get("task_name")
                new_task_name = slots.get("new_task_name")
                new_due_date = slots.get("new_due_date")

                if not new_task_name and not new_due_date:
                    return {
                        "error": "Either a new task name or a new due date is required for updating a task."
                    }

                matching_task = self.db.get_task_by_exact_name(task_name)

                if not matching_task:
                    logging.critical(
                        f"Task '{task_name}' not found was found in determine action."
                    )
                    return {"error": "Critical error task, was found before now no."}

                if new_task_name:
                    matching_task["task_name"] = new_task_name

                if new_due_date:
                    matching_task["due_date"] = new_due_date

                self.db.remove_task_by_exact_name(task_name)
                self.db.add_task(
                    Task(
                        task_name=matching_task["task_name"],
                        due_date=matching_task["due_date"],
                        normalized_due_date=normalize_date(matching_task["due_date"]),
                        completed=False,
                    )
                )

                message = f"Task '{task_name}' updated"
                if new_task_name:
                    message += f" with new name: '{new_task_name}'"
                if new_due_date:
                    message += f"{' and' if new_task_name else ' with'} new due date: {new_due_date}"
                message += "."

                return {"message": message}

            else:
                logging.critical(f"Unknown action: {action}")
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logging.critical(f"Error executing action {action}: {e}")
            return {"error": str(e)}

    def _track_conversation_context(self, intent, action):
        context = {"intent": intent, "action": action}
        self.current_context = context

        if intent != "acknowledgment":
            self.pending_suggestions = {}

    def _determine_action_with_model(
        self, intent: str, slots: dict
    ) -> tuple[str, dict]:
        model_input = {
            "intent": intent,
            "slots": slots,
            "state": {"tasks": self.db.get_tasks()},
            "active_intents": self.active_intents,
        }

        if intent in ["fallback", "acknowledgment"]:
            return intent, {}

        prompt_key = f"DM_intents_{intent}"

        logging.info(
            f"Passing model input to DM: {model_input} with prompt key: {prompt_key}"
        )

        dm_output = format_tokenize_and_generate_response(
            self.config.prompts[prompt_key],
            json.dumps(model_input, indent=2),
            self.config,
        )

        try:
            dm_output_json = extract_json_from_text(dm_output)
            logging.info(f"\nDM MODEL OUTPUT:\n{dm_output_json}\n")

            action_required = dm_output_json.get("action_required", "fallback")
            additional_info = dm_output_json.get("additional_info", {})

            return action_required, additional_info

        except Exception as e:
            logging.error(f"Error processing model output: {e}")
            return "fallback", {}

    def _generate_mixed_initiative_suggestions(
        self, action: str, intent: str, slots: dict, additional_info: dict = None
    ) -> dict:
        suggestions = {}
        additional_info = additional_info or {}

        # Get all tasks from the database
        all_tasks = self.db.get_tasks()

        if action == TaskAction.COMPLETE_TASK.value:
            # After completing a task, suggest other pending tasks
            completed_task_name = slots.get("task_name")
            pending_tasks = [
                task for task in all_tasks if not task.get("completed", False)
            ]

            # Don't suggest tasks if we just completed the last one
            if pending_tasks and completed_task_name:
                # Sort by due date (if available)
                pending_tasks_with_dates = [
                    task for task in pending_tasks if task.get("normalized_due_date")
                ]

                if pending_tasks_with_dates:
                    pending_tasks_with_dates.sort(
                        key=lambda x: x.get("normalized_due_date", "9999-12-31")
                    )
                    next_task = pending_tasks_with_dates[0]

                    # Only suggest if it's different from the one we just completed
                    if next_task["task_name"].lower() != completed_task_name.lower():
                        suggestions["suggest_next_task"] = {
                            "task_name": next_task["task_name"],
                            "due_date": next_task.get("due_date"),
                            "normalized_due_date": next_task.get("normalized_due_date"),
                            "suggestion_type": "complete",
                        }
                elif pending_tasks:
                    # If no tasks with dates, suggest the most recently added task
                    next_task = pending_tasks[-1]
                    if next_task["task_name"].lower() != completed_task_name.lower():
                        suggestions["suggest_next_task"] = {
                            "task_name": next_task["task_name"],
                            "suggestion_type": "complete",
                        }

        elif action == TaskAction.ADD_TASK.value:
            # After adding a task with a deadline, suggest adding a related preparatory task
            task_name = slots.get("task_name", "")
            due_date = slots.get("due_date")

            # Check if this task already contains preparation-related words
            preparation_words = [
                "prepare",
                "preparation",
                "prep",
                "get ready",
                "plan",
                "review",
            ]
            is_already_preparatory = any(
                prep_word in task_name.lower() for prep_word in preparation_words
            )

            if (
                due_date
                and not is_already_preparatory
                and (
                    "deadline" in task_name.lower()
                    or "submit" in task_name.lower()
                    or "deliver" in task_name.lower()
                    or "report" in task_name.lower()
                )
            ):
                # Suggest a preparatory task before the deadline
                suggestions["suggest_preparatory_task"] = {
                    "related_to": task_name,
                    "suggested_task": f"prepare for {task_name}",
                    "suggestion_type": "add",
                    "original_due_date": due_date,
                }

        elif action == TaskAction.DELETE_TASK.value:
            # After deleting a task, suggest deleting similar tasks
            task_name = slots.get("task_name", "")

            # Find similar tasks (but not the same task)
            similar_tasks = []

            if task_name:
                # Break down the task name into meaningful parts
                task_words = set(task_name.lower().split())

                for task in all_tasks:
                    # Don't suggest the same task (that was just deleted)
                    if task["task_name"].lower() == task_name.lower():
                        continue

                    # Check for similarity
                    target_task_words = set(task["task_name"].lower().split())
                    common_words = task_words.intersection(target_task_words)
                    meaningful_common_words = [w for w in common_words if len(w) > 2]

                    if meaningful_common_words:
                        similarity_score = len(meaningful_common_words) / max(
                            len(task_words), len(target_task_words)
                        )
                        if similarity_score > 0.2:
                            similar_tasks.append(task)

                if similar_tasks:
                    suggestions["suggest_delete_similar"] = {
                        "similar_tasks": similar_tasks[:2],  # Limit to 2 suggestions
                        "suggestion_type": "delete",
                    }

        elif action == TaskAction.RETRIEVE_TASKS.value:
            # For task retrieval, suggest task management actions based on what they have
            if len(all_tasks) > 3:
                # Find overdue tasks
                today = datetime.now().strftime("%Y-%m-%d")
                overdue_tasks = [
                    task
                    for task in all_tasks
                    if not task.get("completed", False)
                    and task.get("normalized_due_date")
                    and task.get("normalized_due_date") < today
                ]

                if overdue_tasks:
                    suggestions["suggest_overdue"] = {
                        "overdue_tasks": overdue_tasks[:2],  # Limit to 2 suggestions
                        "suggestion_type": "complete",
                    }
                else:
                    # Find the most urgent task (closest due date)
                    pending_tasks = [
                        task
                        for task in all_tasks
                        if not task.get("completed", False)
                        and task.get("normalized_due_date")
                    ]

                    if pending_tasks:
                        # Sort by due date
                        pending_tasks.sort(
                            key=lambda x: x.get("normalized_due_date", "9999-12-31")
                        )
                        most_urgent_task = pending_tasks[0]

                        suggestions["suggest_urgent_task"] = {
                            "task": most_urgent_task,
                            "suggestion_type": "complete",
                        }

        elif action == TaskAction.UPDATE_TASK.value:
            # After updating a task, provide related suggestions
            task_name = slots.get("task_name", "")
            new_task_name = slots.get("new_task_name")
            new_due_date = slots.get("new_due_date")
            task = additional_info.get("task", {})

            # If user pushed back a deadline, suggest creating intermediate milestones
            if new_due_date and task.get("due_date"):
                old_date = normalize_date(task.get("due_date"))
                new_date = normalize_date(new_due_date)

                if old_date and new_date and new_date > old_date:
                    # User pushed back the deadline, suggest creating a milestone
                    suggestions["suggest_milestone"] = {
                        "original_task": task_name,
                        "new_due_date": new_due_date,
                        "old_due_date": task.get("due_date"),
                        "suggestion_type": "add_milestone",
                    }

            # If this looks like a recurring task (has numbers, days, or "weekly"/"monthly" in name)
            recurring_indicators = [
                "weekly",
                "monthly",
                "daily",
                "meeting",
                "report",
                "#",
                "week",
                "session",
            ]
            if new_task_name or task_name:
                task_to_check = new_task_name or task_name
                if any(
                    indicator in task_to_check.lower()
                    for indicator in recurring_indicators
                ):
                    # Find similar tasks that might be part of the same series
                    similar_tasks = []
                    words_to_match = [
                        w
                        for w in task_to_check.lower().split()
                        if len(w) > 3 and w not in recurring_indicators
                    ]

                    for t in all_tasks:
                        if t["task_name"].lower() == task_name.lower():
                            continue
                        if any(
                            word in t["task_name"].lower() for word in words_to_match
                        ):
                            similar_tasks.append(t)

                    if similar_tasks:
                        suggestions["suggest_similar_updates"] = {
                            "similar_tasks": similar_tasks[:1],
                            "current_task": task_name,
                            "suggestion_type": "update_similar",
                        }

        return suggestions

    def _handle_pending_suggestion_response(self) -> Dict[str, Any]:
        """Handle a response to a mixed-initiative suggestion."""
        if not self.pending_suggestions:
            return {
                "action_required": "acknowledgment",
                "additional_info": {},
                "original_intent": "acknowledgment",
            }

        suggestion_type = None
        suggestion_data = None

        # Find the first applicable suggestion
        for key, data in self.pending_suggestions.get("suggestions", {}).items():
            suggestion_type = data.get("suggestion_type")
            suggestion_data = data
            break

        if not suggestion_type:
            return {
                "action_required": "acknowledgment",
                "additional_info": {},
                "original_intent": "acknowledgment",
            }

        # Handle different types of suggestions
        if suggestion_type == "add":
            # Create a preparatory task
            task_name = suggestion_data.get("suggested_task")

            # Calculate a due date one day before the original due date if possible
            original_due_date = suggestion_data.get("original_due_date")
            due_date = None

            if original_due_date:
                # Try to set due date to one day earlier
                if "tomorrow" in original_due_date.lower():
                    due_date = "today"
                elif "friday" in original_due_date.lower():
                    due_date = "Thursday"
                elif "thursday" in original_due_date.lower():
                    due_date = "Wednesday"
                elif "wednesday" in original_due_date.lower():
                    due_date = "Tuesday"
                elif "tuesday" in original_due_date.lower():
                    due_date = "Monday"
                elif "monday" in original_due_date.lower():
                    due_date = "Friday"
                else:
                    due_date = "tomorrow"

            new_slots = {"task_name": task_name, "due_date": due_date}

            result = self.execute_action(TaskAction.ADD_TASK.value, new_slots)

            self.pending_suggestions = {}

            return {
                "action_required": "suggestion_accepted",
                "additional_info": {
                    "suggestion_type": suggestion_type,
                    "task_name": task_name,
                    "due_date": due_date,
                },
                "original_intent": "add_task",
                "message": result.get(
                    "message",
                    f"Added task '{task_name}' {f'with due date: {due_date}' if due_date else ''}.",
                ),
            }

        elif suggestion_type == "complete":
            # Mark a suggested task as complete
            task_name = None

            # Check for different suggestion structures
            if "task_name" in suggestion_data:
                task_name = suggestion_data.get("task_name")
            elif "overdue_tasks" in suggestion_data:
                # Handle overdue task suggestion
                overdue_tasks = suggestion_data.get("overdue_tasks", [])
                if overdue_tasks and len(overdue_tasks) > 0:
                    task_name = overdue_tasks[0].get("task_name")
            elif "next_task" in suggestion_data:
                task_name = suggestion_data.get("next_task", {}).get("task_name")

            if task_name:
                new_slots = {"task_name": task_name}
                result = self.execute_action(TaskAction.COMPLETE_TASK.value, new_slots)

                # Clear the pending suggestion
                self.pending_suggestions = {}

                return {
                    "action_required": "suggestion_accepted",
                    "additional_info": {
                        "suggestion_type": suggestion_type,
                        "task_name": task_name,
                    },
                    "original_intent": "complete_task",
                    "message": result.get(
                        "message", f"Task '{task_name}' marked as completed."
                    ),
                }

        elif suggestion_type == "delete":
            # Delete a suggested task
            similar_tasks = suggestion_data.get("similar_tasks", [])

            if similar_tasks and len(similar_tasks) > 0:
                task_name = similar_tasks[0].get("task_name")

                if task_name:
                    new_slots = {"task_name": task_name, "confirmation": True}
                    result = self.execute_action(
                        TaskAction.DELETE_TASK.value, new_slots
                    )

                    # Clear the pending suggestion
                    self.pending_suggestions = {}

                    return {
                        "action_required": "suggestion_accepted",
                        "additional_info": {
                            "suggestion_type": suggestion_type,
                            "task_name": task_name,
                        },
                        "original_intent": "delete_task",
                        "message": result.get(
                            "message", f"Task '{task_name}' deleted."
                        ),
                    }

        elif suggestion_type == "update_similar":
            # Update a similar task with the same changes
            similar_tasks = suggestion_data.get("similar_tasks", [])
            original_slots = self.pending_suggestions.get("original_slots", {})

            if similar_tasks and len(similar_tasks) > 0 and original_slots:
                task_name = similar_tasks[0].get("task_name")
                new_task_name = original_slots.get("new_task_name")
                new_due_date = original_slots.get("new_due_date")

                if task_name:
                    new_slots = {
                        "task_name": task_name,
                        "new_task_name": new_task_name,
                        "new_due_date": new_due_date,
                    }
                    result = self.execute_action(
                        TaskAction.UPDATE_TASK.value, new_slots
                    )

                    # Clear the pending suggestion
                    self.pending_suggestions = {}

                    return {
                        "action_required": "suggestion_accepted",
                        "additional_info": {
                            "suggestion_type": suggestion_type,
                            "task_name": task_name,
                            "new_task_name": new_task_name,
                            "new_due_date": new_due_date,
                        },
                        "original_intent": "update_task",
                        "message": result.get(
                            "message", f"Task '{task_name}' updated."
                        ),
                    }
