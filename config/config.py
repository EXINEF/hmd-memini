from enum import Enum


class TaskIntent(Enum):
    ADD_TASK = "add_task"
    COMPLETE_TASK = "complete_task"
    DELETE_TASK = "delete_task"
    RETRIEVE_TASKS = "retrieve_tasks"
    UPDATE_TASK = "update_task"


class TaskAction(Enum):
    ADD_TASK = "add_task"
    COMPLETE_TASK = "complete_task"
    DELETE_TASK = "delete_task"
    RETRIEVE_TASKS = "retrieve_tasks"
    UPDATE_TASK = "update_task"

    TASK_ALREADY_EXISTS = "task_already_exists"
    REQ_TASK_NAME = "req_task_name"
    REQ_TASK_UPDATE_DETAILS = "req_task_update_details"
    TASK_NOT_FOUND = "task_not_found"
    CONFIRM_DELETE_TASK = "confirm_delete_task"
    MULTIPLE_TASKS_FOUND = "multiple_tasks_found"


EXECUTABLE_ACTIONS = [
    TaskAction.ADD_TASK.value,
    TaskAction.COMPLETE_TASK.value,
    TaskAction.CONFIRM_DELETE_TASK.value,
    TaskAction.DELETE_TASK.value,
    TaskAction.RETRIEVE_TASKS.value,
    TaskAction.UPDATE_TASK.value,
]

TASK_INTENTS = {
    "add_task": "Add a new task",
    "update_task": "Update an existing task",
    "complete_task": "Mark a task as complete",
    "delete_task": "Delete a task",
    "retrieve_tasks": "Retrieve all tasks",
}

ALLOWED_MULTIPLE_INTENTS = {"add_task", "complete_task"}

ERROR_INTENTS = {"fallback": "Default fallback for unrecognized input"}

ALL_INTENTS = {**TASK_INTENTS, **ERROR_INTENTS}

MAX_HISTORICAL_CONTEXT_LENGTH = 2
MAX_NEW_TOKENS = 1024
